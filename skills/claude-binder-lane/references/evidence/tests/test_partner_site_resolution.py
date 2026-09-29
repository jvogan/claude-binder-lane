"""Focused coverage for derived target contacts from a named partner complex."""

from __future__ import annotations

import json

import pytest

from claude_binder.adapters import binder_metrics
from claude_binder.make_target_inputs import main as make_target_inputs_main, read_structure
from claude_binder.partner_site import (
    ComplexNotFoundError,
    LocatedComplex,
    PartnerComplexSpec,
    RcsbComplexLocator,
    SiteResolutionError,
    resolve_partner_site,
)


def _atom(serial: int, chain: str, residue: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  ALA {chain}{residue:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
    )


FIXTURE_COMPLEX = "\n".join(
    [
        _atom(1, "A", 1, 0.0),
        _atom(2, "A", 2, 10.0),
        _atom(3, "A", 3, 20.0),
        _atom(4, "B", 1, 4.9),
        _atom(5, "B", 2, 14.9),
        _atom(6, "B", 3, 26.1),
        "END",
    ]
) + "\n"


class _StaticComplexLocator:
    def __init__(self, complex_: LocatedComplex):
        self.complex = complex_

    def locate(self, specification: PartnerComplexSpec) -> LocatedComplex:
        return self.complex


def _fixture_complex() -> LocatedComplex:
    return LocatedComplex(
        entry_id="1ABC",
        structure=binder_metrics.parse_pdb_atoms(FIXTURE_COMPLEX),
        coordinate_text=FIXTURE_COMPLEX,
        target_chains=("A",),
        partner_chains=("B",),
        structure_status="experimental",
        experimental_methods=("X-RAY DIFFRACTION",),
        source="fixture",
        source_url="https://example.test/1ABC",
        partner_entity_description="Partner",
        # This fixture stands for a complex resolved from a deposition record,
        # so it states both checks rather than relying on a default.
        partner_match="exact-entity-name",
        target_match="deposited-entity-accession",
    )


def test_partner_complex_derives_the_exact_heavy_atom_contact_set() -> None:
    resolution = resolve_partner_site(
        PartnerComplexSpec(
            target_accession="P12345",
            partner_name="Partner",
            target_chain="A",
            partner_chains=("B",),
        ),
        _StaticComplexLocator(_fixture_complex()),
    )

    assert resolution.contact_residues == ("A:1", "A:2")
    artifact = resolution.artifact()
    assert artifact["contact_cutoff_angstrom"] == 5.0
    assert artifact["atom_selection"] == "heavy-atoms"
    assert artifact["target_chain"] == "A"
    assert artifact["partner_chains"] == ["B"]
    assert artifact["contact_residue_count"] == 2
    assert artifact["structure_status"] == "experimental"
    assert artifact["confidence"]["level"] == "high"


def test_missing_complex_refuses_instead_of_deriving_a_target_surface(monkeypatch) -> None:
    locator = RcsbComplexLocator()
    monkeypatch.setattr(locator, "_search_entries", lambda specification: [])

    with pytest.raises(ComplexNotFoundError, match="no complex"):
        locator.locate(
            PartnerComplexSpec(target_accession="P12345", partner_name="Partner")
        )


def test_local_partner_complex_writes_campaign_site_and_resolution_artifact(tmp_path) -> None:
    complex_path = tmp_path / "fixture-complex.pdb"
    complex_path.write_text(FIXTURE_COMPLEX, encoding="utf-8")
    residue_map = tmp_path / "residue-map.json"
    site_path = tmp_path / "site.json"
    resolution_path = tmp_path / "site-resolution.json"
    copied_complex = tmp_path / "campaign-target.pdb"

    code = make_target_inputs_main(
        [
            "--out",
            str(residue_map),
            "--site-out",
            str(site_path),
            "--resolution-out",
            str(resolution_path),
            "--complex-out",
            str(copied_complex),
            "--complex-structure",
            str(complex_path),
            "--complex-kind",
            "experimental",
            "--target-accession",
            "P12345",
            "--partner",
            "Partner",
            "--chain",
            "A",
            "--partner-chain",
            "B",
        ]
    )

    assert code == 0
    site = json.loads(site_path.read_text(encoding="utf-8"))
    assert site["reference_contact_residues"] == ["A:1", "A:2"]
    assert site["resolution_artifact_path"] == "site-resolution.json"
    artifact = json.loads(resolution_path.read_text(encoding="utf-8"))
    assert artifact["structure_id"] == "fixture-complex"
    assert artifact["target_chain"] == "A"
    assert artifact["partner_chains"] == ["B"]
    assert artifact["contact_residue_count"] == 2
    assert copied_complex.read_text(encoding="utf-8") == FIXTURE_COMPLEX


def test_partner_complex_accepts_and_records_a_custom_positive_cutoff(tmp_path) -> None:
    complex_path = tmp_path / "fixture-complex.pdb"
    complex_path.write_text(FIXTURE_COMPLEX, encoding="utf-8")
    residue_map = tmp_path / "residue-map.json"
    site_path = tmp_path / "site.json"
    resolution_path = tmp_path / "site-resolution.json"

    code = make_target_inputs_main(
        [
            "--out", str(residue_map),
            "--site-out", str(site_path),
            "--resolution-out", str(resolution_path),
            "--complex-structure", str(complex_path),
            "--complex-kind", "experimental",
            "--target-accession", "P12345",
            "--partner", "Partner",
            "--chain", "A",
            "--partner-chain", "B",
            "--contact-cutoff-angstrom", "6.2",
        ]
    )

    assert code == 0
    site = json.loads(site_path.read_text(encoding="utf-8"))
    artifact = json.loads(resolution_path.read_text(encoding="utf-8"))
    assert site["contact_cutoff_angstrom"] == 6.2
    assert site["reference_contact_residues"] == ["A:1", "A:2", "A:3"]
    assert artifact["contact_cutoff_angstrom"] == 6.2
    assert artifact["reference_contact_residues"] == ["A:1", "A:2", "A:3"]
    assert artifact["contact_cutoff_source"].startswith("operator-supplied")


@pytest.mark.parametrize("cutoff", ["0", "-1", "nan", "inf"])
def test_partner_complex_refuses_a_non_positive_or_non_finite_cutoff(
    tmp_path, cutoff: str
) -> None:
    complex_path = tmp_path / "fixture-complex.pdb"
    complex_path.write_text(FIXTURE_COMPLEX, encoding="utf-8")
    residue_map = tmp_path / "residue-map.json"

    code = make_target_inputs_main(
        [
            "--out", str(residue_map),
            "--complex-structure", str(complex_path),
            "--complex-kind", "experimental",
            "--target-accession", "P12345",
            "--partner", "Partner",
            "--chain", "A",
            "--partner-chain", "B",
            "--contact-cutoff-angstrom", cutoff,
        ]
    )

    assert code == 1
    assert not residue_map.exists()


def test_default_complex_copy_keeps_the_source_format_suffix(tmp_path) -> None:
    """The copied complex has to be readable at the suffix the summary prints.

    The default name was always `.cif`. A PDB complex was copied verbatim under
    that name, and the campaign then pointed `structure_path` at PDB text that
    the mmCIF reader refuses. The existing local-complex test passes an explicit
    `--complex-out`, so it never reached the default.
    """
    complex_path = tmp_path / "fixture-complex.pdb"
    complex_path.write_text(FIXTURE_COMPLEX, encoding="utf-8")
    residue_map = tmp_path / "residue-map.json"

    code = make_target_inputs_main(
        [
            "--out",
            str(residue_map),
            "--complex-structure",
            str(complex_path),
            "--complex-kind",
            "experimental",
            "--target-accession",
            "P12345",
            "--partner",
            "Partner",
            "--chain",
            "A",
            "--partner-chain",
            "B",
        ]
    )

    assert code == 0
    copied = residue_map.with_name("residue-map.reference-complex.pdb")
    assert copied.read_text(encoding="utf-8") == FIXTURE_COMPLEX
    assert not residue_map.with_name("residue-map.reference-complex.cif").exists()
    # The package's own reader accepts the file the campaign is told to point at.
    assert read_structure(copied).chain_ids() == ["A", "B"]


def test_a_caller_supplied_complex_claims_no_check_it_did_not_run(tmp_path) -> None:
    """The receipt for a local complex may not borrow the RCSB wording.

    Nothing reads a deposition record on this route. The accession, the partner
    chain, and the coordinate provenance are all caller assertions, so all three
    basis lines said something untrue and the level said `high`.
    """
    complex_path = tmp_path / "fixture-complex.pdb"
    complex_path.write_text(FIXTURE_COMPLEX, encoding="utf-8")
    residue_map = tmp_path / "residue-map.json"
    resolution_path = tmp_path / "site-resolution.json"

    code = make_target_inputs_main(
        [
            "--out", str(residue_map),
            "--resolution-out", str(resolution_path),
            "--complex-structure", str(complex_path),
            "--complex-kind", "experimental",
            "--target-accession", "P12345",
            "--partner", "Partner",
            "--chain", "A",
            "--partner-chain", "B",
        ]
    )

    assert code == 0
    artifact = json.loads(resolution_path.read_text(encoding="utf-8"))
    basis = artifact["confidence"]["basis"]
    assert artifact["confidence"]["level"] == "medium"
    for claim in (
        "target accession exactly matches the deposited polymer entity",
        "the caller pinned the partner chain and it belongs to a non-target protein entity",
        "the deposited complex reports an experimental method",
    ):
        assert claim not in basis
    assert all("no deposition record was read" in line for line in basis)
    # "user-supplied" is not an experimental method, so the field stays empty.
    assert artifact["experimental_methods"] == []


def test_the_cutoff_citation_states_the_rule_it_cites() -> None:
    """A receipt cannot cite a path its reader does not hold.

    The constant named a file under `notes/campaign/published-source/`, which is
    a different checkout. It resolves nowhere in this package and nowhere in an
    installed copy.
    """
    from claude_binder.partner_site import PUBLISHED_CUTOFF_SOURCE

    assert "notes/campaign" not in PUBLISHED_CUTOFF_SOURCE
    assert "INSILICO.md" in PUBLISHED_CUTOFF_SOURCE
    assert "5 Angstrom" in PUBLISHED_CUTOFF_SOURCE


def test_an_unregistered_provenance_value_is_refused() -> None:
    """A future locator cannot invent a basis line by inventing a match value."""
    with pytest.raises(SiteResolutionError, match="target_match must be one of"):
        LocatedComplex(
            entry_id="1ABC",
            structure=binder_metrics.parse_pdb_atoms(FIXTURE_COMPLEX),
            coordinate_text=FIXTURE_COMPLEX,
            target_chains=("A",),
            partner_chains=("B",),
            structure_status="experimental",
            experimental_methods=(),
            source="fixture",
            source_url=None,
            target_match="verified-by-vibes",
        )


def _icode_atom(serial: int, chain: str, residue: int, icode: str, x: float) -> str:
    line = list(" " * 80)
    def put(value: str, start: int) -> None:
        for offset, char in enumerate(value):
            line[start + offset] = char
    put("ATOM  ", 0)
    put(f"{serial:5d}", 6)
    put(" CA ", 12)
    put("ALA", 17)
    put(chain, 21)
    put(f"{residue:4d}", 22)
    put(icode or " ", 26)
    put(f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}", 30)
    put("  1.00 20.00", 54)
    put(" C", 76)
    return "".join(line).rstrip()


# A:52 and A:52A each contact one partner atom. A:53 contacts none.
INSERTION_CODE_COMPLEX = "\n".join(
    [
        _icode_atom(1, "A", 52, " ", 0.0),
        _icode_atom(2, "A", 52, "A", 10.0),
        _icode_atom(3, "A", 53, " ", 30.0),
        _icode_atom(4, "B", 1, " ", 3.0),
        _icode_atom(5, "B", 2, " ", 13.0),
        "END",
    ]
) + "\n"


def test_partner_contacts_keep_an_insertion_code_distinct(tmp_path) -> None:
    """52 and 52A are different residues, so both belong in the contact set.

    This refused any target chain carrying an insertion code, saying the
    campaign residue format could not represent one. That stopped being true
    when `Residue.label` began carrying the code: `lane.RESIDUE_RE` and
    `target_prep_adapter.RESIDUE_RE` both accept it, and
    `_expand_residue_labels` returns it. The refusal named a limitation that
    no longer existed.
    """
    complex_path = tmp_path / "insertion-code.pdb"
    complex_path.write_text(INSERTION_CODE_COMPLEX, encoding="utf-8")
    residue_map = tmp_path / "residue-map.json"
    site_path = tmp_path / "site.json"

    code = make_target_inputs_main(
        [
            "--out", str(residue_map),
            "--site-out", str(site_path),
            "--complex-structure", str(complex_path),
            "--complex-kind", "experimental",
            "--target-accession", "P12345",
            "--partner", "Partner",
            "--chain", "A",
            "--partner-chain", "B",
        ]
    )

    assert code == 0
    site = json.loads(site_path.read_text(encoding="utf-8"))
    assert site["reference_contact_residues"] == ["A:52", "A:52A"]
    mapping = json.loads(residue_map.read_text(encoding="utf-8"))["source_to_cleaned"]
    assert list(mapping) == ["A:52", "A:52A", "A:53"]
