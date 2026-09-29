"""Run the target-input helper the way a scientist runs it.

The module used ``os.path.relpath`` and ``sys.stderr`` without importing
either, so both the success path and the error path raised NameError. The
adapter import test could not see it, because that test globs ``adapters/*.py``
and this module sits beside the package root.

These tests call ``main`` with argv, which is the only way the two broken lines
are reached. Importing the module is not enough, because a bare name resolves
at call time.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from claude_binder.make_target_inputs import main


TWO_RESIDUE_PDB = """\
ATOM      1  N   MET A   1      11.104  13.207  10.000  1.00 20.00           N
ATOM      2  CA  MET A   1      12.560  13.207  10.000  1.00 20.00           C
ATOM      3  C   MET A   1      13.100  14.600  10.000  1.00 20.00           C
ATOM      4  O   MET A   1      12.400  15.600  10.000  1.00 20.00           O
ATOM      5  N   ALA A   2      14.400  14.700  10.000  1.00 20.00           N
ATOM      6  CA  ALA A   2      15.100  16.000  10.000  1.00 20.00           C
ATOM      7  C   ALA A   2      16.600  15.800  10.000  1.00 20.00           C
ATOM      8  O   ALA A   2      17.100  14.700  10.000  1.00 20.00           O
END
"""


# Author numbering distinguishes 52 from 52A. `binder_metrics` carries the
# insertion code through `Residue.label`, so the helper has to as well.
INSERTION_CODE_PDB = """\
ATOM      1  CA  ALA A  52       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  ALA A  52A     10.000   0.000   0.000  1.00 20.00           C
ATOM      3  CA  ALA A  53      30.000   0.000   0.000  1.00 20.00           C
END
"""


class MakeTargetInputsCliTests(unittest.TestCase):
    def test_a_supplied_structure_and_site_produce_a_relative_residue_map(self) -> None:
        """The success path writes the pointer with os.path.relpath."""
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            structure = work / "target.pdb"
            structure.write_text(TWO_RESIDUE_PDB, encoding="utf-8")
            residue_map = work / "residue_map.json"
            site = work / "site.json"

            code = main(
                [
                    "--structure", str(structure),
                    "--chain", "A",
                    "--out", str(residue_map),
                    "--site-out", str(site),
                    "--surface-residues", "A:1,A:2",
                ]
            )

            self.assertEqual(code, 0)
            block = json.loads(site.read_text(encoding="utf-8"))
            # Relative, so the site block survives being moved with its map.
            self.assertEqual(block["residue_map_path"], "residue_map.json")
            self.assertEqual(block["reference_contact_residues"], ["A:1", "A:2"])

    def test_a_missing_structure_reports_the_path_and_returns_one(self) -> None:
        """The error path writes to sys.stderr and does not raise."""
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            code = main(
                [
                    "--structure", str(work / "absent.pdb"),
                    "--chain", "A",
                    "--out", str(work / "residue_map.json"),
                ]
            )
            self.assertEqual(code, 1)

    def test_explicit_site_refuses_invalid_contact_cutoffs_before_writing(self) -> None:
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as raw:
                work = Path(raw)
                structure = work / "target.pdb"
                structure.write_text(TWO_RESIDUE_PDB, encoding="utf-8")
                residue_map = work / "residue-map.json"
                site = work / "site.json"

                code = main(
                    [
                        "--structure", str(structure),
                        "--chain", "A",
                        "--surface-residues", "A:1",
                        "--contact-cutoff-angstrom", value,
                        "--out", str(residue_map),
                        "--site-out", str(site),
                    ]
                )

                self.assertEqual(code, 1)
                self.assertFalse(residue_map.exists())
                self.assertFalse(site.exists())

    def test_no_supplied_site_refuses_rather_than_choosing_one(self) -> None:
        """Choosing the site is the scientist's decision, so the helper stops."""
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            structure = work / "target.pdb"
            structure.write_text(TWO_RESIDUE_PDB, encoding="utf-8")
            code = main(
                [
                    "--structure", str(structure),
                    "--chain", "A",
                    "--out", str(work / "residue_map.json"),
                ]
            )
            self.assertEqual(code, 1)

    def test_map_only_writes_the_map_without_requiring_or_writing_a_site(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            structure = work / "target.pdb"
            structure.write_text(TWO_RESIDUE_PDB, encoding="utf-8")
            residue_map = work / "residue_map.json"
            default_site = residue_map.with_suffix(".site.json")

            code = main(
                [
                    "--structure", str(structure),
                    "--chain", "A",
                    "--out", str(residue_map),
                    "--map-only",
                ]
            )

            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(residue_map.read_text(encoding="utf-8")),
                {
                    "schema_version": 1,
                    "source_to_cleaned": {"A:1": "A:1", "A:2": "A:2"},
                },
            )
            self.assertFalse(default_site.exists())

    def test_map_only_refuses_site_options_instead_of_silently_ignoring_them(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            structure = work / "target.pdb"
            structure.write_text(TWO_RESIDUE_PDB, encoding="utf-8")
            residue_map = work / "residue_map.json"
            site = work / "site.json"

            code = main(
                [
                    "--structure", str(structure),
                    "--chain", "A",
                    "--out", str(residue_map),
                    "--site-out", str(site),
                    "--map-only",
                ]
            )

            self.assertEqual(code, 1)
            self.assertFalse(residue_map.exists())
            self.assertFalse(site.exists())


    def test_an_insertion_code_survives_the_explicit_site(self) -> None:
        """A residue label carrying an insertion code has to reach the site block.

        The surface sort read the number with `int(label.split(":")[1])`, so
        `A:52A` raised a bare ValueError and the command printed a traceback
        rather than a refusal. Every residue pattern in the tree accepts an
        insertion code, so there was nothing to refuse.
        """
        with tempfile.TemporaryDirectory() as raw:
            work = Path(raw)
            structure = work / "target.pdb"
            structure.write_text(INSERTION_CODE_PDB, encoding="utf-8")
            site = work / "site.json"

            code = main(
                [
                    "--structure", str(structure),
                    "--chain", "A",
                    "--out", str(work / "residue_map.json"),
                    "--site-out", str(site),
                    "--surface-residues", "A:52,A:52A",
                ]
            )

            self.assertEqual(code, 0)
            block = json.loads(site.read_text(encoding="utf-8"))
            # 52 sorts before 52A, which is the order the deposited file carries.
            self.assertEqual(block["reference_contact_residues"], ["A:52", "A:52A"])


if __name__ == "__main__":
    unittest.main()
