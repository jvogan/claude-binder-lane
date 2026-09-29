"""Free checks for the user-owned Modal execution route.

These checks inspect local plan data and injected session capabilities. They do
not create a Modal handle, submit a job, download weights, or start a GPU.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .adapters import modal_platform


APPROVAL_BRIEFING = (
    "Jobs run on your own Modal account. Your budget pays.",
    "Two approvals are coming: one to set up the compute, one to submit the job.",
    "Declining either card aborts cleanly. Receipts record where the campaign stopped.",
    "The first job builds the image and fills the weight cache. It is the slow job.",
    "Artifacts survive only through promotion. The workspace is deleted six hours after it goes idle.",
)


@dataclass
class PreflightReport:
    """The results of the free ordered checks."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def record(self, name: str, *, ok: bool, error: str | None = None, warning: str | None = None, notice: str | None = None) -> None:
        self.checks.append({"name": name, "ok": ok, "error": error, "warning": warning, "notice": notice})
        if error:
            self.errors.append(error)
        if warning:
            self.warnings.append(warning)
        if notice:
            self.notices.append(notice)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "notices": self.notices,
            "checks": self.checks,
        }


def _platform(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("modal_platform")
    return value if isinstance(value, Mapping) else {}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_object(path: Path) -> bool:
    try:
        return isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
    except (OSError, json.JSONDecodeError):
        return False


def _adapter_map(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(adapter.get("adapter_id")): adapter
        for adapter in plan.get("adapters", [])
        if isinstance(adapter, Mapping) and isinstance(adapter.get("adapter_id"), str)
    }


def preflight_plan(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    link_ok: bool,
    installed_skills: Sequence[str] | None = None,
    compute: Any | None = None,
    package_resolver: Callable[[str], bool] | None = None,
    environment_resolver: Callable[[str, str], bool] | None = None,
    render_smoke: Callable[[str], str | None] | None = None,
    wait_for_notification: Callable[[], Mapping[str, Any]] | None = None,
    promote_artifacts: Callable[..., Any] | None = None,
    selected_stage_ids: Sequence[str] | None = None,
) -> PreflightReport:
    """Run the free checks needed by the selected Modal work.

    A profile can describe more tools than one materialized or resumed route
    uses.  Only selected stages and their adapters may gate that route.
    """
    report = PreflightReport()
    platform = _platform(config)
    profile = config.get("profile", {})
    profile_id = profile.get("profile_id", "<name>") if isinstance(profile, Mapping) else "<name>"

    report.record(
        "profile-integrity",
        ok=link_ok,
        error=None if link_ok else f"Profile {profile_id} failed the static linker: artifact <A> is read by stage <S> and written by nothing. Fix the profile or pick another.",
    )

    plan_stages = [stage for stage in plan.get("stages", []) if isinstance(stage, Mapping)]
    selected = (
        {str(stage_id) for stage_id in selected_stage_ids}
        if selected_stage_ids is not None
        else {str(stage.get("stage_id")) for stage in plan_stages}
    )
    active_stages = [stage for stage in plan_stages if str(stage.get("stage_id")) in selected]
    adapters = _adapter_map(plan)
    active_modal_stages = [
        stage
        for stage in active_stages
        if isinstance(stage.get("adapter_id"), str)
        and stage["adapter_id"] in adapters
        and modal_platform.uses_modal_platform(adapters[stage["adapter_id"]])
    ]

    inventory = set(installed_skills or platform.get("installed_skills", []))
    roster = platform.get("roster_tools", [])
    roster_ok = isinstance(roster, list)
    if isinstance(roster, list):
        for entry in roster:
            if not isinstance(entry, Mapping):
                roster_ok = False
                continue
            tool = entry.get("tool_id")
            decision = entry.get("decision", "<D>")
            if not isinstance(tool, str) or tool not in inventory:
                roster_ok = False
                report.record(
                    "roster-resolution",
                    ok=True,
                    warning=(
                        f"Optional roster tool {tool} from decision {decision} did not "
                        "resolve in this session. The selected plan does not depend on "
                        "that catalogue entry; choose or install it only if you select it."
                    ),
                )
    if roster_ok:
        report.record("roster-resolution", ok=True)

    targets = config.get("targets", [])
    target_ok = isinstance(targets, list) and bool(targets)
    if target_ok:
        for target in targets:
            if not isinstance(target, Mapping):
                target_ok = False
                continue
            path = Path(str(target.get("structure_path", "")))
            expected = target.get("structure_sha256")
            try:
                if not path.is_file():
                    raise ValueError("file is absent")
                observed = _hash(path)
                if isinstance(expected, str) and expected != observed:
                    raise ValueError("sha256 differs from the recorded target")
            except (OSError, ValueError) as exc:
                target_ok = False
                report.record("target-parse", ok=False, error=f"Target file {path} did not load: {exc}. Re-run target preparation.")
    if target_ok:
        report.record("target-parse", ok=True)

    site_ok = target_ok
    if isinstance(targets, list):
        for target in targets:
            site = target.get("site", {}) if isinstance(target, Mapping) else {}
            path = Path(str(site.get("residue_map_path", ""))) if isinstance(site, Mapping) else Path()
            expected = site.get("residue_map_sha256") if isinstance(site, Mapping) else None
            if not path.is_file() or not _json_object(path) or not isinstance(expected, str) or _hash(path) != expected:
                site_ok = False
                report.record("site-map", ok=False, error=f"No valid site selection at {path}. Run site selection before generating.")
    if site_ok:
        report.record("site-map", ok=True)

    route_ok = True
    for stage in active_stages:
        text = json.dumps(stage, sort_keys=True)
        if "fal.run" in text and platform.get("allow_provider_route") is not True:
            route_ok = False
            report.record("route-scan", ok=False, error=f"Stage {stage.get('stage_id')} calls external endpoint fal.run. This campaign runs on session tools and your own Modal account. Remove the stage or enable the opt-in provider route.")
    if route_ok:
        report.record("route-scan", ok=True)

    bindings_value = platform.get("bindings", [])
    bindings = bindings_value if isinstance(bindings_value, list) else []
    active_bindings: list[Mapping[str, Any]] = []
    selected_binding_indexes: set[int] = set()
    binding_selection_errors: list[str] = []
    for stage in active_modal_stages:
        stage_id = str(stage["stage_id"])
        adapter_id = str(stage["adapter_id"])
        adapter_matches = [
            (index, binding)
            for index, binding in enumerate(bindings)
            if isinstance(binding, Mapping)
            and str(binding.get("adapter_id")) == adapter_id
        ]
        exact_matches = [
            (index, binding)
            for index, binding in adapter_matches
            if str(binding.get("stage_id")) == stage_id
        ]
        if len(exact_matches) == 1:
            chosen = exact_matches[0]
        elif len(exact_matches) > 1:
            binding_selection_errors.append(
                f"Modal stage {stage_id} has more than one exact environment binding "
                f"for adapter {adapter_id}. Keep one binding for that stage."
            )
            continue
        elif len(adapter_matches) == 1:
            # A predictor adapter can serve both screen and rescore stages. The
            # adapter carries the environment and image identity; a unique
            # platform binding supplies the session-side limits for both.
            chosen = adapter_matches[0]
        elif not adapter_matches:
            binding_selection_errors.append(
                f"Modal stage {stage_id} has no environment binding for adapter "
                f"{adapter_id}. Add one binding before paid preflight."
            )
            continue
        else:
            binding_stage_ids = sorted(
                str(binding.get("stage_id")) for _, binding in adapter_matches
            )
            binding_selection_errors.append(
                f"Modal stage {stage_id} has ambiguous shared-adapter bindings for "
                f"{adapter_id}: {binding_stage_ids}. Add an exact binding for the "
                "selected stage."
            )
            continue
        index, binding = chosen
        if index not in selected_binding_indexes:
            selected_binding_indexes.add(index)
            active_bindings.append(binding)

    for error in binding_selection_errors:
        report.record("modal-environment", ok=False, error=error)

    concurrency_ok = True
    for binding in active_bindings:
        if not isinstance(binding, Mapping):
            concurrency_ok = False
            continue
        requested = binding.get("concurrency", 1)
        allowed = binding.get("concurrency_limit", 10)
        if not isinstance(requested, int) or not isinstance(allowed, int) or requested > allowed:
            concurrency_ok = False
            report.record("concurrency-fit", ok=False, error=f"Stage {binding.get('stage_id', '<S>')} requests {requested} concurrent jobs; this session allows {allowed}. Lower the stage concurrency.")
    if concurrency_ok:
        report.record("concurrency-fit", ok=True)

    packages = sorted(
        {
            str(package)
            for binding in active_bindings
            for package in (
                binding.get("packages", [])
                if isinstance(binding.get("packages", []), list)
                else []
            )
        }
    )
    package_ok = True
    for package in packages:
        if package_resolver is None:
            package_ok = False
            report.record(
                "package-resolution",
                ok=False,
                error=(
                    f"Package resolution was not measured for {package}: no "
                    "package_resolver callback is bound to this session. Bind "
                    "package_resolver before paid preflight."
                ),
            )
            continue
        try:
            resolved = package_resolver(str(package))
        except Exception as exc:
            package_ok = False
            report.record(
                "package-resolution",
                ok=False,
                error=(
                    f"Package resolution was not measured for {package}: "
                    f"package_resolver raised {type(exc).__name__}: {exc}. Fix the "
                    "callback, then run preflight again."
                ),
            )
            continue
        if not resolved:
            package_ok = False
            report.record(
                "package-resolution",
                ok=False,
                error=(
                    f"Package {package} did not resolve through the bound "
                    "package_resolver. Install it in the selected environment or "
                    "choose another tool."
                ),
            )
    if package_ok:
        report.record("package-resolution", ok=True)

    account_ok = compute is not None and callable(getattr(getattr(compute, "compute", None), "create", None))
    report.record(
        "modal-account",
        ok=account_ok,
        error=None
        if account_ok
        else (
            "Modal submission capability was not measured: no bound "
            "host.compute.create capability is available in this process. Bind the "
            "Claude Science control-plane host before paid preflight."
        ),
        notice="A connected account is not measured authorization. Run preflight with a bound compute_details reader to measure it, and the first approval card confirms it again before paid dispatch." if account_ok else None,
    )

    binding_ok = (
        isinstance(bindings_value, list)
        and not binding_selection_errors
        and bool(active_bindings)
    )
    if not binding_ok:
        report.record(
            "modal-environment",
            ok=False,
            error=(
                "The plan declares no Modal environment bindings. Select a Modal "
                "profile or materialize one with adapter bindings before paid preflight."
            ),
        )
    for binding in active_bindings:
        if not isinstance(binding, Mapping):
            binding_ok = False
            continue
        adapter = adapters.get(str(binding.get("adapter_id", "")))
        if adapter is None:
            binding_ok = False
            report.record(
                "modal-environment",
                ok=False,
                error=(
                    f"Modal binding {binding.get('adapter_id', '<adapter>')} names no "
                    "adapter in the materialized plan. Recompose and rematerialize "
                    "the plan."
                ),
            )
            continue
        environment = adapter.get("environment", {})
        selected = environment.get(modal_platform.ENVIRONMENT_KEY) if isinstance(environment, Mapping) else None
        if not isinstance(selected, str) or not selected:
            binding_ok = False
            report.record(
                "modal-environment",
                ok=False,
                error=(
                    f"Modal adapter {binding.get('adapter_id', '<adapter>')} declares "
                    f"no {modal_platform.ENVIRONMENT_KEY}. Populate the binding and "
                    "rematerialize the plan."
                ),
            )
            continue
        image = environment.get(modal_platform.IMAGE_KEY) if isinstance(environment, Mapping) else None
        if not isinstance(image, str) or modal_platform.IMAGE_REF_RE.fullmatch(image) is None:
            binding_ok = False
            report.record("image-reference", ok=False, error=f"Image reference {image} is invalid for this provider. Fix the reference before dispatching.")
            continue
        weights = binding.get("weights", {})
        hydrated = isinstance(weights, Mapping) and weights.get("hydrated") is True
        if not hydrated:
            report.record("weight-cache", ok=True, warning=f"Weights for {selected} are not cached. The first job downloads them; expect extra minutes. Cache them with compute-env-setup to skip this next time.")
        if environment_resolver is None:
            binding_ok = False
            report.record(
                "modal-environment",
                ok=False,
                error=(
                    "Modal environment resolution was not measured for "
                    f"{selected} -> {image}: no environment_resolver callback is "
                    "bound to this session. Bind environment_resolver before paid "
                    "preflight."
                ),
            )
            continue
        try:
            environment_resolved = environment_resolver(selected, image)
        except Exception as exc:
            binding_ok = False
            report.record(
                "modal-environment",
                ok=False,
                error=(
                    "Modal environment resolution was not measured for "
                    f"{selected} -> {image}: environment_resolver raised "
                    f"{type(exc).__name__}: {exc}. Fix the callback, then run "
                    "preflight again."
                ),
            )
            continue
        if not environment_resolved:
            binding_ok = False
            report.record(
                "modal-environment",
                ok=False,
                error=(
                    f"Modal environment {selected} did not resolve to image {image} "
                    "through the bound environment_resolver. Use the environment and "
                    "image recorded in the compute_details ledger, then rematerialize "
                    "the plan."
                ),
            )
            continue
        try:
            modal_platform.provider_params(adapter, timeout_seconds=int(binding.get("timeout_seconds", 1)))
        except (TypeError, ValueError) as exc:
            binding_ok = False
            report.record(
                "modal-environment",
                ok=False,
                error=f"Modal binding {binding.get('adapter_id', '<adapter>')} is incomplete: {exc}",
            )
    if binding_ok:
        report.record("modal-environment", ok=True)
        report.record("image-reference", ok=True)

    provider = config.get("provider", {})
    budget = provider.get("budget", {}) if isinstance(provider, Mapping) else {}
    budget_ok = isinstance(budget, Mapping) and isinstance(budget.get("maximum_spend_usd"), (int, float)) and float(budget["maximum_spend_usd"]) > 0
    report.record("budget-ceiling", ok=budget_ok, error=None if budget_ok else "Set a dollar ceiling before generating. The lane asks for it before anything dispatches.")

    promotion = platform.get("promotion_destination")
    promotion_ok = isinstance(promotion, str) and bool(promotion) and promotion != "__REQUIRED__"
    report.record("persistence", ok=promotion_ok, error=None if promotion_ok else "Artifacts vanish when the workspace idles out, six hours after your last action. Set a promotion destination before starting.")
    report.record(
        "completion-notifications",
        ok=callable(wait_for_notification),
        error=None
        if callable(wait_for_notification)
        else "Modal completion needs the kernel wait_for_notification tool. Bind it before starting.",
    )
    report.record(
        "workspace-promotion",
        ok=callable(promote_artifacts),
        error=None
        if callable(promote_artifacts)
        else "Modal completion needs the kernel workspace promotion tool. Bind it before starting.",
    )

    selected_picture_stages = [
        str(stage.get("stage_id"))
        for stage in active_stages
        if any(
            isinstance(output, Mapping) and output.get("kind") == "image"
            for output in stage.get("outputs", [])
        )
    ]
    picture_owner = platform.get("picture_owner_stage")
    if not selected_picture_stages:
        report.record(
            "picture-owner",
            ok=True,
            notice="The selected stages produce no pictures, so no picture owner is required.",
        )
        report.record(
            "render-smoke",
            ok=True,
            notice="The selected stages produce no pictures, so the renderer smoke check was skipped.",
        )
    else:
        if not isinstance(picture_owner, str) or picture_owner not in selected_picture_stages:
            picture_owner = selected_picture_stages[0] if len(selected_picture_stages) == 1 else None
        owner_ok = isinstance(picture_owner, str) and bool(picture_owner)
        report.record(
            "picture-owner",
            ok=owner_ok,
            error=None
            if owner_ok
            else "More than one selected stage produces pictures. Set modal_platform.picture_owner_stage to the renderer to probe.",
        )
        if not owner_ok:
            report.record("render-smoke", ok=False, error="The renderer smoke check has no picture-owning stage.")
        elif render_smoke is None:
            report.record("render-smoke", ok=False, error="Renderer readiness was not measured: no free renderer probe was supplied.")
        else:
            try:
                failure = render_smoke(str(picture_owner))
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
            report.record("render-smoke", ok=failure is None, error=None if failure is None else f"Renderer failed on a smoke frame: {failure}.")

    briefing = platform.get("approval_briefing")
    briefing_matches = isinstance(briefing, list) and tuple(briefing) == APPROVAL_BRIEFING
    report.record(
        "approval-briefing",
        ok=True,
        warning=None
        if briefing_matches
        else "The stored briefing differs from the current wording. The executor prints the canonical approval briefing before dispatch.",
    )
    return report
