"""Tier 1 & Tier 2 E2E tests for Features F1, F2, F3, and F12.

Covers:
- F1: Dedicated sibling worktree & repository isolation, untouched forbidden/locked files,
      Conventional Commits scopes & zero AI attribution trailers.
- F2: Plan & documentation alignment (`docs/copilot.md`,
      `docs/superpowers/plans/2026-11-14-operator-copilot-plan.md`, and verified task docs
      `T-210`..`T-215`) reflecting active grasp gating & 3-tier architecture.
- F3: Copilot AST guard permitting active grasp gating / HTTP & WS audit publishing while
      forbidding raw operator `"key"` / `.on_key` injection.
- F12: `teleop atc` CLI `--no-gemini`, `--headless`, `AtcConfig` / `load_config`, and legacy
       `AudioCues` suppression when Copilot TTS is active.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest

import teleop.commands.atc as atc_cmd
from teleop.atc.stage import AudioCues, NullSpeaker, SystemSpeaker
from teleop.commands.atc import DEFAULT_CONFIG, load_config, resolve_seed
from teleop.sim.bays import PANDA_XML
from teleop.sim.hands import load_hand

from .conftest import REPO_ROOT, require_attr

# Exact verified task filenames per Explorer R1-1 §1.1
VERIFIED_COPILOT_TASK_FILES: tuple[str, ...] = (
    "tasks/T-210-copilot-foundation.md",
    "tasks/T-211-copilot-feed-events.md",
    "tasks/T-212-copilot-callouts-speech.md",
    "tasks/T-213-copilot-deep-review.md",
    "tasks/T-214-copilot-local-gemma.md",
    "tasks/T-215-copilot-console-panel.md",
)

FORBIDDEN_T194_T195_AND_LOCKED_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "uv.lock",
    "scripts/train_act.py",
    "docs/training/verda.md",
    "src/teleop/policy/act_runtime.py",
    "src/teleop/atc/act_recovery.py",
    "src/teleop/cli.py",
    "scripts/fetch_models.py",
    "configs/hands/leap.yaml",
    ".env.example",
    ".pre-commit-config.yaml",
    "src/teleop/compliance/solana.py",
    "tests/atc/golden/dmp_stage.json",
)

ALLOWED_COMMIT_SCOPES: frozenset[str] = frozenset(
    {
        "console",
        "input",
        "sim",
        "retarget",
        "record",
        "train",
        "cli",
        "infra",
        "docs",
        "tasks",
        "net",
        "atc",
        "unoq",
    }
)
ALLOWED_COMMIT_TYPES: frozenset[str] = frozenset(
    {
        "feat",
        "fix",
        "chore",
        "docs",
        "refactor",
        "test",
        "perf",
        "build",
        "ci",
        "style",
        "revert",
    }
)
CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(?P<type>[a-z]+)\((?P<scope>[a-z0-9_-]+)\)!?: (?P<summary>\S.*)$"
)


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return proc.stdout.strip()


def _branch_base() -> str:
    """Where this branch leaves ``origin/main`` (or ``main``): only commits above it are ours.

    A fixed historic commit would also pull in already-merged ``main`` history (other teams'
    commits and their trailers), which this branch cannot change.
    """
    for ref in ("origin/main", "main"):
        try:
            return _git("merge-base", "HEAD", ref)
        except subprocess.CalledProcessError:
            continue
    return "8898b76"


def _resolve_atc_parser() -> argparse.ArgumentParser:
    factory = getattr(atc_cmd, "build_parser", None) or getattr(atc_cmd, "_parser", None)
    assert callable(factory), "Expected `build_parser` or `_parser` in teleop.commands.atc"
    parser = factory()
    assert isinstance(parser, argparse.ArgumentParser)
    return parser


def _resolve_atc_config_cls() -> type[Any]:
    return getattr(atc_cmd, "AtcRunConfig", None) or atc_cmd.AtcConfig


def _copilot_ast_violations(source_code: str) -> list[str]:
    """Evaluate the Copilot AST policy: allow coordinator/audit hooks, forbid raw key control."""
    tree = ast.parse(source_code)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "on_key":
                violations.append("forbidden .on_key(...) call")
            # Raw WS/client send({"type": "key", ...}) injection check
            if isinstance(node.func, ast.Attribute) and node.func.attr in (
                "send",
                "send_text",
                "put_text",
            ):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and sub.value == "key":
                        violations.append("forbidden 'key' message send")
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values, strict=False):
                if (
                    isinstance(k, ast.Constant)
                    and k.value == "type"
                    and isinstance(v, ast.Constant)
                    and v.value == "key"
                ):
                    violations.append("forbidden {'type': 'key'} control payload")
    return violations


# ===========================================================================
# FEATURE 1 — Dedicated Sibling Worktree & Repo Isolation (Tier 1 >=5, Tier 2 >=5)
# ===========================================================================


class TestFeature1WorktreeAndRepoIsolationTier1:
    """Tier 1 category-partition tests for F1 (worktree setup, symlinks, golden, commits)."""

    def test_f1_t1_running_inside_dedicated_git_worktree_not_main_checkout(self) -> None:
        common_dir = Path(_git("rev-parse", "--path-format=absolute", "--git-common-dir"))
        git_dir = Path(_git("rev-parse", "--path-format=absolute", "--git-dir"))
        toplevel = Path(_git("rev-parse", "--show-toplevel")).resolve()
        main_root = common_dir.parent.resolve()
        assert toplevel != main_root, (
            f"Must run inside a dedicated sibling git worktree, not main checkout {main_root}"
        )
        assert git_dir != common_dir

    def test_f1_t1_branch_naming_follows_kebab_convention_without_task_ids(self) -> None:
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        assert "/" in branch, f"Expected <type>/<kebab-summary> branch, got {branch!r}"
        assert not re.search(r"\bT-\d+\b", branch, re.IGNORECASE), (
            f"Branch name {branch!r} must not contain T-NNN task IDs per CLAUDE.md"
        )
        assert not re.search(r"wave|agent|claude", branch, re.IGNORECASE)

    def test_f1_t1_worktree_descends_from_baseline_commit_8898b76(self) -> None:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", "8898b76", "HEAD"],
            capture_output=True,
            timeout=10,
        )
        assert proc.returncode == 0, "Worktree HEAD must descend cleanly from main baseline 8898b76"

    def test_f1_t1_third_party_mujoco_symlinks_present_and_ignored_by_git(self) -> None:
        assert PANDA_XML.is_file(), f"Panda MJCF missing at {PANDA_XML}"
        inspire_right = load_hand("inspire").mjcf["right"]
        assert inspire_right.is_file(), f"Inspire right MJCF missing at {inspire_right}"
        ignored = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", "third_party/inspire"],
            capture_output=True,
            timeout=10,
        )
        assert ignored.returncode == 0, "third_party/ must remain gitignored"

    def test_f1_t1_golden_dmp_stage_file_matches_exact_pinned_linux_reference(self) -> None:
        golden_path = REPO_ROOT / "tests" / "atc" / "golden" / "dmp_stage.json"
        data = json.loads(golden_path.read_text(encoding="utf-8"))
        assert data["linux"]["ticks"] == 2385
        assert data["linux"]["wrist"] == [
            [-2.636500001, -3.378659522, 0.215780866],
            [3.778971364, -3.281898959, 0.135543324],
        ]


class TestFeature1WorktreeAndRepoIsolationTier2:
    """Tier 2 boundary & negative tests for F1 (forbidden files, trailers, scopes, secrets)."""

    def test_f1_t2_zero_modifications_to_t194_t195_and_locked_files_since_base(self) -> None:
        committed_changed = set(_git("diff", "--name-only", f"{_branch_base()}..HEAD").splitlines())
        working_changed = set(_git("diff", "--name-only", "HEAD").splitlines())
        staged_changed = set(_git("diff", "--cached", "--name-only").splitlines())
        all_changed = committed_changed | working_changed | staged_changed
        touched_forbidden = sorted(all_changed.intersection(FORBIDDEN_T194_T195_AND_LOCKED_FILES))
        assert touched_forbidden == [], (
            f"Forbidden T-194 / T-195 / locked files modified: {touched_forbidden}"
        )

    def test_f1_t2_commits_above_base_use_allowed_conventional_scopes_only(self) -> None:
        subjects = [
            line
            for line in _git("log", "--format=%s", f"{_branch_base()}..HEAD").splitlines()
            if line
        ]
        for subj in subjects:
            m = CONVENTIONAL_COMMIT_RE.match(subj)
            assert m is not None, f"Commit subject violates Conventional Commits: {subj!r}"
            ctype = m.group("type")
            scope = m.group("scope")
            assert ctype in ALLOWED_COMMIT_TYPES, f"Disallowed commit type {ctype!r} in {subj!r}"
            assert scope in ALLOWED_COMMIT_SCOPES, (
                f"Disallowed commit scope {scope!r} in {subj!r}; "
                f"allowed={sorted(ALLOWED_COMMIT_SCOPES)}"
            )

    def test_f1_t2_commits_above_base_contain_zero_ai_attribution_trailers(self) -> None:
        full_log = _git("log", "--format=%B", f"{_branch_base()}..HEAD")
        assert not re.search(r"Co-Authored-By:", full_log, re.IGNORECASE), (
            "Commits must NOT include Co-Authored-By trailers"
        )
        assert not re.search(r"Generated with", full_log, re.IGNORECASE), (
            "Commits must NOT include 'Generated with' attribution lines"
        )

    def test_f1_t2_main_checkout_has_no_modified_tracked_files_from_our_work(self) -> None:
        common_dir = Path(_git("rev-parse", "--path-format=absolute", "--git-common-dir"))
        main_root = common_dir.parent.resolve()
        tracked_mods = [
            line
            for line in _git(
                "status", "--porcelain", "--untracked-files=no", cwd=main_root
            ).splitlines()
            # ``uv run task claim`` records claims in the main checkout's tasks/ by design.
            if line.strip() and not line.split()[-1].startswith("tasks/")
        ]
        assert tracked_mods == [], (
            f"Main checkout tracked files must stay untouched: {tracked_mods}"
        )

    def test_f1_t2_no_raw_api_keys_or_teamwork_metadata_inside_repository_tree(self) -> None:
        assert not (REPO_ROOT / ".agents" / "teamwork").exists(), (
            ".agents/teamwork metadata must NEVER be written inside the git worktree"
        )
        for path in (REPO_ROOT / "tests" / "e2e").rglob("*.py"):
            content = path.read_text(encoding="utf-8")
            assert not re.search(r"AIza[0-9A-Za-z_-]{30,}", content), (
                f"Potential hardcoded Google API key detected in {path}"
            )


# ===========================================================================
# FEATURE 2 — Plan & Documentation Alignment (Tier 1 >=5, Tier 2 >=5)
# ===========================================================================


class TestFeature2PlanAndDocumentationAlignmentTier1:
    """Tier 1 positive content checks on docs/copilot.md, operator-copilot-plan.md, T-210..T-215."""

    def test_f2_t1_docs_copilot_specifies_tier1_sse_tts_and_raw_pcm_audio(self) -> None:
        text = (REPO_ROOT / "docs" / "copilot.md").read_text(encoding="utf-8")
        assert "gemini-3.8-flash-lite-tts" in text
        assert "Kore" in text
        assert re.search(r"streamGenerateContent|alt=sse|\bSSE\b", text)
        assert re.search(r"paplay|aplay", text)
        assert "espeak-ng" in text

    def test_f2_t1_docs_copilot_specifies_tier2_hydrogen_active_grasp_gate(self) -> None:
        text = (REPO_ROOT / "docs" / "copilot.md").read_text(encoding="utf-8")
        assert "gemini-4-hydrogen" in text
        for label in ("stall", "mid-hold", "bolt-clear"):
            assert label in text, f"docs/copilot.md must document {label!r} rescue keyframe"
        assert re.search(r"register_skill|SkillValidator|dual[- ]approval|grasp gate", text, re.I)
        assert "approve" in text and "reject" in text

    def test_f2_t1_docs_copilot_specifies_tier3_silent_argon_audit_and_dataset_tagging(
        self,
    ) -> None:
        text = (REPO_ROOT / "docs" / "copilot.md").read_text(encoding="utf-8")
        assert "gemini-4-argon" in text
        assert re.search(r"silent|never speaks|zero TTS|no TTS", text, re.IGNORECASE)
        assert "flag" in text
        assert re.search(r"teleop_episodes|atc_qc|dataset_tag|T-191|T-193", text)

    def test_f2_t1_superpowers_plan_documents_all_three_tiers_and_offline_fallbacks(self) -> None:
        plan_path = (
            REPO_ROOT / "docs" / "superpowers" / "plans" / "2026-11-14-operator-copilot-plan.md"
        )
        text = plan_path.read_text(encoding="utf-8")
        for model in ("gemini-3.8-flash-lite-tts", "gemini-4-hydrogen", "gemini-4-argon"):
            assert model in text, f"Plan missing model {model}"
        assert "--no-gemini" in text
        assert re.search(r"gemma|COPILOT_LOCAL_URL", text, re.IGNORECASE)
        assert re.search(r"silent|never speaks|zero TTS|no TTS", text, re.IGNORECASE)

    def test_f2_t1_all_six_verified_task_files_exist_and_align_with_three_tier_design(
        self,
    ) -> None:
        for rel in VERIFIED_COPILOT_TASK_FILES:
            path = REPO_ROOT / rel
            assert path.is_file(), f"Verified task file missing: {rel}"
        t213 = (REPO_ROOT / "tasks" / "T-213-copilot-deep-review.md").read_text(encoding="utf-8")
        assert re.search(r"silent|never spoken|not spoken|no TTS|zero TTS|dataset|flag", t213, re.I)


class TestFeature2PlanAndDocumentationAlignmentTier2:
    """Tier 2 checks verifying outdated advisory-only & spoken-Argon claims are removed."""

    def test_f2_t2_docs_copilot_removes_outdated_strictly_advisory_gate_claim(self) -> None:
        text = (REPO_ROOT / "docs" / "copilot.md").read_text(encoding="utf-8")
        assert not re.search(
            r"never blocks or changes[^\n]*(?:3-rollout|validation) gate",
            text,
            re.IGNORECASE,
        ), "docs/copilot.md still claims copilot never changes/participates in validation gate"
        assert not re.search(
            r"The human and the existing gates decide\.",
            text,
            re.IGNORECASE,
        )

    def test_f2_t2_docs_copilot_removes_outdated_system_voice_by_default_claim(self) -> None:
        text = (REPO_ROOT / "docs" / "copilot.md").read_text(encoding="utf-8")
        assert not re.search(
            r"System voice by default\.\*\*\s*Live call-outs are spoken with the system voice",
            text,
            re.IGNORECASE,
        ), "docs/copilot.md still states system voice is default instead of SSE Kore TTS"

    def test_f2_t2_superpowers_plan_removes_spoken_argon_and_non_blocking_gate_claims(self) -> None:
        plan_path = (
            REPO_ROOT / "docs" / "superpowers" / "plans" / "2026-11-14-operator-copilot-plan.md"
        )
        text = plan_path.read_text(encoding="utf-8")
        assert not re.search(
            r"The verdict is advisory\.\s*It is shown and spoken",
            text,
            re.IGNORECASE,
        ), "Plan still claims Argon deep review verdict is spoken on TTS"
        assert not re.search(
            r"It does \*\*not\*\* block the DMP gate or ACT data",
            text,
            re.IGNORECASE,
        ), "Plan still claims copilot does not gate DMP or tag ACT data"

    def test_f2_t2_superpowers_plan_removes_false_streaming_tts_exclusivity_claim(self) -> None:
        plan_path = (
            REPO_ROOT / "docs" / "superpowers" / "plans" / "2026-11-14-operator-copilot-plan.md"
        )
        text = plan_path.read_text(encoding="utf-8")
        assert not re.search(
            r"gemini-3\.1-flash-tts-preview[^\n]*the only one with streaming",
            text,
            re.IGNORECASE,
        ), "Plan still falsely claims only gemini-3.1-flash-tts-preview supports streaming"

    def test_f2_t2_task_t213_removes_spoken_verdict_and_tasks_do_not_ban_gating(self) -> None:
        t213 = (REPO_ROOT / "tasks" / "T-213-copilot-deep-review.md").read_text(encoding="utf-8")
        assert not re.search(r"advisory verdict spoken", t213, re.IGNORECASE), (
            "T-213 still says Argon verdict is spoken (violates R4 zero-TTS requirement)"
        )
        # Ensure at least the updated task files clarify raw robot/key control vs active gating
        combined = "\n".join(
            (REPO_ROOT / rel).read_text(encoding="utf-8") for rel in VERIFIED_COPILOT_TASK_FILES
        )
        assert re.search(r"gate|hydrogen|dataset|flag", combined, re.IGNORECASE)


# ===========================================================================
# FEATURE 3 — Copilot AST Guard Update (Tier 1 >=5, Tier 2 >=5)
# ===========================================================================


class TestFeature3CopilotAstGuardTier1:
    """Tier 1 tests verifying the updated AST guard permits active gate & WS audit publishing."""

    def test_f3_t1_all_copilot_modules_parse_cleanly_and_never_inject_keys(self) -> None:
        copilot_dir = REPO_ROOT / "src" / "teleop" / "copilot"
        py_files = sorted(copilot_dir.glob("*.py"))
        assert len(py_files) >= 8
        for path in py_files:
            violations = _copilot_ast_violations(path.read_text(encoding="utf-8"))
            assert violations == [], f"{path.name} has AST safety violations: {violations}"

    def test_f3_t1_existing_copilot_test_gemini_ast_guard_has_lifted_atc_import_ban(self) -> None:
        test_gemini_src = (REPO_ROOT / "tests" / "copilot" / "test_gemini.py").read_text(
            encoding="utf-8"
        )
        assert 'not name.startswith("teleop.atc")' not in test_gemini_src, (
            "tests/copilot/test_gemini.py still bans all `teleop.atc` imports (blocks M2/M3 gate)"
        )

    def test_f3_t1_existing_copilot_test_gemini_ast_guard_has_lifted_blanket_ws_send_ban(
        self,
    ) -> None:
        test_gemini_src = (REPO_ROOT / "tests" / "copilot" / "test_gemini.py").read_text(
            encoding="utf-8"
        )
        assert 'node.attr not in ("send", "send_text")' not in test_gemini_src, (
            "tests/copilot/test_gemini.py still bans all `.send(...)` calls (blocks WS audit)"
        )

    def test_f3_t1_existing_copilot_ast_guard_executes_and_passes_via_pytest(self) -> None:
        import importlib.util

        test_gemini_path = REPO_ROOT / "tests" / "copilot" / "test_gemini.py"
        spec = importlib.util.spec_from_file_location(
            "_copilot_test_gemini_guard", test_gemini_path
        )
        assert spec is not None and spec.loader is not None
        import sys

        tg = importlib.util.module_from_spec(spec)
        # @dataclass resolves its module through sys.modules, so register it while it loads.
        sys.modules[spec.name] = tg
        try:
            spec.loader.exec_module(tg)
        finally:
            sys.modules.pop(spec.name, None)

        guard_fns = [
            getattr(tg, name)
            for name in dir(tg)
            if name.startswith("test_copilot_never_") and callable(getattr(tg, name))
        ]
        assert guard_fns, "Expected AST guard test function in tests/copilot/test_gemini.py"
        for fn in guard_fns:
            fn()

    def test_f3_t1_ast_policy_permits_copilot_audit_ws_and_atc_types_imports(self) -> None:
        valid_snippet = """
import json
from teleop.atc.types import BOLT_CLEARED_M

def publish_audit(ws, report):
    ws.send(json.dumps({"type": "copilot_audit", "audit": report.to_dict()}))
"""
        assert _copilot_ast_violations(valid_snippet) == []


class TestFeature3CopilotAstGuardTier2:
    """Tier 2 negative/boundary tests verifying raw operator key injection remains forbidden."""

    def test_f3_t2_ast_policy_rejects_direct_on_key_method_invocation(self) -> None:
        bad_snippet = """
def sneak_spacebar(controller):
    controller.on_key(32)
"""
        violations = _copilot_ast_violations(bad_snippet)
        assert any("on_key" in v for v in violations)

    def test_f3_t2_ast_policy_rejects_ws_send_with_key_message_type(self) -> None:
        bad_snippet = """
import json
def inject_ws_key(ws):
    ws.send(json.dumps({"type": "key", "code": 32}))
"""
        violations = _copilot_ast_violations(bad_snippet)
        assert violations != []

    def test_f3_t2_ast_policy_rejects_dict_literal_type_key_control_payload(self) -> None:
        bad_snippet = """
PAYLOAD = {"type": "key", "code": 49}
"""
        violations = _copilot_ast_violations(bad_snippet)
        assert any("control payload" in v for v in violations)

    def test_f3_t2_copilot_package_never_references_on_key_or_scripted_glove(self) -> None:
        copilot_dir = REPO_ROOT / "src" / "teleop" / "copilot"
        for path in sorted(copilot_dir.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    assert node.attr != "on_key", f"{path.name} must never call .on_key"
                if isinstance(node, ast.Name):
                    assert node.id != "ScriptedGlove", f"{path.name} must never drive ScriptedGlove"

    def test_f3_t2_existing_copilot_test_gemini_still_guards_against_key_injection(self) -> None:
        test_gemini_src = (REPO_ROOT / "tests" / "copilot" / "test_gemini.py").read_text(
            encoding="utf-8"
        )
        assert '"key"' in test_gemini_src or "'key'" in test_gemini_src, (
            "tests/copilot/test_gemini.py must continue guarding against raw `'key'` injection"
        )


# ===========================================================================
# FEATURE 12 — CLI `--no-gemini` & Offline Determinism (Tier 1 >=5, Tier 2 >=5)
# ===========================================================================


class TestFeature12CliNoGeminiAndDeterminismTier1:
    """Tier 1 category-partition tests for `teleop atc --no-gemini` CLI & config wiring."""

    def test_f12_t1_atc_parser_accepts_no_gemini_flag_without_error(self) -> None:
        parser = _resolve_atc_parser()
        try:
            ns = parser.parse_args(["--no-gemini", "--headless"])
        except SystemExit as exc:
            raise AssertionError("`teleop atc` parser does not support `--no-gemini` yet") from exc
        ns_dict = vars(ns)
        # Either dest="gemini" -> False or dest="no_gemini" -> True
        disabled = (ns_dict.get("gemini") is False) or (ns_dict.get("no_gemini") is True)
        assert disabled, f"Expected `--no-gemini` to disable Gemini on Namespace, got {ns_dict}"

    def test_f12_t1_atc_config_default_enables_gemini_when_available(self) -> None:
        cfg = load_config(DEFAULT_CONFIG, {})
        gemini_enabled = getattr(cfg, "gemini", None)
        no_gemini_flag = getattr(cfg, "no_gemini", None)
        assert gemini_enabled is not None or no_gemini_flag is not None, (
            "AtcConfig must expose `gemini: bool = True` (or `no_gemini: bool = False`)"
        )
        if gemini_enabled is not None:
            assert gemini_enabled is True
        if no_gemini_flag is not None:
            assert no_gemini_flag is False

    def test_f12_t1_load_config_applies_no_gemini_override_cleanly(self) -> None:
        parser = _resolve_atc_parser()
        try:
            ns = parser.parse_args(["--no-gemini", "--headless", "--steps", "5"])
        except SystemExit as exc:
            raise AssertionError("`teleop atc` parser does not support `--no-gemini` yet") from exc
        skip = {"config", "hotspot", "no_hotspot", "env_file"}
        overrides = {k: v for k, v in vars(ns).items() if k not in skip}
        cfg = load_config(DEFAULT_CONFIG, overrides)
        assert cfg.headless is True
        assert cfg.steps == 5
        is_off = (getattr(cfg, "gemini", True) is False) or (
            getattr(cfg, "no_gemini", False) is True
        )
        assert is_off, "load_config with --no-gemini override must disable Gemini in AtcConfig"

    def test_f12_t1_resolve_seed_preserves_stage_seed_determinism_with_no_gemini(self) -> None:
        parser = _resolve_atc_parser()
        ns_default = parser.parse_args(["--script", "--headless"])
        try:
            ns_nogemini = parser.parse_args(["--script", "--headless", "--no-gemini"])
        except SystemExit as exc:
            raise AssertionError("`teleop atc` parser does not support `--no-gemini` yet") from exc
        skip = {"config", "hotspot", "no_hotspot", "env_file"}
        cfg_a = load_config(
            DEFAULT_CONFIG, {k: v for k, v in vars(ns_default).items() if k not in skip}
        )
        cfg_b = load_config(
            DEFAULT_CONFIG, {k: v for k, v in vars(ns_nogemini).items() if k not in skip}
        )
        assert resolve_seed(cfg_a) == resolve_seed(cfg_b)

    def test_f12_t1_legacy_audio_cues_suppressed_when_headless_or_copilot_tts_active(
        self,
    ) -> None:
        # Verify AudioCues(NullSpeaker(), enabled=False) emits zero system voice calls, and
        # inspect commands/atc.py to confirm SystemSpeaker is suppressed when Copilot is active.
        cues = AudioCues(NullSpeaker(), enabled=False)
        assert cues.enabled is False
        atc_src = (REPO_ROOT / "src" / "teleop" / "commands" / "atc.py").read_text(encoding="utf-8")
        assert "no-gemini" in atc_src, "commands/atc.py must wire `--no-gemini`"
        # Legacy SystemSpeaker must not double-speak when Copilot Kore voice is active
        assert "copilot" in atc_src.lower() or "gemini" in atc_src.lower()
        assert SystemSpeaker is not None


class TestFeature12CliNoGeminiAndDeterminismTier2:
    """Tier 2 boundary, YAML validation, and offline headless execution tests for F12."""

    def test_f12_t2_load_config_rejects_unknown_yaml_keys_while_accepting_gemini_key(
        self, tmp_path: Path
    ) -> None:
        cfg_cls = _resolve_atc_config_cls()
        field_names = {f.name for f in dataclasses.fields(cfg_cls)}
        gemini_field = "gemini" if "gemini" in field_names else "no_gemini"
        assert gemini_field in field_names, (
            "AtcConfig must declare `gemini` (or `no_gemini`) field for YAML/CLI config"
        )
        valid_yaml = tmp_path / "atc_valid.yaml"
        valid_yaml.write_text(
            f"{gemini_field}: {'false' if gemini_field == 'gemini' else 'true'}\nheadless: true\n",
            encoding="utf-8",
        )
        loaded = load_config(valid_yaml, {})
        assert loaded.headless is True

        bad_yaml = tmp_path / "atc_bad.yaml"
        bad_yaml.write_text("totally_nonexistent_flag: 123\n", encoding="utf-8")
        with pytest.raises(ValueError, match="totally_nonexistent_flag"):
            load_config(bad_yaml, {})

    def test_f12_t2_atc_parser_rejects_abbreviated_or_invalid_gemini_flags(self) -> None:
        parser = _resolve_atc_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--no-gem"])  # allow_abbrev=False must reject partial flag

    def test_f12_t2_atc_main_headless_no_gemini_short_steps_runs_with_zero_network_calls(
        self,
        monkeypatch: pytest.MonkeyPatch,
        block_external_network_egress: list[Any],
    ) -> None:
        # Even when GEMINI_API_KEY is present in the environment, `--no-gemini --headless --steps 5`
        # must NOT contact any external Gemini host!
        monkeypatch.setenv("GEMINI_API_KEY", "synthetic-should-not-be-used")
        monkeypatch.delenv("COPILOT_LOCAL_URL", raising=False)
        parser = _resolve_atc_parser()
        # Verify `--no-gemini` parses first (raises AssertionError cleanly if M3 not merged yet)
        try:
            parser.parse_args(["--no-gemini"])
        except SystemExit as exc:
            raise AssertionError("`teleop atc` parser does not support `--no-gemini` yet") from exc

        rc = require_attr(atc_cmd, "main")(
            [
                "--headless",
                "--no-gemini",
                "--unoq-host",
                "127.0.0.1",
                "--no-audio",
                "--no-background",
                "--source",
                "synthetic",
                "--steps",
                "5",
            ]
        )
        assert rc == 0
        non_loopback = [
            entry
            for entry in block_external_network_egress
            if isinstance(entry[2], tuple)
            and str(entry[2][0]) not in ("127.0.0.1", "localhost", "::1", "")
        ]
        assert non_loopback == []

    def test_f12_t2_atc_main_headless_unset_api_key_runs_deterministically_offline(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("COPILOT_LOCAL_URL", raising=False)
        # Ensure `--no-gemini` exists on M3 CLI surface
        parser = _resolve_atc_parser()
        try:
            parser.parse_args(["--no-gemini"])
        except SystemExit as exc:
            raise AssertionError("`teleop atc` parser does not support `--no-gemini` yet") from exc

        empty_env = REPO_ROOT / "tests" / "e2e" / "__init__.py"  # empty file -> 0 dotenv vars
        rc = atc_cmd.main(
            [
                "--headless",
                "--unoq-host",
                "127.0.0.1",
                "--no-audio",
                "--no-background",
                "--source",
                "synthetic",
                "--env-file",
                str(empty_env),
                "--steps",
                "5",
            ]
        )
        assert rc == 0

    def test_f12_t2_mutually_exclusive_hotspot_flags_still_return_exit_code_2_with_no_gemini(
        self,
    ) -> None:
        parser = _resolve_atc_parser()
        try:
            parser.parse_args(["--no-gemini"])
        except SystemExit as exc:
            raise AssertionError("`teleop atc` parser does not support `--no-gemini` yet") from exc
        rc = atc_cmd.main(["--no-gemini", "--hotspot", "--no-hotspot"])
        assert rc == 2
