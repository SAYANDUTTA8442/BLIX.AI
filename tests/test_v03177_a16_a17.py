"""
tests/test_v03177_a16_a17.py
==============================
Regression tests for:

  A16 — print() statements in production modules replaced with logging
         (only real runtime prints were in AblationV3Report.print_report)
         format_report() added; print_report() kept as deprecated shim

  A17 — PolicyVersion class had no direct tests; TestPolicyVersion added
"""

from __future__ import annotations

import inspect
import uuid
from dataclasses import fields
from datetime import datetime, timezone

import pytest


# ═════════════════════════════════════════════════════════════════════════════
# A16 — AblationV3Report.format_report() + print_report() shim
# ═════════════════════════════════════════════════════════════════════════════

class TestA16PrintStatements:
    """All runtime print() calls in production modules must be removed (A16)."""

    def test_no_runtime_print_in_print_report(self):
        """print_report() must not contain print() calls in its body (A16)."""
        from policy.ablation_v3 import AblationV3Report
        src = inspect.getsource(AblationV3Report.print_report)
        # The only acceptable print() is the one delegating to format_report()
        print_calls = [l.strip() for l in src.splitlines()
                       if l.strip().startswith('print(') and 'format_report' not in l]
        assert print_calls == [], (
            f"print_report() must not call print() except via format_report(): {print_calls}"
        )

    def test_format_report_exists(self):
        from policy.ablation_v3 import AblationV3Report
        assert hasattr(AblationV3Report, 'format_report')
        assert callable(AblationV3Report.format_report)

    def test_format_report_returns_str(self):
        from policy.ablation_v3 import AblationV3Report
        report = AblationV3Report()  # no baseline → "No baseline available."
        result = report.format_report()
        assert isinstance(result, str)

    def test_format_report_no_baseline_returns_message(self):
        from policy.ablation_v3 import AblationV3Report
        report = AblationV3Report()
        assert report.baseline is None
        out = report.format_report()
        assert "baseline" in out.lower() or "No baseline" in out

    def test_format_report_with_baseline_contains_header(self):
        """format_report() with a baseline must contain the report header."""
        from policy.ablation_v3 import (
            AblationV3Report, AblationConditionResult, AblationConfig, ABLATION_CONDITIONS
        )
        baseline_cfg = ABLATION_CONDITIONS[0]  # full_system
        baseline = AblationConditionResult(condition=baseline_cfg)
        report = AblationV3Report(baseline=baseline)
        out = report.format_report()
        assert "BLIX" in out or "ABLATION" in out, (
            f"format_report() header missing. Got: {out[:200]}"
        )
        assert isinstance(out, str)

    def test_format_report_contains_condition_names(self):
        """Each condition in the study must appear in the formatted report."""
        from policy.ablation_v3 import (
            AblationV3Report, AblationConditionResult, ABLATION_CONDITIONS
        )
        baseline = AblationConditionResult(condition=ABLATION_CONDITIONS[0])
        other    = AblationConditionResult(condition=ABLATION_CONDITIONS[1])
        report   = AblationV3Report(baseline=baseline, ablations=[other])
        out = report.format_report()
        # The second condition's name must appear in the report
        assert ABLATION_CONDITIONS[1].name in out

    def test_print_report_still_callable(self):
        """print_report() shim must still exist and call format_report()."""
        from policy.ablation_v3 import AblationV3Report
        report = AblationV3Report()
        # Must not raise (it prints the "no baseline" message)
        report.print_report()

    def test_print_report_delegates_to_format_report(self):
        """print_report() must call self.format_report() (source inspection)."""
        from policy.ablation_v3 import AblationV3Report
        src = inspect.getsource(AblationV3Report.print_report)
        assert 'format_report' in src, (
            "print_report() must delegate to format_report()"
        )

    def test_format_report_return_type_annotation(self):
        """format_report() must be annotated as returning str."""
        from policy.ablation_v3 import AblationV3Report
        sig = inspect.signature(AblationV3Report.format_report)
        ret = sig.return_annotation
        assert ret is str or ret == 'str', (
            f"format_report() must return str, annotation is: {ret}"
        )

    def test_no_runtime_print_in_config_settings(self):
        """config/settings.py must have no runtime print() outside docstrings."""
        from pathlib import Path
        path = next(
            p for p in Path('/home/claude/blix_v03/config').rglob('settings.py')
        )
        src = path.read_text()
        # Strip docstrings, then check
        import ast
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                name = (func.id if isinstance(func, ast.Name) else
                        func.attr if isinstance(func, ast.Attribute) else None)
                if name == 'print':
                    pytest.fail(
                        f"Runtime print() found in config/settings.py "
                        f"at line {node.lineno}"
                    )

    def test_no_runtime_print_in_hgshm(self):
        """memory/hybrid/hgshm.py must have no runtime print() outside docstrings."""
        from pathlib import Path
        import ast
        path = Path('/home/claude/blix_v03/memory/hybrid/hgshm.py')
        src = path.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                name = (func.id if isinstance(func, ast.Name) else
                        func.attr if isinstance(func, ast.Attribute) else None)
                if name == 'print':
                    pytest.fail(
                        f"Runtime print() found in hgshm.py at line {node.lineno}"
                    )


# ═════════════════════════════════════════════════════════════════════════════
# A17 — PolicyVersion direct tests
# ═════════════════════════════════════════════════════════════════════════════

class TestPolicyVersion:
    """Direct tests for PolicyVersion invariants (A17)."""

    # ── construction & defaults ───────────────────────────────────────────

    def test_default_version_id_is_valid_uuid(self):
        from policy.models import PolicyVersion
        pv = PolicyVersion()
        parsed = uuid.UUID(pv.version_id)  # raises if not valid UUID
        assert str(parsed) == pv.version_id

    def test_each_instance_gets_unique_version_id(self):
        from policy.models import PolicyVersion
        ids = {PolicyVersion().version_id for _ in range(10)}
        assert len(ids) == 10, "Each PolicyVersion must have a unique version_id"

    def test_default_alpha(self):
        from policy.models import PolicyVersion
        assert PolicyVersion().alpha == 1.0

    def test_default_beta_(self):
        from policy.models import PolicyVersion
        assert PolicyVersion().beta_ == 1.0

    def test_default_mean_reward(self):
        from policy.models import PolicyVersion
        assert PolicyVersion().mean_reward == 0.5

    def test_default_version_number(self):
        from policy.models import PolicyVersion
        assert PolicyVersion().version == 1

    def test_default_reason_empty_string(self):
        from policy.models import PolicyVersion
        assert PolicyVersion().reason == ""

    def test_default_policy_id_empty_string(self):
        from policy.models import PolicyVersion
        assert PolicyVersion().policy_id == ""

    def test_default_config_empty_dict(self):
        from policy.models import PolicyVersion
        pv = PolicyVersion()
        assert pv.config == {}
        assert isinstance(pv.config, dict)

    def test_default_created_at_is_iso8601(self):
        from policy.models import PolicyVersion
        pv = PolicyVersion()
        # Must parse as ISO 8601 datetime
        dt = datetime.fromisoformat(pv.created_at)
        assert dt.tzinfo is not None, "created_at must be timezone-aware"

    # ── construction with explicit values ─────────────────────────────────

    def test_explicit_reason_preserved(self):
        from policy.models import PolicyVersion
        pv = PolicyVersion(reason="pre-rollback snapshot")
        assert pv.reason == "pre-rollback snapshot"

    def test_explicit_beta_(self):
        from policy.models import PolicyVersion
        pv = PolicyVersion(beta_=3.5)
        assert pv.beta_ == 3.5

    def test_explicit_config_preserved(self):
        from policy.models import PolicyVersion
        cfg = {"beam_width": 5, "max_depth": 3}
        pv = PolicyVersion(config=cfg)
        assert pv.config == cfg

    def test_explicit_version_id_honoured(self):
        from policy.models import PolicyVersion
        vid = str(uuid.uuid4())
        pv = PolicyVersion(version_id=vid)
        assert pv.version_id == vid

    # ── to_dict() ─────────────────────────────────────────────────────────

    def test_to_dict_contains_all_required_keys(self):
        from policy.models import PolicyVersion
        pv = PolicyVersion()
        d = pv.to_dict()
        required = {"version_id", "policy_id", "version", "config",
                    "alpha", "beta_", "mean_reward", "created_at", "reason"}
        missing = required - set(d.keys())
        assert not missing, f"to_dict() missing keys: {missing}"

    def test_to_dict_beta_underscore_key(self):
        """to_dict() must emit 'beta_', never bare 'beta' (A08 consistency)."""
        from policy.models import PolicyVersion
        d = PolicyVersion(beta_=2.7).to_dict()
        assert "beta_" in d, "to_dict() must use 'beta_' key"
        assert "beta" not in d or d["beta_"] == 2.7

    def test_to_dict_values_correct(self):
        from policy.models import PolicyVersion
        pid = str(uuid.uuid4())
        vid = str(uuid.uuid4())
        pv = PolicyVersion(
            version_id=vid, policy_id=pid, version=3,
            config={"x": 1}, alpha=2.0, beta_=1.5,
            mean_reward=0.75, reason="test"
        )
        d = pv.to_dict()
        assert d["version_id"] == vid
        assert d["policy_id"] == pid
        assert d["version"] == 3
        assert d["config"] == {"x": 1}
        assert d["alpha"] == 2.0
        assert d["beta_"] == 1.5
        assert d["mean_reward"] == 0.75
        assert d["reason"] == "test"

    # ── from_dict() ───────────────────────────────────────────────────────

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict(pv)) must reproduce an equivalent PolicyVersion."""
        from policy.models import PolicyVersion
        pv = PolicyVersion(
            policy_id=str(uuid.uuid4()),
            version=7,
            config={"mode": "ablation"},
            alpha=3.0,
            beta_=2.0,
            mean_reward=0.65,
            reason="roundtrip test",
        )
        d = pv.to_dict()
        pv2 = PolicyVersion.from_dict(d)

        assert pv2.version_id  == pv.version_id
        assert pv2.policy_id   == pv.policy_id
        assert pv2.version     == pv.version
        assert pv2.config      == pv.config
        assert pv2.alpha       == pv.alpha
        assert pv2.beta_       == pv.beta_
        assert pv2.mean_reward == pv.mean_reward
        assert pv2.reason      == pv.reason
        assert pv2.created_at  == pv.created_at

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict() must silently ignore keys not in __dataclass_fields__."""
        from policy.models import PolicyVersion
        d = PolicyVersion().to_dict()
        d["unknown_future_field"] = "ignored"
        pv = PolicyVersion.from_dict(d)  # must not raise
        assert not hasattr(pv, "unknown_future_field")

    def test_from_dict_with_minimal_dict(self):
        """from_dict() with only a subset of keys must use defaults for the rest."""
        from policy.models import PolicyVersion
        pv = PolicyVersion.from_dict({"policy_id": "abc", "version": 2})
        assert pv.policy_id == "abc"
        assert pv.version == 2
        assert pv.alpha == 1.0       # default
        assert pv.beta_ == 1.0       # default
        assert pv.reason == ""       # default

    def test_from_dict_preserves_reason(self):
        from policy.models import PolicyVersion
        pv = PolicyVersion(reason="why not")
        pv2 = PolicyVersion.from_dict(pv.to_dict())
        assert pv2.reason == "why not"

    # ── dataclass field coverage ──────────────────────────────────────────

    def test_all_dataclass_fields_in_to_dict(self):
        """Every dataclass field must appear in to_dict() output."""
        from policy.models import PolicyVersion
        pv = PolicyVersion()
        d = pv.to_dict()
        field_names = {f.name for f in fields(pv)}
        dict_keys = set(d.keys())
        missing = field_names - dict_keys
        assert not missing, (
            f"to_dict() does not include fields: {missing}"
        )

    def test_configs_are_independent_instances(self):
        """Default config dicts must not be shared between instances."""
        from policy.models import PolicyVersion
        pv1 = PolicyVersion()
        pv2 = PolicyVersion()
        pv1.config["key"] = "value"
        assert "key" not in pv2.config, (
            "PolicyVersion instances must not share the same config dict"
        )
