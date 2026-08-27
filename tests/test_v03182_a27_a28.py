"""
tests/test_v03182_a27_a28.py
==============================
Regression tests for:

  A27 — UserMemory.store_interaction() now accepts response_summary param
  A28 — export_config_snapshot() handles pyyaml absence gracefully
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def user_mem(tmp_path):
    from memory.hybrid.hgshm import HGSHM
    from memory.user.user_memory import UserMemory
    h = HGSHM(tmp_path)
    um = UserMemory(hgshm=h, user_id="test_user")
    yield um
    h.close()


# ═════════════════════════════════════════════════════════════════════════════
# A27 — store_interaction() response_summary parameter
# ═════════════════════════════════════════════════════════════════════════════

class TestStoreInteractionResponseSummary:

    def test_signature_has_response_summary(self):
        import inspect
        from memory.user.user_memory import UserMemory
        sig = inspect.signature(UserMemory.store_interaction)
        assert 'response_summary' in sig.parameters, (
            "store_interaction() must have a response_summary parameter (A27)"
        )

    def test_response_summary_default_is_empty_string(self):
        import inspect
        from memory.user.user_memory import UserMemory
        sig = inspect.signature(UserMemory.store_interaction)
        default = sig.parameters['response_summary'].default
        assert default == "", "response_summary default must be empty string"

    def test_backward_compat_no_response_summary(self, user_mem):
        """Calling without response_summary must work identically to before."""
        node = user_mem.store_interaction("what is Python?", True)
        assert "Query: what is Python?" in node.text
        assert "Response:" not in node.text

    def test_response_summary_appended_to_text(self, user_mem):
        """When response_summary is given, it must appear in the stored text."""
        node = user_mem.store_interaction(
            "explain recursion", True,
            response_summary="Recursion is when a function calls itself."
        )
        assert "Response: Recursion is when a function calls itself." in node.text

    def test_response_summary_stored_in_metadata(self, user_mem):
        """response_summary must also appear in node.metadata."""
        node = user_mem.store_interaction(
            "explain recursion", True,
            response_summary="Recursion is when a function calls itself."
        )
        assert "response_summary" in node.metadata
        assert node.metadata["response_summary"].startswith("Recursion")

    def test_response_summary_truncated_to_200(self, user_mem):
        """response_summary must be truncated to 200 chars in both text and metadata."""
        long_summary = "x" * 300
        node = user_mem.store_interaction("q", True, response_summary=long_summary)
        # Metadata truncated
        assert len(node.metadata["response_summary"]) == 200
        # Text truncated (Response: + 200 chars of x)
        assert "x" * 200 in node.text
        assert "x" * 201 not in node.text

    def test_empty_response_summary_not_in_text(self, user_mem):
        """Empty response_summary must not add 'Response:' to the stored text."""
        node = user_mem.store_interaction("q", True, response_summary="")
        assert "Response:" not in node.text

    def test_response_summary_empty_metadata_when_empty(self, user_mem):
        """Empty response_summary → metadata['response_summary'] is empty string."""
        node = user_mem.store_interaction("q", True, response_summary="")
        assert node.metadata.get("response_summary", None) == ""

    def test_accepted_interaction_has_lower_importance(self, user_mem):
        """Accepted interactions get importance=0.6 (corrections get 0.9)."""
        node = user_mem.store_interaction("q", True,
            response_summary="Good answer.")
        assert node.importance == pytest.approx(0.6)

    def test_corrected_interaction_has_higher_importance(self, user_mem):
        """Corrections get importance=0.9."""
        node = user_mem.store_interaction("q", False,
            correction="Actually...", response_summary="Wrong answer.")
        assert node.importance == pytest.approx(0.9)

    def test_response_summary_and_correction_both_in_text(self, user_mem):
        """Both response_summary and correction must appear in the stored text."""
        node = user_mem.store_interaction(
            "what is 2+2?", False,
            correction="It is 4",
            response_summary="The answer is 5"
        )
        assert "Response: The answer is 5" in node.text
        assert "Correction: It is 4" in node.text

    def test_response_summary_ordering(self, user_mem):
        """Response must come before Correction in the stored text."""
        node = user_mem.store_interaction(
            "q", False,
            correction="fix",
            response_summary="original"
        )
        response_idx = node.text.index("Response:")
        correction_idx = node.text.index("Correction:")
        assert response_idx < correction_idx, (
            "Response must appear before Correction in node text"
        )

    def test_accepted_status_in_text(self, user_mem):
        node = user_mem.store_interaction("q", True, response_summary="ok")
        assert "[ACCEPTED]" in node.text

    def test_corrected_status_in_text(self, user_mem):
        node = user_mem.store_interaction("q", False,
            response_summary="wrong", correction="right")
        assert "[CORRECTED]" in node.text

    def test_metadata_still_contains_accepted_flag(self, user_mem):
        """Existing metadata field 'accepted' must still be present."""
        node = user_mem.store_interaction("q", True, response_summary="ans")
        assert "accepted" in node.metadata
        assert node.metadata["accepted"] is True

    def test_metadata_still_contains_user_id(self, user_mem):
        node = user_mem.store_interaction("q", True, response_summary="ans")
        assert node.metadata.get("user_id") == "test_user"

    def test_query_truncated_to_100(self, user_mem):
        long_q = "q" * 150
        node = user_mem.store_interaction(long_q, True)
        # The node text must contain at most 100 q's from the query
        assert "q" * 100 in node.text
        assert "q" * 101 not in node.text or "Response:" in node.text


# ═════════════════════════════════════════════════════════════════════════════
# A28 — export_config_snapshot() YAML failure handling
# ═════════════════════════════════════════════════════════════════════════════

class TestExportConfigSnapshotYamlHandling:

    def test_normal_export_writes_json(self, tmp_path):
        from config.settings import export_config_snapshot
        export_config_snapshot(tmp_path)
        assert (tmp_path / "config_snapshot.json").exists()

    def test_normal_export_writes_yaml(self, tmp_path):
        from config.settings import export_config_snapshot
        export_config_snapshot(tmp_path)
        assert (tmp_path / "config_snapshot.yaml").exists()

    def test_normal_export_returns_dict(self, tmp_path):
        from config.settings import export_config_snapshot
        snap = export_config_snapshot(tmp_path)
        assert isinstance(snap, dict)
        assert "generated_at" in snap

    def test_json_content_valid(self, tmp_path):
        from config.settings import export_config_snapshot
        export_config_snapshot(tmp_path)
        content = (tmp_path / "config_snapshot.json").read_text()
        parsed = json.loads(content)
        assert "adma" in parsed or "profile" in parsed

    def test_yaml_absent_json_still_written(self, tmp_path):
        """If pyyaml is absent, JSON must still be written successfully."""
        import config.settings as cs
        original_ya = cs._YAML_AVAILABLE
        original_yaml = cs.yaml
        cs._YAML_AVAILABLE = False
        cs.yaml = None
        try:
            cs.export_config_snapshot(tmp_path)
        finally:
            cs._YAML_AVAILABLE = original_ya
            cs.yaml = original_yaml
        assert (tmp_path / "config_snapshot.json").exists()

    def test_yaml_absent_logs_warning(self, tmp_path, caplog):
        """A WARNING must be logged when pyyaml is absent."""
        import logging, sys
        import config.settings as cs
        original_ya = cs._YAML_AVAILABLE
        original_yaml = cs.yaml
        original_sysmod = sys.modules.get('yaml')
        cs._YAML_AVAILABLE = False
        cs.yaml = None
        sys.modules['yaml'] = None  # type: ignore  # blocks `import yaml as _yaml`
        try:
            with caplog.at_level(logging.WARNING):
                cs.export_config_snapshot(tmp_path)
        finally:
            cs._YAML_AVAILABLE = original_ya
            cs.yaml = original_yaml
            if original_sysmod is None:
                sys.modules.pop('yaml', None)
            else:
                sys.modules['yaml'] = original_sysmod
        yaml_warnings = [r for r in caplog.records
                         if r.levelno >= logging.WARNING and "yaml" in r.message.lower()]
        assert yaml_warnings, (
            "A WARNING must be logged when pyyaml is absent (A28)"
        )

    def test_yaml_absent_does_not_raise(self, tmp_path):
        """Missing pyyaml must not raise — export completes partially."""
        import config.settings as cs
        original_ya = cs._YAML_AVAILABLE
        original_yaml = cs.yaml
        cs._YAML_AVAILABLE = False
        cs.yaml = None
        try:
            snap = cs.export_config_snapshot(tmp_path)  # must not raise
        finally:
            cs._YAML_AVAILABLE = original_ya
            cs.yaml = original_yaml
        assert isinstance(snap, dict)

    def test_no_output_dir_returns_dict(self):
        """Calling without output_dir must return dict without writing files."""
        from config.settings import export_config_snapshot
        snap = export_config_snapshot(None)
        assert isinstance(snap, dict)

    def test_yaml_available_flag_exists(self):
        import config.settings as cs
        assert hasattr(cs, '_YAML_AVAILABLE')
        assert isinstance(cs._YAML_AVAILABLE, bool)

    def test_pyyaml_in_pyproject(self):
        """pyyaml must be listed as a formal dependency."""
        pyproject = (PROJECT_ROOT / 'pyproject.toml').read_text()
        assert 'pyyaml' in pyproject.lower(), (
            "pyyaml must be listed in pyproject.toml dependencies (A28)"
        )

    def test_yaml_write_exception_handled(self, tmp_path):
        """Any exception during yaml.dump must be caught and logged, not raised."""
        import config.settings as cs

        class BadYaml:
            @staticmethod
            def dump(*a, **kw):
                raise OSError("disk full simulation")

        original_ya = cs._YAML_AVAILABLE
        original_yaml = cs.yaml
        cs._YAML_AVAILABLE = True
        cs.yaml = BadYaml  # type: ignore
        try:
            snap = cs.export_config_snapshot(tmp_path)  # must not raise
        finally:
            cs._YAML_AVAILABLE = original_ya
            cs.yaml = original_yaml
        assert isinstance(snap, dict)
        assert (tmp_path / "config_snapshot.json").exists()
