"""
tests/test_v03183_b01_b02_b03.py
===================================
Regression tests for:

  B01 — PolicyStore.update_atomic() atomic read-modify-write
  B02 — pyproject.toml version aligned with blix.__version__
  B03 — No hardcoded /home/claude paths in test files
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════════════
# B01 — PolicyStore.update_atomic()
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def store_with_policy(tmp_path):
    from policy.store import PolicyStore
    from policy.models import PolicyRecord, PolicyType, PolicyDomain
    store = PolicyStore(memory_dir=tmp_path)
    p = PolicyRecord(
        policy_id=str(uuid.uuid4()),
        name="atomic-test",
        policy_type=PolicyType.PLANNER_CONFIG,
        domain=PolicyDomain.SYSTEM,
        config={},
        alpha=1.0,
        beta_=1.0,
    )
    store.save(p)
    yield store, p
    store.close()


class TestUpdateAtomicExists:

    def test_method_exists(self, store_with_policy):
        store, _ = store_with_policy
        assert hasattr(store, 'update_atomic')
        assert callable(store.update_atomic)

    def test_signature(self, store_with_policy):
        import inspect
        store, _ = store_with_policy
        sig = inspect.signature(store.update_atomic)
        params = list(sig.parameters)
        assert 'policy_id' in params
        assert 'mutator' in params

    def test_returns_policy_record(self, store_with_policy):
        from policy.models import PolicyRecord
        store, p = store_with_policy
        result = store.update_atomic(p.policy_id, lambda r: None)
        assert isinstance(result, PolicyRecord)

    def test_get_docstring_mentions_update_atomic(self, store_with_policy):
        from policy.store import PolicyStore
        doc = PolicyStore.get.__doc__ or ""
        assert "update_atomic" in doc, (
            "get() docstring must mention update_atomic() as the preferred "
            "alternative for concurrent use (B01)"
        )


class TestUpdateAtomicSerial:

    def test_mutator_applied(self, store_with_policy):
        store, p = store_with_policy
        result = store.update_atomic(p.policy_id, lambda r: setattr(r, 'alpha', 5.0))
        assert result.alpha == pytest.approx(5.0)

    def test_mutation_persisted(self, store_with_policy):
        store, p = store_with_policy
        store.update_atomic(p.policy_id, lambda r: setattr(r, 'alpha', 7.5))
        reloaded = store.get(p.policy_id)
        assert reloaded.alpha == pytest.approx(7.5)

    def test_incremental_mutation(self, store_with_policy):
        store, p = store_with_policy
        def bump(r): r.alpha += 0.5
        store.update_atomic(p.policy_id, bump)
        store.update_atomic(p.policy_id, bump)
        result = store.get(p.policy_id)
        assert result.alpha == pytest.approx(2.0)  # 1.0 + 0.5 + 0.5

    def test_config_mutation_persisted(self, store_with_policy):
        store, p = store_with_policy
        def set_cfg(r): r.config = {"mode": "updated"}
        store.update_atomic(p.policy_id, set_cfg)
        result = store.get(p.policy_id)
        assert result.config == {"mode": "updated"}

    def test_missing_policy_raises_value_error(self, store_with_policy):
        store, _ = store_with_policy
        with pytest.raises(ValueError, match="not found"):
            store.update_atomic("nonexistent-policy-id", lambda r: None)

    def test_missing_does_not_corrupt_db(self, store_with_policy):
        store, p = store_with_policy
        try:
            store.update_atomic("nonexistent", lambda r: None)
        except ValueError:
            pass
        # Original record must be intact
        reloaded = store.get(p.policy_id)
        assert reloaded is not None
        assert reloaded.alpha == pytest.approx(1.0)


class TestUpdateAtomicRollback:

    def test_mutator_exception_causes_rollback(self, store_with_policy):
        store, p = store_with_policy

        def bad_mutator(r):
            r.alpha = 999.0
            raise RuntimeError("deliberate failure")

        with pytest.raises(RuntimeError):
            store.update_atomic(p.policy_id, bad_mutator)

        reloaded = store.get(p.policy_id)
        assert reloaded.alpha == pytest.approx(1.0), (
            "Rollback must undo partial mutation: alpha must remain 1.0"
        )

    def test_rollback_on_arbitrary_exception(self, store_with_policy):
        store, p = store_with_policy

        def raise_value_error(r):
            r.beta_ = 50.0
            raise ValueError("bad value")

        with pytest.raises(ValueError):
            store.update_atomic(p.policy_id, raise_value_error)

        reloaded = store.get(p.policy_id)
        assert reloaded.beta_ == pytest.approx(1.0)

    def test_rollback_leaves_other_records_intact(self, tmp_path):
        from policy.store import PolicyStore
        from policy.models import PolicyRecord, PolicyType, PolicyDomain

        store = PolicyStore(memory_dir=tmp_path)
        p1 = PolicyRecord(
            policy_id=str(uuid.uuid4()), name="p1",
            policy_type=PolicyType.PLANNER_CONFIG,
            domain=PolicyDomain.SYSTEM, config={}, alpha=1.0, beta_=1.0,
        )
        p2 = PolicyRecord(
            policy_id=str(uuid.uuid4()), name="p2",
            policy_type=PolicyType.PLANNER_CONFIG,
            domain=PolicyDomain.SYSTEM, config={}, alpha=2.0, beta_=1.0,
        )
        store.save(p1)
        store.save(p2)

        try:
            store.update_atomic(p1.policy_id, lambda r: (_ for _ in ()).throw(RuntimeError()))
        except RuntimeError:
            pass

        # p2 must be untouched
        assert store.get(p2.policy_id).alpha == pytest.approx(2.0)
        store.close()


class TestUpdateAtomicConcurrent:

    def test_10_threads_no_lost_updates(self, tmp_path):
        """10 concurrent threads each increment alpha by 0.1 → final must be 1.0."""
        from policy.store import PolicyStore
        from policy.models import PolicyRecord, PolicyType, PolicyDomain

        store = PolicyStore(memory_dir=tmp_path)
        p = PolicyRecord(
            policy_id=str(uuid.uuid4()), name="concurrent",
            policy_type=PolicyType.PLANNER_CONFIG,
            domain=PolicyDomain.SYSTEM, config={}, alpha=0.0, beta_=1.0,
        )
        store.save(p)
        errors = []

        def increment():
            try:
                store.update_atomic(
                    p.policy_id,
                    lambda r: setattr(r, 'alpha', r.alpha + 0.1)
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors, f"Thread errors: {errors}"
        final = store.get(p.policy_id)
        assert final.alpha == pytest.approx(1.0, abs=1e-6), (
            f"Lost updates detected: expected alpha=1.0, got {final.alpha}"
        )
        store.close()

    def test_20_threads_no_lost_updates(self, tmp_path):
        """Stress with 20 threads incrementing beta_ by 0.05 → final=2.0."""
        from policy.store import PolicyStore
        from policy.models import PolicyRecord, PolicyType, PolicyDomain

        store = PolicyStore(memory_dir=tmp_path)
        p = PolicyRecord(
            policy_id=str(uuid.uuid4()), name="stress",
            policy_type=PolicyType.PLANNER_CONFIG,
            domain=PolicyDomain.SYSTEM, config={}, alpha=1.0, beta_=0.0,
        )
        store.save(p)
        errors = []

        def bump_beta():
            try:
                store.update_atomic(
                    p.policy_id,
                    lambda r: setattr(r, 'beta_', r.beta_ + 0.05)
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=bump_beta) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors
        final = store.get(p.policy_id)
        assert final.beta_ == pytest.approx(1.0, abs=1e-6), (
            f"Lost updates: expected beta_=1.0, got {final.beta_}"
        )
        store.close()

    def test_concurrent_different_policies_no_interference(self, tmp_path):
        """Updates to separate policies must not interfere."""
        from policy.store import PolicyStore
        from policy.models import PolicyRecord, PolicyType, PolicyDomain

        store = PolicyStore(memory_dir=tmp_path)
        policies = []
        for i in range(5):
            p = PolicyRecord(
                policy_id=str(uuid.uuid4()), name=f"p{i}",
                policy_type=PolicyType.PLANNER_CONFIG,
                domain=PolicyDomain.SYSTEM, config={}, alpha=0.0, beta_=1.0,
            )
            store.save(p)
            policies.append(p)

        errors = []
        def update_all():
            for p in policies:
                try:
                    store.update_atomic(p.policy_id, lambda r: setattr(r, 'alpha', r.alpha + 1.0))
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=update_all) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors
        for p in policies:
            final = store.get(p.policy_id)
            assert final.alpha == pytest.approx(4.0, abs=1e-6), (
                f"Policy {p.name}: expected alpha=4.0, got {final.alpha}"
            )
        store.close()


# ═════════════════════════════════════════════════════════════════════════════
# B02 — Version alignment
# ═════════════════════════════════════════════════════════════════════════════

class TestVersionAlignment:

    def test_blix_version_exists(self):
        import blix
        assert hasattr(blix, '__version__'), "blix.__version__ must be defined (B02)"

    def test_blix_version_is_string(self):
        import blix
        assert isinstance(blix.__version__, str)

    def test_blix_version_semver_format(self):
        import blix, re
        assert re.match(r'^\d+\.\d+\.\d+', blix.__version__), (
            f"__version__ must follow semver: {blix.__version__!r}"
        )

    def test_pyproject_version_matches_blix_version(self):
        import blix
        pyproject = (PROJECT_ROOT / 'pyproject.toml').read_text()
        expected_line = f'version = "{blix.__version__}"'
        assert expected_line in pyproject, (
            f"pyproject.toml must contain: {expected_line}\n"
            f"blix.__version__={blix.__version__!r}"
        )

    def test_pyproject_not_old_version(self):
        pyproject = (PROJECT_ROOT / 'pyproject.toml').read_text()
        assert 'version = "0.3.13"' not in pyproject, (
            "pyproject.toml must not still say 0.3.13 (B02)"
        )

    def test_blix_init_exists(self):
        init = PROJECT_ROOT / 'blix' / '__init__.py'
        assert init.exists(), "blix/__init__.py must exist (B02)"


# ═════════════════════════════════════════════════════════════════════════════
# B03 — No hardcoded /home/claude paths
# ═════════════════════════════════════════════════════════════════════════════

class TestNoHardcodedPaths:

    # Patterns that indicate a genuine hardcoded path (not docstring references)
    _HARDCODED_PATTERNS = (
        "sys.path.insert(0,",  # combined with /home check below
        "Path('/home/",
        'Path("/home/',
    )

    def _check_file(self, path: Path) -> list[str]:
        """Return lines with genuine hardcoded /home/claude path usage."""
        src = path.read_text()
        return [
            f"{path.name}:{i+1}: {line.rstrip()}"
            for i, line in enumerate(src.splitlines())
            if any((pat in line and "/home/claude" in line) for pat in self._HARDCODED_PATTERNS)
        ]

    def test_test_gap_fixes_no_hardcoded_paths(self):
        hits = self._check_file(PROJECT_ROOT / 'tests' / 'test_gap_fixes.py')
        assert not hits, "Hardcoded paths found:\n" + "\n".join(hits)

    def test_test_v03177_no_hardcoded_paths(self):
        hits = self._check_file(PROJECT_ROOT / 'tests' / 'test_v03177_a16_a17.py')
        assert not hits, "Hardcoded paths found:\n" + "\n".join(hits)

    def test_test_v03182_no_hardcoded_paths(self):
        hits = self._check_file(PROJECT_ROOT / 'tests' / 'test_v03182_a27_a28.py')
        assert not hits, "Hardcoded paths found:\n" + "\n".join(hits)

    def test_no_hardcoded_paths_in_any_test_file(self):
        """Broad scan across all test files for genuine hardcoded paths."""
        test_dir = PROJECT_ROOT / 'tests'
        all_hits = []
        for py_file in sorted(test_dir.glob('*.py')):
            hits = self._check_file(py_file)
            all_hits.extend(hits)
        assert not all_hits, (
            "Genuine hardcoded paths found:\n" + "\n".join(all_hits)
        )

    def test_project_root_resolves_correctly(self):
        """PROJECT_ROOT must point to the blix_v03 directory."""
        assert (PROJECT_ROOT / 'pyproject.toml').exists(), (
            f"PROJECT_ROOT does not point to project root: {PROJECT_ROOT}"
        )
        assert (PROJECT_ROOT / 'policy').is_dir()
        assert (PROJECT_ROOT / 'memory').is_dir()

    def test_test_dir_resolves_correctly(self):
        test_dir = PROJECT_ROOT / 'tests'
        assert test_dir.is_dir()
        assert (test_dir / 'test_v03183_b01_b02_b03.py').exists()
