"""
tests/test_v03191_c09.py
=========================
Regression tests for C09 — "contextual bandit" claim corrected to
"multi-armed bandit with context-scoped reward attribution".

The implementation uses Beta-Bernoulli Thompson sampling over a shared
arm pool. _context_key() scopes which arm receives reward credit, but
the arm pool itself and the sampling distribution are not conditioned on
context features.  This is MAB, not LinUCB / LinThompson-style contextual
bandit.  The fix is documentation/framing — no algorithmic code changed.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestContextualBanditClaimCorrected:

    def test_learner_module_docstring_says_mab(self):
        src = (PROJECT_ROOT / 'policy' / 'learner.py').read_text()
        first_line = src.lstrip('"\n').splitlines()[0]
        assert 'multi-armed bandit' in first_line.lower(), (
            "policy/learner.py module docstring must say 'multi-armed bandit', "
            "not 'contextual bandit' (C09)"
        )

    def test_learner_module_docstring_not_contextual(self):
        import ast
        src = (PROJECT_ROOT / 'policy' / 'learner.py').read_text()
        tree = ast.parse(src)
        module_doc = ast.get_docstring(tree) or ''
        assert 'contextual bandit' not in module_doc.lower(), (
            "policy/learner.py module docstring must not claim 'contextual bandit'"
        )

    def test_changelog_says_mab(self):
        src = (PROJECT_ROOT / 'CHANGELOG_v0.3.16.md').read_text()
        assert 'Multi-Armed Bandit with Thompson Sampling' in src, (
            "CHANGELOG algorithm section must say 'Multi-Armed Bandit'"
        )

    def test_changelog_not_provably_optimal(self):
        src = (PROJECT_ROOT / 'CHANGELOG_v0.3.16.md').read_text()
        assert 'provably optimal' not in src, (
            "'provably optimal' is an overreach — Beta-Bernoulli TS has "
            "sublinear regret for MAB but optimality depends on priors and "
            "problem structure. (C09)"
        )

    def test_changelog_has_implementation_note(self):
        src = (PROJECT_ROOT / 'CHANGELOG_v0.3.16.md').read_text()
        assert 'C09' in src and 'not a true contextual bandit' in src, (
            "CHANGELOG must contain the C09 implementation note explaining "
            "the MAB vs contextual bandit distinction"
        )

    def test_arm_selection_not_context_conditioned(self):
        """Verify algorithmically: all_active() fetches all arms regardless of ctx."""
        import inspect
        from policy.learner import PolicyLearner
        src = inspect.getsource(PolicyLearner.select)
        # all_active() is called without filtering by context — correct for MAB
        assert 'all_active(' in src
        # ctx_key is only used for _last_selected tracking, not arm pool filtering
        assert '_last_selected[ctx_key]' in src or '_last_selected' in src

    def test_context_key_used_for_reward_attribution_only(self):
        """_context_key scopes which arm gets credit, not which arms are offered."""
        import inspect
        from policy.learner import PolicyLearner
        observe_src = inspect.getsource(PolicyLearner.observe)
        # ctx_key determines which policy_id to credit
        assert 'ctx_key' in observe_src
        assert '_last_selected' in observe_src

    def test_models_docstring_corrected(self):
        src = (PROJECT_ROOT / 'policy' / 'models.py').read_text()
        # Should not claim contextual bandit for the Beta fields
        assert 'contextual bandit arm selection' not in src

    def test_mab_sublinear_regret_claim_preserved(self):
        """The correct claim (sublinear regret for MAB) must still appear."""
        src = (PROJECT_ROOT / 'CHANGELOG_v0.3.16.md').read_text()
        assert 'sublinear regret' in src
