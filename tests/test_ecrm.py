"""
Unit tests for ECRM Memory Store.

Classification: COMPATIBILITY
Tests the legacy ecrm/ module. Behavior preserved deliberately.
Canonical replacement: researchforge.memory.ecrm
"""
import pytest

pytestmark = pytest.mark.COMPATIBILITY

from ecrm.memory_store import ECRMMemoryStore, MemoryRecord
from ecrm.res_scorer import compute_res, memory_utility
from ecrm.negative_transfer import NTRDetector
from ecrm.embedder import cosine_similarity, embed_text


class TestMemoryStore:
    def setup_method(self):
        self.store = ECRMMemoryStore(dim=384)

    def test_store_and_retrieve(self):
        rec = self.store.store(
            text="Testing batch normalization for convergence",
            outcome={"score": 0.85, "success": True},
        )
        assert rec.record_id is not None
        results = self.store.retrieve("batch normalization convergence", top_k=1)
        assert len(results) == 1
        assert results[0][0].record_id == rec.record_id

    def test_retrieve_empty(self):
        results = self.store.retrieve("anything", top_k=5)
        assert results == []

    def test_store_multiple_and_order(self):
        self.store.store("hypothesis about learning rate", outcome={"score": 0.5, "success": False})
        self.store.store("hypothesis about batch size", outcome={"score": 0.8, "success": True})
        self.store.store("hypothesis about optimizer Adam vs SGD", outcome={"score": 0.9, "success": True})
        results = self.store.retrieve("Adam optimizer learning rate", top_k=2)
        assert len(results) == 2
        # scores should be in descending order
        assert results[0][1] >= results[1][1]

    def test_has_similar_failure(self):
        self.store.store(
            text="Using dropout=0.9 for regularization",
            outcome={"score": 0.2, "success": False},
            failure_flags=["Underfitting"],
        )
        # Very similar query
        assert self.store.has_similar_failure("dropout 0.9 for regularization", threshold=0.4)

    def test_consolidation(self):
        for i in range(5):
            self.store.store(f"Old stale record {i}", outcome={"score": 0.0, "success": False})
        # Force retention_score to 0 by setting very high threshold
        self.store._retain_threshold = 999.0
        removed = self.store.consolidate()
        assert removed == 5

    def test_stats(self):
        self.store.store("success experiment", outcome={"score": 0.9, "success": True})
        self.store.store("failed experiment", outcome={"score": 0.1, "success": False})
        s = self.store.stats()
        assert s["total_records"] == 2
        assert s["successful"] == 1
        assert s["failed"] == 1

    def test_neutral_run_is_not_retained(self):
        record = self.store.store(
            "neutral repeated experiment", outcome={"score": 0.5, "success": True, "baseline": 0.5}
        )
        assert len(self.store) == 0
        assert record.embedding is None

    def test_hash_embedding_preserves_token_similarity(self):
        related = cosine_similarity(
            embed_text("digits SVC regularization"), embed_text("digits SVC tuning")
        )
        unrelated = cosine_similarity(
            embed_text("digits SVC regularization"), embed_text("satellite segmentation transformer")
        )
        assert related > unrelated

    def test_contextual_retrieval_downweights_incompatible_success(self):
        self.store.store(
            "use focal loss for an imbalanced ECG task",
            outcome={"score": 0.92, "success": True, "baseline": 0.70},
            context={"domain": "healthcare", "task_id": "ecg", "data_regime": "imbalanced"},
        )
        self.store.store(
            "avoid focal loss because it hurt balanced image classification",
            outcome={"score": 0.45, "success": False, "baseline": 0.75},
            failure_flags=["LowPerformance"],
            context={"domain": "vision", "task_id": "images", "data_regime": "balanced"},
        )
        matches = self.store.retrieve_for_context(
            "focal loss for ECG classification",
            {"domain": "healthcare", "task_id": "ecg", "data_regime": "imbalanced"},
        )
        assert matches[0][0].outcome["success"] is True

    def test_decision_support_penalizes_relevant_negative_evidence(self):
        self.store.store(
            "increase learning rate for unstable ECG training",
            outcome={"score": 0.1, "success": False, "baseline": 0.4},
            failure_flags=["Divergence"],
            context={"domain": "healthcare", "task_id": "ecg"},
        )
        support = self.store.decision_support(
            "increase learning rate for unstable ECG training",
            {"domain": "healthcare", "task_id": "ecg"},
        )
        assert support["negative"] > support["positive"]
        assert support["adjustment"] < 0

    def test_duplicate_procedural_lesson_is_consolidated(self):
        kwargs = {
            "text": "increase SVC regularization on small digits data",
            "outcome": {"score": 0.9, "success": True, "baseline": 0.8},
            "context": {"domain": "vision", "task_id": "digits"},
        }
        first = self.store.store(**kwargs)
        second = self.store.store(**kwargs)
        assert len(self.store) == 1
        assert first.record_id == second.record_id
        assert second.evidence_count == 2

    def test_save_load(self, tmp_path):
        self.store.store("test record", outcome={"score": 0.7, "success": True})
        path = str(tmp_path / "memory.json")
        self.store.save(path)
        store2 = ECRMMemoryStore(dim=384)
        store2.load(path)
        assert len(store2) == 1


class TestRESSCore:
    def test_res_empty_outcomes(self):
        assert compute_res([]) == 0.0

    def test_res_positive_outcome(self):
        res = compute_res([0.8, 0.85, 0.82])
        assert res > 0

    def test_memory_utility(self):
        assert memory_utility(0.9, 0.7) == pytest.approx(0.2, abs=1e-6)
        assert memory_utility(0.6, 0.7) < 0  # worse with memory = negative utility


class TestNTRDetector:
    def setup_method(self):
        self.detector = NTRDetector(threshold=0.3)

    def test_no_memory_uses(self):
        assert self.detector.ntr_for_strategy("s1") == 0.0

    def test_harmful_memory_use(self):
        self.detector.record("s1", used_memory=True, baseline_score=0.8, achieved_score=0.6)
        assert self.detector.ntr_for_strategy("s1") == 1.0
        assert self.detector.is_harmful("s1")

    def test_helpful_memory_use(self):
        self.detector.record("s2", used_memory=True, baseline_score=0.7, achieved_score=0.9)
        assert self.detector.ntr_for_strategy("s2") == 0.0
        assert not self.detector.is_harmful("s2")

    def test_mixed_uses(self):
        self.detector.record("s3", used_memory=True, baseline_score=0.7, achieved_score=0.9)
        self.detector.record("s3", used_memory=True, baseline_score=0.7, achieved_score=0.6)
        ntr = self.detector.ntr_for_strategy("s3")
        assert ntr == pytest.approx(0.5, abs=1e-6)
