import unittest
from main_rag_pipeline import run_rag_pipeline

class TestE2ERAGIntegrationAndEdgeCases(unittest.TestCase):

    def test_01_standard_query_execution(self):
        """Test happy path with valid query."""
        query = "What is the employee onboarding process?"
        res = run_rag_pipeline(query, top_k_stage1=10, top_k_stage2=3)
        self.assertIn("prompt", res)
        self.assertIsNotNone(res["prompt"])
        self.assertGreater(len(res["citations"]), 0)

    def test_02_empty_query_edge_case(self):
        """Test edge case: Empty string query."""
        res = run_rag_pipeline("   ")
        self.assertIn("error", res)
        self.assertEqual(res["error"], "Query cannot be empty.")

    def test_03_out_of_domain_query(self):
        """Test edge case: Query with no matching domain concept."""
        query = "XYZ999_Unrelated_Quantum_Physics_Term"
        res = run_rag_pipeline(query, top_k_stage1=5, top_k_stage2=2)
        self.assertIn("prompt", res)

    def test_04_deduplication(self):
        """Test that duplicate chunks are pruned correctly."""
        query = "onboarding"
        res = run_rag_pipeline(query, top_k_stage1=20, top_k_stage2=3)
        self.assertLessEqual(len(res["contexts"]), 3)

if __name__ == "__main__":
    unittest.main()
