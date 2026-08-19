import unittest
from main_rag_pipeline import run_rag_pipeline

class TestEndToEndRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.test_query = "What is the employee onboarding process?"

    def test_pipeline_execution(self):
        """Test full pipeline execution, citation map generation, and prompt output."""
        output = run_rag_pipeline(self.test_query, top_k_stage1=10, top_k_stage2=3)

        # 1. Assert keys exist in pipeline result dictionary
        self.assertIn("query", output)
        self.assertIn("prompt", output)
        self.assertIn("citations", output)
        self.assertIn("contexts", output)

        # 2. Assert citation map contains expected document keys
        citations = output["citations"]
        self.assertGreater(len(citations), 0, "Citation map should not be empty.")
        self.assertIn("Doc 1", citations, "Doc 1 key missing in citation mapping.")

        # 3. Assert grounded prompt contains critical constraints
        prompt = output["prompt"]
        self.assertIn("CRITICAL INSTRUCTIONS", prompt)
        self.assertIn(self.test_query, prompt)

if __name__ == "__main__":
    unittest.main()
