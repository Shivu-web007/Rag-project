import unittest
from stage1_retrieval import get_stage1_candidates

class TestStage1Retrieval(unittest.TestCase):

    def setUp(self):
        self.test_query = "What is the onboarding process?"
        self.top_k = 20

    def test_candidate_retrieval_count_and_structure(self):
        """Test if Stage-1 retrieves documents and matches expected candidate structure."""
        results = get_stage1_candidates(self.test_query, top_k=self.top_k)

        # 1. Assert results are returned as a list
        self.assertIsInstance(results, list, "Retrieval output should be a list.")

        # 2. Assert retrieved candidates count is at most top_k
        self.assertLessEqual(len(results), self.top_k, f"Retrieved candidates should not exceed {self.top_k}")

        # 3. Assert each result tuple contains (Document, score)
        if len(results) > 0:
            doc, score = results[0]
            self.assertTrue(hasattr(doc, "page_content"), "Candidate document must have page_content.")
            self.assertTrue(hasattr(doc, "metadata"), "Candidate document must have metadata.")
            self.assertIsInstance(score, (float, int), "Similarity score should be numeric.")

if __name__ == "__main__":
    unittest.main()
