import unittest
import numpy as np
from langchain_core.documents import Document
from stage2_reranker import rerank_candidates

class TestStage2Reranker(unittest.TestCase):

    def setUp(self):
        self.test_query = "What is the employee onboarding process?"
        # Mock Stage-1 candidate documents
        self.mock_stage1_candidates = [
            (Document(page_content="The office cafeteria is located on the second floor.", metadata={"source": "faq.pdf"}), 0.85),
            (Document(page_content="Employee onboarding requires background verification and submitting documents.", metadata={"source": "sop.pdf"}), 0.20),
            (Document(page_content="Annual leave policy allows 18 paid leaves per year.", metadata={"source": "hr.pdf"}), 0.90),
            (Document(page_content="Onboarding step 2 includes IT asset allocation and email account setup.", metadata={"source": "sop.pdf"}), 0.35)
        ]
        self.top_k = 2

    def test_reranker_filtering_and_ranking(self):
        """Test if Stage-2 reranks candidates correctly and limits to top_k."""
        results = rerank_candidates(self.test_query, self.mock_stage1_candidates, top_k=self.top_k)

        # 1. Check returned count matches top_k
        self.assertEqual(len(results), self.top_k, f"Reranker output count should equal top_k ({self.top_k}).")

        # 2. Verify structure of (Document, score)
        doc, score = results[0]
        self.assertTrue(hasattr(doc, "page_content"), "Reranked doc must have page_content.")
        self.assertTrue(isinstance(score, (float, int, np.floating, np.number)), "Reranking score must be numeric.")

        # 3. Verify descending score order (Rank 1 score >= Rank 2 score)
        self.assertGreaterEqual(results[0][1], results[1][1], "Candidates should be sorted in descending order of scores.")

        # 4. Verify most relevant doc is reranked to top
        top_doc_content = results[0][0].page_content
        self.assertIn("onboarding", top_doc_content.lower(), "Top ranked document should be relevant to onboarding.")

if __name__ == "__main__":
    unittest.main()
