from main_rag_pipeline import run_rag_pipeline


# ==========================================================
# RAG EVALUATION
# ==========================================================

test_questions = [

    "What is the onboarding process?",

    "What is the leave policy?",

    "What is IT support?",

    "What documents are required for onboarding?",

    "What is the employee salary?"
]


print("=" * 70)
print("RAG PIPELINE EVALUATION")
print("=" * 70)


for index, question in enumerate(
    test_questions,
    start=1
):

    print("\n")
    print("=" * 70)
    print(f"TEST {index}")
    print("=" * 70)

    print(
        f"\nQuestion: {question}"
    )

    run_rag_pipeline(
        question
    )


print("\n")
print("=" * 70)
print("RAG EVALUATION COMPLETED")
print("=" * 70)