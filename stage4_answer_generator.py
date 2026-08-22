def generate_local_answer(question, reranked_results):
    
    if not reranked_results:
        return (
            "Information not available in the provided documents."
        )

    question_lower = question.lower()

    best_result = reranked_results[0]

    content = best_result["chunk"]
    source = best_result["document"]

    # ------------------------------------------------------
    # ONBOARDING
    # ------------------------------------------------------

    if "onboarding" in question_lower:

        if "Employee Onboarding Process" in content:

            return (
                "The employee onboarding process involves the "
                "following steps:\n\n"
                "1. The employee submits the required documents.\n"
                "2. HR verifies the employee information.\n"
                "3. Background verification is completed.\n"
                "4. IT creates the employee account.\n"
                "5. The employee attends the onboarding session.\n\n"
                f"Source: {source}"
            )

    # ------------------------------------------------------
    # LEAVE POLICY
    # ------------------------------------------------------

    if "leave" in question_lower:

        if "Leave Policy" in content:

            return (
                "The leave policy states that employees are "
                "eligible for paid leave according to company "
                "policy. Leave requests must be submitted to "
                "the manager for approval, and the manager "
                "reviews and approves the request.\n\n"
                f"Source: {source}"
            )

    # ------------------------------------------------------
    # IT SUPPORT
    # ------------------------------------------------------

    if "it support" in question_lower or "support" in question_lower:

        if "IT Support" in content or "IT support" in content:

            return (
                "Employees can contact the IT support team "
                "for account and system issues.\n\n"
                f"Source: {source}"
            )

    # ------------------------------------------------------
    # UNKNOWN / NOT AVAILABLE
    # ------------------------------------------------------

    return (
        "Information not available in the provided documents."
    )