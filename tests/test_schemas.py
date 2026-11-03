import pytest
from pydantic import ValidationError

from swe_agent.schemas import CodeCitation, ImplementationPlan, IssueAnalysis, PlanStep


def _step(path: str) -> dict:
    return {
        "step_number": 1,
        "file_path": path,
        "action": "modify",
        "description": "lowercase the email",
        "rationale": "make lookup case-insensitive",
    }


def test_valid_analysis():
    a = IssueAnalysis.model_validate(
        {
            "summary": "Login is case sensitive",
            "issue_type": "bug",
            "root_cause_hypothesis": "find_user compares emails directly",
            "affected_areas": ["src/auth/login.py"],
            "acceptance_criteria": ["Login works with any email casing"],
            "confidence": 0.8,
        }
    )
    assert a.needs_clarification is False


def test_confidence_must_be_between_0_and_1():
    with pytest.raises(ValidationError):
        IssueAnalysis.model_validate(
            {
                "summary": "x",
                "issue_type": "bug",
                "root_cause_hypothesis": "x",
                "affected_areas": [],
                "acceptance_criteria": ["x"],
                "confidence": 1.5,
            }
        )


@pytest.mark.parametrize(
    "bad_path",
    ["../secrets.txt", "/etc/passwd", "C:\\Windows\\system32", "src/../../x.py", ""],
)
def test_plan_step_rejects_unsafe_paths(bad_path):
    with pytest.raises(ValidationError):
        PlanStep.model_validate(_step(bad_path))


def test_plan_step_normalizes_backslashes():
    step = PlanStep.model_validate(_step("src\\auth\\login.py"))
    assert step.file_path == "src/auth/login.py"


def test_plan_needs_at_least_one_step():
    with pytest.raises(ValidationError):
        ImplementationPlan.model_validate({"goal": "g", "steps": [], "test_strategy": "t"})


def test_citation_label_and_order():
    c = CodeCitation(file_path="a.py", start_line=3, end_line=9)
    assert c.label == "a.py:3-9"
    with pytest.raises(ValidationError):
        CodeCitation(file_path="a.py", start_line=9, end_line=3)