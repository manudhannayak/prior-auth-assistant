import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from coverage_agent import check_coverage  # noqa: E402
from drafting_agent import draft_with_template  # noqa: E402
from extraction_agent import extract_rule_based  # noqa: E402
from graph import run_pipeline  # noqa: E402

SAMPLE_LUMBAR_REFERRAL = """
Patient ID: MRN-12345
Diagnosis: Chronic low back pain

Clinical Notes:
Low back pain for 8 weeks. Completed 6 weeks of physical therapy and NSAIDs
with minimal improvement. Requesting MRI lumbar spine without contrast.
"""

SAMPLE_UNSUPPORTED_REFERRAL = """
Patient ID: MRN-99999
Diagnosis: Acute ankle sprain

Clinical Notes:
Patient twisted ankle playing basketball yesterday. Mild swelling, no
fracture on exam. Requesting MRI lumbar spine without contrast.
"""


def test_extract_rule_based_finds_patient_id_and_procedure():
    result = extract_rule_based(SAMPLE_LUMBAR_REFERRAL)
    assert result.patient_id == "MRN-12345"
    assert result.requested_procedure_code == "72148"
    assert result.backend == "rule_based"


def test_extract_rule_based_handles_missing_procedure():
    result = extract_rule_based("Patient ID: MRN-1\nDiagnosis: Fatigue\nNo specific procedure requested.")
    assert result.requested_procedure_code is None


def test_coverage_no_pa_needed_for_office_visit():
    result = check_coverage("99213", "routine follow-up visit")
    assert result.requires_prior_auth is False
    assert result.recommendation == "no_pa_needed"


def test_coverage_unknown_procedure_code():
    result = check_coverage("00000", "some notes")
    assert result.recommendation == "unknown_procedure"


def test_coverage_approve_draft_when_criteria_documented():
    extracted = extract_rule_based(SAMPLE_LUMBAR_REFERRAL)
    result = check_coverage(extracted.requested_procedure_code, extracted.clinical_notes)
    assert result.requires_prior_auth is True
    # At least the conservative-therapy criterion should be detected as met.
    assert any(result.criteria_evidence.values())


def test_draft_with_template_includes_patient_and_procedure():
    extracted = extract_rule_based(SAMPLE_LUMBAR_REFERRAL).to_dict()
    coverage = check_coverage(extracted["requested_procedure_code"], extracted["clinical_notes"]).to_dict()
    letter = draft_with_template(extracted, coverage)
    assert "MRN-12345" in letter
    assert "MRI lumbar spine" in letter
    assert "review" in letter.lower() and "before submission" in letter.lower()


def test_full_graph_pipeline_end_to_end():
    result = run_pipeline(SAMPLE_LUMBAR_REFERRAL)
    assert result["extracted"]["requested_procedure_code"] == "72148"
    assert result["coverage"]["requires_prior_auth"] is True
    assert "letter" in result["draft_result"]
    assert len(result["draft_result"]["letter"]) > 0


def test_full_graph_pipeline_unsupported_procedure_mismatch():
    # Procedure keyword ("MRI lumbar spine") appears but clinical notes don't
    # support it (ankle sprain) -- should still run without error and flag
    # for more info rather than silently approving.
    result = run_pipeline(SAMPLE_UNSUPPORTED_REFERRAL)
    assert result["coverage"]["recommendation"] in ("request_more_info", "unknown_procedure")
