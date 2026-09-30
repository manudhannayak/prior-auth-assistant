"""
Coverage agent: looks up the payer's policy for the requested procedure and
checks the referral's clinical notes against that policy's criteria.

This is a deterministic, auditable rule engine rather than an LLM call --
coverage determinations need to be explainable and traceable to a specific
policy criterion, which is exactly the kind of decision you do NOT want a
black-box model making silently. The LLM-backed agents in this pipeline are
used for language tasks (extraction, drafting); this one is pure logic.
"""
import json
import os
from dataclasses import asdict, dataclass

DEFAULT_POLICY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "payer_policies.json")

# Lightweight keyword signals used to judge whether the referral's free text
# plausibly satisfies each criterion. This is intentionally simple/auditable
# -- a production system would replace this with structured EHR fields
# rather than keyword matching against free text.
_CRITERION_SIGNALS = {
    "conservative treatment attempted for at least 4 weeks": ["4 week", "conservative", "nsaid", "physical therapy"],
    "conservative therapy (pt, nsaids) attempted": ["physical therapy", "pt for", "nsaid", "6 week"],
    "progressive neurological deficit": ["neurological deficit", "weakness", "numbness", "cauda equina"],
    "imaging (mri or x-ray) confirming": ["mri", "x-ray", "xray", "confirmed", "tear"],
    "conservative management (pt, nsaids, activity modification)": ["physical therapy", "nsaid", "activity modification", "6 week"],
    "inadequate response to at least one conventional therapy": ["failed", "inadequate response", "did not respond", "methotrexate", "corticosteroid"],
    "documented neurological symptoms": ["headache", "seizure", "focal deficit", "neurological symptom"],
    "confirmed diagnosis": ["diagnosis", "diagnosed with", "confirmed"],
}


@dataclass
class CoverageResult:
    procedure_code: str | None
    procedure_name: str | None
    requires_prior_auth: bool
    criteria: list[str]
    criteria_evidence: dict[str, bool]
    recommendation: str  # "approve_draft", "request_more_info", "no_pa_needed", "unknown_procedure"

    def to_dict(self) -> dict:
        return asdict(self)


def load_policies(path: str = DEFAULT_POLICY_PATH) -> dict[str, dict]:
    with open(path) as f:
        data = json.load(f)
    return {policy["procedure_code"]: policy for policy in data["policies"]}


def _check_criterion(criterion: str, clinical_notes: str) -> bool:
    notes_lower = clinical_notes.lower()
    criterion_lower = criterion.lower()

    signals = None
    for key, key_signals in _CRITERION_SIGNALS.items():
        if key in criterion_lower:
            signals = key_signals
            break

    if signals is None:
        # No known signal set matches this criterion text; be conservative.
        return False
    return any(signal in notes_lower for signal in signals)


def check_coverage(
    procedure_code: str | None,
    clinical_notes: str,
    policies_path: str = DEFAULT_POLICY_PATH,
) -> CoverageResult:
    policies = load_policies(policies_path)

    if procedure_code is None or procedure_code not in policies:
        return CoverageResult(
            procedure_code=procedure_code,
            procedure_name=None,
            requires_prior_auth=True,
            criteria=[],
            criteria_evidence={},
            recommendation="unknown_procedure",
        )

    policy = policies[procedure_code]

    if not policy["requires_prior_auth"]:
        return CoverageResult(
            procedure_code=procedure_code,
            procedure_name=policy["procedure_name"],
            requires_prior_auth=False,
            criteria=[],
            criteria_evidence={},
            recommendation="no_pa_needed",
        )

    evidence = {c: _check_criterion(c, clinical_notes) for c in policy["criteria"]}
    all_met = all(evidence.values()) if evidence else False
    recommendation = "approve_draft" if all_met else "request_more_info"

    return CoverageResult(
        procedure_code=procedure_code,
        procedure_name=policy["procedure_name"],
        requires_prior_auth=True,
        criteria=policy["criteria"],
        criteria_evidence=evidence,
        recommendation=recommendation,
    )
