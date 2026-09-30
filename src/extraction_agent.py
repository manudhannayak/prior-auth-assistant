"""
Extraction agent: pulls structured patient/procedure details out of a
free-text referral note.

Uses an LLM (OpenAI or Anthropic) for extraction when an API key is
configured, and falls back to a deterministic rule-based extractor
otherwise, so the multi-agent pipeline is runnable and testable with zero
API keys.
"""
import json
import os
import re
from dataclasses import asdict, dataclass

PROCEDURE_KEYWORDS = {
    "70551": ["mri brain", "brain mri", "mri of the brain"],
    "72148": ["mri lumbar spine", "lumbar mri", "mri of the lumbar spine", "mri lumbar"],
    "29881": ["knee arthroscopy", "arthroscopic meniscectomy", "meniscectomy"],
    "J1745": ["infliximab"],
    "99213": ["office visit"],
    "71046": ["chest x-ray", "chest xray", "cxr"],
}


@dataclass
class ExtractedReferral:
    raw_text: str
    patient_id: str | None
    diagnosis: str | None
    requested_procedure_code: str | None
    requested_procedure_name: str | None
    clinical_notes: str
    backend: str = "rule_based"

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_patient_id(text: str) -> str | None:
    match = re.search(r"\b(?:MRN|Patient ID)[:\s]*([A-Za-z0-9-]+)", text, re.IGNORECASE)
    return match.group(1) if match else None


def _extract_diagnosis(text: str) -> str | None:
    match = re.search(r"Diagnosis[:\s]*(.+)", text, re.IGNORECASE)
    if match:
        return match.group(1).splitlines()[0].strip()
    return None


def _extract_procedure(text: str) -> tuple[str | None, str | None]:
    lowered = text.lower()
    for code, keywords in PROCEDURE_KEYWORDS.items():
        for kw in keywords:
            if kw in lowered:
                return code, kw
    return None, None


def extract_rule_based(referral_text: str) -> ExtractedReferral:
    procedure_code, procedure_name = _extract_procedure(referral_text)
    return ExtractedReferral(
        raw_text=referral_text,
        patient_id=_extract_patient_id(referral_text),
        diagnosis=_extract_diagnosis(referral_text),
        requested_procedure_code=procedure_code,
        requested_procedure_name=procedure_name,
        clinical_notes=" ".join(referral_text.split()),
        backend="rule_based",
    )


def extract_with_llm(referral_text: str) -> ExtractedReferral:
    """Structured extraction via an LLM. Requires OPENAI_API_KEY or ANTHROPIC_API_KEY."""
    prompt = (
        "Extract the following fields from this clinical referral note as "
        "compact JSON: patient_id, diagnosis, requested_procedure_code "
        "(a CPT/HCPCS code if identifiable), requested_procedure_name. "
        f"Use null for anything not stated.\n\nReferral note:\n{referral_text}"
    )

    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
    elif os.getenv("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(response.content[0].text)
    else:
        raise RuntimeError("No LLM API key found.")

    return ExtractedReferral(
        raw_text=referral_text,
        patient_id=data.get("patient_id"),
        diagnosis=data.get("diagnosis"),
        requested_procedure_code=data.get("requested_procedure_code"),
        requested_procedure_name=data.get("requested_procedure_name"),
        clinical_notes=" ".join(referral_text.split()),
        backend="llm",
    )


def extract(referral_text: str) -> ExtractedReferral:
    if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        try:
            return extract_with_llm(referral_text)
        except Exception:
            result = extract_rule_based(referral_text)
            result.backend = "rule_based (llm call failed)"
            return result
    return extract_rule_based(referral_text)
