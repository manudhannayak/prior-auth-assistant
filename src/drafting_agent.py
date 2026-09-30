"""
Drafting agent: turns the extracted referral details + coverage-check
result into a prior authorization request letter addressed to the payer.

Uses an LLM for more natural prose when configured, falling back to a
deterministic template otherwise (still fully usable, just less
conversational -- and arguably preferable for a document that may be
audited).
"""
import os


def draft_with_template(extracted: dict, coverage: dict) -> str:
    criteria_lines = "\n".join(
        f"  - {'[MET] ' if met else '[NOT DOCUMENTED] '}{criterion}"
        for criterion, met in coverage.get("criteria_evidence", {}).items()
    ) or "  (no specific criteria on file for this procedure)"

    return f"""PRIOR AUTHORIZATION REQUEST

Patient ID: {extracted.get('patient_id') or '[NOT PROVIDED -- please attach patient identifiers]'}
Diagnosis: {extracted.get('diagnosis') or '[NOT PROVIDED]'}
Requested Procedure: {coverage.get('procedure_name') or extracted.get('requested_procedure_name') or '[UNKNOWN -- please confirm CPT/HCPCS code]'} ({coverage.get('procedure_code') or 'code not identified'})

Clinical Summary:
{extracted.get('clinical_notes', '')}

Payer Coverage Criteria Reviewed:
{criteria_lines}

Recommendation: {coverage.get('recommendation')}

This request was drafted by an AI assistant for physician/staff review and
must be verified against the patient's chart before submission to the payer.
""".strip()


def draft_with_llm(extracted: dict, coverage: dict) -> str:
    """Requires OPENAI_API_KEY or ANTHROPIC_API_KEY."""
    prompt = (
        "Draft a concise, professional prior authorization request letter to "
        "a health insurance payer using the structured data below. Reference "
        "which coverage criteria are documented and which are missing. End "
        "with a note that this draft requires clinician review before "
        f"submission.\n\nExtracted referral data: {extracted}\n\n"
        f"Coverage check result: {coverage}"
    )

    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    raise RuntimeError("No LLM API key found.")


def draft(extracted: dict, coverage: dict) -> dict:
    if os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        try:
            return {"backend": "llm", "letter": draft_with_llm(extracted, coverage)}
        except Exception as e:
            return {
                "backend": "template (llm call failed)",
                "letter": draft_with_template(extracted, coverage),
                "error": str(e),
            }
    return {"backend": "template", "letter": draft_with_template(extracted, coverage)}
