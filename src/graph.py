"""
LangGraph orchestration wiring the three agents into a single pipeline:

    extract (referral text -> structured fields)
        -> check_coverage (structured fields -> payer policy match)
            -> draft (structured fields + coverage result -> PA letter)

Each node is independently testable (see src/*_agent.py + tests/), and the
graph itself just sequences them and carries state between steps.

Usage:
    python src/graph.py --referral_file data/sample_referral.txt
"""
import argparse
from typing import TypedDict

from langgraph.graph import END, StateGraph

from coverage_agent import check_coverage
from drafting_agent import draft
from extraction_agent import extract


class PriorAuthState(TypedDict, total=False):
    referral_text: str
    extracted: dict
    coverage: dict
    draft_result: dict


def extract_node(state: PriorAuthState) -> PriorAuthState:
    extracted = extract(state["referral_text"])
    return {"extracted": extracted.to_dict()}


def coverage_node(state: PriorAuthState) -> PriorAuthState:
    extracted = state["extracted"]
    coverage = check_coverage(extracted.get("requested_procedure_code"), extracted.get("clinical_notes", ""))
    return {"coverage": coverage.to_dict()}


def draft_node(state: PriorAuthState) -> PriorAuthState:
    draft_result = draft(state["extracted"], state["coverage"])
    return {"draft_result": draft_result}


def build_graph():
    graph = StateGraph(PriorAuthState)
    graph.add_node("extract", extract_node)
    graph.add_node("check_coverage", coverage_node)
    graph.add_node("draft", draft_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "check_coverage")
    graph.add_edge("check_coverage", "draft")
    graph.add_edge("draft", END)

    return graph.compile()


def run_pipeline(referral_text: str) -> PriorAuthState:
    app = build_graph()
    return app.invoke({"referral_text": referral_text})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--referral_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.referral_file) as f:
        referral_text = f.read()

    result = run_pipeline(referral_text)

    print("=== Extracted ===")
    print(result["extracted"])
    print("\n=== Coverage check ===")
    print(result["coverage"])
    print("\n=== Draft letter ===")
    print(result["draft_result"]["letter"])


if __name__ == "__main__":
    main()
