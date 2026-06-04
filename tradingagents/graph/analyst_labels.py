"""Display names for LangGraph analyst nodes (underscore keys → titled labels)."""

ANALYST_GRAPH_LABELS: dict[str, str] = {
    "broad_market": "Broad Market",
    "capital_flow": "Capital Flow",
    "market": "Market",
    "social": "Social",
    "news": "News",
    "fundamentals": "Fundamentals",
}


def analyst_node_name(analyst_type: str) -> str:
    label = ANALYST_GRAPH_LABELS.get(
        analyst_type, analyst_type.replace("_", " ").title()
    )
    return f"{label} Analyst"


def msg_clear_node_name(analyst_type: str) -> str:
    label = ANALYST_GRAPH_LABELS.get(
        analyst_type, analyst_type.replace("_", " ").title()
    )
    return f"Msg Clear {label}"
