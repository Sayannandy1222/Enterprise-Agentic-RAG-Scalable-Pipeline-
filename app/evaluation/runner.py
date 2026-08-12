from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.retrieval import (
    evaluate_retrieval,
)


DATASET_PATH = (
    Path(__file__).parent
    / "data"
    / "rag_eval.json"
)


def load_dataset(
    path: Path = DATASET_PATH,
) -> list[dict[str, Any]]:
    """
    Load and validate the retrieval evaluation dataset.
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    cases = payload.get(
        "cases",
        [],
    )

    if not isinstance(cases, list):
        raise ValueError(
            "evaluation dataset 'cases' must be a list"
        )

    validated: list[
        dict[str, Any]
    ] = []

    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(
                f"case {index} must be an object"
            )

        case_id = case.get("id")
        query = case.get("query")
        relevant = case.get(
            "relevant_chunk_ids"
        )

        if not case_id:
            raise ValueError(
                f"case {index} is missing 'id'"
            )

        if not query:
            raise ValueError(
                f"{case_id} is missing 'query'"
            )

        if not isinstance(
            relevant,
            list,
        ) or not relevant:
            raise ValueError(
                f"{case_id} must contain "
                "'relevant_chunk_ids'"
            )

        validated.append(
            {
                "id": str(case_id),
                "query": str(query),
                "relevant_chunk_ids": [
                    str(value)
                    for value in relevant
                ],
            }
        )

    return validated


def main() -> None:
    """
    Run a deterministic smoke evaluation of the retrieval metrics.
    """

    cases = load_dataset()

    retrieved = [
        ["chunk-a", "chunk-x", "chunk-y"],
        ["chunk-x", "chunk-b", "chunk-y"],
        ["chunk-c", "chunk-x", "chunk-y"],
    ]

    relevant = [
        case["relevant_chunk_ids"]
        for case in cases
    ]

    metrics = evaluate_retrieval(
        retrieved,
        relevant,
        ks=(3, 5),
    )

    print(
        "RAG Retrieval Evaluation"
    )
    print(
        "========================"
    )
    print(
        f"Cases:      {len(cases)}"
    )
    print(
        f"Recall@3:   "
        f"{metrics['recall_at_3']:.4f}"
    )
    print(
        f"Recall@5:   "
        f"{metrics['recall_at_5']:.4f}"
    )
    print(
        f"MRR:        "
        f"{metrics['mrr']:.4f}"
    )

    print()
    print(
        "Per-case results"
    )
    print(
        "----------------"
    )

    for case, retrieved_ids in zip(
        cases,
        retrieved,
    ):
        case_metrics = evaluate_retrieval(
            [retrieved_ids],
            [case["relevant_chunk_ids"]],
            ks=(3, 5),
        )

        print(
            f"{case['id']}: "
            f"Recall@3="
            f"{case_metrics['recall_at_3']:.4f}, "
            f"Recall@5="
            f"{case_metrics['recall_at_5']:.4f}, "
            f"MRR="
            f"{case_metrics['mrr']:.4f}"
        )

        print(
            "  Retrieved: "
            + ", ".join(
                retrieved_ids
            )
        )


if __name__ == "__main__":
    main()
