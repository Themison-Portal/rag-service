"""
evals/test_uc_protocol.py

Regression suite for the UC-protocol RAG assistant, built from
evals/datasets/qa_pairs.json (50 hand-labeled cases).

Run:
    deepeval test run evals/test_uc_protocol.py

Requires:
    ANTHROPIC_API_KEY set (used both as the judge model and,
    presumably, as your generator — see note in claude_judge.py
    about correlated blind spots).
"""

import json
import os
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    GEval,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from .claude_judge import ClaudeJudge  # wraps Claude as the judge LLM
from .rag_client import (
    run_rag_pipeline,
)  # YOUR system: (question, history=None) -> (answer, chunks)
from .refusal_phrase_metric import RefusalPhraseMetric

DATA_PATH = Path(__file__).parent / "datasets" / "qa_pairs.json"
judge = ClaudeJudge()


# --------------------------------------------------------------------------
# Load + split fixture data by category
# --------------------------------------------------------------------------


def _load_qa_pairs():
    with open(DATA_PATH) as f:
        return json.load(f)


ALL_PAIRS = _load_qa_pairs()

FACTUAL = [q for q in ALL_PAIRS if q["category"] in ("Factual", "Completeness", "Version")]
UNANSWERABLE = [q for q in ALL_PAIRS if q["category"] == "Unanswerable"]
AMBIGUOUS = [q for q in ALL_PAIRS if q["category"] == "Ambiguous"]
FOLLOWUP = [q for q in ALL_PAIRS if q["category"] == "Follow-up"]
OUT_OF_SCOPE = [
    q for q in ALL_PAIRS if q["category"] == "Out-of-scope"
]  # tracked only, never asserted


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

correctness_metric = GEval(
    name="Correctness",
    model=judge,
    threshold=0.7,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    criteria=(
        "Determine if 'actual_output' is factually correct and complete relative to "
        "'expected_output'. For questions about stratification factors, safety committee "
        "names, or country lists, the answer MUST reflect the CURRENT protocol version "
        "(v1.1), not a superseded value, unless it explicitly labels the older value as "
        "historical/superseded."
    ),
)

faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=judge)
context_precision_metric = ContextualPrecisionMetric(threshold=0.7, model=judge)

# Deterministic — SYSTEM_PROMPT contractually guarantees this exact phrase
# when context doesn't support an answer, so this is a string match, not
# something worth spending a judge call on.
refusal_metric = RefusalPhraseMetric()

clarify_metric = GEval(
    name="AsksClarification",
    model=judge,
    threshold=0.7,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    criteria=(
        "The question as asked is ambiguous (underspecified, multiple valid readings). "
        "Pass if 'actual_output' asks a clarifying question, OR gives all plausible "
        "readings explicitly labeled. Fail if it silently guesses one interpretation."
    ),
)

followup_context_metric = GEval(
    name="HoldsConversationContext",
    model=judge,
    threshold=0.7,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    criteria=(
        "'input' contains a two-turn exchange separated by '->'. Pass only if "
        "'actual_output' (the answer to turn 2) correctly resolves using the context "
        "established in turn 1, matching 'expected_output'."
    ),
)


# --------------------------------------------------------------------------
# A / B / C — factual, completeness, current-vs-historical
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item",
    FACTUAL,
    ids=[f"{q['id']:02d}_{q['category']}" for q in FACTUAL],
)
def test_factual_and_version_questions(item):
    actual_output, retrieved_chunks = run_rag_pipeline(item["question"])

    test_case = LLMTestCase(
        input=item["question"],
        actual_output=actual_output,
        expected_output=item["expected_answer"],
        retrieval_context=retrieved_chunks,
        additional_metadata={"source": item["source"], "category": item["category"]},
    )
    assert_test(test_case, [correctness_metric, faithfulness_metric, context_precision_metric])


# --------------------------------------------------------------------------
# D — unanswerable: correct behaviour is refusal
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item",
    UNANSWERABLE,
    ids=[f"{q['id']:02d}_unanswerable" for q in UNANSWERABLE],
)
def test_unanswerable_questions_are_refused(item):
    actual_output, _ = run_rag_pipeline(item["question"])

    test_case = LLMTestCase(
        input=item["question"],
        actual_output=actual_output,
    )
    assert_test(test_case, [refusal_metric])


# --------------------------------------------------------------------------
# E — ambiguous: correct behaviour is asking for clarification
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item",
    AMBIGUOUS,
    ids=[f"{q['id']:02d}_ambiguous" for q in AMBIGUOUS],
)
def test_ambiguous_questions_prompt_clarification(item):
    actual_output, _ = run_rag_pipeline(item["question"])

    test_case = LLMTestCase(
        input=item["question"],
        actual_output=actual_output,
    )
    assert_test(test_case, [clarify_metric])


# --------------------------------------------------------------------------
# F — follow-up chains: needs conversation history threaded through
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item",
    FOLLOWUP,
    ids=[f"{q['id']:02d}_followup" for q in FOLLOWUP],
)
def test_followup_chains_hold_context(item):
    turn1, turn2 = [t.strip() for t in item["question"].split("->")]

    turn1_answer, _ = run_rag_pipeline(turn1)
    turn2_answer, turn2_chunks = run_rag_pipeline(turn2, history=[(turn1, turn1_answer)])

    test_case = LLMTestCase(
        input=f"{turn1} -> {turn2}",
        actual_output=turn2_answer,
        expected_output=item["expected_answer"],
        retrieval_context=turn2_chunks,
    )
    assert_test(test_case, [followup_context_metric])


# --------------------------------------------------------------------------
# G — out-of-scope: NEVER asserted, just logged for visibility.
# The protocol doc's notes are explicit that these should track, not fail.
# --------------------------------------------------------------------------


def test_out_of_scope_questions_are_logged(capsys):
    for item in OUT_OF_SCOPE:
        actual_output, _ = run_rag_pipeline(item["question"])
        print(f"[out-of-scope #{item['id']}] Q: {item['question']}\n  -> {actual_output}\n")
    # intentionally no assertion — see docstring above
