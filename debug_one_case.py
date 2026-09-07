import json
from deepeval.test_case import LLMTestCase
from deepeval.models import AnthropicModel
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCaseParams
from evals.rag_client import run_rag_pipeline

with open("evals/datasets/qa_pairs.json") as f:
    qa = json.load(f)
item = qa[0]  # question #1

answer, context = run_rag_pipeline(item["question"])
print("QUESTION:", item["question"])
print("EXPECTED:", item["expected_answer"])
print("ACTUAL:", answer)
print("---")

judge = AnthropicModel(model="claude-sonnet-4-6", temperature=0)
metric = GEval(
    name="Correctness", model=judge, threshold=0.7,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    criteria="Determine if actual_output is factually correct and complete relative to expected_output.",
)
test_case = LLMTestCase(input=item["question"], actual_output=answer, expected_output=item["expected_answer"])
metric.measure(test_case)
print("SCORE:", metric.score)
print("REASON:", metric.reason)
