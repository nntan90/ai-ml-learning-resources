# Notebook: evaluation_deep_eval

> Source: https://github.com/NirDiamant/RAG_Techniques/blob/HEAD/evaluation/evaluation_deep_eval.ipynb

---

```python
from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
```

### Test Correctness

```python
correctness_metric = GEval(
    name="Correctness",
    model="gpt-4o",
    evaluation_params=[
        LLMTestCaseParams.EXPECTED_OUTPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=[
        "Determine whether the actual output is factually correct based on the expected output."
    ],

)

gt_answer = "Madrid is the capital of Spain."
pred_answer = "MadriD."

test_case_correctness = LLMTestCase(
    input="What is the capital of Spain?",
    expected_output=gt_answer,
    actual_output=pred_answer,
)

correctness_metric.measure(test_case_correctness)
print(correctness_metric.score)
```

### Test faithfulness

```python
question = "what is 3+3?"
context = ["6"]
generated_answer = "6"

faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-4",
    include_reason=False
)

test_case = LLMTestCase(
    input = question,
    actual_output=generated_answer,
    retrieval_context=context

)

faithfulness_metric.measure(test_case)
print(faithfulness_metric.score)
print(faithfulness_metric.reason)


```

### Test contextual relevancy 

```python
actual_output = "then go somewhere else."
retrieval_context = ["this is a test context","mike is a cat","if the shoes don't fit, then go somewhere else."]
gt_answer = "if the shoes don't fit, then go somewhere else."

relevance_metric = ContextualRelevancyMetric(
    threshold=1,
    model="gpt-4",
    include_reason=True
)
relevance_test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output,
    retrieval_context=retrieval_context,
    expected_output=gt_answer,

)

relevance_metric.measure(relevance_test_case)
print(relevance_metric.score)
print(relevance_metric.reason)
```

```python
new_test_case = LLMTestCase(
    input="What is the capital of Spain?",
    expected_output="Madrid is the capital of Spain.",
    actual_output="MadriD.",
    retrieval_context=["Madrid is the capital of Spain."]
)
```

### Test two different cases together with several metrics together

```python
evaluate(
    test_cases=[relevance_test_case, new_test_case],
    metrics=[correctness_metric, faithfulness_metric, relevance_metric]
)
```

### Funcion to create multiple LLMTestCases based on four lists: 
* Questions
* Ground Truth Answers
* Generated Answers
* Retrieved Documents - Each element is a list

```python
def create_deep_eval_test_cases(questions, gt_answers, generated_answers, retrieved_documents):
    return [
        LLMTestCase(
            input=question,
            expected_output=gt_answer,
            actual_output=generated_answer,
            retrieval_context=retrieved_document
        )
        for question, gt_answer, generated_answer, retrieved_document in zip(
            questions, gt_answers, generated_answers, retrieved_documents
        )
    ]
```

![](https://europe-west1-rag-techniques-views-tracker.cloudfunctions.net/rag-techniques-tracker?notebook=evaluation--evaluation-deep-eval)