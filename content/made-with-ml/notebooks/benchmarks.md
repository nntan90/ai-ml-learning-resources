# Notebook: benchmarks

> Source: https://github.com/GokuMohandas/Made-With-ML/blob/HEAD/notebooks/benchmarks.ipynb

---

<div align="center">
<h1><img width="30" src="https://madewithml.com/static/images/rounded_logo.png">&nbsp;<a href="https://madewithml.com/">Made With ML</a></h1>
    <h3>ML for Developers</h3>
    Design · Develop · Deploy · Iterate
</div>

<br>

<div align="center">
    <a target="_blank" href="https://madewithml.com"><img src="https://img.shields.io/badge/Subscribe-40K-brightgreen"></a>&nbsp;
    <a target="_blank" href="https://github.com/GokuMohandas/MadeWithML"><img src="https://img.shields.io/github/stars/GokuMohandas/MadeWithML.svg?style=social&label=Star"></a>&nbsp;
    <a target="_blank" href="https://www.linkedin.com/in/goku"><img src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social"></a>&nbsp;
    <a target="_blank" href="https://twitter.com/GokuMohandas"><img src="https://img.shields.io/twitter/follow/GokuMohandas.svg?label=Follow&style=social"></a>
    <br>
    🔥&nbsp; Among the <a href="https://github.com/GokuMohandas/MadeWithML" target="_blank">top ML</a> repositories on GitHub
</div>

<br>
<hr>

# Generative AI

In our [Made With ML course](https://madewithml.com/) we will be fine-tuning an LLM for a supervised classification task. The specific class of LLMs we'll be using is called [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)). Bert models are encoder-only models and are the gold-standard for supervised NLP tasks. However, you may be wondering how do all the (much larger) LLM, created for generative applications, fare ([GPT 4](https://openai.com/research/gpt-4), [Falcon 40B](https://huggingface.co/tiiuae/falcon-40b), [Llama 2](https://ai.meta.com/llama/), etc.)?

> We chose the smaller BERT model for our course because it's easier to train and fine-tune. However, the workflow for fine-tuning the larger LLMs are quite similar as well. They do require much more compute but Ray abstracts away the scaling complexities involved with that.

## Set up

```python
!pip install openai==0.27.8 tqdm==4.65.0 -q
```

You'll need to first sign up for an [OpenAI account](https://platform.openai.com/signup) and then grab your API key from [here](https://platform.openai.com/account/api-keys).

```python
import openai
openai.api_key = "YOUR_API_KEY"
```

### Load data

```python
import pandas as pd
```

```python
# Load training data
DATASET_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
train_df = pd.read_csv(DATASET_LOC)
train_df.head()
```

```python
# Unique labels
tags = train_df.tag.unique().tolist()
tags
```

```python
# Load inference dataset
HOLDOUT_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/holdout.csv"
test_df = pd.read_csv(HOLDOUT_LOC)
```

### Utilities

We'll define a few utility functions to make the OpenAI request and to store our predictions. While we could perform batch prediction by loading samples until the context length is reached, we'll just perform one at a time since it's not too many data points and we can have fully deterministic behavior (if you insert new data, etc.). We'll also added some reliability in case we overload the endpoints with too many request at once.

```python
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from sklearn.metrics import precision_recall_fscore_support
import time
from tqdm import tqdm
```

```python
# Query OpenAI endpoint
system_content = "you only answer in rhymes"  # system content (behavior)
assistant_content = ""  # assistant content (context)
user_content = "how are you"  # user content (message)
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=[
        {"role": "system", "content": system_content},
        {"role": "assistant", "content": assistant_content},
        {"role": "user", "content": user_content},
    ],
)
print (response.to_dict()["choices"][0].to_dict()["message"]["content"])
```

Now let's create a function that can predict tags for a given sample.

```python
def get_tag(model, system_content="", assistant_content="", user_content=""):
    try:
        # Get response from OpenAI
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": user_content},
            ],
        )
        predicted_tag = response.to_dict()["choices"][0].to_dict()["message"]["content"]
        return predicted_tag

    except (openai.error.ServiceUnavailableError, openai.error.APIError) as e:
        return None
```

```python
# Get tag
model = "gpt-3.5-turbo-0613"
system_context = f"""
    You are a NLP prediction service that predicts the label given an input's title and description.
    You must choose between one of the following labels for each input: {tags}.
    Only respond with the label name and nothing else.
    """
assistant_content = ""
user_context = "Transfer learning with transformers: Using transformers for transfer learning on text classification tasks."
tag = get_tag(model=model, system_content=system_context, assistant_content=assistant_content, user_content=user_context)
print (tag)
```

Next, let's create a function that can predict tags for a list of inputs.

```python
# List of dicts w/ {title, description} (just the first 3 samples for now)
samples = test_df[["title", "description"]].to_dict(orient="records")[:3]
samples
```

```python
def get_predictions(inputs, model, system_content, assistant_content=""):
    y_pred = []
    for item in tqdm(inputs):
        # Convert item dict to string
        user_content = str(item)

        # Get prediction
        predicted_tag = get_tag(
            model=model, system_content=system_content,
            assistant_content=assistant_content, user_content=user_content)

        # If error, try again after pause (repeatedly until success)
        while predicted_tag is None:
            time.sleep(30)  # could also do exponential backoff
            predicted_tag = get_tag(
                model=model, system_content=system_content,
                assistant_content=assistant_content, user_content=user_content)

        # Add to list of predictions
        y_pred.append(predicted_tag)

    return y_pred
```

```python
# Get predictions for a list of inputs
get_predictions(inputs=samples, model=model, system_content=system_context)
```

Next we'll define a function that can clean our predictions in the event that it's not the proper format or has hallucinated a tag outside of our expected tags.

```python
def clean_predictions(y_pred, tags, default="other"):
    for i, item in enumerate(y_pred):
        if item not in tags:  # hallucinations
            y_pred[i] = default
        if item.startswith("'") and item.endswith("'"):  # GPT 4 likes to places quotes
            y_pred[i] = item[1:-1]
    return y_pred
```

> Open AI has now released [function calling](https://openai.com/blog/function-calling-and-other-api-updates) and [custom instructions](https://openai.com/blog/custom-instructions-for-chatgpt) which is worth exploring to avoid this manual cleaning.

Next, we'll define a function that will plot our ground truth labels and predictions.

```python
def plot_tag_dist(y_true, y_pred):
    # Distribution of tags
    true_tag_freq = dict(Counter(y_true))
    pred_tag_freq = dict(Counter(y_pred))
    df_true = pd.DataFrame({"tag": list(true_tag_freq.keys()), "freq": list(true_tag_freq.values()), "source": "true"})
    df_pred = pd.DataFrame({"tag": list(pred_tag_freq.keys()), "freq": list(pred_tag_freq.values()), "source": "pred"})
    df = pd.concat([df_true, df_pred], ignore_index=True)

    # Plot
    plt.figure(figsize=(10, 3))
    plt.title("Tag distribution", fontsize=14)
    ax = sns.barplot(x="tag", y="freq", hue="source", data=df)
    ax.set_xticklabels(list(true_tag_freq.keys()), rotation=0, fontsize=8)
    plt.legend()
    plt.show()
```

And finally, we'll define a function that will combine all the utilities above to predict, clean and plot our results.

```python
def evaluate(test_df, model, system_content, assistant_content, tags):
    # Predictions
    y_test = test_df.tag.to_list()
    test_samples = test_df[["title", "description"]].to_dict(orient="records")
    y_pred = get_predictions(
        inputs=test_samples, model=model,
        system_content=system_content, assistant_content=assistant_content)
    y_pred = clean_predictions(y_pred=y_pred, tags=tags)

    # Performance
    metrics = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    performance = {"precision": metrics[0], "recall": metrics[1], "f1": metrics[2]}
    print(json.dumps(performance, indent=2))
    plot_tag_dist(y_true=y_test, y_pred=y_pred)
    return y_pred, performance
```

## Benchmarks

Now we're ready to start benchmarking our different LLMs with different context.

```python
y_pred = {"zero_shot": {}, "few_shot": {}}
performance = {"zero_shot": {}, "few_shot": {}}
```

### Zero-shot learning

We'll start with zero-shot learning which involves providing the model with the `system_content` that tells it how to behave but no examples of the behavior.

```python
system_content = f"""
    You are a NLP prediction service that predicts the label given an input's title and description. 
    You must choose between one of the following labels for each input: {tags}. 
    Only respond with the label name and nothing else.
    """
```

```python
# Zero-shot with GPT 3.5
method = "zero_shot"
model = "gpt-3.5-turbo-0613"
y_pred[method][model], performance[method][model] = evaluate(
    test_df=test_df, model=model, system_content=system_content,
    assistant_content="", tags=tags)
```

```python
# Zero-shot with GPT 4
method = "zero_shot"
model = "gpt-4-0613"
y_pred[method][model], performance[method][model] = evaluate(
    test_df=test_df, model=model, system_content=system_content,
    assistant_content="", tags=tags)
```

### Few-shot learning

Now, we'll be adding a `assistant_context` with a few samples from our training data for each class. The intuition here is that we're giving the model a few examples (few-shot learning) of what each class looks like so that it can learn to generalize better.

```python
# Create additional context with few samples from each class
num_samples = 2
additional_context = []
cols_to_keep = ["title", "description", "tag"]
for tag in tags:
    samples = train_df[cols_to_keep][train_df.tag == tag][:num_samples].to_dict(orient="records")
    additional_context.extend(samples)
additional_context
```

```python
# Add additional context
assistant_content = f"""Here are some examples with the correct labels: {additional_context}"""
print (assistant_content)
```

> We could increase the number of samples by increasing the context length. We could also retrieve better few-shot samples by extracting examples from the training data that are similar to the current sample (ex. similar unique vocabulary).

```python
# Few-shot with GPT 3.5
method = "few_shot"
model = "gpt-3.5-turbo-0613"
y_pred[method][model], performance[method][model] = evaluate(
    test_df=test_df, model=model, system_content=system_content,
    assistant_content=assistant_content, tags=tags)
```

```python
# Few-shot with GPT 4
method = "few_shot"
model = "gpt-4-0613"
y_pred[method][model], performance[method][model] = evaluate(
    test_df=test_df, model=model, system_content=system_content,
    assistant_content=assistant_content, tags=tags)
```

As we can see, few shot learning performs better than it's respective zero shot counter part. GPT 4 has had considerable improvements in reducing hallucinations but for our supervised task this comes at an expense of high precision but lower recall and f1 scores. When GPT 4 is not confident, it would rather predict `other`.

## OSS LLMs

So far, we've only been using closed-source models from OpenAI. While these are *currently* the gold-standard, there are many open-source models that are rapidly catching up ([Falcon 40B](https://huggingface.co/tiiuae/falcon-40b), [Llama 2](https://ai.meta.com/llama/), etc.). Before we see how these models perform on our task, let's first consider a few reasons why we should care about open-source models.

- **data ownership**: you can serve your models and pass data to your models, without having to share it with a third-party API endpoint.
- **fine-tune**: with access to our model's weights, we can actually fine-tune them, as opposed to experimenting with fickle prompting strategies.
- **optimization**: we have full freedom to optimize our deployed models for inference (ex. quantization, pruning, etc.) to reduce costs.

```python
# Coming soon in August!
```

## Results

```python
print(json.dumps(performance, indent=2))
```

```python
# Transform data into a new dictionary with four keys
by_model_and_context = {}
for context_type, models_data in performance.items():
    for model, metrics in models_data.items():
        key = f"{model}_{context_type}"
        by_model_and_context[key] = metrics
```

```python
# Extracting the model names and the metric values
models = list(by_model_and_context.keys())
metrics = list(by_model_and_context[models[0]].keys())

# Plotting the bar chart with metric scores on top of each bar
fig, ax = plt.subplots(figsize=(10, 4))
width = 0.2
x = range(len(models))

for i, metric in enumerate(metrics):
    metric_values = [by_model_and_context[model][metric] for model in models]
    ax.bar([pos + width * i for pos in x], metric_values, width, label=metric)
    # Displaying the metric scores on top of each bar
    for pos, val in zip(x, metric_values):
        ax.text(pos + width * i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks([pos + width for pos in x])
ax.set_xticklabels(models, rotation=0, ha='center', fontsize=8)
ax.set_ylabel('Performance')
ax.set_title('GPT Benchmarks')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()
```

Our best model is GPT 4 with few shot learning at an f1 score of ~92%. We will see in the [Made With ML course](https://madewithml.com/) how fine-tuning an LLM with a proper training dataset to change the actual weights of the last N layers (as opposed to the hard prompt tuning here) will yield similar/slightly better results to GPT 4 (at a fraction of the model size and inference costs).

However, the best system might actually be a combination of using these few-shot hard prompt LLMs alongside fine-tuned LLMs. For example, our fine-tuned LLMs in the course will perform well when the test data is similar to the training data (similar distributions of vocabulary, etc.) but may not perform well on out of distribution. Whereas, these hard prompted LLMs, by themselves or augmented with additional context (ex. arXiv plugins in our case), could be used when our primary fine-tuned model is not so confident.