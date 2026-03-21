# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/transformers/feedback/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/feedback/experiment.ipynb)                    

## Feedback Transformer

This is an experiment training Shakespeare dataset with Feedback Transformer.

Install the `labml-nn` package

```python
!pip install labml-nn
```

Imports

```python
from labml import experiment
from labml_nn.transformers.feedback.experiment import Configs
```

Create an experiment

```python
experiment.create(name="feedback_transformer")
```

Initialize configurations

```python
conf = Configs()
```

Set experiment configurations and assign a configurations dictionary to override configurations

```python
experiment.configs(conf,
                    {'tokenizer': 'character',
                    'text': 'tiny_shakespeare',
                    'optimizer.learning_rate': 1.0,
                    'optimizer.optimizer': 'Noam',
                    'prompt': 'It is',
                    'prompt_separator': '',

                    'model': 'feedback_transformer',

                    'train_loader': 'shuffled_train_loader',
                    'valid_loader': 'shuffled_valid_loader',

                    'seq_len': 64,
                    'epochs': 128,
                    'batch_size': 80,
                    'inner_iterations': 25})
```

Set PyTorch models for loading and saving

```python
experiment.add_pytorch_models({'model': conf.model})
```

Start the experiment and run the training loop.

```python
# Start the experiment
with experiment.start():
    conf.run()
```