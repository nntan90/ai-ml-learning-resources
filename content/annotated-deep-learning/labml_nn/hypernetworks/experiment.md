# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/hypernetworks/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/hypernetworks/experiment.ipynb)                    

## HyperLSTM

This is an experiment training Shakespear dataset with HyperLSTM from paper HyperNetworks.

```python
!pip install labml-nn
```

```python
from labml import experiment
from labml_nn.hypernetworks.experiment import Configs
```

```python
# Create experiment
experiment.create(name="hyper_lstm", comment='')
# Create configs
conf = Configs()
# Load configurations
experiment.configs(conf,
                    # A dictionary of configurations to override
                    {'tokenizer': 'character',
                    'text': 'tiny_shakespeare',
                    'optimizer.learning_rate': 2.5e-4,
                    'optimizer.optimizer': 'Adam',
                    'prompt': 'It is',
                    'prompt_separator': '',

                    'rnn_model': 'hyper_lstm',

                    'train_loader': 'shuffled_train_loader',
                    'valid_loader': 'shuffled_valid_loader',

                    'seq_len': 512,
                    'epochs': 128,
                    'batch_size': 2,
                    'inner_iterations': 25})


# Set models for saving and loading
experiment.add_pytorch_models({'model': conf.model})

conf.init()
```

```python
# Start the experiment
with experiment.start():
    # `TrainValidConfigs.run`
    conf.run()
```