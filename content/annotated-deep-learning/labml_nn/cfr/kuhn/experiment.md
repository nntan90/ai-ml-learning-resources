# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/cfr/kuhn/experiment.ipynb

---

[![Github](https://img.shields.io/github/stars/labmlai/annotated_deep_learning_paper_implementations?style=social)](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/cfr/kuhn/experiment.ipynb)                    

## [Counterfactual Regret Minimization (CFR)](https://nn.labml.ai/cfr/index.html) on Kuhn Poker

This is an experiment learning to play Kuhn Poker with Counterfactual Regret Minimization CFR algorithm.

Install the `labml-nn` package

```python
%%capture
!pip install labml-nn
```

Imports

```python
from labml import experiment, analytics
from labml_nn.cfr.analytics import plot_infosets
from labml_nn.cfr.kuhn import Configs
from labml_nn.cfr.infoset_saver import InfoSetSaver
```

Create an experiment, we only write tracking information to `sqlite` to speed things up.
Since the algorithm iterates fast and we track data on each iteration, writing to
other destinations such as Tensorboard can be relatively time consuming.
SQLite is enough for our analytics.

```python
experiment.create(name='kuhn_poker', writers={'sqlite'})
```

Initialize configurations

```python
conf = Configs()
```

Set experiment configurations and assign a configurations dictionary to override configurations

```python
experiment.configs(conf, {'epochs': 1_000_000})
```

Start the experiment and run the training loop.

```python
# Start the experiment
with experiment.start():
    conf.cfr.iterate()
```

```python
inds = analytics.runs(experiment.get_uuid())
```

```python
# dir(inds)
```

```python
plot_infosets(inds['average_strategy.*'], width=600, height=500).display()
```

```python
analytics.scatter(inds.average_strategy_Q_b, inds.average_strategy_Kb_b,
                  width=400, height=400)
```