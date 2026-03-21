# Notebook: experiment

> Source: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/HEAD/labml_nn/lora/experiment.ipynb

---

```python
!pip install labml-nn
```

```python
from labml_nn.lora.experiment import Trainer
from labml import experiment
```

```python
experiment.create(name="lora_gpt2")
```

```python
trainer = Trainer()
```

```python
experiment.configs(trainer)
```

```python
trainer.initialize()
```

```python
with experiment.start():
    trainer.run()
```