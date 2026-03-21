# Notebook: notebook

> Source: https://github.com/microsoft/ML-For-Beginners/blob/HEAD/8-Reinforcement/2-Gym/notebook.ipynb

---

## CartPole Skating

> **Problem**: If Peter wants to escape from the wolf, he needs to be able to move faster than him. We will see how Peter can learn to skate, in particular, to keep balance, using Q-Learning.

First, let's install the gym and import required libraries:

```python
#code block 1
```

## Create a cartpole environment

```python
#code block 2
```

To see how the environment works, let's run a short simulation for 100 steps.

```python
#code block 3
```

During simulation, we need to get observations in order to decide how to act. In fact, `step` function returns us back current observations, reward function, and the `done` flag that indicates whether it makes sense to continue the simulation or not:

```python
#code block 4
```

We can get min and max value of those numbers:

```python
#code block 5
```

## State Discretization

```python
#code block 6
```

Let's also explore other discretization method using bins:

```python
#code block 7
```

Let's now run a short simulation and observe those discrete environment values.

```python
#code block 8
```

## Q-Table Structure

```python
#code block 9
```

## Let's Start Q-Learning!

```python
#code block 10
```

```python
#code block 11
```

## Plotting Training Progress

```python
plt.plot(rewards)
```

From this graph, it is not possible to tell anything, because due to the nature of stochastic training process the length of training sessions varies greatly. To make more sense of this graph, we can calculate **running average** over series of experiments, let's say 100. This can be done conveniently using `np.convolve`:

```python
#code block 12
```

## Varying Hyperparameters and Seeing the Result in Action

Now it would be interesting to actually see how the trained model behaves. Let's run the simulation, and we will be following the same action selection strategy as during training: sampling according to the probability distribution in Q-Table: 

```python
# code block 13
```


## Saving result to an animated GIF

If you want to impress your friends, you may want to send them the animated GIF picture of the balancing pole. To do this, we can invoke `env.render` to produce an image frame, and then save those to animated GIF using PIL library:

```python
from PIL import Image
obs = env.reset()
done = False
i=0
ims = []
while not done:
   s = discretize(obs)
   img=env.render(mode='rgb_array')
   ims.append(Image.fromarray(img))
   v = probs(np.array([Qbest.get((s,a),0) for a in actions]))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
   i+=1
env.close()
ims[0].save('images/cartpole-balance.gif',save_all=True,append_images=ims[1::2],loop=0,duration=5)
print(i)
```