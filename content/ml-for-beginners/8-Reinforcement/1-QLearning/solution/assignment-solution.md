# Notebook: assignment-solution

> Source: https://github.com/microsoft/ML-For-Beginners/blob/HEAD/8-Reinforcement/1-QLearning/solution/assignment-solution.ipynb

---

# Peter and the Wolf: Realistic Environment

In our situation, Peter was able to move around almost without getting tired or hungry. In more realistic world, we has to sit down and rest from time to time, and also to feed himself. Let's make our world more realistic, by implementing the following rules:

1. By moving from one place to another, Peter loses **energy** and gains some **fatigue**.
2. Peter can gain more energy by eating apples.
3. Peter can get rid of fatigue by resting under the tree or on the grass (i.e. walking into a board location with a tree or grass - green field)
4. Peter needs to find and kill the wolf
5. In order to kill the wolf, Peter needs to have certain levels of energy and fatigue, otherwise he loses the battle.


```python
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from rlboard import *
```

```python
width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

## Defining state

In our new game rules, we need to keep track of energy and fatigue at each board state. Thus we will create an object `state` that will carry all required information about current problem state, including state of the board, current levels of energy and fatigue, and whether we can win the wolf while at terminal state:

```python
class state:
    def __init__(self,board,energy=10,fatigue=0,init=True):
        self.board = board
        self.energy = energy
        self.fatigue = fatigue
        self.dead = False
        if init:
            self.board.random_start()
        self.update()

    def at(self):
        return self.board.at()

    def update(self):
        if self.at() == Board.Cell.water:
            self.dead = True
            return
        if self.at() == Board.Cell.tree:
            self.fatigue = 0
        if self.at() == Board.Cell.apple:
            self.energy = 10

    def move(self,a):
        self.board.move(a)
        self.energy -= 1
        self.fatigue += 1
        self.update()

    def is_winning(self):
        return self.energy > self.fatigue
```

Let's try to solve the problem using random walk and see if we succeed:

```python
def random_policy(state):
    return random.choice(list(actions))

def walk(board,policy):
    n = 0 # number of steps
    s = state(board)
    while True:
        if s.at() == Board.Cell.wolf:
            if s.is_winning():
                return n # success!
            else:
                return -n # failure!
        if s.at() == Board.Cell.water:
            return 0 # died
        a = actions[policy(m)]
        s.move(a)
        n+=1

walk(m,random_policy)
```

```python
def print_statistics(policy):
    s,w,n = 0,0,0
    for _ in range(100):
        z = walk(m,policy)
        if z<0:
            w+=1
        elif z==0:
            n+=1
        else:
            s+=1
    print(f"Killed by wolf = {w}, won: {s} times, drown: {n} times")

print_statistics(random_policy)
```

## Reward Function


```python
def reward(s):
    r = s.energy-s.fatigue
    if s.at()==Board.Cell.wolf:
        return 100 if s.is_winning() else -100
    if s.at()==Board.Cell.water:
        return -100
    return r
```

## Q-Learning algorithm

The actual learning algorithm stays pretty much unchanged, we just use `state` instead of just board position.

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

```python
def probs(v,eps=1e-4):
    v = v-v.min()+eps
    v = v/v.sum()
    return v
```

```python

from IPython.display import clear_output

lpath = []

for epoch in range(10000):
    clear_output(wait=True)
    print(f"Epoch = {epoch}",end='')

    # Pick initial point
    s = state(m)
    
    # Start travelling
    n=0
    cum_reward = 0
    while True:
        x,y = s.board.human
        v = probs(Q[x,y])
        while True:
            a = random.choices(list(actions),weights=v)[0]
            dpos = actions[a]
            if s.board.is_valid(s.board.move_pos(s.board.human,dpos)):
                break 
        s.move(dpos)
        r = reward(s)
        if abs(r)==100: # end of game
            print(f" {n} steps",end='\r')
            lpath.append(n)
            break
        alpha = np.exp(-n / 3000)
        gamma = 0.5
        ai = action_idx[a]
        Q[x,y,ai] = (1 - alpha) * Q[x,y,ai] + alpha * (r + gamma * Q[x+dpos[0], y+dpos[1]].max())
        n+=1
```

```python
m.plot(Q)
```

## Results

Let's see if we were successful training Peter to fight the wolf!

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

We now see much less cases of drowning, but Peter is still not always able to kill the wolf. Try to experiment and see if you can improve this result by playing with hyperparameters.

```python
plt.plot(lpath)
```