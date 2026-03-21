# Notebook: lecture3_DQN

> Source: https://github.com/norhum/reinforcement-learning-from-scratch/blob/HEAD/lecture3_DQN.ipynb

---

<a href="https://colab.research.google.com/github/norhum/reinforcement-learning-from-scratch/blob/main/lecture3_DQN.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Note:**
If you're using Google Colab, it’s recommended to switch the runtime to **GPU** for faster training later.
You can do this by going to **Runtime > Change runtime type > Hardware accelerator > GPU**.
## Limitations of Q-Tables and the Need for Function Approximation

Q-learning and SARSA using Q-tables worked well for our 10x10 GridWorld. We could explicitly store a Q-value for every possible state-action pair. However, this tabular approach breaks down quickly as problems become more complex:

1.  **Huge State Spaces:** Imagine games like Chess or Go. The number of possible board configurations (states) is astronomically large. Creating a table to hold a value for every state is impossible due to memory limitations.
2.  **Continuous States:** What if the state isn't a discrete grid cell number, but continuous values like sensor readings from a robot (position, velocity, joint angles)? You can't create a table row for every possible combination of real numbers.

This challenge is often called the **curse of dimensionality**. As the number of dimensions describing the state (or the range of values within those dimensions) increases, the size of the state space explodes, making tabular methods impractical.

### Solution: Function Approximation

We need a way to **generalize**. Instead of storing a value for every *exact* state, can we learn a function that *estimates* the Q-value based on the state's features? Even for states the agent hasn't encountered before?

This is where **function approximation** comes in. We can use powerful function approximators, like **Neural Networks**, to learn the Q-value function.

### Deep Q-Networks (DQN)

The core idea of **Deep Q-Networks (DQN)** is precisely this: use a neural network to approximate the Q-value function.

*   **Input:** Some representation of the current state *s*.
*   **Output:** Estimated Q-values for *all possible actions* from that state *s*.

Instead of looking up `Q[s, a]` in a table, we feed state `s` into the network and get a vector of Q-values, like `[Q(s, action_0), Q(s, action_1), Q(s, action_2), ...]`.

Let's build a basic DQN agent to solve our original (simple) GridWorld problem.

---

### DQN Implementation for GridWorld

First, ensure the necessary imports are included. The `GridWorld` environment class remains unchanged from the previous section.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from tabulate import tabulate

class GridWorld:
    def __init__(self, size=10):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_position = 0
        return self.agent_position

    def step(self, action):
        if action == 0 and self.agent_position % self.size > 0: # left
            self.agent_position -= 1
        elif action == 1 and self.agent_position % self.size < self.size - 1: # right
            self.agent_position += 1
        elif action == 2 and self.agent_position >= self.size: # up
            self.agent_position -= self.size
        elif action == 3 and self.agent_position < self.size * (self.size - 1): # down
            self.agent_position += self.size

        done = self.agent_position == self.size * self.size - 1
        reward = 10 if done else -1

        return self.agent_position, reward, done
```

#### The DQN Agent Class

The "brain" of our DQN agent will be encapsulated in a class, `DeepQLearningAgent`. This class will hold the neural network, the optimizer, and the logic for action selection and training.

```python
class DeepQLearningAgent:
    """
    A basic Deep Q-Network Agent.
    Uses a simple MLP to approximate Q-values.
    """
    def __init__(self, state_size, action_size,
                 learning_rate=0.001, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min
        self.lr = learning_rate

        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # --- Q-Network Definition ---
        # A simple Multi-Layer Perceptron (MLP)
        self.q_network = nn.Sequential(
            nn.Linear(self.state_size, 24), # Input layer (state size)
            nn.ReLU(),                     # Activation function
            nn.Linear(24, 24),             # Hidden layer
            nn.ReLU(),                     # Activation function
            nn.Linear(24, self.action_size)# Output layer (Q-values for each action)
        ).to(self.device) # Move the network to the chosen device

        # --- Optimizer and Loss Function ---
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss() # Mean Squared Error Loss

    def get_action(self, state):
        """
        Selects action using epsilon-greedy policy based on the Q-network.
        """
        # Exploration: choose a random action
        if random.random() < self.epsilon: # Use random.random() for standard epsilon check
            return random.randrange(self.action_size) # Return random action index

        # Exploitation: choose the best action based on Q-network
        with torch.no_grad(): # Disable gradient calculation for inference
            # Convert state to one-hot tensor
            state_tensor = F.one_hot(torch.tensor(state, device=self.device), num_classes=self.state_size).float()
            # Unsqueeze to add batch dimension (network expects batches)
            state_tensor = state_tensor.unsqueeze(0)

            # Get Q-values from the network
            q_values = self.q_network(state_tensor)

            # Choose the action with the highest Q-value
            action = torch.argmax(q_values).item()
            return action

    def train(self, state, action, reward, next_state, done):
        """
        Performs a single training step on the Q-network.
        """
        # Convert inputs to tensors on the correct device
        state_tensor = F.one_hot(torch.tensor(state), num_classes=self.state_size).float().unsqueeze(0).to(self.device)
        next_state_tensor = F.one_hot(torch.tensor(next_state), num_classes=self.state_size).float().unsqueeze(0).to(self.device)
        action_tensor = torch.tensor([action], device=self.device) # Action as tensor index
        reward_tensor = torch.tensor([reward], device=self.device).float()
        done_tensor = torch.tensor([done], device=self.device).float() # Use float for multiplication trick

        # --- Calculate Predicted Q-values ---
        # Get Q-values for *all* actions in the current state
        current_q_values_all = self.q_network(state_tensor)
        # Select the Q-value for the *action actually taken*
        # gather(1, action_tensor.unsqueeze(-1)) selects Q-value corresponding to action
        predicted_q_value = current_q_values_all.gather(1, action_tensor.unsqueeze(-1)).squeeze(-1)

        # --- Calculate Target Q-values ---
        with torch.no_grad(): # Target calculation shouldn't affect gradients
            # Get Q-values for the *next* state from the same network
            next_q_values_all = self.q_network(next_state_tensor)
            # Find the maximum Q-value among possible next actions
            max_next_q_value = torch.max(next_q_values_all, dim=1)[0] # Get max value, ignore index
            # Calculate TD Target: R + gamma * max_a'(Q(S', a'))
            # Use (1 - done_tensor) trick: if done, future value is 0
            target_q_value = reward_tensor + (self.gamma * max_next_q_value * (1 - done_tensor))

        # --- Calculate Loss ---
        loss = self.loss_fn(predicted_q_value, target_q_value)

        # --- Perform Gradient Descent Step ---
        self.optimizer.zero_grad() # Clear old gradients
        loss.backward()           # Calculate gradients
        self.optimizer.step()      # Update network weights
```

#### Understanding the `DeepQLearningAgent` Class:

*   **`__init__`**:
    *   Stores hyperparameters: learning rate (`lr`), discount factor (`gamma`), and parameters for epsilon decay (`epsilon`, `epsilon_decay`, `epsilon_min`).
    *   Sets the `device` (GPU or CPU).
    *   Defines the `q_network` using `nn.Sequential`. It's a simple Multi-Layer Perceptron (MLP) with ReLU activations, taking the state size as input and outputting `action_size` Q-values.
    *   Moves the network to the selected `device`.
    *   Initializes the `AdamW` optimizer and `MSELoss` function.
*   **`get_action(self, state)`**:
    *   Implements epsilon-greedy: with probability `epsilon`, returns a random action index.
    *   Otherwise (exploitation):
        *   Converts the integer `state` into a **one-hot encoded tensor**. For a 10x10 grid (100 states), state 5 becomes a tensor of size 100 with a '1' at index 5 and zeros elsewhere. This is a standard way to feed categorical states to an MLP.
        *   Adds a batch dimension using `unsqueeze(0)` as PyTorch models typically expect batches.
        *   Uses `with torch.no_grad():` to disable gradient calculations (important for performance during inference).
        *   Passes the `state_tensor` through the `q_network` to get the Q-values for all actions.
        *   Returns the index of the action with the highest Q-value using `torch.argmax()`.
*   **`train(self, state, action, reward, next_state, done)`**:
    *   This method performs one step of the Q-learning update using the network.
    *   Converts all inputs (`state`, `action`, `reward`, `next_state`, `done`) into appropriate tensors on the `device`. States are one-hot encoded.
    *   **Prediction:** Passes the `state_tensor` through `q_network` to get Q-values for all actions. It then uses `gather()` to select the specific Q-value corresponding to the `action` that was actually taken. This is the value we want to compare against the target.
    *   **Target Calculation:**
        *   Uses `torch.no_grad()` as targets should be treated as fixed values for the loss calculation.
        *   Passes the `next_state_tensor` through the **same `q_network`** to get the Q-values for the next state.
        *   Finds the maximum Q-value among actions in the next state (`torch.max(...)[0]`).
        *   Calculates the TD Target: `reward + gamma * max_next_q * (1 - done)`. The `(1 - done)` term elegantly zeroes out the future value if the episode terminated.
    *   **Loss Calculation:** Computes the Mean Squared Error (`loss_fn`) between the `predicted_q_value` (for the action taken) and the calculated `target_q_value`.
    *   **Gradient Update:** Performs the standard PyTorch training steps: `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.

#### The Training Loop Function (`train_dqn_agent`)

Now, let's define the main function that orchestrates the training process over many episodes.

```python
def train_dqn_agent(episodes=500, grid_size=10):
    """
    Trains the DeepQLearningAgent on the GridWorld environment.

    Args:
        episodes (int): Number of training episodes.
        grid_size (int): Size of the GridWorld (e.g., 10 for 10x10).

    Returns:
        DeepQLearningAgent: The trained agent instance.
        list: List of total rewards per episode.
        list: List of exploration rates (epsilon) per episode.
    """
    env = GridWorld(size=grid_size)
    state_size = env.size**2
    action_size = 4

    # Initialize the agent
    agent = DeepQLearningAgent(state_size=state_size, action_size=action_size)

    rewards_history = []
    epsilon_history = []
    print_every = 100 # How often to print progress

    print(f"Starting DQN Training for {episodes} episodes...")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0 # prevent infinite loops in case of issues
        max_steps_per_episode = env.size * env.size * 2 # Heuristic limit

        while not done and step_count < max_steps_per_episode:
            # 1. Agent chooses action
            action = agent.get_action(state)

            # 2. Environment executes action
            next_state, reward, done = env.step(action)

            # 3. Agent learns from the experience
            loss = agent.train(state, action, reward, next_state, done) # Store loss if needed

            # 4. Update state and total reward
            state = next_state
            total_reward += reward
            step_count += 1

        # --- Epsilon Decay ---
        # Decrease exploration rate after each episode
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # --- Logging ---
        rewards_history.append(total_reward)
        epsilon_history.append(agent.epsilon)

        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(rewards_history[-print_every:]) # Avg over last 'print_every' episodes
            print(f"Episode {episode+1}/{episodes} | Avg Reward (Last {print_every}): {avg_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    print(f"DQN Training finished.")
    return agent, rewards_history, epsilon_history
```

#### Understanding `train_dqn_agent`:

*   Initializes the `GridWorld` environment and the `DeepQLearningAgent`.
*   Loops through the specified number of `episodes`.
*   **Inner `while` loop:** Runs steps within an episode until `done` is true (or a step limit is hit).
    *   Calls `agent.get_action()` to select the next move.
    *   Calls `env.step()` to interact with the environment.
    *   Calls `agent.train()` to update the Q-network based on the `(state, action, reward, next_state, done)` transition.
    *   Updates the current `state`.
*   **Epsilon Decay:** After each episode, it reduces `agent.epsilon` by multiplying by `agent.epsilon_decay`, ensuring it doesn't drop below `agent.epsilon_min`. This makes the agent explore less and exploit more as training progresses.
*   Logs rewards and prints progress periodically.
*   Returns the trained `agent` and histories of rewards and epsilon values.

---

### Running the Basic DQN

Let's set the seed and train our basic DQN agent.

```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("\n--- Training Basic Deep Q-Network Agent ---")
trained_agent_dqn, dqn_rewards_history, dqn_epsilon_history = train_dqn_agent(episodes=500) # Using 500 episodes like before

def moving_average(x, window=50):
    return [torch.tensor(x[max(0, i-window):i+1]).float().mean().item() for i in range(len(x))]

# --- Plot Rewards ---
dqn_rewards_smoothed = moving_average(dqn_rewards_history, window=50)

# Calculate optimal reward for this plot context
grid_size_dqn = 10 # Assuming 10x10
shortest_path_len_dqn = 2 * (grid_size_dqn - 1)
optimal_reward_dqn = (shortest_path_len_dqn - 1) * (-1) + 10

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1) # Plot rewards
plt.plot(dqn_rewards_smoothed, label=f"DQN (Smoothed, Window=50)")
plt.axhline(y=optimal_reward_dqn, color='red', linestyle='--', label=f'Optimal Reward ({optimal_reward_dqn})')
plt.xlabel("Episodes")
plt.ylabel("Average Reward (Moving Avg)")
plt.title("DQN Performance (Simple GridWorld)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2) # Plot epsilon decay
plt.plot(dqn_epsilon_history, label="Epsilon Value")
plt.xlabel("Episodes")
plt.ylabel("Exploration Rate (Epsilon)")
plt.title("Epsilon Decay")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
```

Looking at the plots, we can see the agent's average reward increases over time, ideally approaching the optimal value, demonstrating successful learning. The epsilon plot confirms that the exploration rate decreases steadily as intended.

### Visualizing the Learned Policy and Values

What has the DQN agent actually learned? We can visualize the results by examining the Q-values the network predicts for each state.

*   **State Value Map:** For each state, find the maximum Q-value predicted by the network across all actions. This represents the network's estimate of the value (expected future discounted reward) of being in that state.
*   **Optimal Policy Map:** For each state, find the action that corresponds to the highest predicted Q-value. This shows the greedy policy learned by the agent.

```python
# --- Visualization Helper Functions ---
def print_q_value_map(agent, size):
    """Prints a map showing the maximum Q-value for each state."""
    state_size = agent.state_size
    q_value_map = np.zeros((size, size))
    print(f"\n--- Maximum Predicted Q-Value Map ({size}x{size}) ---")
    if state_size != size*size:
        print(f"Error: Agent state size {state_size} doesn't match grid {size}x{size}.")
        return

    with torch.no_grad():
        for state in range(state_size):
            if state >= state_size: continue # Should not happen if state_size is correct
            state_tensor = F.one_hot(torch.tensor(state), num_classes=state_size).float().unsqueeze(0).to(agent.device)
            q_values = agent.q_network(state_tensor)
            max_q_value = torch.max(q_values).item()
            row, col = state // size, state % size
            if row < size and col < size: # Bounds check for safety
                q_value_map[row, col] = max_q_value

    table = tabulate(q_value_map, tablefmt="grid", numalign="right", floatfmt=".2f")
    print(table)

def print_optimal_policy_map(agent, size):
    """Prints a map showing the best action (policy) for each state."""
    state_size = agent.state_size
    policy_map = np.full((size, size), '?', dtype=object)
    action_symbols = {0: '←', 1: '→', 2: '↑', 3: '↓', 'goal': 'G'}
    goal_state = state_size - 1
    print(f"\n--- Optimal Policy Map (Agent's Best Action) ({size}x{size}) ---")
    if state_size != size*size:
        print(f"Error: Agent state size {state_size} doesn't match grid {size}x{size}.")
        return

    with torch.no_grad():
        for state in range(state_size):
            if state >= state_size: continue
            row, col = state // size, state % size
            if row >= size or col >= size: continue # Bounds check

            if state == goal_state:
                policy_map[row, col] = action_symbols['goal']
                continue

            state_tensor = F.one_hot(torch.tensor(state), num_classes=state_size).float().unsqueeze(0).to(agent.device)
            q_values = agent.q_network(state_tensor)
            best_action_index = torch.argmax(q_values).item()
            policy_map[row, col] = action_symbols.get(best_action_index, '?') # Use .get for safety

    table = tabulate(policy_map, tablefmt="grid", stralign="center")
    print(table)

# --- Visualize DQN Results ---
print("\n--- Visualizing Learned Values and Policy from DQN ---")
print_q_value_map(trained_agent_dqn, grid_size_dqn)
print_optimal_policy_map(trained_agent_dqn, grid_size_dqn)
```

<!-- Code Cell Output: Prints the Q-value map and the optimal policy map derived from the trained DQN. -->

The State Value Map should show a gradient, with values generally increasing as states get closer to the goal (state 99). The Optimal Policy Map should show arrows indicating the learned path towards the goal 'G' from most states.

---

### Basic DQN Recap

We successfully transitioned from tabular methods (Q-tables) to function approximation using a Deep Q-Network. By training a neural network to estimate Q-values, we can handle larger state spaces more effectively. We implemented a basic DQN agent, trained it on the simple GridWorld using epsilon decay, and visualized its learned value function and policy.

However, this basic DQN implementation has stability issues, especially when applied to more complex problems. The next section explores these issues and introduces key improvements: Experience Replay and Fixed Q-Targets.

---

Alright, so our first DQN agent did a pretty decent job on the simple, open GridWorld. But let's be real, most interesting problems aren't quite that straightforward. What happens when we add some common challenges, like obstacles or a bit of randomness? Let's make our GridWorld tougher and see how our agent handles it.

We'll tweak our `GridWorld` class to add two things:
*   **Walls:** Impassable squares the agent just can't enter.
*   **Stochasticity:** A bit of randomness in movement – like a slippery floor!

### Defining a Harder GridWorld

Here's how we'll change the `GridWorld` code:
*   `__init__`: Now accepts `stochasticity` (probability of random move) and a list of `walls` (integer state indices).
*   `step`:
    *   At the beginning, there's a `stochasticity` chance the chosen `action` is overridden by a random one.
    *   After calculating the potential `next_position`, it checks if that position is in the `walls` set. If it is, the agent doesn't move (position remains unchanged) but still incurs the -1 step penalty.

Let's define some walls and visualize this more challenging environment.

```python
# Helper function to visualize grid with walls
import numpy as np
from tabulate import tabulate

def print_grid_with_walls_and_rewards(size, walls=None):
    reward_map = np.full((size, size), -1, dtype=int)

    goal_state_index = size * size - 1
    goal_row = goal_state_index // size
    goal_col = goal_state_index % size
    reward_map[goal_row, goal_col] = 10

    grid = np.full((size, size), '.', dtype='<U1')

    if walls is not None:
        for wall in walls:
            row = wall // size
            col = wall % size
            grid[row, col] = 'X'

    grid[goal_row, goal_col] = 'G'

    for row in range(size):
        for col in range(size):
            if grid[row, col] == '.':
                grid[row, col] = str(reward_map[row, col])

    print(f"\nGrid Map with Walls and Rewards ({size}x{size}):")
    table = tabulate(grid.tolist(), tablefmt="grid", numalign="center", stralign="center")
    print(table)

walls = [20,21,22,23,24,25,26,27,59,58,57,56,55,54,53,52,62,72,82,75,85,95]
print_grid_with_walls_and_rewards(size=10, walls=walls)
```

You can see how these 'X's create barriers the agent has to navigate around.

Now, let's define the updated `GridWorld` class incorporating these changes.

```python
import torch
import random

class GridWorld:
    def __init__(self, size=10, stochasticity=0.2, walls=None):
        self.size = size
        self.stochasticity = stochasticity
        self.manual_walls = walls # Store the list passed in
        self.observation_space_n = size*size
        self.action_space_n = 4
        self.reset()

    def reset(self):
        self.agent_position = 0
        self.goal_position = self.size * self.size - 1

        # Initialize walls as a set for efficient lookup
        if self.manual_walls is not None:
            self.walls = set(self.manual_walls)
        else:
            self.walls = set()

        return self.agent_position # Return initial state

    def step(self, action):
        # Apply stochasticity: sometimes take a random action
        if random.random() < self.stochasticity:
            action = random.randrange(self.action_space_n)

        # Calculate potential next position based on (potentially random) action
        next_position = self.agent_position
        current_pos = self.agent_position
        row, col = current_pos // self.size, current_pos % self.size

        if action == 0 and col > 0:
            next_position -= 1  # Left
        elif action == 1 and col < self.size - 1:
            next_position += 1  # Right
        elif action == 2 and row > 0:
            next_position -= self.size  # Up
        elif action == 3 and row < self.size - 1:
            next_position += self.size  # Down

        # Check for walls: only update position if next_position is not a wall
        if next_position not in self.walls:
            self.agent_position = next_position
        # If it's a wall, self.agent_position remains unchanged

        # Determine reward and done status
        done = self.agent_position == self.goal_position
        reward = 10 if done else -1

        return self.agent_position, reward, done
```

### Testing Basic DQN on the Harder GridWorld (Conceptual)

Okay, so we've made the environment significantly trickier. What happens if we take our *exact same basic DQN agent* (the `DeepQLearningAgent` class *without* Experience Replay and Target Networks) and try to train it on this new, harder `GridWorld` with walls and stochasticity?

*(Note: We won't run this specific experiment code, I have ran it and it did not train at all. We'll just describe the likely outcome here.)*

**Expected Outcome:**
If we were to run the basic DQN training loop (`train_dqn_agent` using the basic `DeepQLearningAgent`) on this harder environment, the results would likely be poor. We would expect to see a learning curve that:
*   Starts very low (agent bumping into walls, acting randomly).
*   Fails to improve significantly, possibly bouncing around erratically.
*   Does *not* converge towards the optimal reward for this harder grid.

The plot would look "bad," showing a failure to learn an effective strategy, even though the exploration rate decays as expected.

### Why Does Basic DQN Fail Here?

The script highlights two main reasons why the basic DQN struggles with added complexity:

1.  **Moving Target Problem:** The target Q-value (`Reward + gamma * max Q(next_state)`) is calculated using the *same network* (`q_network`) that is being updated. Every weight update changes the target itself, leading to instability and oscillations, preventing convergence.
2.  **Correlated & Scarce Data:**
    *   The agent learns from experiences sequentially (`s, a, r, s', done`). These are highly correlated.
    *   In a harder environment, finding the goal (positive reward sequence) might be rare. Basic DQN learns from each experience *once* and then discards it. These rare successful experiences get drowned out by more common, less informative experiences (like hitting walls or moving randomly), making learning inefficient.

### Stabilizing DQN: Experience Replay and Fixed Q-Targets

Luckily, these problems can be addressed with two key techniques:

1.  **Experience Replay:**
    *   **Mechanism:** Store experiences `(s, a, r, s', done)` in a large memory buffer (`replay_buffer`). Instead of training on the last experience, sample a random mini-batch from this buffer for each training step.
    *   **Benefits:** Breaks temporal correlations by mixing old and new experiences. Allows the agent to learn multiple times from rare but important transitions. Increases data efficiency.

2.  **Fixed Q-Targets:**
    *   **Mechanism:** Use *two* networks:
        *   `q_network` (Online Network): Updated frequently, used to select actions.
        *   `target_q_network`: A copy of `q_network`. Its weights are frozen and only updated periodically (e.g., every `target_update_frequency` steps) by copying weights from `q_network`.
    *   **Target Calculation:** Use the `target_q_network` to calculate the `max Q(S', a')` part of the TD target: `target = reward + gamma * max(TargetNetwork(S')) * (1 - done)`.
    *   **Benefits:** Provides a stable target for the `q_network` to learn towards, reducing oscillations caused by the moving target problem.

These techniques are almost always used together in modern DQN implementations.

---

### Enhanced DQN Implementation (with Replay Buffer & Target Network)

Let's modify our agent to include these improvements. We'll define the network architecture separately for clarity.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random # For sampling replay buffer
from collections import deque # Efficient replay buffer implementation (alternative to list)

# --- Q-Network Architecture ---
class QNetwork(nn.Module):
    """ Simple MLP for Q-value approximation """
    def __init__(self, state_size, action_size, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state_one_hot):
        x = F.relu(self.fc1(state_one_hot))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

# --- Enhanced DQN Agent Class ---
class DeepQLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001,
                 discount_factor=0.99, exploration_rate=1.0,
                 exploration_decay=0.998, exploration_min=0.01,
                 target_update_frequency=100, # Frequency to update target net
                 buffer_size=10000, batch_size=64): # Replay buffer params

        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = exploration_min
        self.target_update_freq = target_update_frequency
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.train_step_counter = 0 # To track when to update target network

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # --- Create Q-Network and Target Network ---
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_q_network = QNetwork(state_size, action_size).to(self.device)
        # Initialize target network with same weights as q_network
        self.update_target_network()
        self.target_q_network.eval() # Set target network to evaluation mode

        # --- Optimizer and Loss ---
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # --- Replay Buffer ---
        # Using deque for efficient appends and pops from both ends
        self.replay_buffer = deque(maxlen=self.buffer_size)

    def update_target_network(self):
        """ Copy weights from q_network to target_q_network """
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def get_action(self, state):
        """ Epsilon-greedy action selection using the online q_network """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = F.one_hot(torch.tensor(state, device=self.device), num_classes=self.state_size).float().unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values, dim=1).item()
                return action

    def store_transition(self, state, action, reward, next_state, done):
        """ Store experience in replay buffer and trigger training if buffer is ready """

        self.replay_buffer.append((state, action, reward, next_state, done))

        # Check if buffer has enough samples to start training
        if len(self.replay_buffer) >= self.batch_size:
            self.train_from_replay() # Call training method

            # Increment step counter after a training step is performed
            self.train_step_counter += 1

            # Periodically update the target network
            if self.train_step_counter % self.target_update_freq == 0:
                self.update_target_network()


    def train_from_replay(self):
        """ Sample a minibatch from replay buffer and perform training step """
        # Don't train if buffer is smaller than batch size
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a random minibatch of experiences
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        # Unzip the batch into separate lists/tuples
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert batch elements to tensors
        states_tensor = F.one_hot(torch.tensor(states, device=self.device), num_classes=self.state_size).float()
        next_states_tensor = F.one_hot(torch.tensor(next_states, device=self.device), num_classes=self.state_size).float()
        actions_tensor = torch.tensor(actions, device=self.device).long().unsqueeze(1) # Need shape [batch_size, 1] for gather
        rewards_tensor = torch.tensor(rewards, device=self.device).float().unsqueeze(1) # Shape [batch_size, 1]
        dones_tensor = torch.tensor(dones, device=self.device).bool().unsqueeze(1) # Shape [batch_size, 1]

        # --- Calculate Predicted Q-values ---
        # Get Q-values for the actions actually taken in the states from the batch
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor)

        # --- Calculate Target Q-values ---
        with torch.no_grad():
            # Get next state Q-values from the *target* network
            next_q_values_target = self.target_q_network(next_states_tensor)
            # Find the max Q-value among actions for each next state
            max_next_q_values = next_q_values_target.max(1, keepdim=True)[0] # Shape [batch_size, 1]
            # Calculate target: R + gamma * max Q_target(S', a') * (NOT done)
            # Use ~dones_tensor for element-wise NOT
            target_q_values = rewards_tensor + self.gamma * max_next_q_values * (~dones_tensor)

        # --- Loss Calculation ---
        loss = self.loss_fn(current_q_values, target_q_values)

        # --- Gradient Update ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

#### Understanding the Enhanced `DeepQLearningAgent`:

*   **`__init__`**:
    *   Now takes `buffer_size`, `batch_size`, `target_update_frequency` hyperparameters.
    *   Initializes *two* `QNetwork` instances: `q_network` and `target_q_network`.
    *   Calls `update_target_network()` to synchronize them initially. `target_q_network` is set to `eval()` mode.
    *   Initializes `replay_buffer` (using `deque` for efficiency) and `train_step_counter`.
*   **`update_target_network`**: Helper method to copy weights from `q_network` to `target_q_network`.
*   **`get_action`**: Unchanged, still uses `q_network` for decisions.
*   **`store_transition`**: Replaces the old `train` method.
    *   Appends the experience `(s, a, r, s', done)` to the `replay_buffer`.
    *   Manages buffer size (removes oldest if full).
    *   **Crucially:** Only if the buffer has enough samples (`>= batch_size`), it calls `train_from_replay()`.
    *   Increments `train_step_counter` *after* a potential training step.
    *   Calls `update_target_network()` every `target_update_frequency` training steps.
*   **`train_from_replay`**:
    *   Samples a `minibatch` randomly from the `replay_buffer`.
    *   Converts the batch data into tensors (states are one-hot encoded).
    *   Calculates `current_q_values` using `q_network` and `gather`.
    *   Calculates `target_q_values` using `reward` and `max_next_q_values` obtained from the **`target_q_network`**. The `(~dones_tensor)` handles terminal states correctly for the batch.
    *   Computes loss and performs the optimizer step on `q_network`.

#### The Enhanced Training Loop

The main training loop function now only needs to call `agent.store_transition` instead of `agent.train`.

```python
def train_dqn_agent(episodes=1000, grid_size=10, stochasticity=0.2, walls=None):
    """ Trains the enhanced DQN agent with Replay Buffer and Target Network """

    env = GridWorld(size=grid_size, stochasticity=stochasticity, walls=walls)
    agent = DeepQLearningAgent(state_size=env.size**2, action_size=4)

    episode_rewards = []
    print_every = 50

    print(f"Starting Enhanced DQN Training for {episodes} episodes...")

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            # Store transition, which internally triggers training and target updates
            agent.store_transition(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            step_count += 1

        # Epsilon decay after each episode
        agent.epsilon = max(
            agent.epsilon_min,
            agent.epsilon * agent.epsilon_decay
        )

        episode_rewards.append(total_reward)

        # Logging progress
        if (episode + 1) % print_every == 0:
            avg_reward = np.mean(episode_rewards[-print_every:])
            print(f"Episode {episode+1}/{episodes}: Steps = {step_count}, Total Reward = {total_reward:.2f}, Avg Reward (Last {print_every}) = {avg_reward:.2f}, Exp Rate = {agent.epsilon:.3f}")

    print("Enhanced DQN Training finished.")
    return agent, episode_rewards # Return agent and rewards for analysis
```

The key change is replacing `agent.train(...)` with `agent.store_transition(...)`. The agent now manages its own training frequency and target updates internally based on the buffer state and step counter.

---

### Running the Enhanced DQN on the Harder GridWorld

Let's train this improved agent on the GridWorld with walls and stochasticity.

```python
# --- Setup and Training ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

walls = [20,21,22,23,24,25,26,27, 59,58,57,56,55,54,53,52, 62,72,82, 75,85,95]

print("\n--- Training Enhanced DQN Agent (Replay Buffer + Target Network) ---")
trained_agent_enhanced, enhanced_rewards = train_dqn_agent(
    episodes=1000,
    grid_size=10,
    stochasticity=0.2,
    walls=walls
)

# --- Plotting Results ---
def moving_average(data, window_size=50):
    if not data or window_size <= 0: return []
    return np.convolve(np.array(data), np.ones(window_size)/window_size, mode='valid') # Use 'valid' mode

optimal_reward_hard = -27

plt.figure(figsize=(10, 5))
plt.plot(enhanced_rewards, label='Episode Reward', alpha=0.6)
ma_rewards = moving_average(enhanced_rewards, window_size=50)
ma_episodes = np.arange(len(ma_rewards)) + 50
plt.plot(ma_episodes, ma_rewards, label='Moving Average (50 episodes)', linewidth=2)

plt.axhline(y=optimal_reward_hard, color='red', linestyle='--', label=f'Reference Reward ({optimal_reward_hard})')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Enhanced DQN Training Rewards (Hard GridWorld)")
plt.legend()
plt.grid(True)
plt.show()
```

#### Analyzing Enhanced DQN Results

Okay, now this looks much better! Let's break down the reward plot for the enhanced DQN:

*   **Initial Phase:** The average reward likely stays low and flat for a while. The agent is exploring randomly, hitting walls, experiencing stochastic slips, and importantly, filling up the replay buffer. Little effective learning happens until the buffer has enough diverse experiences.
*   **Learning Phase:** Once the buffer is sufficiently full (>= `batch_size`), `train_from_replay` starts getting called regularly. The combination of sampling diverse experiences (Experience Replay) and learning towards stable targets (Fixed Q-Targets) allows the agent to learn much more effectively. We should see the average reward start to shoot up significantly.
*   **Convergence Phase:** Towards the end, the curve likely starts to flatten out again, hopefully at a much higher level than the basic DQN achieved. This indicates the agent's performance is converging.

The final average reward might still be somewhat below the theoretical optimum (calculated as -27 for this specific hard grid) due to the inherent `stochasticity` (20% chance of random moves) preventing perfect execution of the learned policy. However, the agent has clearly learned a robust and effective strategy for this much harder environment, demonstrating the power of Experience Replay and Fixed Q-Targets.

You can check the optimal policy map to check that the agent has learned the correct path.

```python
import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

def print_optimal_policy_map(agent, size, walls=None):
    state_size = agent.state_size
    policy_map = np.full((size, size), '?', dtype=object)

    action_symbols = {
        0: '←',
        1: '→',
        2: '↑',
        3: '↓',
        'goal': 'G',
        'wall': 'X'
    }

    wall_set = set(walls) if walls else set()
    goal_state = state_size - 1

    with torch.no_grad():
        for state in range(state_size):
            row = state // size
            col = state % size

            if state == goal_state:
                policy_map[row, col] = action_symbols['goal']
            elif state in wall_set:
                policy_map[row, col] = action_symbols['wall']
            else:
                state_tensor = F.one_hot(torch.tensor(state), num_classes=state_size).float().to(agent.device)
                q_values = agent.q_network(state_tensor)
                best_action_index = torch.argmax(q_values).item()
                policy_map[row, col] = action_symbols[best_action_index]

    print(f"Optimal Policy Map with Walls and Goal ({size}x{size}):")
    table = tabulate(policy_map.tolist(), tablefmt="grid", stralign="center")
    print(table)

walls = [20,21,22,23,24,25,26,27,59,58,57,56,55,54,53,52,62,72,82,75,85,95]
print_optimal_policy_map(trained_agent_enhanced, size=10, walls=walls)

```


**Off-Policy Nature:** It's worth noting that DQN, even with these enhancements, remains an **off-policy** algorithm. The use of the replay buffer means it can learn from experiences generated by older versions of its policy, not just the very latest one.

This comparison highlights why Experience Replay and Target Networks are essential components for building stable and effective DQN agents capable of tackling more complex RL problems.

---

### Transitioning from Value-Based to Policy-Based Methods

Alright, so far on our RL adventure, we've pretty much stuck to one main approach. Think back to Q-learning, SARSA, and our Deep Q-Network. What were they all trying to do?

They were all about learning a **value function** – figuring out a score (like Q-values) that tells us how good it is to take action A when we're in state S. This score represented the expected total future reward. Then, the agent derived its policy indirectly by picking the action with the highest score (with some exploration).

That approach works great, but there's a whole other family of RL algorithms that tackles the problem differently: **Policy Gradient methods**.

Instead of learning values first, Policy Gradient methods aim to learn the **policy directly**.

The idea is to represent the agent's policy using a function – often another neural network. You feed the current state into this "policy network," and it directly outputs information about which action to take:
*   For **discrete** actions (like Left/Right): It might output the *probability* of taking each action.
*   For **continuous** actions (like steering angle): It might output the parameters (e.g., mean and standard deviation) of a probability distribution from which to sample the action.

The network *is* the policy. The learning goal then becomes: how do we adjust the network's parameters directly so that the actions it chooses lead to higher cumulative rewards over time? This often involves techniques that estimate the "gradient" of the policy's performance with respect to its parameters.