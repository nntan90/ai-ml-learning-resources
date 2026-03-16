# 🎮 Reinforcement Learning

Tài liệu về Reinforcement Learning — từ lý thuyết nền tảng (Sutton & Barto) đến Deep RL hiện đại (DQN, A3C, PPO).

---

## 📚 Repos trong chủ đề này

| Repo | Stars | Tác giả | Mô tả |
|------|-------|---------|-------|
| [awesome-rl](https://github.com/aikorea/awesome-rl) | ⭐ 9.6K+ | aikorea | Curated list of RL resources — lectures, books, papers, code |
| [RL-Algorithms-From-Scratch](https://github.com/KhashayarRahimi/RL-From-Scratch) | ⭐ 72 | KhashayarRahimi | 9 notebooks implement RL từ đầu theo Sutton & Barto |

---

## 1. aikorea/awesome-rl

> **GitHub**: https://github.com/aikorea/awesome-rl  
> **Stars**: ~9,600 ⭐  
> **Tác giả**: aikorea  
> **License**: MIT

### Mô tả

Danh sách tổng hợp các tài nguyên học Reinforcement Learning chất lượng cao: bài giảng, sách, papers, và code. Được cộng đồng AI Việt Nam và thế giới đóng góp.

### 🎓 Bài giảng (Lectures)

| Khóa học | Giảng viên | Link |
|----------|-----------|------|
| Introduction to Reinforcement Learning | David Silver (DeepMind) | [YouTube](https://www.youtube.com/watch?v=2pWv7GOvuf0) |
| Deep RL Bootcamp | UC Berkeley | [Website](https://sites.google.com/view/deep-rl-bootcamp) |
| CS285 Deep RL | Sergey Levine (Berkeley) | [Course Page](http://rail.eecs.berkeley.edu/deeprlcourse/) |
| CS234 Reinforcement Learning | Emma Brunskill (Stanford) | [Course Page](http://web.stanford.edu/class/cs234/) |
| Advanced Deep Learning & RL | UCL / DeepMind | [YouTube](https://www.youtube.com/playlist?list=PLqYmG7hTraZDNJre23vqCGIVpfZ_K2RZs) |

### 📖 Sách (Books)

| Sách | Tác giả | Ghi chú |
|------|---------|---------|
| **Reinforcement Learning: An Introduction** (2nd ed.) | Sutton & Barto | Kinh điển — free PDF online |
| Algorithms for Reinforcement Learning | Csaba Szepesvári | Ngắn gọn, toán học |
| Deep Reinforcement Learning Hands-On | Maxim Lapan | Thực hành với PyTorch |

### 📄 Papers theo chủ đề

#### Dynamic Programming
| Paper | Mô tả |
|-------|-------|
| Bellman (1957) | Dynamic Programming gốc |
| Policy Iteration | Howard (1960) |
| Value Iteration | Bellman optimality |

#### Monte Carlo Methods
| Paper | Mô tả |
|-------|-------|
| Monte Carlo Control | First-visit / Every-visit MC |

#### Temporal Difference (TD)
| Paper | Mô tả |
|-------|-------|
| TD(λ) | Sutton (1988) — seminal TD paper |
| SARSA | On-policy TD control |
| Q-Learning | Watkins & Dayan (1992) |

#### Policy Gradient
| Paper | Mô tả |
|-------|-------|
| REINFORCE | Williams (1992) |
| Actor-Critic | Advantage function |
| PPO | Schulman et al. (2017) — widely used |
| TRPO | Schulman et al. (2015) |

#### Deep RL
| Paper | Mô tả |
|-------|-------|
| DQN | Mnih et al. (2013/2015) — Atari games |
| Double DQN | Van Hasselt et al. (2016) |
| Dueling DQN | Wang et al. (2016) |
| A3C | Mnih et al. (2016) — asynchronous |
| A2C | Synchronous variant of A3C |
| PPO | Proximal Policy Optimization |
| SAC | Soft Actor-Critic — continuous actions |
| TD3 | Twin Delayed DDPG |
| AlphaGo/AlphaZero | DeepMind — game mastery |

---

## 2. KhashayarRahimi/RL-Algorithms-From-Scratch

> **GitHub**: https://github.com/KhashayarRahimi/RL-From-Scratch  
> **Stars**: ~72 ⭐  
> **Tác giả**: KhashayarRahimi  
> **License**: MIT

### Mô tả

9 Jupyter notebooks implement các thuật toán RL từ đầu bằng Python thuần, theo đúng trình tự sách **Sutton & Barto "Reinforcement Learning: An Introduction"**. Lý tưởng cho người muốn hiểu sâu cơ chế hoạt động của từng thuật toán.

### Nội dung (9 Notebooks)

| Notebook | Chương S&B | Chủ đề |
|----------|-----------|--------|
| 01 | Ch. 4 | **Dynamic Programming** — Policy Evaluation, Policy Improvement, Policy Iteration, Value Iteration |
| 02 | Ch. 5 | **Monte Carlo Methods** — First-visit MC, Every-visit MC, MC Control, Off-policy MC |
| 03 | Ch. 6 | **Temporal Difference Learning** — TD(0), SARSA, Q-Learning |
| 04 | Ch. 7 | **n-step Bootstrapping** — n-step TD, n-step SARSA, n-step Q(σ) |
| 05 | Ch. 8 | **Planning & Learning** — Dyna-Q, Prioritized Sweeping, Monte Carlo Tree Search |
| 06 | Ch. 9 | **Function Approximation (Value-Based)** — Linear FA, Semi-gradient TD |
| 07 | Ch. 10 | **Function Approximation (Control)** — Semi-gradient SARSA, Episodic semi-gradient |
| 08 | Ch. 12 | **Eligibility Traces** — TD(λ), SARSA(λ), True online TD(λ) |
| 09 | Ch. 13 | **Policy Gradient** — REINFORCE, REINFORCE with Baseline, Actor-Critic |

### Điểm nổi bật
- Code rõ ràng, có comment giải thích từng bước
- Môi trường: OpenAI Gym (CartPole, MountainCar, FrozenLake, GridWorld)
- Visualizations: learning curves, value functions, policies
- Hoàn toàn implement from scratch — không dùng RL libraries

---

## 🗺️ Lộ trình học Reinforcement Learning

```
1. Nền tảng Toán học (prerequisite)
   └── Probability & Statistics
   └── Linear Algebra cơ bản
   └── → 01-math-foundations.md

2. RL Foundations
   └── awesome-rl: Sutton & Barto book (free PDF)
   └── awesome-rl: David Silver lectures (10 videos)

3. Implement từ đầu
   └── RL-From-Scratch: Notebook 01 (Dynamic Programming)
   └── RL-From-Scratch: Notebook 02 (Monte Carlo)
   └── RL-From-Scratch: Notebook 03 (TD Learning / Q-Learning)

4. Advanced Tabular Methods
   └── RL-From-Scratch: Notebooks 04-05 (n-step, Planning)

5. Function Approximation & Neural Networks
   └── RL-From-Scratch: Notebooks 06-08
   └── awesome-rl: DQN paper (Mnih et al. 2015)

6. Policy Gradient & Deep RL
   └── RL-From-Scratch: Notebook 09 (Policy Gradient)
   └── awesome-rl: PPO, A3C, SAC papers
   └── awesome-rl: CS285 Berkeley (Sergey Levine)

7. Ứng dụng thực tế
   └── awesome-rl: Game AI (Atari, AlphaGo)
   └── awesome-rl: Robotics, NLP, Finance
```

---

## 🔗 Tài nguyên liên quan

- **Neural Networks nền tảng**: [03-neural-networks.md](./03-neural-networks.md) — cần hiểu NN trước khi học Deep RL
- **Math Foundations**: [01-math-foundations.md](./01-math-foundations.md)
- **AI Agents** (liên quan): [08-ai-agents.md](./08-ai-agents.md) — nhiều agent frameworks dùng RL
