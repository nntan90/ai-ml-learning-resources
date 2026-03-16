# Neural Networks & Deep Learning

Từ nền tảng backpropagation đến các kiến trúc hiện đại.

---

## 🧠 Neural Networks: Zero to Hero — Karpathy

**Repo:** [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) | ⭐ 20,904  
**Author:** Andrej Karpathy (ex-OpenAI, ex-Tesla AI)  
**Format:** YouTube videos + Jupyter notebooks, build từng thứ từ scratch

### Chương trình (8 lectures)
| # | Tiêu đề | Nội dung | Link |
|---|---------|---------|------|
| 1 | **Micrograd: Backpropagation** | Tự build autograd engine. Hiểu sâu backprop qua scalars. | [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0) |
| 2 | **Makemore — Bigrams** | Character-level LM, `torch.Tensor`, train/sample/loss | [YouTube](https://www.youtube.com/watch?v=PaCmpygFfXo) |
| 3 | **Makemore — MLP** | MLP, learning rate, hyperparameters, train/dev/test | [YouTube](https://youtu.be/TCH_1BHY58I) |
| 4 | **Makemore — BatchNorm** | Forward activations, backward gradients, Batch Normalization | [YouTube](https://youtu.be/P6sfmUTpUmc) |
| 5 | **Makemore — Backprop Ninja** | Manual backprop qua toàn bộ network (không dùng autograd) | [YouTube](https://youtu.be/q8SA3rM6ckI) |
| 6 | **Makemore — WaveNet** | Deep MLP → CNN, `torch.nn`, DeepMind WaveNet-style | [YouTube](https://youtu.be/t3YJ5hKiMQ0) |
| 7 | **Build GPT from Scratch** | Transformer theo "Attention is All You Need", GPT-2/3 | [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| 8 | **GPT Tokenizer (BPE)** | Build BPE tokenizer từ scratch, encode/decode | [YouTube](https://www.youtube.com/watch?v=zduSFxRajkE) |

---

## 📝 Annotated Deep Learning Paper Implementations

**Repo:** [labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) | ⭐ 66,005  
**Website:** [nn.labml.ai](https://nn.labml.ai) — code + mathematical notes side-by-side

### Implementations theo category

#### Transformers
- Multi-headed Attention, Flash Attention (Triton), Transformer XL
- Rotary Positional Embeddings (RoPE), ALiBi, GPT Architecture
- Switch Transformer (MoE), Vision Transformer (ViT), MLP-Mixer
- **LoRA** (Low-Rank Adaptation) — fine-tuning efficient

#### Diffusion Models
- DDPM, DDIM, Latent Diffusion, Stable Diffusion

#### GANs
- Original GAN, DCGAN, CycleGAN, WGAN, StyleGAN 2

#### Recurrent Networks
- LSTM, Recurrent Highway Networks, HyperLSTM, Sketch RNN

#### CNNs & Vision
- ResNet, ConvMixer, Capsule Networks, U-Net

#### Reinforcement Learning
- PPO (Proximal Policy Optimization)
- DQN (Deep Q-Networks)

#### Optimizers
- Adam, AMSGrad, RAdam, AdaBelief, Sophia-G

#### Normalization
- Batch Norm, Layer Norm, Group Norm, DeepNorm

#### Advanced
- Knowledge Distillation, LLM.int8() Quantization
- Sampling: Greedy / Temperature / Top-k / Nucleus
- Zero3 Memory Optimization
