# AI & Machine Learning Learning Compendium — Part 2
> Tổng hợp từ 11 GitHub repositories — dành cho LLM Notebook learning
> Last updated: March 2026

---

## Mục lục

6. [Neural Networks: Zero to Hero — Karpathy](#6-neural-networks-zero-to-hero)
7. [Hands-On Large Language Models — O'Reilly Book](#7-hands-on-large-language-models)
8. [Prompt Engineering Guide — DAIR.AI](#8-prompt-engineering-guide)
9. [AI Agents for Beginners — Microsoft](#9-ai-agents-for-beginners-microsoft)
10. [GenAI Agents — NirDiamant](#10-genai-agents-nirdiamanth)
11. [ML for Beginners — Microsoft](#11-ml-for-beginners-microsoft)
12. [100 Days of ML Code](#12-100-days-of-ml-code)
13. [All Algorithms in Python — TheAlgorithms](#13-all-algorithms-in-python)
14. [Mathematics for Machine Learning — Book](#14-mathematics-for-machine-learning)
15. [Made With ML — MLOps](#15-made-with-ml)
16. [Annotated Deep Learning Paper Implementations](#16-annotated-deep-learning-paper-implementations)

---

## 6. Neural Networks: Zero to Hero

**Source:** [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
**Stars:** 20,904 ⭐ | **Author:** Andrej Karpathy (ex-OpenAI, ex-Tesla AI)
**Language:** Jupyter Notebook

### Giới thiệu

Khóa học neural networks từ nền tảng tuyệt đối, build từ zero. Dạy bằng cách **code từng thành phần** — không dùng black box. Mỗi bài là YouTube video + Jupyter notebook.

---

### Chương trình học (8 lectures)

| Lecture | Tiêu đề | Nội dung chính | Link |
|---------|---------|----------------|------|
| 1 | **Micrograd: Backpropagation** | Tự xây dựng autograd engine từ đầu. Hiểu sâu backprop qua scalar operations. | [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0) · [Notebook](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd) |
| 2 | **Makemore Part 1: Bigrams** | Character-level language model bằng bigrams. Giới thiệu `torch.Tensor`, framework language modeling (training, sampling, loss). | [YouTube](https://www.youtube.com/watch?v=PaCmpygFfXo) · [Notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb) |
| 3 | **Makemore Part 2: MLP** | Multilayer Perceptron character-level LM. Basics: learning rate, hyperparameters, train/dev/test splits, under/overfitting. | [YouTube](https://youtu.be/TCH_1BHY58I) · [Notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb) |
| 4 | **Makemore Part 3: BatchNorm** | Internals của deep MLP — forward activations, backward gradients, pitfalls của improper scaling. Giới thiệu **Batch Normalization**. | [YouTube](https://youtu.be/P6sfmUTpUmc) · [Notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part3_bn.ipynb) |
| 5 | **Makemore Part 4: Backprop Ninja** | Manual backprop qua cross entropy, linear layers, tanh, BatchNorm, embedding table — không dùng `loss.backward()`. | [YouTube](https://youtu.be/q8SA3rM6ckI) · [Notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb) · [Colab](https://colab.research.google.com/drive/1WV2oi2fh9XXyldh02wupFQX0wh5ZC-z-) |
| 6 | **Makemore Part 5: WaveNet** | Deepening MLP thành tree-like structure → CNN architecture tương tự WaveNet (DeepMind 2016). Học cách dùng `torch.nn`. | [YouTube](https://youtu.be/t3YJ5hKiMQ0) · [Notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part5_cnn1.ipynb) |
| 7 | **Build GPT from Scratch** | Xây dựng GPT hoàn chỉnh theo paper "Attention is All You Need" và GPT-2/GPT-3 của OpenAI. | [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| 8 | **GPT Tokenizer (BPE)** | Xây dựng tokenizer từ scratch. BPE algorithm, encode/decode, tại sao tokenization gây ra nhiều weird behaviors trong LLMs. | [YouTube](https://www.youtube.com/watch?v=zduSFxRajkE) · [minBPE](https://github.com/karpathy/minbpe) · [Colab](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L) |

### Repos liên quan của Karpathy
- [micrograd](https://github.com/karpathy/micrograd) — tiny autograd engine (25 lines)
- [makemore](https://github.com/karpathy/makemore) — character-level LM
- [minBPE](https://github.com/karpathy/minbpe) — BPE tokenizer

### Lộ trình học đề xuất
1. **Lecture 1** (micrograd) → hiểu backprop
2. **Lecture 2-6** (makemore) → từ bigrams → CNN
3. **Lecture 7** → build GPT
4. **Lecture 8** → build tokenizer

---

## 7. Hands-On Large Language Models

**Source:** [HandsOnLLM/Hands-On-Large-Language-Models](https://github.com/HandsOnLLM/Hands-On-Large-Language-Models)
**Stars:** 24,030 ⭐ | **Publisher:** O'Reilly | **Authors:** Jay Alammar & Maarten Grootendorst
**Language:** Jupyter Notebook

### Giới thiệu

Code repo chính thức cho sách *Hands-On Large Language Models* (O'Reilly). Được gọi là **"The Illustrated LLM Book"** với gần **300 custom figures**. Endorsed bởi Andrew Ng, Nils Reimers (creator của sentence-transformers), và Josh Starmer (StatQuest).

---

### Nội dung sách (12 chapters)

| Chapter | Chủ đề | Notebook | Nội dung |
|---------|--------|----------|---------|
| 1 | **Introduction to Language Models** | [Colab](https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter01/Chapter%201%20-%20Introduction%20to%20Language%20Models.ipynb) | Tổng quan về LMs, lịch sử, kiến trúc cơ bản |
| 2 | **Tokens and Embeddings** | [Colab](https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter02/Chapter%202%20-%20Tokens%20and%20Token%20Embeddings.ipynb) | Tokenization, word/token embeddings, vector representations |
| 3 | **Looking Inside Transformer LLMs** | [Colab](https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter03/Chapter%203%20-%20Looking%20Inside%20LLMs.ipynb) | Kiến trúc Transformer bên trong: attention, FFN, layer norm |
| 4 | **Text Classification** | [Colab](https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter04/Chapter%204%20-%20Text%20Classification.ipynb) | Phân loại văn bản với LLMs |
| 5 | **Text Clustering & Topic Modeling** | [Colab](https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter05/Chapter%205%20-%20Text%20Clustering%20and%20Topic%20Modeling.ipynb) | Clustering với embeddings, BERTopic |
| 6 | **Prompt Engineering** | [Colab](https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter06/Chapter%206%20-%20Prompt%20Engineering.ipynb) | Kỹ thuật prompting, zero-shot, few-shot, CoT |
| 7 | **Advanced Text Generation** | [Colab](https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter07/Chapter%207%20-%20Advanced%20Text%20Generation%20Techniques%20and%20Tools.ipynb) | Sampling, temperature, top-k/p, output control |
| 8 | **Semantic Search & RAG** | [Colab](https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter08/Chapter%208%20-%20Semantic%20Search.ipynb) | Vector search, embeddings, RAG pipeline |
| 9 | **Multimodal LLMs** | [Colab](https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter09/Chapter%209%20-%20Multimodal%20Large%20Language%20Models.ipynb) | Vision-Language models, image+text |
| 10 | **Creating Text Embedding Models** | [Colab](https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter10/Chapter%2010%20-%20Creating%20Text%20Embedding%20Models.ipynb) | Training embedding models, contrastive learning |
| 11 | **Fine-tuning Representation Models** | [Colab](https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter11/Chapter%2011%20-%20Fine-Tuning%20BERT.ipynb) | Fine-tune BERT cho classification |
| 12 | **Fine-tuning Generation Models** | [Colab](https://colab.research.google.com/github/HandsOnLLM/Hands-On-Large-Language-Models/blob/main/chapter12/Chapter%2012%20-%20Fine-tuning%20Generation%20Models.ipynb) | Fine-tune LLMs (LoRA, PEFT, instruction tuning) |

### Bonus Content
- [A Visual Guide to Mamba](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state) — State Space Models
- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) — INT4/INT8
- [A Visual Guide to MoE](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) — Mixture of Experts
- [A Visual Guide to Reasoning LLMs](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-reasoning-llms) — Chain-of-thought, o1
- [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1)
- [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)

---

## 8. Prompt Engineering Guide

**Source:** [dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
**Stars:** 71,722 ⭐ | **Language:** MDX | **Website:** [promptingguide.ai](https://www.promptingguide.ai/)
**Topics:** `prompt-engineering` `llms` `rag` `agents` `chatgpt` `generative-ai`

### Giới thiệu

Hướng dẫn toàn diện nhất về Prompt Engineering — 71K+ stars, featured trên Wall Street Journal và Forbes. Hỗ trợ 13 ngôn ngữ, 3M+ learners.

---

### Kỹ thuật Prompting

#### Cơ bản
| Kỹ thuật | Mô tả | Link |
|----------|-------|------|
| **Zero-Shot Prompting** | Yêu cầu LLM thực hiện task không có ví dụ nào | [Guide](https://www.promptingguide.ai/techniques/zeroshot) |
| **Few-Shot Prompting** | Cung cấp vài ví dụ (demonstrations) trong prompt | [Guide](https://www.promptingguide.ai/techniques/fewshot) |
| **Chain-of-Thought (CoT)** | Hướng dẫn LLM suy luận từng bước trước khi trả lời | [Guide](https://www.promptingguide.ai/techniques/cot) |
| **Self-Consistency** | Sample nhiều reasoning paths, lấy kết quả nhiều nhất | [Guide](https://www.promptingguide.ai/techniques/consistency) |
| **Generate Knowledge** | Tạo kiến thức liên quan trước khi trả lời câu hỏi | [Guide](https://www.promptingguide.ai/techniques/knowledge) |

#### Nâng cao
| Kỹ thuật | Mô tả | Link |
|----------|-------|------|
| **Tree of Thoughts (ToT)** | Khám phá nhiều suy luận song song, backtrack khi cần | [Guide](https://www.promptingguide.ai/techniques/tot) |
| **Prompt Chaining** | Kết nối nhiều prompts thành pipeline | [Guide](https://www.promptingguide.ai/techniques/prompt_chaining) |
| **ReAct** | Reasoning + Acting — kết hợp suy luận với tool use | [Guide](https://www.promptingguide.ai/techniques/react) |
| **RAG (Retrieval Augmented Generation)** | Retrieval + Generation cho grounded answers | [Guide](https://www.promptingguide.ai/techniques/rag) |
| **ART (Automatic Reasoning & Tool-use)** | Tự động tạo reasoning + gọi tools | [Guide](https://www.promptingguide.ai/techniques/art) |
| **APE (Automatic Prompt Engineer)** | Tự động tối ưu prompt bằng LLM | [Guide](https://www.promptingguide.ai/techniques/ape) |
| **Active-Prompt** | Chọn ra examples uncertainty cao để annotate | [Guide](https://www.promptingguide.ai/techniques/activeprompt) |
| **DSP (Directional Stimulus)** | Cung cấp hint/stimulus cho LLM | [Guide](https://www.promptingguide.ai/techniques/dsp) |
| **PAL (Program-Aided Language)** | Dùng code/interpreter để giải quyết bài toán | [Guide](https://www.promptingguide.ai/techniques/pal) |
| **Multimodal CoT** | Chain-of-thought cho multimodal inputs | [Guide](https://www.promptingguide.ai/techniques/multimodalcot) |
| **Graph Prompting** | Prompting với cấu trúc graph | [Guide](https://www.promptingguide.ai/techniques/graph) |

---

### Ứng dụng Prompting

| Ứng dụng | Link |
|----------|------|
| Function Calling | [Guide](https://www.promptingguide.ai/applications/function_calling) |
| Generating Data / Synthetic Data for RAG | [Guide](https://www.promptingguide.ai/applications/synthetic_rag) |
| Code Generation | [Guide](https://www.promptingguide.ai/applications/coding) |
| Generating Textbooks | [Guide](https://www.promptingguide.ai/applications/generating_textbooks) |

---

### Prompt Hub (Thư viện mẫu)

| Category | Link |
|----------|------|
| Classification | [Hub](https://www.promptingguide.ai/prompts/classification) |
| Coding | [Hub](https://www.promptingguide.ai/prompts/coding) |
| Reasoning | [Hub](https://www.promptingguide.ai/prompts/reasoning) |
| Text Summarization | [Hub](https://www.promptingguide.ai/prompts/text-summarization) |
| Question Answering | [Hub](https://www.promptingguide.ai/prompts/question-answering) |
| Mathematics | [Hub](https://www.promptingguide.ai/prompts/mathematics) |
| Image Generation | [Hub](https://www.promptingguide.ai/prompts/image-generation) |
| Adversarial Prompting | [Hub](https://www.promptingguide.ai/prompts/adversarial-prompting) |

---

### Rủi ro và Hạn chế

| Vấn đề | Mô tả | Link |
|--------|-------|------|
| **Adversarial Prompting** | Prompt injection, jailbreaking, leaking | [Guide](https://www.promptingguide.ai/risks/adversarial) |
| **Factuality** | Hallucination, factual errors | [Guide](https://www.promptingguide.ai/risks/factuality) |
| **Biases** | Stereotypes, confirmation bias trong outputs | [Guide](https://www.promptingguide.ai/risks/biases) |

---

### LLM Settings quan trọng
- **Temperature** — Randomness (0 = deterministic, 1+ = creative)
- **Top-p (Nucleus Sampling)** — Lấy tokens trong top-p probability mass
- **Top-k** — Lấy k tokens có xác suất cao nhất
- **Max Length** — Độ dài tối đa output
- **Stop Sequences** — Ký tự dừng generation

---

## 9. AI Agents for Beginners (Microsoft)

**Source:** [microsoft/ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners)
**Stars:** 54,146 ⭐ | **Language:** Jupyter Notebook
**Topics:** `agentic-ai` `ai-agents` `autogen` `semantic-kernel` `agentic-rag`

### Giới thiệu

Khóa học chính thức từ Microsoft: 12+ lessons để bắt đầu xây dựng AI Agents. Dạy từ concepts → frameworks → production.

---

### Curriculum (15 lessons + bonus)

| Lesson | Chủ đề | Nội dung |
|--------|--------|---------|
| 00 | **Course Setup** | Environment setup, tools |
| 01 | **Intro to AI Agents** | Agent là gì? Tại sao quan trọng? |
| 02 | **Agentic Frameworks** | AutoGen, Semantic Kernel, LangChain so sánh |
| 03 | **Agentic Design Patterns** | ReAct, Plan-Execute, Reflection patterns |
| 04 | **Tool Use** | Function calling, tool integration |
| 05 | **Agentic RAG** | RAG trong agentic systems |
| 06 | **Trustworthy Agents** | Safety, reliability, evaluations |
| 07 | **Planning Design** | Task decomposition, planning strategies |
| 08 | **Multi-Agent** | Multi-agent systems, coordination |
| 09 | **Metacognition** | Agents thinking about their own thinking |
| 10 | **AI Agents in Production** | Deployment, monitoring, scaling |
| 11 | **Agentic Protocols** | MCP, A2A, communication protocols |
| 12 | **Context Engineering** | Managing context windows, memory |
| 13 | **Agent Memory** | Short-term, long-term, episodic memory |
| 14 | **Microsoft Agent Framework** | Azure AI Agent Service |
| 15 | **Browser Use** | Web browsing agents |

### Frameworks được dạy
- **AutoGen** (Microsoft) — multi-agent conversation framework
- **Semantic Kernel** (Microsoft) — enterprise AI orchestration
- **LangChain / LangGraph** — popular open-source framework
- **Azure AI Agent Service** — managed cloud service

---

## 10. GenAI Agents (NirDiamant)

**Source:** [NirDiamant/GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents)
**Stars:** 20,583 ⭐ | **Language:** Jupyter Notebook
**Topics:** `agents` `langchain` `langgraph` `llm` `genai` `openai`

### Giới thiệu

45+ implementations của GenAI agents — từ beginner đến advanced. Mỗi agent là một tutorial hoàn chỉnh với code thực tế.

---

### Danh sách đầy đủ 45 Agents

#### 🌱 Beginner

| # | Agent | Framework | Link |
|---|-------|-----------|------|
| 1 | **Simple Conversational Agent** | LangChain / PydanticAI | [Notebook](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_conversational_agent.ipynb) |
| 2 | **Simple Q&A Agent** | LangChain | [Notebook](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_question_answering_agent.ipynb) |
| 3 | **Simple Data Analysis Agent** | LangChain / PydanticAI | [Notebook](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/simple_data_analysis_agent_notebook.ipynb) |

#### 🔧 Framework Tutorials

| # | Agent | Framework | Link |
|---|-------|-----------|------|
| 4 | **Intro to LangGraph** | LangGraph | [Notebook](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/langgraph-tutorial.ipynb) |
| 5 | **Model Context Protocol (MCP)** | MCP | [Notebook](https://github.com/NirDiamant/GenAI_Agents/blob/main/all_agents_tutorials/mcp-tutorial.ipynb) |

#### 🎓 Educational

| # | Agent | Đặc điểm |
|---|-------|---------|
| 6 | **ATLAS: Academic Task System** | Multi-agent: Coordinator + Planner + Notewriter + Advisor |
| 7 | **Scientific Paper Agent** | Literature review tự động, CORE API, PDFplumber |
| 8 | **Chiron: Feynman Learning Agent** | Adaptive learning, 70% understanding threshold, checkpoint system |

#### 💼 Business

| # | Agent | Tính năng nổi bật |
|---|-------|-----------------|
| 9 | Customer Support Agent | Query categorization + sentiment analysis |
| 10 | Essay Grading Agent | Multi-criteria grading (relevance, grammar, structure, depth) |
| 11 | Travel Planning Agent | Stateful multi-step itinerary generation |
| 12 | GenAI Career Assistant | Multi-agent: Learning + Resume + Interview + Job Search |
| 13 | Project Manager Assistant | Task gen + dependency mapping + Gantt chart + risk assessment |
| 14 | Contract Analysis (ClauseAI) | Clause analysis + Pinecone vector store + compliance check |
| 15 | E2E Testing Agent | Natural language → Playwright browser automation |

#### 🎨 Creative

| # | Agent | Tính năng |
|---|-------|---------|
| 16 | GIF Animation Generator | Text → animation pipeline |
| 17 | TTS Poem Generator | Text classification + speech synthesis |
| 18 | Music Compositor | AI music composition |
| 19 | Content Intelligence | Multi-platform content generation |
| 20 | Business Meme Generator | Brand-aligned memes |
| 21 | Murder Mystery Game | Procedural story generation |

#### 📊 Analysis

| # | Agent | Đặc điểm |
|---|-------|---------|
| 22 | Memory-Enhanced Conversational | Short/long-term memory integration |
| 23 | Multi-Agent Collaboration | Historical research + data analysis |
| 24 | Self-Improving Agent | Learns from interactions |
| 25 | Internet Search Agent | Web research + summarization |
| 26 | Research Team (AutoGen) | Multi-agent research collaboration |
| 27 | Sales Call Analyzer | Audio transcription + NLP |
| 28 | Weather Emergency System | Real-time data + disaster management |
| 29 | Self-Healing Codebase | Error detection + automated fixes |
| 30 | DataScribe (Database Agent) | Database exploration + query planning |
| 31 | Memory-Enhanced Email | Email triage + response generation |

#### 📰 News & Media

| # | Agent | Tính năng |
|---|-------|---------|
| 32 | News TL;DR | News summarization + API integration |
| 33 | AInsight | AI/ML news aggregation |
| 34 | Journalism Assistant | Fact-checking + bias detection |
| 35 | Blog Writer (Swarm) | Collaborative content (OpenAI Swarm) |
| 36 | Podcast Generator | Content search + audio generation |

#### 🛍️ Shopping & Task Management

| # | Agent | Tính năng |
|---|-------|---------|
| 37 | ShopGenie | Product comparison + recommendations |
| 38 | Car Buyer Agent | Web data + decision support |
| 39 | Taskifier | Work style analysis + task breakdown |
| 40 | Grocery Management | Inventory + recipes (CrewAI) |

#### 🔍 QA & Advanced

| # | Agent | Tính năng |
|---|-------|---------|
| 41 | LangGraph Inspector | System testing + vulnerability detection |
| 42 | EU Green Deal Bot | Regulatory compliance FAQ |
| 43 | Systematic Review | Academic paper processing |
| 44 | **Controllable RAG Agent** | Complex Q&A + deterministic graph (custom) |

---

## 11. ML for Beginners (Microsoft)

**Source:** [microsoft/ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners)
**Stars:** 84,477 ⭐ | **Language:** Jupyter Notebook
**Description:** 12 weeks, 26 lessons, 52 quizzes — classic ML for all

### Giới thiệu

Chương trình ML 12 tuần của Microsoft dành cho người mới bắt đầu — **không dùng Deep Learning**, tập trung vào **classic ML** với Scikit-Learn. Có 52 quizzes, nhiều ngôn ngữ.

---

### Curriculum (12 tuần)

| Module | Thư mục | Nội dung |
|--------|---------|---------|
| 1 | **Introduction to ML** | `1-Introduction/` | ML là gì? Lịch sử, fairness, techniques overview |
| 2 | **Regression** | `2-Regression/` | Linear Regression, Logistic Regression — dùng dữ liệu giá bí đỏ Bắc Mỹ |
| 3 | **Web App** | `3-Web-App/` | Deploy ML model lên web app với Flask |
| 4 | **Classification** | `4-Classification/` | Logistic Regression, KNN, SVM, Decision Tree — dùng dữ liệu ẩm thực |
| 5 | **Clustering** | `5-Clustering/` | K-Means, dùng dữ liệu nhạc Nigeria |
| 6 | **NLP** | `6-NLP/` | Bag of Words, TF-IDF, Sentiment, Language translation |
| 7 | **Time Series** | `7-TimeSeries/` | ARIMA, dữ liệu điện năng |
| 8 | **Reinforcement Learning** | `8-Reinforcement/` | Q-Learning, CartPole |
| 9 | **Real World ML** | `9-Real-World/` | Case studies, industry applications |

### Điểm đặc biệt
- Classic ML only (không Deep Learning)
- Project-based: mỗi chủ đề có dataset thực tế
- 52 pre/post-lesson quizzes
- Sketchnote visuals cho mỗi bài
- Đa ngôn ngữ (translations)
- Tools: Python, R, Scikit-Learn

---

## 12. 100 Days of ML Code

**Source:** [Avik-Jain/100-Days-Of-ML-Code](https://github.com/Avik-Jain/100-Days-Of-ML-Code)
**Stars:** 49,921 ⭐
**Topics:** `machine-learning` `deep-learning` `linear-regression` `svm` `naive-bayes` `infographics`

### Giới thiệu

Challenge học ML trong 100 ngày với các **infographics trực quan** và code implementations. Nổi tiếng nhờ visual explanations rõ ràng.

---

### Chương trình học (Day by Day)

| Ngày | Chủ đề | Nội dung |
|------|--------|---------|
| Day 1 | **Data Preprocessing** | Import data, handle missing values, encode categorical, split train/test |
| Day 2 | **Simple Linear Regression** | Concept, OLS, fitting model |
| Day 3 | **Multiple Linear Regression** | Multiple features, dummy variables, backward elimination |
| Day 4-5 | **Logistic Regression** | Classification, sigmoid function |
| Day 6-8 | **K-Nearest Neighbors (KNN)** | Algorithm, choosing K |
| Day 9-11 | **Support Vector Machine (SVM)** | Hyperplane, kernel trick, margin |
| Day 12-13 | **Naive Bayes** | Bayes theorem, Gaussian NB |
| Day 14-15 | **Decision Tree** | Entropy, information gain, CART |
| Day 16-18 | **Random Forest** | Ensemble, bagging, feature importance |
| Day 19-20 | **K-Means Clustering** | Elbow method, centroids |
| Day 21 | **Linear Discriminant Analysis** | Dimensionality reduction cho classification |
| Day 22-25 | **Neural Networks Basics** | Perceptron, activation functions, backprop |
| Day 26-30 | **Deep Learning Intro** | CNNs, RNNs concepts |
| Day 31+ | **Advanced Topics** | PCA, NLP basics, evaluation metrics |

### Điểm đặc biệt
- **Infographics** đẹp cho từng algorithm (dễ hiểu nhanh)
- Code Python sạch với Scikit-Learn
- Datasets thực tế đi kèm
- Thư mục `Info-graphs/` chứa visual references

---

## 13. All Algorithms in Python

**Source:** [TheAlgorithms/Python](https://github.com/TheAlgorithms/Python)
**Stars:** 218,751 ⭐ | **Language:** Python
**Topics:** `algorithm` `python` `sorting` `searches` `interview` `education`

### Giới thiệu

Repo lớn nhất GitHub về implementations thuật toán bằng Python — 218K+ stars. Mỗi thuật toán được implement từ đầu, có docstrings và test cases.

---

### Danh mục thuật toán

#### Searches (Tìm kiếm)
- Linear Search, Binary Search, Jump Search, Interpolation Search
- Exponential Search, Fibonacci Search, Ternary Search
- Hash Table Search

#### Sorting (Sắp xếp)
- Bubble Sort, Selection Sort, Insertion Sort, Merge Sort
- Quick Sort, Heap Sort, Counting Sort, Radix Sort
- Shell Sort, Tim Sort, Patience Sort, Strand Sort

#### Data Structures (Cấu trúc dữ liệu)
- Linked List (Singly, Doubly, Circular)
- Stack, Queue, Deque
- Binary Tree, BST, AVL Tree, Red-Black Tree
- Heap (Min/Max), Trie, Graph
- Hash Map, LRU Cache

#### Graph Algorithms
- BFS, DFS, Dijkstra, Bellman-Ford, Floyd-Warshall
- Kruskal, Prim (Minimum Spanning Tree)
- Topological Sort, Strongly Connected Components
- A* Search

#### Dynamic Programming
- Fibonacci, Knapsack, Longest Common Subsequence
- Longest Increasing Subsequence, Matrix Chain Multiplication
- Coin Change, Edit Distance, Floyd-Warshall

#### Math & Number Theory
- GCD, LCM, Sieve of Eratosthenes (Primes)
- Modular Exponentiation, Fast Fourier Transform
- Matrix operations

#### Machine Learning
- Linear Regression, Logistic Regression
- K-Means Clustering, KNN
- Decision Tree, Random Forest (basic)
- Naive Bayes, Neural Network (từ scratch)

#### String Algorithms
- KMP, Rabin-Karp, Z-algorithm
- Manacher's (Longest Palindrome)
- Levenshtein Distance

#### Các chủ đề khác
- Backtracking (N-Queens, Sudoku)
- Divide & Conquer
- Greedy Algorithms
- Bit Manipulation

---

## 14. Mathematics for Machine Learning

**Source:** [mml-book/mml-book.github.io](https://github.com/mml-book/mml-book.github.io)
**Stars:** 15,183 ⭐ | **Authors:** Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong
**Publisher:** Cambridge University Press | **Website:** [mml-book.com](https://mml-book.com)

### Giới thiệu

Sách giáo khoa toán học nền tảng cho Machine Learning — **miễn phí PDF**. Không cover advanced ML mà tập trung vào **mathematical prerequisites** cần thiết để đọc các ML books khác.

---

### Cấu trúc sách

#### Part I: Mathematical Foundations (Nền tảng Toán học)

| Chương | Chủ đề | Nội dung |
|--------|--------|---------|
| 2 | **Linear Algebra** | Vectors, matrices, linear maps, eigendecomposition |
| 3 | **Analytic Geometry** | Norms, inner products, distances, projections, rotations |
| 4 | **Matrix Decompositions** | Determinant, trace, eigenvalues, SVD, matrix approximation |
| 5 | **Vector Calculus** | Differentiation, gradients, Jacobian, Hessian, backpropagation |
| 6 | **Probability & Distributions** | Probability rules, distributions, Bayes theorem, conjugate priors |
| 7 | **Continuous Optimization** | Gradient descent, convex optimization, Lagrangian duality |

#### Part II: ML Algorithms sử dụng nền tảng trên

| Chương | ML Algorithm | Toán dùng |
|--------|-------------|-----------|
| 8 | **Linear Regression** | Least squares, MLE, Bayesian regression |
| 9 | **PCA (Dimensionality Reduction)** | Linear algebra, SVD |
| 10 | **Density Estimation (GMM)** | Probability, EM algorithm |
| 11 | **Classification with SVM** | Convex optimization, kernel methods |

---

### Jupyter Notebooks kèm theo

| Tutorial | Notebook | Solution |
|----------|----------|---------|
| Linear Regression | [Colab](https://colab.research.google.com/github/mml-book/mml-book.github.io/blob/master/tutorials/tutorial_linear_regression.ipynb) | [Solution](https://colab.research.google.com/github/mml-book/mml-book.github.io/blob/master/tutorials/tutorial_linear_regression.solution.ipynb) |
| PCA | [Colab](https://colab.research.google.com/github/mml-book/mml-book.github.io/blob/master/tutorials/tutorial_pca.ipynb) | [Solution](https://colab.research.google.com/github/mml-book/mml-book.github.io/blob/master/tutorials/tutorial_pca.solution.ipynb) |
| Gaussian Mixture Model | [Colab](https://colab.research.google.com/github/mml-book/mml-book.github.io/blob/master/tutorials/tutorial_gmm.ipynb) | [Solution](https://colab.research.google.com/github/mml-book/mml-book.github.io/blob/master/tutorials/tutorial_gmm.solution.ipynb) |

### Tải sách
- **PDF miễn phí:** [mml-book.com](https://mml-book.com)
- **GitHub exercises:** [mml-book.github.io](https://github.com/mml-book/mml-book.github.io)

---

## 15. Made With ML

**Source:** [GokuMohandas/Made-With-ML](https://github.com/GokuMohandas/Made-With-ML)
**Stars:** 46,801 ⭐ | **Language:** Jupyter Notebook
**Topics:** `mlops` `machine-learning` `pytorch` `ray` `nlp` `llms` `data-engineering`

### Giới thiệu

**MLOps-focused course** — học cách build production-grade ML systems. Focus: Design → Develop → Deploy → Iterate. 40K+ developers đã học.

---

### Curriculum

#### Thiết kế & Phát triển
| Module | Nội dung |
|--------|---------|
| **Data Engineering** | Data ingestion, validation, preprocessing |
| **Exploratory Analysis** | EDA, feature engineering |
| **Modeling** | Training với PyTorch, evaluation |
| **Experiment Tracking** | MLflow (track experiments + model registry) |
| **Hyperparameter Tuning** | Ray Tune, distributed tuning |

#### Deployment & Production
| Module | Nội dung |
|--------|---------|
| **Serving** | Ray Serve — scalable model serving |
| **Testing** | Code tests, data tests, model tests (pytest) |
| **CI/CD** | GitHub Actions — continuous training + deployment |
| **Monitoring** | Production model monitoring |
| **Distributed Training** | Ray Train — scale to multiple GPUs/nodes |

---

### MLOps Stack

| Component | Tool |
|-----------|------|
| Data version control | DVC |
| Experiment tracking | MLflow |
| Hyperparameter tuning | Ray Tune |
| Distributed training | Ray Train |
| Model serving | Ray Serve |
| CI/CD | GitHub Actions |
| Orchestration | Anyscale Jobs |
| Code quality | pre-commit, pytest |
| Environment | Docker, Conda |

---

### Project thực tế

Course xây dựng **ML classifier cho GitHub repository tags** (NLP task):
- Input: title + description của một ML project
- Output: tag (NLP, CV, MLOps, Other)
- Model: Fine-tuned Transformer (DistilBERT)
- Deployed via Ray Serve trên Anyscale

```bash
# Example inference
Input: "Transfer learning with transformers for text classification"
Output: {"prediction": ["natural-language-processing"], "probability": 0.998}
```

---

## 16. Annotated Deep Learning Paper Implementations

**Source:** [labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)
**Stars:** 66,005 ⭐ | **Language:** Python (PyTorch)
**Website:** [nn.labml.ai](https://nn.labml.ai/index.html)
**Topics:** `transformer` `gan` `reinforcement-learning` `pytorch` `lora` `deep-learning`

### Giới thiệu

60+ implementations của DL papers với **side-by-side code + notes** — đọc như book, hiểu như tutorial. Website render code kèm mathematical explanations ngay bên cạnh.

---

### Danh sách đầy đủ implementations

#### Transformers
| Implementation | Link |
|---------------|------|
| JAX Transformer | [nn.labml.ai](https://nn.labml.ai/transformers/jax_transformer/index.html) |
| Multi-headed Attention | [nn.labml.ai](https://nn.labml.ai/transformers/mha.html) |
| Triton Flash Attention | [nn.labml.ai](https://nn.labml.ai/transformers/flash/index.html) |
| Transformer XL + Relative MHA | [nn.labml.ai](https://nn.labml.ai/transformers/xl/index.html) |
| Rotary Positional Embeddings (RoPE) | [nn.labml.ai](https://nn.labml.ai/transformers/rope/index.html) |
| ALiBi (Attention with Linear Biases) | [nn.labml.ai](https://nn.labml.ai/transformers/alibi/index.html) |
| RETRO | [nn.labml.ai](https://nn.labml.ai/transformers/retro/index.html) |
| GPT Architecture | [nn.labml.ai](https://nn.labml.ai/transformers/gpt/index.html) |
| Switch Transformer (MoE) | [nn.labml.ai](https://nn.labml.ai/transformers/switch/index.html) |
| Vision Transformer (ViT) | [nn.labml.ai](https://nn.labml.ai/transformers/vit/index.html) |
| MLP-Mixer | [nn.labml.ai](https://nn.labml.ai/transformers/mlp_mixer/index.html) |
| Masked Language Model | [nn.labml.ai](https://nn.labml.ai/transformers/mlm/index.html) |

#### LLM Tools
| Implementation | Link |
|---------------|------|
| **LoRA** (Low-Rank Adaptation) | [nn.labml.ai](https://nn.labml.ai/lora/index.html) |
| Eleuther GPT-NeoX | [nn.labml.ai](https://nn.labml.ai/neox/index.html) |
| LLM.int8() (Quantization) | [nn.labml.ai](https://nn.labml.ai/neox/utils/llm_int8.html) |

#### Diffusion Models
| Implementation | Link |
|---------------|------|
| DDPM (Denoising Diffusion Probabilistic Models) | [nn.labml.ai](https://nn.labml.ai/diffusion/ddpm/index.html) |
| DDIM | [nn.labml.ai](https://nn.labml.ai/diffusion/stable_diffusion/sampler/ddim.html) |
| Latent Diffusion Models | [nn.labml.ai](https://nn.labml.ai/diffusion/stable_diffusion/latent_diffusion.html) |
| Stable Diffusion | [nn.labml.ai](https://nn.labml.ai/diffusion/stable_diffusion/index.html) |

#### GANs
| Implementation | Link |
|---------------|------|
| Original GAN | [nn.labml.ai](https://nn.labml.ai/gan/original/index.html) |
| DCGAN | [nn.labml.ai](https://nn.labml.ai/gan/dcgan/index.html) |
| CycleGAN | [nn.labml.ai](https://nn.labml.ai/gan/cycle_gan/index.html) |
| Wasserstein GAN (WGAN) | [nn.labml.ai](https://nn.labml.ai/gan/wasserstein/index.html) |
| StyleGAN 2 | [nn.labml.ai](https://nn.labml.ai/gan/stylegan/index.html) |

#### Recurrent Networks
| Implementation | Link |
|---------------|------|
| LSTM | [nn.labml.ai](https://nn.labml.ai/lstm/index.html) |
| Recurrent Highway Networks | [nn.labml.ai](https://nn.labml.ai/recurrent_highway_networks/index.html) |
| HyperNetworks / HyperLSTM | [nn.labml.ai](https://nn.labml.ai/hypernetworks/hyper_lstm.html) |
| Sketch RNN | [nn.labml.ai](https://nn.labml.ai/sketch_rnn/index.html) |

#### CNNs & Vision
| Implementation | Link |
|---------------|------|
| ResNet | [nn.labml.ai](https://nn.labml.ai/resnet/index.html) |
| ConvMixer | [nn.labml.ai](https://nn.labml.ai/conv_mixer/index.html) |
| Capsule Networks | [nn.labml.ai](https://nn.labml.ai/capsule_networks/index.html) |
| U-Net | [nn.labml.ai](https://nn.labml.ai/unet/index.html) |

#### Graph Neural Networks
| Implementation | Link |
|---------------|------|
| GAT (Graph Attention Networks) | [nn.labml.ai](https://nn.labml.ai/graphs/gat/index.html) |
| GATv2 | [nn.labml.ai](https://nn.labml.ai/graphs/gatv2/index.html) |

#### Reinforcement Learning
| Implementation | Link |
|---------------|------|
| PPO (Proximal Policy Optimization) | [nn.labml.ai](https://nn.labml.ai/rl/ppo/index.html) |
| DQN (Deep Q-Networks) | [nn.labml.ai](https://nn.labml.ai/rl/dqn/index.html) |

#### Optimizers
| Implementation | Link |
|---------------|------|
| Adam | [nn.labml.ai](https://nn.labml.ai/optimizers/adam.html) |
| AMSGrad | [nn.labml.ai](https://nn.labml.ai/optimizers/amsgrad.html) |
| Rectified Adam (RAdam) | [nn.labml.ai](https://nn.labml.ai/optimizers/radam.html) |
| AdaBelief | [nn.labml.ai](https://nn.labml.ai/optimizers/ada_belief.html) |
| Sophia-G | [nn.labml.ai](https://nn.labml.ai/optimizers/sophia.html) |

#### Normalization Layers
| Implementation | Link |
|---------------|------|
| Batch Normalization | [nn.labml.ai](https://nn.labml.ai/normalization/batch_norm/index.html) |
| Layer Normalization | [nn.labml.ai](https://nn.labml.ai/normalization/layer_norm/index.html) |
| Group Normalization | [nn.labml.ai](https://nn.labml.ai/normalization/group_norm/index.html) |
| DeepNorm | [nn.labml.ai](https://nn.labml.ai/normalization/deep_norm/index.html) |

#### Advanced Topics
| Implementation | Link |
|---------------|------|
| Knowledge Distillation | [nn.labml.ai](https://nn.labml.ai/distillation/index.html) |
| PonderNet (Adaptive Computation) | [nn.labml.ai](https://nn.labml.ai/adaptive_computation/ponder_net/index.html) |
| Evidential Deep Learning (Uncertainty) | [nn.labml.ai](https://nn.labml.ai/uncertainty/evidence/index.html) |
| LM Sampling: Greedy/Temperature/Top-k/Nucleus | [nn.labml.ai](https://nn.labml.ai/sampling/index.html) |
| Zero3 Memory Optimization | [nn.labml.ai](https://nn.labml.ai/scaling/zero3/index.html) |

---

## Tổng hợp: Learning Roadmap Mở Rộng

```
MATHEMATICS (Foundation)
└── MML Book (Section 14):
    Linear Algebra → Analytic Geometry → Matrix Decomp
    → Calculus → Probability → Optimization

PROGRAMMING FOUNDATION
└── TheAlgorithms/Python (Section 13):
    Sorting · Searching · Data Structures
    Graph Algorithms · Dynamic Programming

CLASSIC ML (Beginner Path)
└── ML-For-Beginners Microsoft (Section 11) — 12 weeks, Scikit-Learn
└── 100 Days of ML Code (Section 12) — Day-by-day với infographics

NEURAL NETWORKS FROM SCRATCH
└── Karpathy nn-zero-to-hero (Section 6):
    Micrograd → Makemore (bigrams → MLP → BatchNorm → WaveNet) → GPT → BPE

DEEP LEARNING PAPERS (Advanced Reference)
└── Annotated DL Implementations (Section 16):
    Transformers · LoRA · Diffusion · GANs · LSTM · ResNet
    PPO · DQN · Adam · BatchNorm · Distillation

LLM & LANGUAGE MODELS
└── Hands-On LLMs Book (Section 7):
    Tokens → Embeddings → Transformer Internals → Classification
    → Prompt Engineering → Semantic Search/RAG → Multimodal → Fine-tuning

PROMPT ENGINEERING
└── DAIR.AI Prompt Engineering Guide (Section 8):
    Zero/Few-Shot → CoT → ToT → ReAct → RAG → Adversarial

AI AGENTS
├── AI Agents for Beginners Microsoft (Section 9) — 15 lessons, AutoGen/SK
└── GenAI Agents NirDiamant (Section 10) — 45 implementations
    Beginner → Framework (LangGraph/MCP) → Business → Creative → Advanced

PRODUCTION ML (MLOps)
└── Made With ML (Section 15):
    Data Engineering → Modeling → MLflow → Ray Tune
    → CI/CD → Ray Serve → Monitoring
```

---

## Quick Reference: Chọn repo theo mục đích

| Mục đích | Repo tốt nhất |
|----------|--------------|
| Hiểu backprop & neural nets từ đầu | Karpathy nn-zero-to-hero |
| Học LLMs có visuals đẹp | Hands-On LLMs Book |
| Làm chủ prompt engineering | DAIR.AI Prompt Engineering Guide |
| Build AI agents production-ready | Microsoft AI Agents for Beginners |
| 45 agent templates copy-paste | GenAI Agents NirDiamant |
| Classic ML từ zero (Scikit-Learn) | Microsoft ML for Beginners |
| Quick visual ML reference (infographics) | 100 Days of ML Code |
| Algorithm implementations Python | TheAlgorithms/Python |
| Toán ML foundation | MML Book |
| Deploy ML to production (MLOps) | Made With ML |
| Đọc paper DL với annotations | Annotated DL Implementations |

---

## Sources (Part 2)

| # | Repository | URL | Stars |
|---|-----------|-----|-------|
| 6 | karpathy/nn-zero-to-hero | https://github.com/karpathy/nn-zero-to-hero | 20,904 |
| 7 | HandsOnLLM/Hands-On-Large-Language-Models | https://github.com/HandsOnLLM/Hands-On-Large-Language-Models | 24,030 |
| 8 | dair-ai/Prompt-Engineering-Guide | https://github.com/dair-ai/Prompt-Engineering-Guide | 71,722 |
| 9 | microsoft/ai-agents-for-beginners | https://github.com/microsoft/ai-agents-for-beginners | 54,146 |
| 10 | NirDiamant/GenAI_Agents | https://github.com/NirDiamant/GenAI_Agents | 20,583 |
| 11 | microsoft/ML-For-Beginners | https://github.com/microsoft/ML-For-Beginners | 84,477 |
| 12 | Avik-Jain/100-Days-Of-ML-Code | https://github.com/Avik-Jain/100-Days-Of-ML-Code | 49,921 |
| 13 | TheAlgorithms/Python | https://github.com/TheAlgorithms/Python | 218,751 |
| 14 | mml-book/mml-book.github.io | https://github.com/mml-book/mml-book.github.io | 15,183 |
| 15 | GokuMohandas/Made-With-ML | https://github.com/GokuMohandas/Made-With-ML | 46,801 |
| 16 | labmlai/annotated_deep_learning_paper_implementations | https://github.com/labmlai/annotated_deep_learning_paper_implementations | 66,005 |
