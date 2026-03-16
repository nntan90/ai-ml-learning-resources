# AI & Machine Learning Learning Compendium
> Tổng hợp từ 5 GitHub repositories chất lượng cao — dành cho LLM Notebook learning
> Last updated: March 2026

---

## Mục lục

1. [RAG Techniques — Kỹ thuật Retrieval-Augmented Generation](#1-rag-techniques)
2. [Awesome Data Science — Khoa học Dữ liệu Toàn diện](#2-awesome-data-science)
3. [Awesome NLP — Xử lý Ngôn ngữ Tự nhiên](#3-awesome-nlp)
4. [Awesome Reinforcement Learning — Tài nguyên RL](#4-awesome-reinforcement-learning)
5. [RL Algorithms From Scratch — Cài đặt RL từ đầu](#5-rl-algorithms-from-scratch)

---

## 1. RAG Techniques

**Source:** [NirDiamant/RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques)
**Stars:** 26,022 ⭐ | **Language:** Jupyter Notebook
**Topics:** `ai` `langchain` `llama-index` `llm` `python` `rag` `tutorials`

### Giới thiệu

Retrieval-Augmented Generation (RAG) kết hợp information retrieval với generative AI để tạo ra các response chính xác và giàu ngữ cảnh hơn. Repository này là một trong những bộ sưu tập RAG tutorial toàn diện nhất hiện nay, với **34 kỹ thuật** được phân loại theo độ khó và mục đích.

---

### Danh sách kỹ thuật RAG theo Category

#### 🌱 Foundational (Nền tảng)

| # | Kỹ thuật | Notebook | Mô tả ngắn |
|---|----------|----------|-------------|
| 1 | **Simple RAG** | [simple_rag.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_rag.ipynb) | RAG cơ bản dành cho người mới bắt đầu. Triển khai basic retrieval queries + incremental learning. |
| 2 | **RAG with CSV** | [simple_csv_rag.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_csv_rag.ipynb) | RAG với dữ liệu CSV, tích hợp OpenAI cho Q&A system. |
| 3 | **Reliable RAG** | [reliable_rag.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/reliable_rag.ipynb) | Nâng cấp Simple RAG với validation & refinement. Kiểm tra relevancy của retrieved documents. |
| 4 | **Optimizing Chunk Size** | [choose_chunk_size.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/choose_chunk_size.ipynb) | Chọn chunk size tối ưu, cân bằng giữa context preservation và retrieval speed. |
| 5 | **Proposition Chunking** | [proposition_chunking.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/proposition_chunking.ipynb) | Chia văn bản thành các propositions (mệnh đề) thay vì chunks cố định. |

#### 🔍 Query Enhancement (Nâng cao Query)

| # | Kỹ thuật | Notebook | Mô tả ngắn |
|---|----------|----------|-------------|
| 6 | **Query Transformations** | [query_transformations.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/query_transformations.ipynb) | Biến đổi query trước khi retrieval để cải thiện kết quả. |
| 7 | **HyDE** (Hypothetical Document Embedding) | [HyDe_Hypothetical_Document_Embedding.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/HyDe_Hypothetical_Document_Embedding.ipynb) | Tạo document giả định từ query rồi dùng embedding của nó để search. |
| 8 | **HyPE** (Hypothetical Prompt Embedding) | [HyPE_Hypothetical_Prompt_Embeddings.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/HyPE_Hypothetical_Prompt_Embeddings.ipynb) | Sinh hypothetical prompts để cải thiện semantic search. |

#### 📚 Context Enrichment (Làm giàu ngữ cảnh)

| # | Kỹ thuật | Notebook | Mô tả ngắn |
|---|----------|----------|-------------|
| 9 | **Contextual Chunk Headers** | [contextual_chunk_headers.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/contextual_chunk_headers.ipynb) | Thêm headers ngữ cảnh vào mỗi chunk để cải thiện retrieval accuracy. |
| 10 | **Relevant Segment Extraction** | [relevant_segment_extraction.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/relevant_segment_extraction.ipynb) | Trích xuất segment liên quan nhất từ documents. |
| 11 | **Context Window Enhancement** | [context_enrichment_window_around_chunk.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/context_enrichment_window_around_chunk.ipynb) | Mở rộng context window xung quanh retrieved chunk. |
| 12 | **Semantic Chunking** | [semantic_chunking.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/semantic_chunking.ipynb) | Chia văn bản dựa theo ngữ nghĩa thay vì kích thước cố định. |
| 13 | **Contextual Compression** | [contextual_compression.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/contextual_compression.ipynb) | Nén context retrieved để loại bỏ thông tin không liên quan. |
| 14 | **Document Augmentation** | [document_augmentation.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/document_augmentation.ipynb) | Augment documents với metadata và synthetic Q&A để tăng retrieval quality. |

#### 🚀 Advanced Retrieval (Retrieval nâng cao)

| # | Kỹ thuật | Notebook | Mô tả ngắn |
|---|----------|----------|-------------|
| 15 | **Fusion Retrieval** | [fusion_retrieval.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/fusion_retrieval.ipynb) | Kết hợp nhiều retrieval strategies (dense + sparse). |
| 16 | **Reranking** | [reranking.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/reranking.ipynb) | Rerank retrieved documents bằng cross-encoder model. |
| 17 | **Hierarchical Indices** | [hierarchical_indices.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/hierarchical_indices.ipynb) | Xây dựng index phân cấp (summary → detail) cho retrieval hiệu quả hơn. |
| 18 | **Dartboard Retrieval** | [dartboard.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/dartboard.ipynb) | Kỹ thuật retrieval target-focused, tối ưu relevance. |
| 19 | **Multi-modal RAG with Captioning** | [multi_model_rag_with_captioning.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/multi_model_rag_with_captioning.ipynb) | RAG với ảnh, dùng image captioning để index visual content. |
| 20 | **Multi-modal RAG with ColPali** | [multi_model_rag_with_colpali.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/multi_model_rag_with_colpali.ipynb) | RAG đa phương thức dùng ColPali model. |

#### 🔁 Iterative Techniques (Kỹ thuật lặp)

| # | Kỹ thuật | Notebook | Mô tả ngắn |
|---|----------|----------|-------------|
| 21 | **Retrieval with Feedback Loop** | [retrieval_with_feedback_loop.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/retrieval_with_feedback_loop.ipynb) | Cải thiện retrieval theo phản hồi (feedback). |
| 22 | **Adaptive Retrieval** | [adaptive_retrieval.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/adaptive_retrieval.ipynb) | Tự động điều chỉnh chiến lược retrieval theo loại query. |

#### 🏗️ Advanced Architecture (Kiến trúc nâng cao)

| # | Kỹ thuật | Notebook | Mô tả ngắn |
|---|----------|----------|-------------|
| 23 | **Graph RAG (LangChain)** | [graph_rag.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/graph_rag.ipynb) | Dùng knowledge graph để tăng cường retrieval. |
| 24 | **Microsoft GraphRAG** | [Microsoft_GraphRag.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/Microsoft_GraphRag.ipynb) | Triển khai GraphRAG của Microsoft. |
| 25 | **RAPTOR** | [raptor.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/raptor.ipynb) | Recursive Abstractive Processing for Tree-Organized Retrieval — xây cây tóm tắt đệ quy. |
| 26 | **Self-RAG** | [self_rag.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/self_rag.ipynb) | LLM tự quyết định khi nào cần retrieval và tự đánh giá output. |
| 27 | **Corrective RAG (CRAG)** | [crag.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/crag.ipynb) | Tự động phát hiện và sửa lỗi trong retrieved documents. |
| 28 | **Agentic RAG** | [Agentic_RAG.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/Agentic_RAG.ipynb) | RAG với agent tự động điều phối retrieval và generation. |

#### 📊 Evaluation (Đánh giá)

| # | Kỹ thuật | Notebook | Mô tả ngắn |
|---|----------|----------|-------------|
| 29 | **DeepEval** | [evaluation_deep_eval.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/evaluation/evaluation_deep_eval.ipynb) | Đánh giá RAG pipeline với framework DeepEval. |
| 30 | **GroUSE** | [evaluation_grouse.ipynb](https://github.com/NirDiamant/RAG_Techniques/blob/main/evaluation/evaluation_grouse.ipynb) | Grounded Unified Summarization Evaluation cho RAG. |

---

### Tài liệu liên quan từ cùng tác giả
- [GenAI Agents Repository](https://github.com/NirDiamant/GenAI_Agents)
- [Agents Towards Production](https://github.com/NirDiamant/agents-towards-production)
- [Prompt Engineering Techniques](https://github.com/NirDiamant/Prompt_Engineering)

---

## 2. Awesome Data Science

**Source:** [academic/awesome-datascience](https://github.com/academic/awesome-datascience)
**Stars:** 28,627 ⭐ | **Topics:** `data-science` `machine-learning` `deep-learning` `data-visualization`

### Giới thiệu

Repository mã nguồn mở về Data Science — con đường tắt để bắt đầu học Data Science một cách có hệ thống.

---

### Lộ trình học Data Science

**Bắt đầu từ đâu?**
- **Python** — ngôn ngữ phổ biến nhất trong khoa học dữ liệu
- **R** — ngôn ngữ domain-specific cho thống kê
- Bộ công cụ core: `Scikit-Learn`, `Pandas`, `NumPy`, `Seaborn/Matplotlib`

---

### Thuật toán và Kỹ thuật ML

#### Supervised Learning (Học có giám sát)
- **Regression:** Linear Regression, Logistic Regression, Ordinary Least Squares, Multivariate Adaptive Regression Splines, Stepwise Regression
- **Classification:**
  - k-Nearest Neighbors (k-NN)
  - Support Vector Machines (SVM)
  - Decision Trees (ID3, C4.5)
- **Ensemble Learning:**
  - Boosting (AdaBoost, XGBoost, LightGBM)
  - Bagging (Random Forest)
  - Stacking

#### Unsupervised Learning (Học không giám sát)
- Clustering: K-Means, DBSCAN, Hierarchical Clustering
- Dimensionality Reduction: PCA, t-SNE, UMAP
- Association Rules: Apriori, FP-Growth

#### Semi-Supervised Learning
- Self-training
- Co-training
- Label Propagation

#### Reinforcement Learning
- Q-Learning, SARSA
- Policy Gradient Methods
- Model-Based RL

#### Data Mining Algorithms
- Frequent Pattern Mining
- Anomaly Detection
- Sequence Mining

#### Deep Learning Architectures
- CNN (Convolutional Neural Networks)
- RNN / LSTM / GRU
- Transformer
- GAN (Generative Adversarial Networks)
- Autoencoder / VAE
- Diffusion Models

---

### Thư viện và Công cụ

#### General Machine Learning
- [Scikit-Learn](https://scikit-learn.org/) — thuật toán ML tổng quát
- [XGBoost](https://xgboost.readthedocs.io/) — gradient boosting
- [LightGBM](https://lightgbm.readthedocs.io/) — fast gradient boosting
- [CatBoost](https://catboost.ai/) — categorical feature support

#### Deep Learning
**PyTorch Ecosystem:**
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Fast.ai](https://www.fast.ai/)

**TensorFlow Ecosystem:**
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [TFX (TensorFlow Extended)](https://www.tensorflow.org/tfx)

#### Data Processing
- [Pandas](https://pandas.pydata.org/) — data manipulation
- [NumPy](https://numpy.org/) — numerical computing
- [Polars](https://www.pola.rs/) — fast DataFrame library
- [Dask](https://dask.org/) — parallel computing
- [Apache Spark (PySpark)](https://spark.apache.org/)

#### Visualization
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Plotly](https://plotly.com/)
- [Bokeh](https://bokeh.org/)
- [Altair](https://altair-viz.github.io/)

#### Model Evaluation & Monitoring
- [Evidently AI](https://www.evidentlyai.com/) — ML monitoring
- [Weights & Biases](https://wandb.ai/) — experiment tracking
- [MLflow](https://mlflow.org/) — ML lifecycle management
- [DVC](https://dvc.org/) — data version control

---

### Khóa học và Tài nguyên học tập

#### Courses miễn phí
| Nền tảng | Khóa học | Link |
|----------|----------|------|
| Kaggle | Data Science, ML, Python | [kaggle.com/learn](https://www.kaggle.com/learn) |
| Coursera | Data Science Specialization (JHU) | [coursera.org](https://www.coursera.org/specializations/jhu-data-science) |
| Coursera | Deep Learning Specialization | [deeplearning.ai](https://www.coursera.org/specializations/deep-learning) |
| MIT OCW | Linear Algebra (Gilbert Strang) | [ocw.mit.edu](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/) |
| fast.ai | Practical Deep Learning | [fast.ai](https://www.fast.ai/) |
| DataCamp | Data Scientist with Python | [datacamp.com](https://www.datacamp.com/tracks/data-scientist-with-python) |
| Udacity | Intro to Deep Learning | [udacity.com](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187) |
| Stanford | CS231n - CNN for Visual Recognition | [cs231n.github.io](https://cs231n.github.io/) |
| IBM | Data Science Course | [skillsbuild.org](https://skillsbuild.org/students/course-catalog/data-science) |

#### Books đáng đọc
- *Python Machine Learning* — Sebastian Raschka
- *Hands-On Machine Learning with Scikit-Learn & TensorFlow* — Aurélien Géron
- *The Elements of Statistical Learning* — Hastie, Tibshirani, Friedman
- *Pattern Recognition and Machine Learning* — Bishop
- *Deep Learning* — Goodfellow, Bengio, Courville

#### Tutorials tiêu biểu
- [Microsoft Data Science for Beginners](https://github.com/microsoft/Data-Science-For-Beginners) — 10 tuần, 20 bài
- [1000 Data Science Projects](https://cloud.blobcity.com/#/ps/explore) — chạy trực tiếp trên browser
- [Minimum Viable Study Plan for ML Interviews](https://github.com/khangich/machine-learning-interview)

---

### AI Agents & LLM Tooling (Mới nhất 2025–2026)
- **Frameworks:** [ADK-Rust](https://github.com/zavora-ai/adk-rust) — production-ready AI agent kit cho Rust
- **MCP Tools:** [Frostbyte MCP](https://github.com/OzorOwn/frostbyte-mcp) — 13 data tools cho AI agents
- **Research:** [BGPT MCP](https://bgpt.pro/mcp) — truy cập scientific papers database
- **Workflow:** [Sim Studio](https://sim.ai) — build & deploy LLMs với interface trực quan

---

## 3. Awesome NLP

**Source:** [awesomelistsio/awesome-nlp](https://github.com/awesomelistsio/awesome-nlp)
**Stars:** 21 ⭐ | **Language:** Python
**Description:** Curated list of frameworks, libraries, tools, datasets, tutorials, and research papers for NLP

---

### Frameworks và Thư viện NLP

| Thư viện | Mô tả | Link |
|----------|-------|------|
| **Hugging Face Transformers** | Thư viện NLP toàn diện nhất — BERT, GPT, RoBERTa và hàng nghìn models | [huggingface.co](https://huggingface.co/transformers/) |
| **spaCy** | NLP nâng cao trong Python, production-ready | [spacy.io](https://spacy.io/) |
| **NLTK** | Natural Language Toolkit — text processing và analysis cơ bản | [nltk.org](https://www.nltk.org/) |
| **Stanford NLP (CoreNLP)** | Suite NLP tools của Stanford | [stanfordnlp.github.io](https://stanfordnlp.github.io/CoreNLP/) |
| **AllenNLP** | NLP research library trên PyTorch | [allennlp.org](https://allennlp.org/) |
| **TextBlob** | NLP đơn giản cho Python | [textblob.readthedocs.io](https://textblob.readthedocs.io/) |
| **Gensim** | Topic modeling và document similarity | [radimrehurek.com/gensim](https://radimrehurek.com/gensim/) |
| **FastText** | Text classification & representation learning nhanh | [fasttext.cc](https://fasttext.cc/) |

---

### Text Processing & Tokenization

| Công cụ | Mô tả |
|---------|-------|
| **BPE (Byte Pair Encoding)** | Subword tokenization — dùng trong GPT, BERT. [Paper](https://arxiv.org/abs/1508.07909) |
| **SentencePiece** | Language-independent tokenization của Google. [GitHub](https://github.com/google/sentencepiece) |
| **spaCy Tokenizer** | Fast, efficient tokenizer tích hợp trong spaCy |
| **Moses Tokenizer** | Tokenizer phổ biến cho machine translation |

---

### Pretrained Language Models

| Model | Mô tả | Paper |
|-------|-------|-------|
| **BERT** | Bidirectional Encoder Representations from Transformers — nền tảng NLP hiện đại | [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) |
| **GPT-3** | Powerful generative LM của OpenAI (175B params) | [arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165) |
| **RoBERTa** | Optimized BERT với robust pretraining | [arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692) |
| **T5** | Text-to-Text Transfer Transformer — mọi NLP task đều là text→text | [arxiv.org/abs/1910.10683](https://arxiv.org/abs/1910.10683) |
| **XLNet** | Generalized autoregressive pretraining, vượt BERT nhiều benchmark | [arxiv.org/abs/1906.08237](https://arxiv.org/abs/1906.08237) |
| **DistilBERT** | BERT nhỏ hơn, nhanh hơn, nhẹ hơn | [arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108) |

---

### NLP Tasks chính

#### Sentiment Analysis (Phân tích cảm xúc)
- [TextBlob](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) — rule-based, tốt cho social media

#### Named Entity Recognition (NER)
- [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities)
- [Stanford NER](https://nlp.stanford.edu/software/CRF-NER.html)

#### Machine Translation
- [OpenNMT](https://opennmt.net/) — neural MT framework
- [Fairseq](https://fairseq.readthedocs.io/) — Facebook AI sequence-to-sequence

#### Text Summarization
- **BART** — Denoising Seq2Seq. [Paper](https://arxiv.org/abs/1910.13461)
- **PEGASUS** — Pre-trained model cho summarization. [Paper](https://arxiv.org/abs/1912.08777)

---

### Datasets NLP quan trọng

| Dataset | Mục đích | Link |
|---------|----------|------|
| **GLUE Benchmark** | Đánh giá NLU systems | [gluebenchmark.com](https://gluebenchmark.com/) |
| **SQuAD** | Reading comprehension & Q&A | [rajpurkar.github.io/SQuAD-explorer](https://rajpurkar.github.io/SQuAD-explorer/) |
| **CoNLL-2003** | Named Entity Recognition | [clips.uantwerpen.be](https://www.clips.uantwerpen.be/conll2003/ner/) |
| **IMDB Reviews** | Sentiment Analysis | [ai.stanford.edu](https://ai.stanford.edu/~amaas/data/sentiment/) |
| **WikiText** | Language Modeling từ Wikipedia | [blog.einstein.ai](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) |

---

### Research Papers nền tảng NLP

| Paper | Năm | Tóm tắt |
|-------|-----|---------|
| **Attention Is All You Need** | 2017 | Giới thiệu kiến trúc Transformer — cách mạng hóa NLP. [arxiv](https://arxiv.org/abs/1706.03762) |
| **BERT** | 2018 | Bidirectional pretraining cho language understanding. [arxiv](https://arxiv.org/abs/1810.04805) |
| **Word2Vec** | 2013 | Efficient word embeddings. [arxiv](https://arxiv.org/abs/1301.3781) |
| **GloVe** | 2014 | Global Vectors for Word Representation. [paper](https://nlp.stanford.edu/pubs/glove.pdf) |
| **ELMo** | 2018 | Deep Contextualized Word Representations. [arxiv](https://arxiv.org/abs/1802.05365) |

---

### Khóa học NLP

| Khóa | Nơi | Link |
|------|-----|------|
| NLP Specialization | Coursera (Deeplearning.ai) | [coursera.org](https://www.coursera.org/specializations/natural-language-processing) |
| CS224N: NLP with Deep Learning | Stanford | [web.stanford.edu/class/cs224n](http://web.stanford.edu/class/cs224n/) |
| NLP Course | Fast.ai | [fast.ai](https://www.fast.ai/) |
| Hugging Face Course | HuggingFace | [huggingface.co/course](https://huggingface.co/course/chapter1) |

#### Books NLP
- *Speech and Language Processing* — Jurafsky & Martin (textbook chuẩn)
- *Natural Language Processing with Python* — Bird, Klein, Loper (NLTK book)
- *Deep Learning for NLP* — Goyal, Pandey, Jain

---

## 4. Awesome Reinforcement Learning

**Source:** [aikorea/awesome-rl](https://github.com/aikorea/awesome-rl)
**Stars:** 9,650 ⭐
**Description:** Curated list of RL resources

---

### Lý thuyết RL — Bài giảng

| Nguồn | Khóa học | Link |
|-------|----------|------|
| DeepMind × UCL | RL Lecture Series 2021 | [deepmind.com](https://deepmind.com/learning-resources/reinforcement-learning-series-2021) |
| UCL | David Silver — COMPM050/COMPGI13 RL | [cs.ucl.ac.uk](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html) |
| UC Berkeley | CS188 AI — MDP & RL (Pieter Abbeel) | YouTube lectures |
| Stanford | CS229 — Lecture 16: RL (Andrew Ng) | [YouTube](https://www.youtube.com/watch?v=RtxI449ZjSc) |
| UC Berkeley | CS294 Deep RL (John Schulman) | [rll.berkeley.edu](http://rll.berkeley.edu/deeprlcourse/) |
| Udacity (Georgia Tech) | CS7642 RL | [udacity.com](https://classroom.udacity.com/courses/ud600) |
| CMU | 10703: Deep RL and Control, 2017 | [katefvision.github.io](https://katefvision.github.io/) |
| MIT | 6.S094: Deep Learning for Self-Driving Cars | [selfdrivingcars.mit.edu](http://selfdrivingcars.mit.edu/) |

---

### Books RL

| Sách | Tác giả | Link |
|------|---------|------|
| **Reinforcement Learning: An Introduction** (2nd Ed, 2018) | Sutton & Barto | [PDF miễn phí](http://incompleteideas.net/book/RLbook2020.pdf) |
| **Algorithms for Reinforcement Learning** | Csaba Szepesvari | [PDF](http://www.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf) |
| **Deep Reinforcement Learning in Action** | Manning | [Manning](https://www.manning.com/books/deep-reinforcement-learning-in-action) |
| **Reinforcement Learning and Optimal Control** (2019) | Dimitri Bertsekas | [MIT](http://web.mit.edu/dimitrib/www/RLbook.html) |
| **Neuro-Dynamic Programming** | Bertsekas & Tsitsiklis | Amazon |

---

### Các phương pháp RL cốt lõi

#### Dynamic Programming (DP)
- **Value Iteration** — tính optimal value function qua Bellman equations
- **Policy Iteration** — xen kẽ policy evaluation và policy improvement
- Foundational paper: Watkins, *Learning from Delayed Rewards* (Cambridge 1989)

#### Monte Carlo Methods
- Không cần model của environment
- Học từ complete episodes
- On-policy và off-policy variants
- Paper: Barto & Duff, *Monte Carlo Inversion and Reinforcement Learning* (NIPS 1994)

#### Temporal-Difference (TD) Learning
- **TD(0)** — basic TD update
- **SARSA** — on-policy TD control. Paper: Rummery & Niranjan (Cambridge 1994)
- **Q-Learning** — off-policy TD. Paper: Watkins (1989)
- **TD(λ)** — với eligibility traces. Paper: Sutton (1988)

#### Function Approximation
- **Linear Function Approximation**
- **LSTD** (Least-Square TD), **LSPI** (Least-Square Policy Iteration)
- **Neural Network Approximation** — nền tảng của Deep RL

#### Policy Gradient Methods
- **REINFORCE** — Monte Carlo policy gradient
- **Natural Actor-Critic** — Peters et al. (ECML 2005)
- **TRPO** (Trust Region Policy Optimization)
- **PPO** (Proximal Policy Optimization)
- Paper cốt lõi: Sutton et al., *Policy Gradient Methods for RL with Function Approximation* (NIPS 1999)

#### Deep Reinforcement Learning
| Phương pháp | Mô tả | Paper |
|-------------|-------|-------|
| **DQN** | Deep Q-Network — chơi Atari bằng RL | [Nature 2015](http://www.readcube.com/articles/10.1038%2Fnature14236) |
| **Double DQN** | Giảm overestimation trong DQN | [arxiv 2015](http://arxiv.org/abs/1509.06461) |
| **Prioritized Experience Replay** | Ưu tiên quan trọng transitions | [arxiv 2015](http://arxiv.org/pdf/1511.05952v2.pdf) |
| **A3C** | Asynchronous Advantage Actor-Critic | [arxiv 2016](https://arxiv.org/abs/1602.01783) |
| **PILCO** | Model-based, data-efficient policy search | [paper](http://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf) |

#### Hierarchical RL
- **Options Framework** — temporal abstraction. Sutton, Precup, Singh (1999)
- **Skill Transfer** — Konidaris & Barto (IJCAI 2007)

---

### Code Repositories RL

| Repo | Mô tả | Link |
|------|-------|------|
| **Sutton & Barto Python Code** | Code Python cho cuốn sách RL An Introduction | [GitHub](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction) |
| **OpenAI Baselines** | Chuẩn implementations của RL algorithms | [GitHub](https://github.com/openai/baselines) |
| **PyTorch Deep RL** | Deep RL implementations với PyTorch | [GitHub](https://github.com/ShangtongZhang/DeepRL) |
| **ChainerRL** | Deep RL với Chainer | [GitHub](https://github.com/chainer/chainerrl) |
| **Jumanji** | Industry-driven RL environments (JAX) | [GitHub](https://github.com/instadeepai/jumanji) |
| **AgentNet** | Deep RL library với Theano+Lasagne | [GitHub](https://github.com/yandexdataschool/AgentNet) |
| **RLCode Examples** | Minimal & clean RL examples | [GitHub](https://github.com/rlcode/reinforcement-learning) |
| **Gold (Golang)** | RL library cho Golang | [GitHub](https://github.com/aunum/gold) |

---

### Ứng dụng RL thực tế

#### Game Playing
- **Backgammon** — TD-Gammon (Tesauro, ACM 1995)
- **Chess** — KnightCap TD(λ) (1999); Giraffe Deep RL (2015)
- **Atari 2600** — DQN, Human-level control (DeepMind, Nature 2015)
- **StarCraft II** — AlphaStar (DeepMind, Nature 2019)

#### Robotics
- Quadrupedal locomotion (Policy Gradient, ICRA 2004)
- Robot motor skill coordination (IROS 2010)
- Autonomous skill acquisition on mobile manipulators (AAAI 2011)
- Robots that adapt like animals (Nature 2015)

#### Control
- Aerobatic helicopter flight (NIPS 2006)
- Autonomous helicopter control (ICRA 2001)

#### Operations Research
- Product delivery optimization
- Cross-channel marketing optimization
- Semiconductor production scheduling

---

### Survey Papers RL

| Paper | Năm | Link |
|-------|-----|------|
| Kaelbling et al., *RL: A Survey* | JAIR 1996 | [paper](https://www.jair.org/index.php/jair/article/download/10166/24110/) |
| Taylor & Stone, *Transfer Learning for RL Domains* | JMLR 2009 | [paper](http://www.jmlr.org/papers/volume10/taylor09a/taylor09a.pdf) |
| Kober et al., *RL in Robotics, A Survey* | IJRR 2013 | [paper](http://www.ias.tu-darmstadt.de/uploads/Publications/Kober_IJRR_2013.pdf) |
| Arulkumaran et al., *A Brief Survey of Deep RL* | IEEE 2017 | [arxiv](https://arxiv.org/abs/1708.05866) |

---

## 5. RL Algorithms From Scratch

**Source:** [KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch](https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch)
**Stars:** 72 ⭐ | **Language:** Jupyter Notebook
**Based on:** *Reinforcement Learning: An Introduction* — Sutton & Barto

### Giới thiệu

Mục tiêu của repo là **demystify** (làm sáng tỏ) cơ chế bên trong của các RL algorithms. Thay vì dùng external RL libraries, mọi thuật toán đều được implement **from scratch** để người học hiểu sâu hơn.

---

### Cấu trúc Notebooks (theo chapters của Sutton & Barto)

| Chapter | Notebook | Nội dung chính |
|---------|----------|----------------|
| Ch. 4 | [Dynamic Programming.ipynb](https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch/blob/main/Dynamic%20Programming.ipynb) | Policy Evaluation, Policy Iteration, Value Iteration |
| Ch. 5 | [Monte Carlo.ipynb](https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch/blob/main/Monte%20Carlo.ipynb) | Monte Carlo Prediction, MC Control (ES), On-policy MC, Off-policy MC |
| Ch. 6 | [Temporal-Difference Learning.ipynb](https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch/blob/main/Temporal-Difference%20Learning.ipynb) | TD(0), SARSA, Q-Learning |
| Ch. 7 | [n-step Bootstrapping.ipynb](https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch/blob/main/n-step%20Bootstrapping.ipynb) | n-step TD, n-step SARSA, n-step Tree Backup |
| Ch. 8 | [Planning and Learning.ipynb](https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch/blob/main/Planning%20and%20Learning.ipynb) | Dyna-Q, Dyna-Q+, Prioritized Sweeping |
| Ch. 9 | [On-policy Prediction with Approximation.ipynb](https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch/blob/main/On-policy%20Prediction%20with%20Approximation.ipynb) | Semi-gradient TD, Gradient MC, Tile Coding |
| Ch. 10 | [On-policy Control with Approximation.ipynb](https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch/blob/main/On-policy%20Control%20with%20Approximation.ipynb) | Semi-gradient SARSA, Differential Semi-gradient SARSA |
| Ch. 12 | [Eligibility Traces.ipynb](https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch/blob/main/Eligibility%20Traces.ipynb) | TD(λ), SARSA(λ), True Online TD(λ) |
| Ch. 13 | [Policy Gradient Methods.ipynb](https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch/blob/main/Policy%20Gradient%20Methods.ipynb) | REINFORCE, REINFORCE with Baseline, Actor-Critic |

---

### Tóm tắt lý thuyết từng chương

#### Dynamic Programming
- **Giả định:** Biết hoàn toàn model của environment (transition probabilities + rewards)
- **Policy Evaluation:** Tính V(s) cho một policy π cho trước bằng Bellman expectation equations
- **Policy Improvement:** Greedify policy dựa trên value function
- **Policy Iteration:** Lặp evaluation + improvement cho đến convergence
- **Value Iteration:** Kết hợp evaluation + improvement trong một bước Bellman optimality update

#### Monte Carlo
- **Giả định:** Không cần biết model, học từ experience (complete episodes)
- **MC Prediction:** Ước tính V(s) = trung bình returns từ episodes đi qua s
- **MC Control (ES):** Exploring Starts — đảm bảo mọi (s,a) được visit
- **On-policy MC:** ε-greedy policy, không cần Exploring Starts
- **Off-policy MC:** Dùng importance sampling, học π trong khi follow b

#### Temporal-Difference Learning
- **Ưu điểm:** Học online từ incomplete episodes (không cần đợi episode kết thúc)
- **TD(0) Update:** V(s) ← V(s) + α[r + γV(s') - V(s)]
- **SARSA (on-policy):** Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- **Q-Learning (off-policy):** Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

#### n-step Bootstrapping
- **Ý tưởng:** Trung gian giữa MC (n=∞) và TD(0) (n=1)
- **n-step return:** G_t^(n) = r_{t+1} + γr_{t+2} + ... + γ^(n-1)r_{t+n} + γ^n V(s_{t+n})
- Tham số n là hyperparameter cần tuning

#### Planning and Learning (Model-based RL)
- **Dyna-Q:** Kết hợp Q-learning + planning từ simulated experience
- **Dyna-Q+:** Bonus cho các (s,a) chưa được visit lâu
- **Prioritized Sweeping:** Ưu tiên update các state có thay đổi lớn

#### Function Approximation
- **Mục đích:** Xử lý state spaces lớn/liên tục (không thể dùng lookup table)
- **Semi-gradient TD:** Gradient descent chỉ trên error, không qua target
- **Tile Coding:** Feature encoding hiệu quả cho continuous state spaces

#### Eligibility Traces
- **TD(λ):** Kết hợp TD và MC. λ=0 → TD(0), λ=1 → MC
- **Eligibility trace e(s):** Theo dõi "credit" cho mỗi state
- **True Online TD(λ):** Cải thiện chính xác hơn TD(λ) truyền thống

#### Policy Gradient
- **Ý tưởng:** Tối ưu trực tiếp policy parameters bằng gradient ascent trên expected return
- **REINFORCE:** Monte Carlo policy gradient
  - ∇J(θ) ∝ G_t ∇ ln π(a_t|s_t, θ)
- **REINFORCE với Baseline:** Giảm variance bằng cách trừ baseline b(s)
- **Actor-Critic:** Actor (policy) + Critic (value function) — giảm variance hơn REINFORCE

---

## Tổng hợp: Bản đồ học tập AI/ML

```
FOUNDATION
├── Linear Algebra, Calculus, Statistics, Probability
├── Python + NumPy + Pandas + Scikit-Learn
└── Data Science Toolbox (Section 2)

CORE ML ALGORITHMS  
├── Supervised: Regression, Classification, Ensemble
├── Unsupervised: Clustering, Dimensionality Reduction
└── Data Mining (Section 2)

DEEP LEARNING
├── CNNs, RNNs, Transformers, GANs
├── PyTorch / TensorFlow / JAX
└── Pretrained Models (Section 3)

NLP SPECIALIZATION
├── Tokenization (BPE, SentencePiece)
├── Pretrained LMs: BERT, GPT, T5, RoBERTa
├── Tasks: NER, Sentiment, MT, Summarization
└── Benchmarks: GLUE, SQuAD (Section 3)

REINFORCEMENT LEARNING
├── Foundations: MDP, Bellman, DP (Ch.4)
├── Model-Free: MC, TD, Q-Learning (Ch.5-6)
├── Advanced: n-step, Planning (Ch.7-8)
├── Deep RL: DQN, A3C, PPO, SAC
├── From Scratch Implementations (Section 5)
└── Resources & Papers (Section 4)

RAG & LLM APPLICATIONS
├── Foundational RAG: Chunking, Simple RAG
├── Query Enhancement: HyDE, HyPE, Transformations
├── Context Enrichment: Semantic Chunking, Compression
├── Advanced: Graph RAG, RAPTOR, Self-RAG, CRAG
├── Evaluation: DeepEval, GroUSE
└── Full reference (Section 1)
```

---

## Sources

| # | Repository | URL | Stars |
|---|-----------|-----|-------|
| 1 | NirDiamant/RAG_Techniques | https://github.com/NirDiamant/RAG_Techniques | 26,022 |
| 2 | academic/awesome-datascience | https://github.com/academic/awesome-datascience | 28,627 |
| 3 | awesomelistsio/awesome-nlp | https://github.com/awesomelistsio/awesome-nlp | 21 |
| 4 | aikorea/awesome-rl | https://github.com/aikorea/awesome-rl | 9,650 |
| 5 | KhashayarRahimi/RL-Algorithms-From-Scratch | https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch | 72 |
