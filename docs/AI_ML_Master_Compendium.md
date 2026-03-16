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


---


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
