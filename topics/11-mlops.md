# ⚙️ MLOps

Tài liệu về MLOps — đưa Machine Learning models từ notebook vào production, với end-to-end workflows thực tế.

---

## 📚 Repos trong chủ đề này

| Repo | Stars | Tác giả | Mô tả |
|------|-------|---------|-------|
| [Made-With-ML](https://github.com/GokuMohandas/Made-With-ML) | ⭐ 46K+ | GokuMohandas | End-to-end MLOps course với NLP project thực tế |

---

## 1. GokuMohandas/Made-With-ML

> **GitHub**: https://github.com/GokuMohandas/Made-With-ML  
> **Stars**: ~46,000 ⭐  
> **Tác giả**: Goku Mohandas  
> **Website**: https://madewithml.com  
> **License**: MIT

### Mô tả

Course MLOps end-to-end từ nổi tiếng nhất trong cộng đồng ML. Dạy cách xây dựng, train, serve, test, và monitor một ML system thực tế trong production. Project xuyên suốt: **NLP classifier phân loại GitHub repositories theo tags**.

---

### 📋 Chương trình học

#### Module 1: Foundations
| Bài học | Chủ đề |
|---------|--------|
| 01 | Python fundamentals for ML |
| 02 | NumPy — numerical computing |
| 03 | Pandas — data manipulation |
| 04 | PyTorch — deep learning framework |

#### Module 2: Data Engineering
| Bài học | Chủ đề |
|---------|--------|
| 05 | **Labeling** — annotation workflows, label studio |
| 06 | **Exploratory Data Analysis (EDA)** |
| 07 | **Preprocessing** — tokenization, feature engineering |
| 08 | **Splitting** — train/val/test, stratification |

#### Module 3: Modeling
| Bài học | Chủ đề |
|---------|--------|
| 09 | **Baselines** — simple models first |
| 10 | **Neural Networks** — LSTM, transformer fine-tuning |
| 11 | **Experiment Tracking** — MLflow |
| 12 | **Hyperparameter Tuning** — Ray Tune |
| 13 | **Distributed Training** — Ray Train |

#### Module 4: Serving
| Bài học | Chủ đề |
|---------|--------|
| 14 | **REST APIs** — FastAPI model serving |
| 15 | **Ray Serve** — scalable model deployment |
| 16 | **Batch Inference** — offline prediction pipelines |

#### Module 5: Testing & Quality
| Bài học | Chủ đề |
|---------|--------|
| 17 | **Code Testing** — pytest, unit tests |
| 18 | **Data Testing** — Great Expectations |
| 19 | **Model Testing** — behavioral tests, slice analysis |
| 20 | **System Testing** — integration & load tests |

#### Module 6: Production
| Bài học | Chủ đề |
|---------|--------|
| 21 | **CI/CD** — GitHub Actions workflows |
| 22 | **Monitoring** — data drift, model degradation |
| 23 | **Feature Store** — Feast |
| 24 | **Orchestration** — Airflow |

---

### 🛠️ MLOps Tech Stack

| Category | Tool | Mô tả |
|----------|------|-------|
| **Experiment Tracking** | MLflow | Log metrics, parameters, artifacts |
| **Hyperparameter Tuning** | Ray Tune | Distributed HPO |
| **Distributed Training** | Ray Train | Multi-GPU/multi-node training |
| **Model Serving** | Ray Serve | Production-grade serving |
| **API Framework** | FastAPI | REST endpoints cho models |
| **CI/CD** | GitHub Actions | Automated test & deploy pipeline |
| **Data Validation** | Great Expectations | Schema & quality checks |
| **Containerization** | Docker | Reproducible environments |
| **Orchestration** | Apache Airflow | DAG-based pipelines |
| **Feature Store** | Feast | Feature management |
| **Version Control** | Git + DVC | Code + data versioning |

---

### 🎯 Project: NLP Classifier cho GitHub Repos

Project xuyên suốt của course: xây dựng hệ thống tự động gán tags cho GitHub repositories dựa trên title và description.

```
Input:  "Transformers-based text classification model for sentiment analysis"
Output: ["nlp", "transformers", "text-classification", "sentiment-analysis"]
```

#### Pipeline đầy đủ

```
1. DATA LAYER
   Raw GitHub data → EDA → Preprocessing → Labeled dataset

2. MODELING LAYER
   Baseline (TF-IDF + LR) → BERT fine-tuning
   MLflow tracking → Ray Tune optimization

3. SERVING LAYER
   FastAPI endpoint → Ray Serve cluster
   Batch inference pipeline

4. PRODUCTION LAYER
   GitHub Actions CI/CD → Monitoring → Drift detection
```

---

### 📐 MLOps Best Practices (từ Made-With-ML)

#### Code Quality
| Practice | Tool |
|----------|------|
| Linting | flake8, black |
| Type checking | mypy |
| Pre-commit hooks | pre-commit |
| Documentation | mkdocs |

#### Testing Pyramid
```
Unit Tests          ← Test individual functions
Integration Tests   ← Test components together
End-to-end Tests    ← Test full pipeline
Load Tests          ← Test under production load
```

#### CI/CD Pipeline (GitHub Actions)
```yaml
On Pull Request:
  → Run unit tests
  → Run linting
  → Check data quality

On Merge to Main:
  → Run full test suite
  → Train model (if data changed)
  → Evaluate model performance
  → Deploy if metrics pass threshold
  → Monitor in production
```

---

### 🗺️ Lộ trình học MLOps

```
1. ML Prerequisites (phải có trước)
   └── Python, NumPy, Pandas → foundations
   └── Basic ML model training → 02-classic-ml.md
   └── Neural Networks → 03-neural-networks.md

2. Made-With-ML: Foundations
   └── Module 1 (Python/NumPy/Pandas/PyTorch review)

3. Data Engineering
   └── Module 2: EDA, preprocessing, data splits
   └── Data validation với Great Expectations

4. Experiment Management
   └── Module 3: MLflow tracking
   └── Hyperparameter tuning với Ray Tune

5. Model Serving
   └── Module 4: FastAPI + Ray Serve
   └── REST APIs cho models

6. Production Systems
   └── Module 5: Testing strategies
   └── Module 6: CI/CD, monitoring, orchestration

7. Advanced MLOps
   └── Feature stores (Feast)
   └── Data versioning (DVC)
   └── Model registries
```

---

## 🔗 Tài nguyên liên quan

- **NLP project context**: [04-nlp.md](./04-nlp.md) — NLP foundations cho project
- **LLM serving**: [05-llm.md](./05-llm.md) — deploying LLMs đặc thù hơn classical ML
- **Data Science**: [10-data-science.md](./10-data-science.md) — DS foundations
- **AI Agents**: [08-ai-agents.md](./08-ai-agents.md) — deploying agents cũng cần MLOps mindset
