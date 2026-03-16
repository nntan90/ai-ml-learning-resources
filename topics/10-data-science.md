# 📊 Data Science

Tổng hợp tài nguyên về Data Science — từ nền tảng thống kê, kỹ năng EDA, đến các công cụ và workflows thực tế trong ngành.

---

## 📚 Repos trong chủ đề này

| Repo | Stars | Tác giả | Mô tả |
|------|-------|---------|-------|
| [awesome-datascience](https://github.com/academic/awesome-datascience) | ⭐ 28K+ | academic/community | Curated list of everything Data Science |

---

## 1. academic/awesome-datascience

> **GitHub**: https://github.com/academic/awesome-datascience  
> **Stars**: ~28,000 ⭐  
> **Tác giả**: Community (academic/awesome-datascience)  
> **License**: MIT

### Mô tả

Danh sách tổng hợp toàn diện nhất về Data Science — bao gồm tất cả: thuật toán, công cụ, khóa học, sách, datasets, và career advice. Được cộng đồng duy trì liên tục từ 2014.

---

### 🧮 Thuật toán (Algorithms)

#### Supervised Learning
| Thuật toán | Mô tả |
|-----------|-------|
| Linear Regression | Dự đoán giá trị liên tục |
| Logistic Regression | Phân loại nhị phân |
| Decision Trees | Cây quyết định |
| Random Forest | Ensemble trees |
| Gradient Boosting | XGBoost, LightGBM, CatBoost |
| Support Vector Machines | SVM — margin maximization |
| K-Nearest Neighbors | KNN — instance-based learning |
| Naive Bayes | Probabilistic classification |
| Neural Networks | → xem [03-neural-networks.md](./03-neural-networks.md) |

#### Unsupervised Learning
| Thuật toán | Mô tả |
|-----------|-------|
| K-Means Clustering | Phân cụm centroid-based |
| DBSCAN | Density-based clustering |
| Hierarchical Clustering | Dendrogram-based |
| PCA | Principal Component Analysis |
| t-SNE | Dimensionality reduction for visualization |
| UMAP | Faster alternative to t-SNE |
| Autoencoders | Neural-based dimensionality reduction |
| Anomaly Detection | Isolation Forest, LOF |

#### Semi-Supervised & Self-Supervised
| Thuật toán | Mô tả |
|-----------|-------|
| Label Propagation | Graph-based semi-supervised |
| Self-training | Bootstrap labeling |
| Contrastive Learning | SimCLR, MoCo |

---

### 🛠️ Công cụ & Thư viện (Tools & Libraries)

#### Python Core Stack
| Tool | Mô tả |
|------|-------|
| **NumPy** | Tính toán số học, array operations |
| **Pandas** | Data manipulation & analysis |
| **Matplotlib** | Visualization cơ bản |
| **Seaborn** | Statistical visualizations |
| **Plotly** | Interactive visualizations |
| **SciPy** | Scientific computing |
| **Statsmodels** | Statistical models & tests |

#### Machine Learning
| Tool | Mô tả |
|------|-------|
| **Scikit-learn** | Classic ML algorithms — chuẩn công nghiệp |
| **XGBoost** | Gradient boosting — Kaggle favorite |
| **LightGBM** | Fast gradient boosting từ Microsoft |
| **CatBoost** | Gradient boosting từ Yandex |

#### Deep Learning
| Tool | Mô tả |
|------|-------|
| **PyTorch** | Research & production DL framework |
| **TensorFlow** | Google's DL framework |
| **Keras** | High-level API (built into TF) |
| **JAX** | Google's high-performance ML |

#### Data Pipeline & Engineering
| Tool | Mô tả |
|------|-------|
| **Apache Spark** | Distributed big data processing |
| **Airflow** | Workflow orchestration |
| **dbt** | Data transformation |
| **Great Expectations** | Data validation |
| **DVC** | Data version control |

#### Notebooks & IDEs
| Tool | Mô tả |
|------|-------|
| **Jupyter Notebook** | Interactive computing |
| **JupyterLab** | Next-gen Jupyter |
| **VS Code** | Editor + Jupyter support |
| **Google Colab** | Free cloud GPU notebooks |
| **Kaggle Notebooks** | Competitive ML environment |

---

### 📚 Khóa học (Courses)

| Khóa học | Platform | Ghi chú |
|----------|----------|---------|
| Machine Learning | Coursera (Andrew Ng) | Nền tảng kinh điển |
| Deep Learning Specialization | Coursera (deeplearning.ai) | 5 courses |
| Practical Deep Learning for Coders | fast.ai | Top-down approach |
| Data Science: Python | Kaggle Learn | Free, ngắn gọn |
| Introduction to Data Science | IBM (Coursera) | Beginner-friendly |
| 6.S191 Introduction to Deep Learning | MIT OpenCourseWare | Chất lượng cao |
| CS50's Introduction to AI | Harvard/edX | Foundations of AI |
| Data Science with Python | DataCamp | Hands-on tracks |

---

### 📖 Sách (Books)

| Sách | Tác giả | Phù hợp cho |
|------|---------|-------------|
| **Python for Data Analysis** | Wes McKinney | Pandas creator — must-read |
| **Hands-On Machine Learning** | Aurélien Géron | Scikit-learn + TF — thực hành |
| **The Elements of Statistical Learning** | Hastie, Tibshirani, Friedman | Lý thuyết sâu |
| **Pattern Recognition and Machine Learning** | Bishop | Bayesian approach |
| **Data Science from Scratch** | Joel Grus | Build from scratch in Python |
| **Storytelling with Data** | Cole Nussbaumer Knaflic | Visualization & communication |
| **Naked Statistics** | Charles Wheelan | Statistics cho non-technical |

---

### 📂 Datasets nổi tiếng

| Dataset | Domain | Mô tả |
|---------|--------|-------|
| **Kaggle Datasets** | Mọi domain | Hàng nghìn datasets miễn phí |
| **UCI ML Repository** | Research | Classic benchmark datasets |
| **ImageNet** | Computer Vision | 1M+ ảnh, 1000 classes |
| **Common Crawl** | NLP | Web text data |
| **OpenAI Gym** | RL | Environments cho RL |
| **MNIST/CIFAR** | Computer Vision | Digit / image classification |
| **IMDb / Yelp Reviews** | NLP | Sentiment analysis |
| **NYC Taxi Data** | Tabular | Time-series & spatial |

---

### 💼 Career & Practice

#### Competitive ML
| Platform | Mô tả |
|----------|-------|
| **Kaggle** | Competitions, notebooks, datasets |
| **DrivenData** | Social good competitions |
| **Zindi** | African data science challenges |

#### Portfolio Projects (Gợi ý)
| Project | Skills thực hành |
|---------|----------------|
| EDA + Visualization | Pandas, Seaborn, storytelling |
| End-to-end ML pipeline | Scikit-learn, feature engineering |
| Time series forecasting | ARIMA, Prophet, LSTM |
| NLP sentiment analysis | → xem [04-nlp.md](./04-nlp.md) |
| Recommendation system | Collaborative filtering, matrix factorization |
| A/B test analysis | Statistics, hypothesis testing |

---

## 🗺️ Lộ trình học Data Science

```
1. Programming Foundation
   └── Python cơ bản: NumPy, Pandas, Matplotlib
   └── 02-classic-ml.md: 100 Days of ML Code

2. Statistics & Math
   └── 01-math-foundations.md: MML Book
   └── Probability, distributions, hypothesis testing

3. Classic Machine Learning
   └── 02-classic-ml.md: ML for Beginners (Microsoft)
   └── Scikit-learn: từng thuật toán

4. EDA & Visualization
   └── awesome-datascience: Pandas profiling
   └── Kaggle Learn: Data Visualization

5. Advanced Topics
   └── Deep Learning → 03-neural-networks.md
   └── NLP → 04-nlp.md
   └── MLOps → 11-mlops.md
```

---

## 🔗 Tài nguyên liên quan

- **Classic ML**: [02-classic-ml.md](./02-classic-ml.md) — ML for Beginners + 100 Days of ML
- **Math Foundations**: [01-math-foundations.md](./01-math-foundations.md)
- **Neural Networks**: [03-neural-networks.md](./03-neural-networks.md)
- **MLOps**: [11-mlops.md](./11-mlops.md) — đưa DS models vào production
