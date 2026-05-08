# Iris Flower Classification

A complete end-to-end ML pipeline that classifies iris flowers into three species using classical supervised learning. 

---

## The Dataset

150 samples. 4 features. 3 species.

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal (cm) |
| Sepal Width | Width of the sepal (cm) |
| Petal Length | Length of the petal (cm) |
| Petal Width | Width of the petal (cm) |

**Target classes:** `Setosa` · `Versicolor` · `Virginica`

---

## Pipeline

```
Load → Explore → Clean → Visualize → Feature Engineering → Train → Evaluate → Tune
```

1. **EDA** — Histograms, scatterplots, and a correlation heatmap to understand feature relationships
2. **Preprocessing** — Duplicate removal, null checks, class distribution validation
3. **Modeling** — Three classifiers trained and benchmarked head-to-head
4. **Hyperparameter Tuning** — GridSearchCV + RFECV pipeline on the best candidate

---

## Results

| Metric | Decision Tree | Random Forest | SVM | RF (Optimized) |
|---|:---:|:---:|:---:|:---:|
| Train Accuracy | 100% | 100% | 97.5% | 100% |
| **Test Accuracy** | **96.7%** | **96.7%** | **100%** | **96.7%** |
| Test F1 Score | 0.967 | 0.967 | 1.000 | 0.967 |

**Winner: SVM (linear kernel)** — perfect test accuracy with a ROC AUC score of `1.0`.

GridSearchCV best params for Random Forest:
```python
{ 'criterion': 'gini', 'max_depth': 5, 'max_features': 'sqrt', 'n_estimators': 100 }
```

---

## Tech Stack

- **Python** — NumPy, Pandas
- **Visualization** — Matplotlib, Seaborn
- **ML** — scikit-learn (Decision Tree, Random Forest, SVM, GridSearchCV, RFECV, Pipeline)

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/itzkoala/iris-flower-classification.git
cd iris-flower-classification

# Set up environment
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Launch
jupyter notebook iris_flower.ipynb
```

---

## Project Structure

```
iris-flower-classification/
├── iris_flower.ipynb   # Full pipeline notebook
└── README.md
```

---
