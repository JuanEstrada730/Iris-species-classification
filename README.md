# Iris Species Classification

**Data Mining Final Project — Universidad de la Costa**
**Professor:** José Escorcia-Gutierrez, Ph.D.
**Department:** Computer Science and Electronics

---

## Purpose

This project implements an end-to-end data mining pipeline to classify Iris flower species (*Iris setosa*, *Iris versicolor*, *Iris virginica*) using a **Random Forest** classifier, deployed as an interactive **Streamlit** dashboard.

---

## Methodology

| Step | Description |
|------|-------------|
| Data Understanding | Explored the UCI Iris dataset: 150 samples, 4 numerical features, 3 balanced classes (50 per species) |
| Preprocessing | Applied `StandardScaler` normalization; 75/25 stratified train-test split |
| Modeling | Random Forest (200 estimators, Gini criterion, `random_state=42`) |
| Validation | Stratified 5-Fold Cross-Validation + held-out test set evaluation |
| Evaluation | Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Feature Importance |
| Deployment | Streamlit + Plotly interactive dashboard |

### Justification of Choices

**Random Forest** was selected because it is an ensemble method that reduces variance through bagging and provides reliable performance on small, structured datasets like Iris. It also yields interpretable feature importance scores via mean Gini decrease. StandardScaler normalization is applied to ensure that distance-based operations within the tree splits are not biased by scale differences between features.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/JuanEstrada730/Iris-species-classification.git
cd iris-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard

```bash
streamlit run Proyect.py
```

The app opens automatically at `http://localhost:8501`

---

## Dashboard Structure

The dashboard is organized into four sections accessible from the sidebar:

**Data Overview** — Dataset KPI cards, descriptive statistics table, class balance bar chart, feature distribution histograms, and box plots per species.

**Exploratory Analysis** — Pearson correlation heatmap, full scatter matrix (pair plot), and an interactive 2D scatter with marginal histograms and box plots.

**Model Performance** — Accuracy, Precision, Recall, and F1-Score metric cards; confusion matrix heatmap; feature importance bar chart; per-class performance table; and a radar chart comparing metrics across species.

**Prediction** — Four sliders for sepal/petal length and width; real-time species prediction with confidence score and probability bars; 3D scatter plot showing the new sample relative to the dataset (configurable axes).

---

## Project Structure

```
iris-classification/
├── Proyect.py          # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## Team Members

- Juan David Estrada

---

## Dataset

UCI Iris Dataset — loaded via `sklearn.datasets.load_iris`

Fisher, R.A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics, 7(2), 179–188.
