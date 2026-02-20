# Machine Learning Assignment — Guidelines (Traditional ML Only)

## Objective
The objective of this assignment is to enable students to collect a local dataset, apply a **traditional (classical) machine learning algorithm**, evaluate and explain the model, and optionally integrate it into a front-end system. :contentReference[oaicite:0]{index=0}

---

## Task Description
Students must identify a real-world problem, collect or compile a dataset, apply a **traditional machine learning algorithm (no deep learning)**, train and evaluate it, and explain the results using XAI techniques.  
**Avoid developing an image processing application.** :contentReference[oaicite:1]{index=1}

---

## Guidelines (Marks Breakdown)

### 1. Problem Definition & Dataset Collection (15 marks)
- Clearly describe the problem and its relevance.
- Explain:
  - Data source (how and from where it was collected)
  - Features and target variable
  - Size of the dataset
- Any preprocessing done (cleaning, encoding, normalization).
- Ensure ethical data use (no personal or sensitive data without consent). :contentReference[oaicite:2]{index=2}

---

### 2. Selection of Traditional Machine Learning Algorithm(s) (15 marks) **(Modified)**
> Original guideline required a “new algorithm not taught in lectures”.  
> **Modified requirement:** Use **traditional ML algorithms only** (no new algorithm requirement), and **compare/tune baselines**.

- Choose **at least one** traditional ML algorithm and **avoid deep learning models**.
- Recommended traditional ML options (examples):
  - Logistic Regression
  - k-NN
  - Decision Tree
  - Random Forest
  - SVM
  - Naive Bayes
  - Gradient Boosting (if considered traditional in your module)
- Justify:
  - Why this algorithm (or set of algorithms) was selected
  - Why it suits your dataset/problem
  - If using multiple models, briefly compare why each is included (as a baseline or candidate)

> Note: This modification preserves the intent of the section (model selection + justification) while removing the “new algorithm” requirement. :contentReference[oaicite:3]{index=3}

---

### 3. Model Training and Evaluation (20 marks)
Explain:
- Train/validation/test split
- Hyperparameter choices
- Performance metrics used (accuracy, F1, RMSE, AUC, etc. depending on task)
- Results obtained and what they indicate
- Include tables, graphs, or plots where appropriate. :contentReference[oaicite:4]{index=4}

---

### 4. Explainability & Interpretation (20 marks)
Apply at least one explainability method, such as:
- SHAP
- LIME
- Feature importance analysis
- Partial Dependence Plots (PDP)

Explain:
- What the model has learned
- Which features are most influential
- Whether the model’s behavior aligns with domain knowledge. :contentReference[oaicite:5]{index=5}

---

### 5. Critical Discussion (10 marks)
Include:
- Limitations of the model
- Data quality issues
- Risks of bias or unfairness
- Potential real-world impact and ethical considerations. :contentReference[oaicite:6]{index=6}

---

### 6. Report Quality & Technical Clarity (10 marks) :contentReference[oaicite:7]{index=7}

---

### 7. Bonus: Front-End Integration (10 marks)
Bonus marks will be awarded for:
- Integrating the trained model into a front-end system (web app, dashboard, mobile app, etc.)
- Allowing users to input data and view predictions/explanations
- Examples: Streamlit app, Flask + HTML, React frontend, etc. :contentReference[oaicite:8]{index=8}

---

## Submission Requirements
Submit the following:

1. **Written Report (PDF)**  
   Including problem description, methodology, results, interpretation, and discussion.

2. **Source Code (ZIP / GitHub link)**  
   Including data preprocessing, training, evaluation, and explainability scripts.

3. **Dataset** (if publicly shareable) **or** a description of how it was obtained.

4. **Demo video (3–5 minutes)** showing the front-end system.

Upload your submission to the Moodle course page. :contentReference[oaicite:9]{index=9}
