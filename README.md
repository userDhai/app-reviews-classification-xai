**app-reviews-classification-xai**

NLP pipeline designed to classify mobile app reviews into three primary categories:
1. Bug Reports
2. Feature Requests
3. User Experience

This project leverages fine-tuned transformer architectures (RoBERTa-Base and RoBERTa-Large), addresses class imbalance through oversampling and weighted loss functions, provides model interpretability via SHAP (Explainable AI), and benchmarks results against zero-shot learning models.

**Features**
* **Text Preprocessing:** Custom cleaning functions to remove URLs, special characters, superfluous whitespace, and character repetition.
* **Class Imbalance Handling:**
  1. Oversampling: Random minority oversampling applied exclusively to the training dataset.
  2. Weighted Loss: Custom PyTorch WeightedTrainer incorporating class weights within Cross-Entropy Loss to penalize misclassification of underrepresented classes.
* **Transformer Fine-Tuning:**
  1. Benchmarks both roberta-base and roberta-large.
  2. Advanced training configurations: Cosine learning rate scheduler, label smoothing, FP16 mixed precision, and early stopping.
* **Zero-Shot Baseline:** Evaluates zero-shot performance using facebook/bart-large-mnli for direct comparison with fine-tuned models.
* **Model Explainability (XAI):** Integrated SHAP (SHapley Additive exPlanations) text plots to highlight token-level contributions to class predictions.

**Tech Stack & Dependencies**
* **Language:** Python 3.x
* **Core Frameworks:**
  * **PyTorch & Hugging Face Transformers** — Model architecture and fine-tuning.
  * **Datasets** — Data loading and mapping.
  * **scikit-learn** — Data splitting, resampling, and evaluation metrics (F1-score, Accuracy, Classification Report).
  * **SHAP** — Explainable AI visualizations.
  * **Pandas & NumPy** — Data manipulation.
