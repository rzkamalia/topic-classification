# Agreement-Based Ensemble Learning for Automated News Topic Classification

## 1. Overview

- **Objective:** Automate news topic classification while retaining human review for uncertain predictions.
- **Approach:** Agreement-based ensemble learning using two independently fine-tuned BERT-based Transformer models.
- **Automatic Tagging:** Assign a topic when both models predict the same category.
- **Human Review:** Route articles to editors when the models disagree.
- **Priority:** Minimize incorrect tags in live topic feeds, even at the cost of lower automation coverage.

## 2. Dataset and Preprocessing

### 2.1 Dataset

- **Source:** [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- **Publication period:** 2012–2022
- **Scope:** 15 most-populated categories
- **Input:** `headline + " " + short_description`
- **Target:** `category`

### 2.2 Data Cleaning

| Processing Stage | Remaining Records |
|---|---:|
| Original dataset | 209,527 |
| Filtered to 15 categories | 148,122 |
| Removed records with missing values | 114,452 |
| Removed conflicting-label records | 114,444 |
| Removed duplicate copies | **114,247** |

- Normalize empty strings, whitespace-only values, and literal `"null"` values into missing values.
- Remove duplicate headline-description pairs and records with conflicting category labels.
- **Limitation:** Missing metadata may cause valid articles to be excluded, potentially introducing selection bias.

### 2.3 Data Splitting

- **Method:** Stratified random splitting to preserve category proportions despite class imbalance.

    | Dataset | Proportion |
    |---|---:|
    | Training | 70% |
    | Validation | 20% |
    | Testing | 10% |

- **Limitation:** Random splitting does not directly measure generalization to future news articles.

## 3. Model Training

### 3.1 Model Selection

| Model | Architecture |
|---|---|
| ModernBERT-base | Bidirectional Transformer encoder |
| DistilBERT-base | Distilled Transformer encoder |

**Why BERT-based Transformer models?**

- **Contextual understanding:** News classification requires interpreting ambiguous terms, named entities, and overlapping topics.
- **Encoder-based architecture:** Produces contextual representations that can be passed directly to a classification head.
- **Pretrained representations:** Leverages existing linguistic knowledge instead of learning text representations from scratch.

**Why ModernBERT and DistilBERT?**

- **ModernBERT:** Modern encoder architecture with strong contextual representation capabilities.
- **DistilBERT:** Smaller model with lower computational requirements.
- **Model diversity:** Different encoder architectures may produce complementary predictions.
- **Deployment:** Model size and inference efficiency are important for processing thousands of articles daily.

### 3.2 Training and Validation

- Fine-tune both models independently using the same training and validation datasets.
- Tokenize using each model's corresponding pretrained tokenizer.
- Select the best checkpoint using validation Macro F1-score.

| Metric | Purpose |
|---|---|
| Cross-Entropy Loss | Monitor optimization and prediction errors |
| Accuracy | Measure overall classification correctness |
| Macro F1-score | Measure balanced performance across categories |

**Why Macro F1-score?**

- The dataset contains imbalanced category distributions.
- Accuracy may be dominated by frequently occurring categories.
- Macro F1 gives equal importance to each category, including less-represented ones.

## 4. Agreement-Based Ensemble Learning

### 4.1 Decision Mechanism

- ModernBERT and DistilBERT independently classify each article.
- **Agreement:** Automatically assign the predicted category.
- **Disagreement:** Route the article to human review.
- Model agreement is used as a reliability signal, but does not guarantee correctness.

### 4.2 Example

| ModernBERT | DistilBERT | Decision |
|---|---|---|
| POLITICS | POLITICS | AUTO_TAG: POLITICS |
| SPORTS | SPORTS | AUTO_TAG: SPORTS |
| BUSINESS | POLITICS | HUMAN_REVIEW |
| TRAVEL | FOOD & DRINK | HUMAN_REVIEW |

## 5. Evaluation and Results

### 5.1 Individual Model Evaluation

Both models were evaluated on the same **11,425 test articles**.

| Metric | DistilBERT | ModernBERT |
|---|---:|---:|
| Test Loss | 0.6612 | **0.6190** |
| Test Accuracy | **80.25%** | 80.07% |
| Test Macro F1 | 0.7389 | **0.7492** |
| Evaluation Time | **18 s** | 48 s |

- DistilBERT achieves slightly higher accuracy.
- ModernBERT achieves higher Macro F1.
- Both models achieve approximately 80% overall accuracy.

### 5.2 Ensemble Evaluation

| Metric | Result |
|---|---:|
| Total Articles | 11,425 |
| Automatically Tagged | 9,615 |
| Human Review Required | 1,810 |
| Correct Auto-tags | 8,415 |
| Incorrect Auto-tags | 1,200 |
| **Automation Coverage** | **84.16%** |
| **Auto-tag Precision** | **87.52%** |
| **Selective Risk** | **12.48%** |
| **Human Review Rate** | **15.84%** |

**Key Findings**

- **Automation Coverage:** 84.16% of articles receive automatic tags.
- **Auto-tag Precision:** 87.52% of automatically tagged articles are correctly classified.
- **Selective Risk:** 12.48% of automatically tagged articles receive incorrect tags.
- **Human Review Rate:** 15.84% of articles require editorial review.
- Agreement-based routing improves correctness among automatically tagged articles compared with tagging all articles using either individual classifier.

**Editorial Risk Assessment**

- Incorrect automatic tags remain a concern: 1,200 articles were incorrectly tagged despite model agreement.
- Model agreement does not guarantee a correct prediction.
- The observed selective risk must be assessed against editorial requirements before production deployment.
