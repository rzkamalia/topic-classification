# Agreement-Based Ensemble Learning for Automated News Topic Classification

## 1. Overview

- **Objective:** Automate news topic classification while retaining human review for uncertain predictions.
- **Approach:** Agreement-based ensemble learning using two independently fine-tuned BERT-based Transformer models.
- **Automatic Tagging:** Assign a topic when both models predict the same category.
- **Human Review:** Flag articles for editorial review when the models disagree; connecting this flag to an editor queue is a deployment step.
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
- **Limitation:** The current `dropna()` removes records with missing values in any column, including `authors`, even though only headline and description are model inputs. This can exclude valid articles and introduce selection bias. A follow-up experiment should require only headline, description, and category to be present.

### 2.3 Data Splitting

- **Method:** Stratified random splitting to preserve category proportions despite class imbalance.
- **Reproducibility:** Both split operations use `random_state=42`. Duplicate and conflicting normalized headline-description pairs are handled before splitting to prevent identical pairs from crossing splits.

    | Dataset | Proportion |
    |---|---:|
    | Training | 70% |
    | Validation | 20% |
    | Testing | 10% |

- **Limitation:** Random splitting does not directly measure generalization to future news articles. Related stories and near-duplicates may still appear across splits, and editorial categories may change over time.

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
- Use only the supplied News Category Dataset for supervised training, initialized from pretrained model weights.

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
- **Disagreement:** Return `category=None` and `requires_human_review=True`. The notebook produces a routing flag; it does not yet submit work to an editor queue.
- Both models are loaded once. Each uses its own saved tokenizer and input length limit, and predictions are compared by category name.
- Softmax scores are returned for inspection, but no confidence threshold or calibration is currently used in the routing decision. Even low-confidence agreement is accepted by the current policy.
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

The saved evaluation records confirm GPU use. Evaluation times are excluded because the previously reported durations could not be verified from those records. A follow-up benchmark should measure processing time on specified hardware.

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
- Auto-tag precision is calculated only for automatically tagged articles; individual model accuracy is calculated for all test articles.

**Editorial Risk Assessment**

- Both models agreed on 1,200 incorrect tags (12.48% of automatic tags). A stricter acceptance policy is needed before automatic publishing.

## 6. Hardware

Training was performed on a rented Vast.ai instance with the following specifications:

- **GPU:** NVIDIA GeForce RTX 3060, 12 GB VRAM
- **CPU:** 8 cores
- **RAM:** 96.4 GB
- **Container disk:** 32 GB

## 7. Next Steps

1. Review overlapping categories such as PARENTS and PARENTING with editors. Merge them only if they represent the same editorial topic; otherwise, clarify their definitions to support consistent labeling.
2. Benchmark the ensemble against LLM API-based services such as OpenAI or JEV, comparing cost per 1,000 articles and estimated monthly cost, alongside prediction quality and response time.
