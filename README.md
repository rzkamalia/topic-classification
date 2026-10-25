# News Topic Classification

## Model Checkpoints

The trained models are available on [Google Drive](https://drive.google.com/drive/folders/1b5aYvqh6QtxfGy7UotBfUH48gEqqaLHf?usp=sharing).

## Repository Structure

```text
.
├── README.md
├── docs.md                       
└── workspace/
    ├── pre_processing.ipynb       # Data audit, cleaning, and 70/20/10 stratified random split
    ├── training.ipynb             # Model training experiments and recorded outputs
    ├── evaluation.ipynb           # Individual model evaluation on the test set
    ├── inference.ipynb            # Two-model inference and ensemble test evaluation
    ├── training.py                # Training loop, validation, and early stopping
    ├── model.py                   # Model/tokenizer loading and device selection
    ├── evaluation.py              # Test metrics, reports, and predictions
    ├── saving.py                  # Save checkpoints, training history, and reports
    ├── data/
    │   ├── News_Category_Dataset_v3.json
    │   ├── preprocessed_results.csv
    │   ├── news_train.csv
    │   ├── news_val.csv
    │   └── news_test.csv
    └── artifacts/
        ├── distilbert-base/      # Model checkpoint and training outputs
        ├── modernbert-base/      # Model checkpoint and training outputs
        └── evaluation/           # Individual and ensemble evaluation results
```

## Notebook Workflow

Use `workspace/` as the working directory so relative paths and local module imports resolve correctly. For Colab or another remote environment, keep the notebooks, Python modules, `data/`, and `artifacts/` together in this layout.

1. `pre_processing.ipynb`: audit and clean the dataset, then export training, validation, and test CSVs.
2. `training.ipynb`: fine-tune the models and save their best checkpoints.
3. `evaluation.ipynb`: evaluate each saved model on the test set.
4. `inference.ipynb`: load both checkpoints, predict individual articles, and measure ensemble performance.

