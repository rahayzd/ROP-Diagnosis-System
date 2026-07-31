# Retinopathy of Prematurity (ROP) Binary Classification

This repository contains the code and experiment notebooks for a bachelor's final-thesis project on automated binary classification of Retinopathy of Prematurity (ROP) from retinal fundus images.

The project evaluates two complementary approaches:

1. **End-to-end transfer learning**, which fine-tunes convolutional neural-network classifiers.
2. **Deep feature extraction plus conventional machine learning**, which extracts features with pretrained CNNs and trains classifiers such as SVM and XGBoost.

The accompanying thesis report is available at [`Final Files/Report and Results.pdf`](Final%20Files/Report%20and%20Results.pdf).

> **Research-use notice:** This code is for academic research and is not a medical device or a substitute for clinical diagnosis. Any clinical use would require independent validation, appropriate approvals, and regulatory review.

## Project overview

The workflow starts with a Kaggle retinal-image dataset and organizes images into `Positive` and `Negative` classes. It supports an original dataset variant, a 10×-expanded variant, and a preprocessed 10× variant. Images are resized to **224 × 224** and data are split stratifiably into train/validation/test subsets (70% / 10% / 20%).

The preprocessing flow was implemented following the image-enhancement recommendations in the dataset article—particularly brightness adjustment, circular cropping with fixed aspect ratio, contrast enhancement, and use of the green channel to make retinal structures more visible. The implementation adapts those recommendations for this classification pipeline.

The preprocessing workflow includes:

- grayscale-border removal and circular retinal cropping;
- brightness assessment and adjustment for darker images;
- green-channel histogram equalization; and
- output image resizing and class-wise organization.

For feature-extraction experiments, pretrained CNN backbones are used to produce image representations. The notebooks then optionally apply standard scaling, PCA (100 components), SMOTE, class weighting, and augmentation before fitting conventional classifiers.

## Repository layout

```text
.
├── Final Files/                         # Final thesis notebooks and report
│   ├── Report and Results.pdf            # Thesis report and recorded results
│   ├── make_10x_dataset.ipynb            # Download/organize the dataset and create 10× variant
│   ├── ROP_DataSet_Pr.ipynb              # Retinal-image preprocessing
│   ├── ROP_Classification_NN_FE.ipynb    # Feature extraction: original data
│   ├── ROP_Classification_NN_FE_10x.ipynb
│   ├── ROP_Classification_NN_FE_10x_PP.ipynb
│   └── ROP_BinaryClassification_CNN_Full.ipynb
├── ROP_Bin_Classification_PP/            # Preprocessed-data experiment copies
├── ROP_Bin_Classification_No_PP/         # Non-preprocessed experiment copies
├── ROP_Bin_Classification_No_PP_Finals/  # Final non-preprocessed experiment copies
├── make_10x_dataset.ipynb                # Top-level copy of dataset preparation notebook
└── ROP_DataSet_Pr.ipynb                  # Top-level copy of preprocessing notebook
```

For the final versions, begin in `Final Files/`. The other directories preserve intermediate and comparison experiments.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `make_10x_dataset.ipynb` | Downloads the Kaggle dataset, copies images into a working directory, and assigns images to `Positive`/`Negative` folders based on filename information. |
| `ROP_DataSet_Pr.ipynb` | Applies retinal cropping, brightness correction, green-channel histogram equalization, and resizing. |
| `ROP_Classification_NN_FE.ipynb` | Feature extraction and classical classification on the original dataset. |
| `ROP_Classification_NN_FE_10x.ipynb` | The same pipeline on the 10× dataset. |
| `ROP_Classification_NN_FE_10x_PP.ipynb` | The same pipeline on the preprocessed 10× dataset. |
| `ROP_BinaryClassification_CNN_Full.ipynb` | End-to-end transfer-learning experiments and evaluation. |

## Models and evaluation

### Feature-extraction pipeline

The notebooks compare pretrained **ResNet50**, **EfficientNetB0**, **VGG16**, **InceptionV3**, **MobileNetV2**, and **DenseNet121** feature extractors (availability varies by notebook). The extracted features are evaluated with:

- Logistic Regression
- RBF SVM
- Random Forest
- XGBoost
- K-Nearest Neighbors
- Gradient Boosting

Metrics include accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and classification reports.

### End-to-end CNN pipeline

The full CNN notebook evaluates transfer-learning backbones including ResNet50, EfficientNetB0, MobileNetV2, DenseNet121, InceptionV3, and VGG16. It reports ROC-AUC, PR-AUC, F1-score, precision, recall, loss, and confusion matrices.

### Recorded experiment highlights

The executed final notebooks contain the detailed outputs. Examples from their recorded summaries include:

| Dataset / approach | Feature extractor + classifier | Accuracy | ROC-AUC |
| --- | --- | ---: | ---: |
| Original dataset | ResNet50 + SVM | 0.7436 | 0.8056 |
| 10× dataset | ResNet50 + SVM | 0.8011 | 0.9016 |
| Preprocessed 10× dataset | EfficientNetB0 + SVM | 0.8560 | 0.8947 |
| End-to-end CNN example | Recorded test run | 0.7895 | 0.7678 |

These values are notebook outputs, not a claim of clinical performance. Consult the report and rerun the notebooks for the complete experimental context.

## Dataset

This project uses the **Retinal Image Dataset of Infants and Retinopathy of Prematurity**. The dataset article describes 6,004 retinal images from 188 newborns and provides the image-enhancement approach that informed this project's preprocessing workflow.

- Dataset article: [Timkovič et al., *Retinal Image Dataset of Infants and Retinopathy of Prematurity*, Scientific Data (2024)](https://www.nature.com/articles/s41597-024-03409-7)
- Dataset download: [Kaggle — Retinal Image Dataset of Infants and ROP](https://www.kaggle.com/datasets/jananowakova/retinal-image-dataset-of-infants-and-rop)

The images themselves are deliberately not included in this repository.

Before using the data, review the dataset's license, terms, provenance, and any privacy/ethics requirements on Kaggle. Do not commit patient data, Kaggle API credentials, or generated image datasets to GitHub.

Expected class-folder structure:

```text
ROP-Data/
└── split_dataset/
    ├── Negative/
    └── Positive/
```

The 10× and preprocessed variants use analogous `10_split_dataset/` and `10_split_dataset_preprocessed/` directories.

## Getting started

The notebooks were authored for **Google Colab** and use Google Drive paths such as `/content/drive/MyDrive/ROP-Data/...`.

1. Open a notebook from `Final Files/` in Google Colab.
2. Mount Google Drive when prompted.
3. Obtain Kaggle API credentials and upload `kaggle.json` only to the runtime; never commit it.
4. Run `make_10x_dataset.ipynb` to download and organize the data, if needed.
5. Run `ROP_DataSet_Pr.ipynb` to create the preprocessed dataset variant.
6. Update the dataset paths in the experiment notebook to match your Drive layout.
7. Run the desired classification notebook from top to bottom.

For local execution, create a Python environment and install the libraries used by the notebooks:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install tensorflow numpy pandas scikit-learn imbalanced-learn xgboost opencv-python tqdm matplotlib seaborn jupyter kaggle
jupyter notebook
```

You will also need to replace Colab/Drive-specific cells and paths with local equivalents.

## Reproducibility notes

- The main experiment notebooks set `SEED = 42`.
- Images are resized to 224 × 224 and use a batch size of 16.
- Data splitting is stratified; some 10× workflows use group-aware splitting to reduce leakage between related images.
- Network weights, GPU hardware, library versions, preprocessing choices, and the exact data split can affect results.

## Citation and contact

If you use or build on this work, please cite the accompanying bachelor's thesis and link to this repository. Add the thesis author, university, department, advisor, year, and preferred citation here before publishing.

## License

No license has been selected yet. Before making the repository public, add a `LICENSE` file that matches how you want others to use the code. Ensure that the chosen license is compatible with the dataset and any third-party materials.
