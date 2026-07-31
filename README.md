# ROP Diagnosis System

Binary classification of Retinopathy of Prematurity (ROP) from infant retinal fundus images. The project evaluates transfer-learning CNNs, then uses the most promising CNN backbones as feature extractors for conventional machine-learning classifiers.

This project has been done as Bachelor's final thesis project.

> **Research-use only.** This project is not a medical device and must not be used as a substitute for clinical diagnosis. 

---
## Dataset sources

This work uses the **Retinal Image Dataset of Infants and Retinopathy of Prematurity**:

- Dataset article: [Timkovic et al., *Retinal Image Dataset of Infants and Retinopathy of Prematurity*, Scientific Data (2024)](https://www.nature.com/articles/s41597-024-03409-7)
- Dataset download: [Kaggle - Retinal Image Dataset of Infants and ROP](https://www.kaggle.com/datasets/jananowakova/retinal-image-dataset-of-infants-and-rop)

The dataset article describes 6,004 retinal images from 188 newborns. Images are not included in this repository. Before using the data, review its license, terms, provenance, and all applicable privacy and ethics requirements.

---
## Experimental design

The study follows a staged evaluation process designed to compare models while reducing patient-level data leakage.

```text
Original dataset (one image per patient)
        |
        +--> Stage 1: evaluate end-to-end CNN classifiers
        |        |
        |        +--> retain the best-performing CNN backbones
        |
        +--> Stage 2: use selected CNNs as feature extractors
        |        +--> train and compare conventional classifiers
        |        +--> retain the best CNN-classifier combinations
        |
10x dataset (ten images per patient)
        |
        +--> patient-oriented Positive/Negative split
        |     (a patient's images stay in only one split)
        +--> re-evaluate selected combinations
        |
Preprocessed 10x dataset
        |
        +--> apply the article-informed preprocessing flow
        +--> re-evaluate the selected combinations
```

### Stage 1 - CNN baseline selection

From the main dataset, **one image per patient** is selected to form the initial experimental dataset. Transfer-learning CNNs are trained as end-to-end binary classifiers to identify the most suitable backbones for the next stage. The full-CNN notebook evaluates ResNet50, EfficientNetB0, MobileNetV2, DenseNet121, InceptionV3, and VGG16 with ImageNet weights and a binary classification head.

### Stage 2 - Feature extraction and classical classification

The selected CNN backbones are then used as fixed feature extractors. Their deep feature vectors are classified with widely used conventional models:

- Logistic Regression
- RBF SVM
- Random Forest
- XGBoost
- K-Nearest Neighbors
- Gradient Boosting

The pipelines support scaling, PCA (100 components), SMOTE, class weighting, and augmentation. Combinations are evaluated using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and classification reports.

### Stage 3 - Patient-oriented 10x dataset

The best CNN/classifier combinations are evaluated on a larger dataset created by selecting **ten images per patient**. The Positive and Negative datasets are split at the **patient level**: images from a patient are assigned to one split only. This prevents related images of the same patient from appearing in both training and test data, reducing data leakage.

### Stage 4 - Preprocessed 10x dataset

The preprocessing flow is applied to the 10x dataset and the selected combinations are tested again. The flow follows and adapts the image-enhancement recommendations in the dataset article, including:

- removing grayscale borders and circular cropping with fixed aspect ratio;
- brightness assessment and correction for darker images;
- contrast enhancement through green-channel histogram equalization; and
- resizing images to 224 x 224 pixels.

The article motivates brightness correction, circular cropping, contrast enhancement, and green-channel analysis to improve the visibility of retinal structures. This repository adapts those methods to a binary-classification workflow.

---
## Results summary

The table shows selected results recorded in the executed notebooks. Full outputs and experiment details are available in the notebooks and in [Report and Results.pdf](Report%20and%20Results.pdf).

| Experiment stage | Selected combination / result | Accuracy | ROC-AUC |
| --- | --- | ---: | ---: |
| One-image-per-patient feature extraction | EfficientNetB0 + Logistic Regression | 0.8205 | 0.8426 |
| One-image-per-patient feature extraction | EfficientNetB0 + SVM | 0.7692 | 0.8457 |
| Patient-oriented 10x dataset | EfficientNetB0 + Random Forest | 0.8481 | 0.9015 |
| Patient-oriented 10x dataset | ResNet50 + Logistic Regression | 0.7652 | 0.9119 |
| Preprocessed patient-oriented 10x dataset | EfficientNetB0 + SVM | **0.8560** | 0.8947 |

For the end-to-end CNN baseline, a recorded ResNet50 test run achieved accuracy of **0.7895**, ROC-AUC of **0.7678**, PR-AUC of **0.6738**, and F1-score of **0.5385**. These are research results from specific data splits and settings, not clinical-performance claims.

---
## Repository contents

| Notebook | Purpose |
| --- | --- |
| `make_10x_dataset.ipynb` | Downloads the Kaggle dataset, organizes it by class, and builds the 10x dataset. |
| `ROP_DataSet_PP.ipynb` | Applies the preprocessing workflow. |
| `ROP_BinaryClassification_CNN_Full.ipynb` | Stage 1: end-to-end transfer-learning CNN evaluation. |
| `ROP_Classification_NN_FE.ipynb` | Stage 2: feature extraction and classical classification on the one-image-per-patient dataset. |
| `ROP_Classification_NN_FE_10x.ipynb` | Stage 3: evaluation on the patient-oriented 10x dataset. |
| `ROP_Classification_NN_FE_10x_PP.ipynb` | Stage 4: evaluation on the preprocessed 10x dataset. |
| `Report and Results.pdf` | Thesis report and detailed results. (Farsi/Persian)|

---
## Running the notebooks

The notebooks were authored for Google Colab and expect Google Drive paths under `/content/drive/MyDrive/ROP-Data/`.

1. Open the required notebook in Google Colab and mount Google Drive.
2. Obtain the dataset through Kaggle; upload `kaggle.json` only to the Colab runtime and never commit it.
3. Run `make_10x_dataset.ipynb` to prepare the dataset variants.
4. Run `ROP_DataSet_Pr.ipynb` to create the preprocessed 10x dataset.
5. Update dataset paths as needed, then run the desired experiment notebook from top to bottom.

Core dependencies include TensorFlow, NumPy, pandas, scikit-learn, imbalanced-learn, XGBoost, OpenCV, matplotlib, seaborn, tqdm, and Kaggle.

---
## Reproducibility notes

- The main experiments use `SEED = 42`, a batch size of 16, and 224 x 224 images.
- Results can vary with library versions, GPU hardware, pretrained weights, preprocessing configuration, and the data split.
- Do not commit images, patient information, dataset copies, or Kaggle credentials to GitHub.

---
Isfahan University of Technology - BSc Final Thesis