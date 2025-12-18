# Human Activity Recognition (HAR)

**Feature Engineering for Machine Learning (ECAC)**
BSc in Informatics Engineering - University of Coimbra (FCTUC)

This project implements a complete Data Science *pipeline* for **Human Activity Recognition** (HAR), using the *FORTH-TRACE* dataset. The system covers everything from the ingestion of raw inertial sensor data to final classification, comparing manual feature engineering against Deep Learning latent representations (*embeddings*).

---

## Author

* **Akcel Soares da Graça** (2022241055)

---

## Project Architecture

The work is divided into two complementary modules:

### Part A: Feature Engineering
Focused on exploratory analysis, cleaning, and information extraction.
* **Data Processing:** *Outlier* detection using univariate (Z-Score, IQR) and multivariate (K-Means) methods.
* **Feature Extraction:** Calculation of temporal, spectral (FFT), and physical metrics from accelerometer, gyroscope, and magnetometer.
* **Dimensionality Reduction:** Implementation of PCA (Principal Component Analysis).
* **Attribute Selection:** Implementation of *Fisher Score* and *ReliefF* algorithms to identify the most relevant features.

### Part B: Machine Learning and Evaluation
Focused on supervised classification of activities 1 to 7 (e.g., *Standing*, *Walking*, *Climbing Stairs*).
* **Data Augmentation:** Class balancing using the **SMOTE** technique to generate synthetic samples.
* **Embeddings:** Automatic feature extraction using the **HARNet5** Deep Learning model (Transfer Learning) on data resampled at 30Hz.
* **Classification:** **k-Nearest Neighbors (kNN)** model with automatic hyperparameter optimization.
* **Validation:** Comparison of *Within-Subject* vs. *Between-Subjects* splitting strategies.
* **Deployment:** "Turnkey" system capable of receiving a raw segment and returning the classification.

---

## Installation and Requirements

### Prerequisites
The project requires **Python 3.8+** and the following libraries:

```bash
pip install numpy pandas matplotlib scipy scikit-learn torch seaborn
