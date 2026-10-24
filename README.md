Swin Transformer for Alzheimer's Disease Classification
This repository contains a complete PyTorch pipeline for multi-class Alzheimer's disease classification using a Swin Transformer (Swin-T) trained on the OASIS Brain MRI Dataset. The model classifies MRI scans into four distinct stages of dementia while addressing severe dataset imbalance through algorithmic cohort capping and inverse loss weighting.

Project Overview
Vision Transformers typically struggle on small, imbalanced medical datasets without massive pre-training. This project adapts the Swin Transformer architecture for medical image analysis, utilizing its shifted window attention mechanism to capture both local anatomical features and global structural changes in brain MRIs.

Hyperparameters were optimized using an automated Optuna sweep, and the final model is serialized via joblib for rapid deployment and inference.

The OASIS Dataset & Class Imbalance
The raw OASIS dataset presents a severe class imbalance (e.g., ~67,200 'Non Demented' scans vs. only 488 'Moderate Dementia' scans). To prevent mode collapse and majority-class bias, this pipeline implements a strict data curation strategy:

Cohort Capping: Majority classes are capped at exactly 1,000 samples.

Minority Preservation: All 488 samples of the rare 'Moderate Dementia' class are preserved.

Stratified Splitting: A deterministic 70/15/15 split (Train/Val/Test) guarantees proportional representation of all four classes across every phase.

Target Classes:

Mild Dementia

Moderate Dementia

Non Demented

Very mild Dementia

Architecture & Training Pipeline
Backbone: Swin-T (Tiny) loaded with default ImageNet weights via PyTorch Image Models (torchvision.models).

Custom Classification Head: Replaces the default head with a high-dropout Sequential block (Dropout -> Linear -> GELU -> Dropout -> Linear) mapping 768 extracted features to the 4 target classes.

Regularization: Heavy data augmentation (Random Rotations, Flips), high label smoothing (0.05), and calculated inverse class penalties applied to the CrossEntropyLoss function.

Optimization: AdamW optimizer paired with a CosineAnnealingLR scheduler to gracefully decay the learning rate and stabilize late-epoch convergence.

Installation & Requirements
Ensure you have Python 3.8+ installed. Install the required dependencies using pip:

Bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn opencv-python optuna joblib
Usage
1. Training the Model
The main script handles data discovery, stratified splitting, and Optuna hyperparameter tuning before launching the final training run.

Bash
python train_swin.py
2. Loading the Model for Inference
The trained model is serialized using joblib. To load the model for inference, ensure the SwinClassifier class definition is present in your active script, then load the .joblib file and map it to your target device.

Python
import joblib
import torch

# Define your device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the entire PyTorch model object from disk
model = joblib.load('swin_classifier.joblib')

# Map to active compute device and set to evaluation mode
model = model.to(device)
model.eval()

print("Swin Transformer ready for inference.")
Evaluation Metrics & Explainability
The pipeline automatically generates a comprehensive suite of evaluation diagnostics on the isolated 15% test set:

Loss & Accuracy Trajectories: Validates parallel tracking to ensure zero overfitting or mode collapse.

Confusion Matrices: Both raw counts and normalized percentages to track inter-class prediction accuracy.

Classification Report: Class-wise Precision, Recall, and F1-Scores to verify minority class performance.

ROC & Precision-Recall Curves: One-vs-Rest multi-class Area Under the Curve (AUC) and Average Precision (AP) scores to evaluate the model's discriminative threshold confidence.
