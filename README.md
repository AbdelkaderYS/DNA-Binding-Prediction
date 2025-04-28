# DNA Binding Prediction System

## Overview
This project implements a 1D Convolutional Neural Network (CNN) to predict DNA-protein binding sites using sequence data, developed for the Kaggle challenge: [DNA Intro to Neural Nets - AIMS Rwanda 2025](https://www.kaggle.com/competitions/dna-intro-to-neural-nets-aims-rwanda-2025/overview). The system employs one-hot encoding for DNA sequences and a deep learning approach to classify whether a DNA sequence is bound by a protein.

## Features
- DNA sequence preprocessing with one-hot encoding
- 1D CNN architecture optimized for binding site prediction
- Regularization techniques to prevent overfitting
- Cross-validation for robust model evaluation
- Performance visualization and analysis

## Technical Details

### Dependencies
- pandas
- numpy
- tensorflow
- scikit-learn
- matplotlib

### Data Processing
- Loads DNA sequence data from CSV files (available at the [Kaggle competition page](https://www.kaggle.com/competitions/dna-intro-to-neural-nets-aims-rwanda-2025/data))
- Converts DNA sequences to one-hot encoded format
- Pads sequences to uniform length for batch processing

## Model Architecture

The DNA binding prediction model is a 1D Convolutional Neural Network (CNN) implemented using TensorFlow/Keras. Below is the detailed architecture:

- **Input Layer**:
  - Input shape: `(max_length, 4)`, where `max_length` is the padded length of the DNA sequence, and 4 represents the one-hot encoded channels (A, C, G, T).

- **Layer 1: Convolutional Layer**:
  - 64 filters, kernel size of 10
  - Activation: ReLU
  - L2 regularization (weight decay: 0.001)
  - Followed by **Dropout** (0.4) to prevent overfitting
  - Followed by **MaxPooling1D** (pool size: 3) for dimensionality reduction

- **Layer 2: Convolutional Layer**:
  - 32 filters, kernel size of 5
  - Activation: ReLU
  - L2 regularization (weight decay: 0.1)
  - Followed by **GlobalMaxPooling1D** to reduce the sequence to a fixed-size vector

- **Dense Layers**:
  - Dense layer with 32 units, ReLU activation
  - **Dropout** (0.5) for further regularization
  - Final dense layer with 1 unit, **sigmoid** activation for binary classification

### Compilation
- **Optimizer**: Adam (learning rate: 0.001235)
- **Loss Function**: Binary cross-entropy
- **Metrics**: Accuracy

### Summary
The model is designed to capture local patterns in DNA sequences through convolutional layers, reduce dimensionality with pooling, and prevent overfitting with dropout and L2 regularization. The final sigmoid output predicts the probability of a DNA sequence being a protein-binding site.

### Training Methodology
- Binary cross-entropy loss function
- Adam optimizer with custom learning rate (0.001235)
- Early stopping and learning rate reduction callbacks
- 5-fold stratified cross-validation

### Evaluation
- Performance metrics: accuracy and loss
- Visualization of training and validation metrics per fold
- Best model selection based on validation accuracy

### Prediction
- Generates binary predictions for test data
- Creates a submission file with predicted binding status

## Usage
1. Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/dna-intro-to-neural-nets-aims-rwanda-2025/overview).
2. Ensure input data is in the correct format and location.
3. Run the script to:
   - Process DNA sequences
   - Train the model with cross-validation
   - Generate predictions on test data
   - Save results to a CSV file

## Output
- A `Final_Submission.csv` file with binding predictions
- Performance metrics displayed during training
- Visualizations of training history showing model convergence
