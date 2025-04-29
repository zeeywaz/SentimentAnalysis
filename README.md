# Sentiment Analysis - TensorFlow & SVM

This project involves building and comparing two sentiment analysis models using an open-source dataset of labeled text reviews. The goal is to classify the sentiment of each review as **positive**, **negative**, or **neutral**.

## Model Details

- **TensorFlow Model**: A deep learning approach using embedding layers, LSTM, and dense layers to capture complex patterns in text.  
  ✅ Achieved an accuracy of **82%**.

- **SVM Model**: A traditional machine learning approach using TF-IDF vectorization and Support Vector Machine for classification.  
  ✅ Achieved an accuracy of **71%**.

## Dataset

The dataset used was sourced from an open-source sentiment analysis corpus. It contains pre-labeled text samples across three sentiment categories.

## Workflow

1. **Preprocessing**: 
   - Tokenization and text cleaning
   - Stopword removal and lemmatization
   - Vectorization using TF-IDF for SVM, and token embedding for TensorFlow

2. **Model Training**:
   - Trained TensorFlow model with LSTM-based architecture
   - Trained SVM model with optimized hyperparameters

3. **Evaluation**:
   - Compared accuracy scores on a test set
   - TensorFlow performed better in handling complex patterns in text

## Key Libraries

- `TensorFlow`, `Keras`
- `scikit-learn`
- `NLTK`
- `NumPy`, `Pandas`

## Results

| Model        | Accuracy |
|--------------|----------|
| TensorFlow   | 82%      |
| SVM          | 71%      |

## screenshots

SVM: ![image](https://github.com/user-attachments/assets/c5a6b7a6-6ca7-4b0d-b779-125d703806e2)



Tensorflow: ![image](https://github.com/user-attachments/assets/e9b97a33-94a8-451a-9aad-470d9540c422)


