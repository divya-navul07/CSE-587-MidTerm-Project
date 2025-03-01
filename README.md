# CSE 587: DEEP LEARNING FOR NLP-MidTerm Project
## ANALYZING MOVIE REVIEW SENTIMENTS USING BIDIRECTIONAL LSTM, CNN and GloVe
This project aims to classify the sentiment of IMDB movie reviews as either positive or negative using the deep learning models. 

I have implemented Two Networks:

**1. Bidirectional LSTM (BiLSTM)**

**2. Convolutional Neural Network (CNN)**

Both the above models used the pre-trained **GloVe embeddings** for word representation and were trained on a balanced dataset of 50,000 IMDB movie reviews dataset.

## **Project Overview**

The overall goal was to develop sentiment analysis models that would be able to classify movie reviews accurately. The dataset comprises of 25,000 positive and 25,000 negative reviews, ensuring balanced training and evaluation. 

**1. Text Preprocessing:** Cleaning, tokenization, and padding/truncation of sequences.

**2. Model Training:** Both of models were trained with the Adam optimizer, binary cross-entropy loss, and early stopping to prevent overfitting.

**3. Evaluation:** Performance metrics (accuracy, precision, recall) were computed.

## **Model Architectures**

**1. Bidirectional Long Short Term Memory (BiLSTM)**

* **Embedding Layer:** Initialized with 100-dimensional GloVe embeddings.

* **LSTM Layers:** Two bidirectional LSTM layers (64 and 32 units) to capture contextual dependencies.

* **Dense Layers:** Fully connected layers with ReLU activation, L2 regularization, and dropout.

* **Output Layer:** Sigmoid activation for binary classification.

**2. Convolutional Neural Network (CNN)**

* **Embedding Layer:** Initialized with 100-dimensional GloVe embeddings.

* **Convolutional Layer:** 128 filters of dimension 5 for local n-gram features capture.

* **Pooling Layer:** Global max pooling for catching vital features.

* **Dense Layers:** Fully connected layers with ReLU activation, L2 regularization, and dropout.

* **Output Layer:** Sigmoid activation for binary output.

## **Dataset Curation**

The dataset was downloaded through IMDB movie reviews dataset via Tensorflow containing 50,000 labeled reviews out of which 25k is positive, 25k negative reviews.

* **Text Cleaning:** HTML tag removal, special characters removal, and lowercase conversion.

* **Tokenization:** Vocabulary limited to 10,000 most common words, OOV tokens replaced with <UNK.

* **Sequence Standardization:** Reviews padded/truncated to 200 words for balanced input dimensions.

## **Training**

1. **BiLSTM:** Trained on 8 epochs with 86.52% validation accuracy.

2. **CNN:** Trained on 8 epochs with 83.70% validation accuracy.

## **Results**

**1. BiLSTM:**

* **Accuracy:** 86.52%

* **Precision:** 87.13% (positive), 86.09% (negative)

* **Recall:** 86.09% (positive), 87.15% (negative)

**2. CNN:**

* **Accuracy:** 83.70%

* **Precision:** 88.86% (positive), 82.99% (negative)

* **Recall:** 77.52% (positive), 83.81% (negative)
  
## **Error Analysis:** 
Both models showed shortcomings in understanding nuanced language when faced with mixed reactions ("Great acting, but painfully slow") and sarcastic evaluations ("Worst masterpiece I've ever seen").

## **Graphical Analysis:**
* Graphs were created to display accuracy and loss by epoch for both the models, demonstrating consistent gains for both models.
* Label distribution also showed the CNN's slight skewness and the BiLSTM's balanced predictions, with sarcasm and mixed sentiments for the majority of misclassifications.

## **Running the Code**

* Clone the repository and install the requirements
* Download the IMDB dataset using TensorFlow Datasets
* Download GloVe Embeddings
* Run the .ipynb file

## **Future Work**

* **Model Improvements:** We can apply transformer-based architectures (e.g., BERT) to reduce the ambiguity and sarcasm of the reviews.

* **Data Augmentation:** Increase the dataset by generating reviews to generalize better.

## **License**
This project is open-source under the MIT License.
