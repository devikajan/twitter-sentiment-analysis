# Twitter Sentiment Analysis (Sentiment140 Dataset)

This project performs sentiment analysis on tweets using the Sentiment140 dataset. It classifies tweets into **positive**, **negative**, or **neutral** sentiment using text preprocessing, TF-IDF vectorization, and machine learning models like **Logistic Regression** and **Naive Bayes**.

📌 Guided by: GeeksforGeeks YouTube tutorial  
💻 Platform Used: Google Colab

---

## 🔍 Dataset

- Source: Kaggle — Sentiment140 Dataset
- Total Tweets: 1.6 million
- Labels:
  - 0 = Negative
  - 2 = Neutral
  - 4 = Positive

---

## 📌 Features

- Text cleaning and preprocessing:
  - Removal of links, mentions, and special characters
  - Lowercasing and tokenization
  - Stopword removal
- TF-IDF vectorization for feature extraction
- Model training using:
  - Logistic Regression
  - Multinomial Naive Bayes
- Evaluation using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- Custom tweet sentiment prediction

---

## 🛠️ Technologies Used

- Google Colab (Python environment)
- Pandas, NumPy — data handling
- NLTK, re — text processing
- Scikit-learn — machine learning models
- Matplotlib — basic visualization

---

## 🚀 How to Run in Google Colab

### Step 1: Open the Colab Notebook  
Use this link to open the notebook:  
https://colab.research.google.com/drive/1E1FeEXq2Yj5Z88dNz-NT5is8GsAfnOwE?usp=sharing

---

### Step 2: Upload the Dataset

Download the file `training.1600000.processed.noemoticon.csv` from Kaggle and upload it manually using this code:

from google.colab import files  
uploaded = files.upload()

---

### Step 3: Load and Prepare the Dataset

Import pandas:  
import pandas as pd

Load the CSV file:  
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None)

Rename the columns:  
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

---

### Step 4: Install Required Libraries (if not installed)

Use the following command to install libraries:  
!pip install nltk scikit-learn matplotlib

---

### Step 5: Continue with These Steps in the Notebook

- Preprocess tweets:
  - Convert text to lowercase
  - Remove URLs, mentions, and special characters
  - Tokenize the text
  - Remove stopwords

- Convert text data into numerical vectors using TF-IDF

- Train models:
  - Logistic Regression
  - Multinomial Naive Bayes

- Evaluate the models using:
  - Accuracy score
  - Confusion matrix
  - Classification report

- Predict sentiment for custom tweets using the trained model

---

## 📊 Sample Output

Example accuracy: 81.5%  
Example confusion matrix:  
[[120 10]  
 [22 148]]

Prediction example:  
Input: "I love this product!"  
Output: Positive

---

## 📚 References

- GeeksforGeeks YouTube - Twitter Sentiment Analysis  
- Sentiment140 Dataset - Kaggle  
- Scikit-learn Documentation

---

## 🧠 Future Work

- Apply deep learning models like LSTM or BERT for better results
- Deploy the model as a web app using Streamlit or Flask
- Improve accuracy by including emoji/emoticon handling and hashtag analysis
- Try advanced vectorization like Word2Vec or BERT embeddings

---

## 👩‍💻 Author

Devika Janardhanan


