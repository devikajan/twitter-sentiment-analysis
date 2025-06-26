# Twitter Sentiment Analysis (Sentiment140 Dataset)

This project performs sentiment analysis on tweets using the [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) dataset. It classifies tweets into positive, negative, or neutral sentiment using text preprocessing, feature extraction (TF-IDF), and machine learning models like Logistic Regression and Naive Bayes.

> 📌 **Guided by:** GeeksforGeeks YouTube tutorial  
> 💻 **Platform Used:** Google Colab

---

## 🔍 Dataset
- **Source:** Kaggle — [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Contains:** 1.6 million tweets with sentiment labels
  - `0` = Negative
  - `2` = Neutral
  - `4` = Positive

---

## 📌 Features
- Text cleaning and preprocessing
- TF-IDF vectorization
- Model training and evaluation
- Accuracy and classification report
- Custom tweet sentiment prediction

---

## 🛠️ Technologies Used
- Google Colab (Python)
- Pandas, NumPy
- NLTK, Regex
- Scikit-learn (ML models)
- Matplotlib

---

## 🚀 How to Run in Google Colab

1. **Open Colab Notebook**  
    https://colab.research.google.com/drive/1E1FeEXq2Yj5Z88dNz-NT5is8GsAfnOwE?usp=sharing

2. **Upload Dataset**  
   Download `training.1600000.processed.noemoticon.csv` from Kaggle and upload it manually to Colab using:
   ```python
   from google.colab import files
   uploaded = files.upload()

   Load and Prepare the Dataset
---

python
Copy code
import pandas as pd

# Load dataset
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None)

# Rename columns
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
Install Required Libraries (if not pre-installed)

python
Copy code
!pip install nltk scikit-learn matplotlib
Proceed with the notebook steps:

Preprocess tweets (tokenization, stopword removal, etc.)

Convert to vectors using TF-IDF

Train machine learning models

Evaluate performance

Predict sentiment for custom tweets

---

📚 References
GeeksforGeeks YouTube - Twitter Sentiment Analysis

Sentiment140 Dataset - Kaggle

Scikit-learn Documentation

---

🧠 Future Work
Apply deep learning models like LSTM or BERT for improved accuracy

Deploy the model as a web app using Streamlit or Flask

Handle emojis, hashtags, and slang in tweets for more accurate sentiment detection

---

👩‍💻 Author
Devika Janardhanan

---

📜 License
This project is licensed under the MIT License.


