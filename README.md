# Twitter Sentiment Analysis (Sentiment140 Dataset)

This project performs sentiment analysis on tweets using the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140). It classifies tweets into **positive**, **negative**, or **neutral** sentiment using text preprocessing, feature extraction (TF-IDF), and machine learning models like **Logistic Regression** and **Naive Bayes**.

> 📌 **Guided by:** GeeksforGeeks YouTube tutorial  
> 💻 **Platform Used:** Google Colab

---

## 🔍 Dataset

- **Source:** [Kaggle — Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Total Tweets:** 1.6 million
- **Labels:**
  - `0` = Negative  
  - `2` = Neutral  
  - `4` = Positive

---

## 📌 Features

- Text cleaning and preprocessing (removal of links, mentions, special characters, etc.)
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

- **Google Colab** (Python environment)
- **Pandas**, **NumPy** — data handling
- **NLTK**, **re** — text processing
- **Scikit-learn** — machine learning models
- **Matplotlib** — basic visualization

---

## 🚀 How to Run in Google Colab

### 1. Open the Colab Notebook  
[Click here to open in Google Colab](https://colab.research.google.com/drive/1E1FeEXq2Yj5Z88dNz-NT5is8GsAfnOwE?usp=sharing)

---

### 2. Upload the Dataset  
Download `training.1600000.processed.noemoticon.csv` from Kaggle and upload it manually:

```python
from google.colab import files
uploaded = files.upload()
3. Load and Prepare the Dataset
python
Copy
Edit
import pandas as pd

# Load dataset
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None)

# Rename columns
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
4. Install Required Libraries (if not pre-installed)
python
Copy
Edit
!pip install nltk scikit-learn matplotlib
5. Continue With These Steps in the Notebook
🔹 Preprocess tweets:

Lowercase conversion

Removing URLs, mentions, special characters

Tokenization

Stopword removal

🔹 Convert text data into numerical form using TF-IDF Vectorizer.

🔹 Train models like Logistic Regression and Naive Bayes.

🔹 Evaluate performance using accuracy, confusion matrix, and classification report.

🔹 Predict sentiment for custom tweets using your trained model.


📊 Sample Output 
Example:
Accuracy: 81.5%
Confusion Matrix
Prediction: "I love this product!" → Positive


📚 References
GeeksforGeeks YouTube - Twitter Sentiment Analysis
Sentiment140 Dataset - Kaggle
Scikit-learn Documentation


🧠 Future Work
🧠 Apply deep learning models like LSTM or BERT for better results.
🌐 Deploy the model as a web app using Streamlit or Flask.
😊 Improve accuracy by including emoji/emoticon handling and hashtag analysis.
🔤 Try different vectorization methods like Word2Vec or BERT embeddings.


👩‍💻 Author
Devika Janardhanan


📜 License
This project is licensed under the MIT License.

