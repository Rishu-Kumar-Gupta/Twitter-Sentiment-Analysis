# Twitter Sentiment Analysis (NLP)

This is a Natural Language Processing (NLP) project that trains a machine learning model to classify tweets as **Positive**, **Negative**, or **Neutral**.

This project demonstrates a complete ML pipeline: from data loading and cleaning to feature engineering, model training, and evaluation.

## Project Overview

The goal of this project is to build and evaluate a model that can accurately predict the sentiment of a given tweet. This is a multi-class classification problem.

* **Data Source:** The project uses the "Twitter Entity Sentiment Analysis" dataset from Kaggle, which contains tweets manually labeled with their sentiment.
* **Preprocessing:** Text data is cleaned and processed using **NLTK**. This includes:
    * Removing URLs, mentions (`@`), and hashtags (`#`)
    * Tokenization (splitting text into individual words)
    * Removing common English stop-words (like "the", "is", "a")
    * Lemmatization (reducing words to their root form, e.g., "running" -> "run")
* **Feature Engineering:** The cleaned text is converted into a numerical format that a machine learning model can understand using **Scikit-learn's `TfidfVectorizer`**.
* **Modeling:** A `Multinomial Naive Bayes` classifier (or another model like `LogisticRegression`) is trained on the processed data.
* **Evaluation:** The model's performance is measured on an unseen test set, evaluated on its accuracy, precision, recall, and F1-score.

## Technologies Used

* **Python**
* **Pandas:** For loading and manipulating data.
* **NLTK (Natural Language Toolkit):** For all text preprocessing tasks.
* **Scikit-learn (sklearn):** For ML modeling, feature engineering, and evaluation.
* **Google Colab / Jupyter Notebook:** As the development environment.

## How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_GITHUB_USERNAME]/[YOUR_REPOSITORY_NAME].git
    ```

2.  **Install dependencies:**
    This project requires the following Python libraries. You can install them using pip:
    ```bash
    pip install pandas nltk scikit-learn
    ```

3.  **Download NLTK Data:**
    The first time you run this, you will need to download the NLTK packages for stopwords, tokenization, and lemmatization.
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

4.  **Get the Dataset:**
    * This project uses the `twitter_training.csv` file.
    * Download it from [Kaggle: Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis).
    * Place the `twitter_training.csv` file in the same root directory as your notebook file.

5.  **Run the Notebook:**
    * Open the `.ipynb` file (e.g., `twitter_sentiment.ipynb`) in Google Colab or a local Jupyter Notebook server.
    * Run the cells from top to bottom.

## Results

The model was evaluated on a 20% test split of the data.

**Final Model Accuracy: [XX.XX]%**

*(**Action:** Run your script and replace `[XX.XX]%` with the accuracy number it prints. Also, copy and paste the classification report it generates below.)*

### Classification Report
- Classification Report --- precision recall f1-score support

Negative       [0.XX]    [0.XX]    [0.XX]    [XXXX]
 Neutral       [0.XX]    [0.XX]    [0.XX]    [XXXX]
Positive       [0.XX]    [0.XX]    [0.XX]    [XXXX]

accuracy                           [0.XX]    [XXXX]
macro avg [0.XX] [0.XX] [0.XX] [XXXX] weighted avg [0.XX] [0.XX] [0.XX] [XXXX]