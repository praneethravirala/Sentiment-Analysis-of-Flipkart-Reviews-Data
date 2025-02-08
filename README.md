# **Sentiment Analysis of Flipkart Reviews**

## **Project Overview**
This project aims to classify user reviews from Flipkart as **positive, negative, or neutral** using **BERT** and **NaiveBayes**. The dataset, sourced from **Hugging Face**, consists of **126,265** reviews in CSV format.

## **Dataset Details**
- **Source:** Hugging Face ([Flipkart Reviews Dataset](https://huggingface.co/datasets/KayEe/flipkart_sentiment_analysis))
- **Features:**
  - `instruction`: Specifies the task (sentiment analysis).
  - `input`: User review text (independent variable).
  - `output`: Sentiment label (`positive`, `negative`, `neutral`) - **Target variable**.

## **Proposed Architecture**
Below is the high-level **system architecture** used for sentiment analysis on Flipkart reviews:

![Proposed Architecture](https://github.com/praneethravirala/Sentiment-Analysis-of-Flipkart-Reviews-Data/blob/main/ProposedArchitecure.jpeg)

### **Workflow Steps:**
1. **Data Sourcing:** Collecting data from Hugging Face.
2. **Data Conversion:** Transforming JSON data into CSV format for ease of processing.
3. **Data Preprocessing:**
   - Removing special characters, punctuation, and stopwords.
   - Converting text to lowercase.
4. **Data Transformation:** Encoding categorical labels (`0: Negative, 1: Neutral, 2: Positive`) and tokenizing text.
5. **Model Selection:** Evaluating different models (BERT, NaiveBayes).
6. **Model Training & Testing:** Training models on processed data.
7. **Model Evaluation:** Assessing model performance using accuracy, precision, recall, and F1-score.

## **Modeling Approaches**
We experiment with:
- **Traditional ML Models:** Naïve Bayes.
- **Deep Learning Models:** BERT (Bidirectional Encoder Representations from Transformers)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score.

## **Implementation & Tools**
- **Programming:** Python
- **Libraries:** `NLTK`, `spaCy`, `Transformers`, `scikit-learn`, `TensorFlow`, `PyTorch`
- **Development Platforms:** Jupyter Notebook, Google Colab

## **Results**
- The sentiment analysis results show that **BERT outperforms Naïve Bayes** in accuracy (**91.66% vs. 85.46%**) and recall (**71.41% vs. 43.99%**), ensuring better detection of actual sentiments.  
- While **Naïve Bayes has higher precision (92.54%)**, its recall is significantly lower, leading to **poor F1-score (46.67%)** compared to **BERT's 73.84%**.  
- **BERT provides a better balance between precision and recall**, making it more suitable for sentiment classification.  

**Conclusion:**  
**BERT is the preferred model** for Flipkart review analysis due to its **superior contextual understanding and overall performance**, while **Naïve Bayes struggles** with capturing nuanced sentiment patterns.

## **Contributors**  
- Praneeth Ravirala  
- Shalvi Sanjay Lale  
- Sagarika Komati Reddy

