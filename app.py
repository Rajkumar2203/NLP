import scrapy
from scrapy.crawler import CrawlerProcess
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from gensim import corpora, models
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
import streamlit as st

class MalaymailSpider(scrapy.Spider):
    name = 'malaymail'
    allowed_domains = ['malaymail.com']
    start_urls = ['https://www.malaymail.com/morearticles/malaysia']

    def parse(self, response):
        for page_num in range(1, 26):
            page_url = f"{response.url}?pgno={page_num}"
            yield scrapy.Request(url=page_url, callback=self.parse_page)

    def parse_page(self, response):
        article_links = response.css('.col-md-3.article-item h2.article-title a::attr(href)').extract()
        for article_link in article_links:
            yield scrapy.Request(article_link, callback=self.parse_article)

    def parse_article(self, response):
        headline = response.css('h1.article-title::text').get()
        date = response.css('div.article-date::text').get()
        paragraphs = response.css('div.article-body p::text').extract()

        # Combine paragraphs into a single text
        article_text = ' '.join(paragraphs)

        # Text summarization function using BART
        summary = generate_summary(article_text)

        # Classify the article using a trained model
        category = classify_text(article_text)

        # Perform topic modeling using LDA
        topics = perform_topic_modeling(article_text)

        return {
            'Headline': headline,
            'Date': date,
            'Paragraphs': paragraphs,
            'Summary': summary,
            'Category': category,
            'Topics': topics
        }

# Function to classify text
def classify_text(text):
    # Placeholder code for demonstration purposes
    # Replace this with your actual labeled dataset for training
    training_data = [
        {'text': 'sample text 1', 'category': 'Category1'},
        {'text': 'sample text 2', 'category': 'Category2'},
        # Add more labeled data...
    ]

    # Extract text and labels
    texts = [item['text'] for item in training_data]
    labels = [item['category'] for item in training_data]

    # Create a TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the training data
    X_train_tfidf = vectorizer.fit_transform(texts)

    # Train a Multinomial Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, labels)

    # Transform the input text using the vectorizer
    text_tfidf = vectorizer.transform([text])

    # Predict the category
    prediction = classifier.predict(text_tfidf)

    return prediction[0]

def perform_topic_modeling(text):
    # Tokenize and preprocess the text
    tokens = [word for word in text.split() if len(word) > 1]

    # Create a dictionary representation of the documents
    dictionary = corpora.Dictionary([tokens])

    # Create a bag-of-words corpus
    corpus = [dictionary.doc2bow(tokens)]

    # Train the LDA model
    lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary)

    # Get the main topics and their probabilities
    topics = lda_model.show_topics(num_topics=3, num_words=5, formatted=False)

    return topics

# Load the pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Function to generate text summary using BART
def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app
def main():
    st.title("Malaymail Web Scraper and NLP App")

    # Create buttons for individual tasks
    if st.button("Topic Modeling"):
        text_input = st.text_area("Enter text for Topic Modeling:")
        if text_input:
            topic_results = perform_topic_modeling(text_input)
            st.table(pd.DataFrame(topic_results))

    if st.button("Text Classification"):
        text_input = st.text_area("Enter text for Text Classification:")
        if text_input:
            classification_result = classify_text(text_input)
            st.write("Text Classification Result:", classification_result)

    if st.button("Text Summarization"):
        text_input = st.text_area("Enter text for Text Summarization:")
        if text_input:
            summarization_result = generate_summary(text_input)
            st.write("Text Summarization Result:", summarization_result)

    # Create a button for an all-in-one task
    if st.button("All in One"):
        results = run_web_scraper()
        st.table(pd.DataFrame(results))

# Function to run the web scraper
def run_web_scraper():
    process = CrawlerProcess(settings={
        'FEEDS': {
            'output.json': {
                'format': 'json',
                'overwrite': True,
                'ensure_ascii': False  # Ensure non-ASCII characters are properly encoded
            },
        },
    })

    process.crawl(MalaymailSpider)
    process.start()

    # Load the results from the scraped data (modify this based on your output format)
    df = pd.read_json("output.json")
    return df.to_dict(orient='records')

if __name__ == "__main__":
    main()
