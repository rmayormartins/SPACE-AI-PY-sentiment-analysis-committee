from transformers import pipeline
import gradio as gr
from textblob import TextBlob
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from afinn import Afinn

#VADER e AFINN
nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()
afinn = Afinn()

#Hugging Face
bert_model = pipeline("sentiment-analysis", model="bert-base-uncased")
#BERT Large
bert_large_model = pipeline("sentiment-analysis", model="bert-large-uncased")
distilbert_model = pipeline("sentiment-analysis", model="distilbert-base-uncased")
siebert_model = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")


def normalize_score(score, range_min, range_max):
    return (score - range_min) / (range_max - range_min)


def analyze_with_bert(text):
    analysis = bert_model(text)
    label, score = map_label(analysis[0]['label']), analysis[0]['score']
    return label, score


def analyze_with_bert_large(text):
    analysis = bert_large_model(text)
    label, score = map_label(analysis[0]['label']), analysis[0]['score']
    return label, score

def analyze_with_distilbert(text):
    analysis = distilbert_model(text)
    label, score = map_label(analysis[0]['label']), analysis[0]['score']
    return label, score

def analyze_with_siebert(text):
    analysis = siebert_model(text)
    return analysis[0]['label'], analysis[0]['score']

def analyze_with_textblob(text):
    analysis = TextBlob(text).sentiment
    label = "POSITIVE" if analysis.polarity > 0 else "NEGATIVE" if analysis.polarity < 0 else "NEUTRAL"
    normalized_score = normalize_score(analysis.polarity, -1, 1)
    return label, normalized_score

def analyze_with_vader(text):
    scores = vader.polarity_scores(text)
    label = "POSITIVE" if scores['compound'] > 0.05 else "NEGATIVE" if scores['compound'] < -0.05 else "NEUTRAL"
    normalized_score = normalize_score(scores['compound'], -1, 1)
    return label, normalized_score

def analyze_with_afinn(text):
    score = afinn.score(text)
    label = "POSITIVE" if score > 0 else "NEGATIVE" if score < 0 else "NEUTRAL"
    normalized_score = normalize_score(score, -5, 5)
    return label, normalized_score

#mapeio BERT e DistilBERT
def map_label(label):
    if label == "LABEL_0":
        return "NEGATIVE"
    elif label == "LABEL_1":
        return "POSITIVE"
    else:
        return "NEUTRAL"


#Comite
def calculate_committee_decision(results):
    #coto voto
    vote_count = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for label, score in results.values():
        vote_count[label] += 1

    #maioria dos votos
    final_label = max(vote_count, key=vote_count.get)
    return final_label, vote_count[final_label] / len(results)




def analyze_text(text):
    results = {
        "BERT Base": analyze_with_bert(text),
        "BERT Large": analyze_with_bert_large(text),
        "DistilBERT": analyze_with_distilbert(text),
        "SiEBERT": analyze_with_siebert(text),
        "TextBlob": analyze_with_textblob(text),
        "VADER": analyze_with_vader(text),
        "AFINN": analyze_with_afinn(text)
    }

    final_label, vote_ratio = calculate_committee_decision(results)
    results["Committee Decision"] = {"label": final_label, "vote_ratio": vote_ratio}
    return results


# Gradio Interface
iface = gr.Interface(
    fn=analyze_text,
    inputs="text",
    outputs="json",
    title="Sentiment-Analysis-Committee",
    description="Enter a text. And the Democratic committee among Sentiment Analysis methods will conduct the vote."
)
iface.launch(debug=True)
