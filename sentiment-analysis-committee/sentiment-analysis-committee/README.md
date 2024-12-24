---
title: sentiment-analysis-committee
emoji: 👥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.12.0"
app_file: app.py
pinned: false
---


# Sentiment Analysis Committee

A comprehensive sentiment analysis tool using multiple methods, including BERT (Base and Large), DistilBERT, SiEBERT, TextBlob, VADER, and AFINN.

## How to Use

Enter text into the interface to receive sentiment analyses from various methods. The committee's decision is based on the majority of votes among the methods.

## Technical Details

This project leverages various natural language processing models to evaluate the sentiment of entered text:

- **BERT Base and BERT Large**: Transformer-based models providing sentiment scores and labels. BERT Large is a larger variant of BERT with more layers, potentially offering more nuanced sentiment analysis.
- **DistilBERT**: A distilled version of BERT, optimized for speed and efficiency.
- **SiEBERT**: A RoBERTa-based model fine-tuned for sentiment analysis.
- **TextBlob**: Utilizes Naive Bayes classifiers, offering straightforward sentiment evaluations.
- **VADER**: Designed for social media and short texts, giving a compound sentiment score.
- **AFINN**: A lexical method assigning scores to words, indicating sentiment intensity.

The final decision of the committee is determined by a majority vote approach, providing a balanced sentiment analysis.

## Additional Information

- Developed by Ramon Mayor Martins (2023)
- E-mail: [rmayormartins@gmail.com](mailto:rmayormartins@gmail.com)
- Homepage: [https://rmayormartins.github.io/](https://rmayormartins.github.io/)
- Twitter: [@rmayormartins](https://twitter.com/rmayormartins)
- GitHub: [https://github.com/rmayormartins](https://github.com/rmayormartins)

## Notes

- The committee's decision is democratic, based on the majority vote from the utilized methods.
- The project is implemented in Python and hosted on Hugging Face Spaces.





