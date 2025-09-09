from transformers import pipeline

# Preload models to cache them
pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
pipeline("ner", model="dslim/bert-base-NER")