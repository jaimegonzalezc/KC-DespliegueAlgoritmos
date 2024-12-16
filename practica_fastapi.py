from fastapi import FastAPI
from pydantic import BaseModel
import base64
import urllib
from typing import Optional
from transformers import pipeline

app = FastAPI()

@app.get("/summarize")
def summarize_text(text: str):
  summarization_pipeline = pipeline("summarization")
  result = summarization_pipeline(text)
  return {"summary": result}

@app.get('/generate_text')
def generate_text(prompt: str): 
    generator = pipeline('text-generation')
    return generator(prompt)

@app.get('/sentiment')
def sentiment_classifier(query): 
    sentiment_pipeline = pipeline('sentiment-analysis')
    return sentiment_pipeline(query)[0]['label']

@app.get('/translate')
def translate_text(text: str): 
    translation_pipeline = pipeline('translation_en_to_de')
    translated_text = translation_pipeline(text)
    return {'Translated Text': translated_text[0]['translation_text']}

@app.get("/add")
def add(n1: float, n2: float):
  return {"Result": n1+n2}

@app.get('/url_encode')
def encode_url(text: str):
    encoded_url = urllib.parse.quote(text)
    return {'Encoded URL': encoded_url}

@app.get('/url_decode')
def decode_url(encoded_text: str):
    decoded_url = urllib.parse.unquote(encoded_text)
    return {'Decoded URL': decoded_url}