---
title: Papluca Multilingual Language Detector
emoji: 🌍
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Papluca Multilingual Language Detector

This application detects the language of an input text using the pretrained
`papluca/xlm-roberta-base-language-detection` model from Hugging Face.

## Features

- Detects 20 supported languages
- Displays the language name
- Displays the language code
- Displays the confidence score
- Uses a Gradio web interface

## Selected Model

The Papluca XLM-RoBERTa model was selected after comparing three pretrained
language-detection models using accuracy, latency, model size, and local
inference cost.

## Limitations

- Very short inputs may produce lower confidence.
- Mixed-language sentences may be difficult to classify.
- The model can only predict its 20 supported languages.