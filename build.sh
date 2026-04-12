#!/bin/bash
pip install -r requirements.txt
python -c "
from sentence_transformers import SentenceTransformer
print('Downloading HuggingFace model...')
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('Model cached successfully.')
"