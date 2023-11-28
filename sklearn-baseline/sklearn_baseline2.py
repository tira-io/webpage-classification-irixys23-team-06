#!/usr/bin/env python3
import argparse
import os
import joblib
import pandas as pd
from bs4 import BeautifulSoup
import re
import json

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def preprocess(content):
    # Assuming you have a preprocessing step for the URL; if not, just return content
    return content

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text(separator=' ')
    text = re.sub(r'[^A-Za-z ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_args():
    parser = argparse.ArgumentParser(description='Webpage classification with sklearn pipeline')
    parser.add_argument("-i", "--input_data", help="Path to the jsonl file for which predictions should be made.", required=True)
    parser.add_argument("-m", "--model", help="The sklearn SGDClassifier model to use for the predictions.", required=True)
    parser.add_argument("-o", "--output", help="Path to the directory to write the results to.", required=True)
    return parser.parse_args()

def load_model(model_file):
    pipeline = joblib.load(model_file)
    return pipeline

def main(input_file, output_dir, model_file):
    # Load datasets
    test_data = load_data(input_file)

    # Extract text from HTML and combine features
    test_data['text'] = test_data['html'].apply(extract_text_from_html)
    url_repeat_count = 25  # Adjust as per your chosen repeat count
    test_data['combined_features'] = (test_data['url'] + ' ') * url_repeat_count + test_data['text']

    # Load the model
    pipeline = load_model(model_file)
    
    # Make predictions on the combined features
    test_predictions = pipeline.predict(test_data['combined_features'])

    # Save the predictions
    test_data['prediction'] = test_predictions
    output_path = os.path.join(output_dir, 'predictions.jsonl')
    test_data[['uid', 'prediction']].to_json(output_path, orient='records', lines=True)

if __name__ == "__main__":
    args = parse_args()
    main(args.input_data, args.output, args.model)
