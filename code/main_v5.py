import re
import json
import csv
import pdfplumber
import logging
import os
import nltk
import spacy
import argparse
import pytesseract
from pdf2image import convert_from_path
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLP models
nltk.download('punkt')
nltk.download('vader_lexicon')
nlp = spacy.load('en_core_web_sm')

# Logging setup
logging.basicConfig(
    filename="email_processor.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Request & Sub-Request Types
REQUEST_TYPES = [
    "Adjustment", "AU Transfer", "Closing Notice", "Commitment Change",
    "Fee Payment", "Money Movement-Inbound", "Money Movement-Outbound"
]

SUB_REQUEST_TYPES = {
    "Closing Notice": ["Reallocation fees", "Amendment Fees", "Reallocation Principal"],
    "Commitment Change": ["Cashless Roll", "Decrease", "Increase"],
    "Fee Payment": ["Ongoing Fee", "Letter of Credit Fee"],
    "Money Movement-Inbound": ["Principal", "Interest", "Principal + Interest", "Principal + Interest + Fee"],
    "Money Movement-Outbound": ["Timebound", "Foreign Currency"]
}

# Load Models
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sia = SentimentIntensityAnalyzer()

summary_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

EMAIL_HISTORY_FILE = "email_history.json"


def load_email_history():
    try:
        if os.path.exists(EMAIL_HISTORY_FILE):
            with open(EMAIL_HISTORY_FILE, 'r') as file:
                return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Email history error: {e}")
    return []


def save_email_history(email_history):
    try:
        with open(EMAIL_HISTORY_FILE, 'w') as file:
            json.dump(email_history, file, indent=4)
    except IOError as e:
        logging.error(f"Error saving email history: {e}")


def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            if not text:
                images = convert_from_path(pdf_path)
                text = "\n".join(pytesseract.image_to_string(img) for img in images)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""


def extract_date_from_text(email_text):
    try:
        match = re.search(r"(\d{1,2}-[A-Za-z]{3}-\d{4})", email_text)
        return match.group(1) if match else "N/A"
    except Exception as e:
        logging.error(f"Error extracting date: {e}")
    return "N/A"


def classify_request_type(email_text):
    try:
        request_descriptions = [
            "Adjustment: General financial modifications.",
            "AU Transfer: Transfer of assets.",
            "Closing Notice: Finalization of financial transactions.",
            "Commitment Change: Alteration in loan commitment.",
            "Fee Payment: Payment of service fees.",
            "Money Movement-Inbound: Incoming funds or loan funding.",
            "Money Movement-Outbound: Outgoing payments or disbursements."
        ]
        classification = classifier(email_text, request_descriptions)
        result1 = classification['labels'][0]
        result = result1.split(":")[0]
        return result, classification['scores'][0]
    except Exception as e:
        logging.error(f"Error classifying request type: {e}")
        return "Unknown", 0.0


def classify_sub_request_type(main_request, email_text):
    try:
        if main_request in SUB_REQUEST_TYPES:
            classification = classifier(email_text, SUB_REQUEST_TYPES[main_request])
            return classification['labels'][0], classification['scores'][0]
    except Exception as e:
        logging.error(f"Error classifying sub-request type: {e}")
    return None, 0.0


def determine_priority(email_text):
    try:
        sentiment_score = sia.polarity_scores(email_text)['compound']
        if sentiment_score < -0.2 or "urgent" in email_text.lower():
            return 'High'
        elif sentiment_score > 0.2:
            return 'Low'
        return 'Medium'
    except Exception as e:
        logging.error(f"Error determining priority: {e}")
        return 'Medium'


def extract_entities(email_text):
    doc = nlp(email_text)
    entities = {ent.label_: ent.text for ent in doc.ents}
    return entities


def extract_reporter(email_text):
    try:
        # Try to extract from "From:" field first
        match = re.search(r"From:\s*([\w\s]+)<([\w.-]+@[\w.-]+)>", email_text)
        if match:
            return match.group(1).strip()

        # Extract the first line after "Regards,"
        signature_match = re.search(r"Regards,[\s\n]+([\w\s]+)", email_text, re.IGNORECASE)
        if signature_match:
            return signature_match.group(1).strip().split("\n")[0]  # Capture only first line

    except Exception as e:
        logging.error(f"Error extracting reporter: {e}")
    return "Unknown"


def write_json(output_data, filename="output.json"):
    try:
        with open(filename, 'w') as json_file:
            json.dump(output_data, json_file, indent=4)
    except Exception as e:
        logging.error(f"Error writing to JSON: {e}")


def write_csv(output_data, filename="output.csv"):
    try:
        fieldnames = ['Short Description', 'Description', 'Priority', 'Requested Date',
                      'Main Request', 'Main Confidence', 'Sub Request', 'Sub Confidence',
                      'Duplicate', 'Reporter']

        with open(filename, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_data)
    except Exception as e:
        logging.error(f"Error writing to CSV: {e}")


def extract_short_description(email_text):
    try:
        # Prioritize extracting actual subject lines
        match = re.search(r"(?:Subject|Re|Regarding):\s*(.+)", email_text, re.IGNORECASE)
        if match:
            return match.group(1).strip().split("\n")[0]  # Get the first line

        # If no subject is found, use zero-shot classification as a fallback
        labels = ["Loan Payment Request", "Fund Transfer", "Interest Adjustment",
                  "Commitment Change", "Fee Payment", "Closing Notice", "General Inquiry"]
        classification = classifier(email_text[:1024], labels)
        return classification['labels'][0]  # Most confident label
    except Exception as e:
        logging.error(f"Error extracting short description: {e}")
    return "Processed Request"


def process_file(file_path):
    try:
        email_history = load_email_history()
        if file_path.lower().endswith('.pdf'):
            email_text = extract_text_from_pdf(file_path)
        else:
            logging.error("Unsupported file type.")
            return
        main_request, main_confidence = classify_request_type(email_text)
        sub_request, sub_confidence = classify_sub_request_type(main_request, email_text)
        requested_date = extract_date_from_text(email_text)
        short_description = extract_short_description(email_text)
        #entities = extract_entities(email_text)
        priority = determine_priority(email_text)
        reporter = extract_reporter(email_text)
        duplicate = email_text in email_history
        output_data = [{
            'Short Description': str(short_description),
            'Description': email_text + "...",
            'Priority': priority,
            'Requested Date': requested_date,
            'Main Request': main_request,
            'Main Confidence': round(main_confidence, 2),
            'Sub Request': sub_request or 'None',
            'Sub Confidence': round(sub_confidence, 2),
            #'Entities': str(entities),
            'Duplicate': 'Yes' if duplicate else 'No',
            'Reporter': reporter
        }]
        write_json(output_data)
        write_csv(output_data)

        print(f"Results saved to output.json and output.csv")

        if not duplicate:
            email_history.append(email_text)
            save_email_history(email_history)
        logging.info(f"Processed file: {file_path} → {main_request} / {sub_request} / Reporter: {reporter}")
    except Exception as e:
        logging.error(f"Error processing file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a PDF or EML file.")
    parser.add_argument("file_path", help="Path to the file")
    args = parser.parse_args()
    process_file(args.file_path)