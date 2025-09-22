import json
import os

from config import PATTERNS_OF_INTENT

def display_analysis_results(analysis):
    """
    Displays the analysis results in a structured, numbered format.
    """
    print("\n" + "="*25 + " ANALYSIS " + "="*25)
    print(f"1. Keyphrase Extractions: {analysis.get('key_phrases', 'N/A')}")
    sentiment_label = analysis.get('sentiment_label', 'N/A').capitalize()
    sentiment_score = analysis.get('sentiment_score', 0)
    print(f"2. Sentiment Analysis: {sentiment_label} (Score: {sentiment_score:.2f})")

    print(f"3. Intent: {analysis.get('intent_detected', 'N/A').capitalize()}")

    print(f"4. Emotion: {analysis.get('emotion_detected', 'N/A').capitalize()}")

    print(f"5. Topic Extraction: {analysis.get('topics_detected', 'N/A')}")

    relevance_score = analysis.get('relevance_score', 0)
    print(f"6. Relevance Score: {relevance_score:.2f}")

    engagement_score = analysis.get('engagement_score', 0)
    engagement_level = analysis.get('engagement_level', 'N/A').capitalize()
    print(f"7. Engagement Score & Level: {engagement_level} (Score: {engagement_score:.2f})")

    reply_source = analysis.get('reply_source', 'N/A').upper()
    print(f"8. Reply Suggestion Source: {reply_source}")

    print("="*60)


def ensure_csv_headers():
    """
    Ensures that all necessary CSV files exist and have their headers written.
    This prevents errors when trying to append data to non-existent or empty files.
    """
    from config import CSV_FILES, CSV_HEADERS
    import csv

    for key in CSV_FILES:
        file_path = CSV_FILES[key]
        header = CSV_HEADERS.get(key)
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        if header and (not os.path.exists(file_path) or os.path.getsize(file_path) == 0):
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                
def update_topic_csv(self, new_topics: list):
    existing_topics = set()
    try:
        with open("config/TOPIC_EXAMPLES_CSV.csv", "r", encoding="utf-8") as f:
            existing_topics = {line.strip().split(",")[0].lower() for line in f.readlines()}
    except FileNotFoundError:
        pass  

    with open("config/TOPIC_EXAMPLES_CSV.csv", "a", encoding="utf-8") as f:
        for topic in new_topics:
            if topic.lower() not in existing_topics:
                f.write(f"{topic},auto_generated\n")


def update_intent_config(self, intent: str, example_text: str):
    """
    Updates the PATTERNS_OF_INTENT dictionary and config file if new intent is found.
    """
    intent = intent.strip().lower()
    if intent not in PATTERNS_OF_INTENT:
        print(f"[INFO] New intent '{intent}' discovered. Adding to config.")
        PATTERNS_OF_INTENT[intent] = [example_text.strip()]
        self.save_intent_config()

def save_intent_config(self):
    """
    Saves the updated PATTERNS_OF_INTENT to config.py or a separate JSON file.
    Recommended: write to a .json file and load into config.py at runtime.
    """
    with open("config/patterns_of_intent.json", "w", encoding="utf-8") as f:
        json.dump(PATTERNS_OF_INTENT, f, indent=2, ensure_ascii=False)