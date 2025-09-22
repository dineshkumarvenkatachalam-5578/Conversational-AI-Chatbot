import csv
import json
import os
import random
import re
import string
from datetime import datetime
from typing import List, Dict, Any, Set, Optional
import nltk
import numpy as np
import spacy
from dotenv import load_dotenv
from groq import Groq
from keybert import KeyBERT
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, Pipeline
from dialogue_tracker import DialogueStateTracker
from web_search import WebSearchService
from mongodb_rag_system import MongoDBRAGSystem

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("[CRITICAL ERROR] GROQ_API_KEY not found in .env file. The application cannot start.")
    exit()
client = Groq(api_key=GROQ_API_KEY)

print("Initializing system configuration and logging paths...")
os.makedirs('logs', exist_ok=True)
CSV_FILES: Dict[str, str] = {
    'feedback': 'logs/feedback_log.csv',
    'exit_feedback': 'logs/exit_feedback_log.csv',
    'rl_data': 'logs/rl_training_data.csv',
    'conversation': 'logs/conversation_log.csv'
}
CSV_HEADERS: Dict[str, List[str]] = {
    'feedback': ['timestamp', 'user_id', 'user_input', 'suggested_replies', 'chosen_reply', 'feedback_score', 'feedback_type'],
    'exit_feedback': ['timestamp', 'user_id', 'star_rating', 'helpful_feedback', 'suggestions_feedback', 'improvements_feedback'],
    'rl_data': ['timestamp', 'user_id', 'observation_json', 'action_taken', 'reward'],
    'conversation': ['timestamp', 'user_id', 'user_name', 'user_input', 'key_phrases', 'sentiment_label', 'sentiment_score', 'intent_detected', 'emotion_detected', 'topics_detected', 'relevance_score', 'engagement_score', 'engagement_level', 'selected_reply']
}
print("Loading NLP resources (spaCy & NLTK)...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ SpaCy 'en_core_web_sm' model loaded.")
except OSError:
    print("[Warning] SpaCy model 'en_core_web_sm' not found. Downloading...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("✓ SpaCy model downloaded and loaded.")

for lib in ['stopwords', 'vader_lexicon', 'punkt']:
    try:
        nltk.data.find(f'corpora/{lib}')
    except LookupError:
        print(f"[Warning] NLTK data '{lib}' not found. Downloading...")
        nltk.download(lib)
        print(f"✓ NLTK '{lib}' downloaded.")
print("✓ All NLP resources are ready.")

class RelevanceScorer:
    def __init__(self, sentence_model: SentenceTransformer, history_length: int = 3):
        """
        Initializes the RelevanceScorer.

        Args:
            sentence_model: A pre-loaded SentenceTransformer model.
            history_length: The number of past user turns to consider for relevance.
        """
        self.model = sentence_model
        self.history_length = history_length
        print("✓ RelevanceScorer initialized.")

    def _cosine_similarity_for_sets(self, set1: Set[str], set2: Set[str]) -> float:
        """
        Calculates the cosine similarity between the centroid embeddings of two sets of phrases.

        Args:
            set1: The first set of strings (keywords or topics).
            set2: The second set of strings.

        Returns:
            A similarity score between 0.0 and 1.0.
        """
        if not set1 or not set2:
            return 0.0
        text1 = " ".join(list(set1))
        text2 = " ".join(list(set2))
        try:
            embeddings = self.model.encode([text1, text2])
            return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        except Exception as e:
            print(f"[Error] Could not calculate set similarity: {e}")
            return 0.0

    def calculate_score(self, current_analysis: Dict[str, Any], conversation_history: List[Dict]) -> float:
        """
        Computes the final relevance score using a weighted model of engineered features.

        Args:
            current_analysis: The analysis dictionary for the current user input.
            conversation_history: The list of all previous analysis dictionaries.

        Returns:
            The final, weighted relevance score.
        """
        if not conversation_history:
            return 1.0  

        user_history = [turn for turn in conversation_history if 'user_input' in turn][-self.history_length:]
        if not user_history:
            return 1.0
        
        current_embedding = self.model.encode([current_analysis['user_input']])[0]
        history_embeddings = self.model.encode([turn['user_input'] for turn in user_history])
        recency_weights = np.linspace(1.0, 0.4, len(history_embeddings))
        semantic_scores = cosine_similarity([current_embedding], history_embeddings)[0]
        weighted_semantic_score = np.average(semantic_scores, weights=recency_weights)

        # Feature 2: Keyword Overlap
        current_keywords = set(current_analysis.get('key_phrases', []))
        history_keywords = set().union(*[set(turn.get('key_phrases', [])) for turn in user_history])
        keyword_overlap_score = self._cosine_similarity_for_sets(current_keywords, history_keywords)

        # Feature 3: Topic Consistency 
        current_topics = set(current_analysis.get('topics_detected', []))
        history_topics = set().union(*[set(turn.get('topics_detected', [])) for turn in user_history])
        topic_consistency_score = self._cosine_similarity_for_sets(current_topics, history_topics)
        
        # Final Weighted Model 
        final_score = (
            weighted_semantic_score * 0.60 + 
            keyword_overlap_score * 0.20 +
            topic_consistency_score * 0.20
        )

        if current_analysis.get('intent_detected') in ['agreement', 'disagreement', 'follow-up', 'appreciation']:
            final_score = min(1.0, final_score + 0.2)

        return float(final_score)

class AdvancedKeyphraseExtractor:
    """
    State-of-the-art keyphrase extractor using KeyBERT + RoBERTa + Advanced NLP.
    Combines the power of RoBERTa embeddings with comprehensive linguistic analysis to
    extract semantically meaningful and nuanced keyphrases.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-roberta-large-v1", nlp_spacy: spacy.Language = None):
        print("Initializing Advanced RoBERTa Keyphrase Extractor...")
        try:
            self.sentence_model = SentenceTransformer(model_name)
            self.kw_model = KeyBERT(model=self.sentence_model)
            print(f"✓ RoBERTa-based KeyBERT initialized successfully with '{model_name}'")
        except Exception as e:
            print(f"[Warning] RoBERTa KeyBERT initialization failed, trying fallback: {e}")
            fallback_model = "sentence-transformers/all-distilroberta-v1"
            self.sentence_model = SentenceTransformer(fallback_model)
            self.kw_model = KeyBERT(model=self.sentence_model)
            print(f"✓ Fallback model '{fallback_model}' loaded.")

        self.nlp = nlp_spacy if nlp_spacy else spacy.load("en_core_web_sm")
        self.BASIC_STOPWORDS = set(stopwords.words('english'))
        self.GENERIC_WORDS = {
            'thing', 'things', 'something', 'anything', 'everything', 'nothing', 'way', 'time',
            'day', 'year', 'place', 'area', 'part', 'number', 'people', 'person', 'man', 'woman',
            'group', 'system', 'program', 'problem', 'example', 'case', 'point', 'fact',
            'information', 'data', 'result', 'study', 'research', 'analysis', 'report', 'method',
            'process', 'concept', 'topic', 'issue', 'aspect', 'detail', 'element', 'feature',
            'component', 'kind', 'type', 'episode', 'series', 'plot', 'twist', 'character',
            'writers', 'edge', 'seats', 'mind', 'level', 'moment', 'talk', 'need', 'seriously',
            'finally', 'finished', 'blown', 'processing', 'tragic', 'brutal', 'beautiful', 'once'
        }
        print("✓ Advanced RoBERTa Keyphrase Extractor ready.")
    
    def preprocess_text(self, text: str) -> str:
        """
        Cleans and standardizes text for processing.
        """
        if not text or len(text.strip()) < 3: return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        return text.strip()
    
    def is_meaningful_token(self, token: spacy.tokens.Token) -> bool:
        text = token.text.lower().strip()
        if (len(text) < 2 or text in self.BASIC_STOPWORDS or text in self.GENERIC_WORDS or token.is_stop or token.is_punct or token.is_space):
            return False
        if token.pos_ in ['NOUN', 'PROPN', 'ADJ']: return True
        return False
    
    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[Dict[str, Any]]:
        if not text or not text.strip(): return []
        doc = self.nlp(text)
        word_count = len([token for token in doc if not token.is_punct])
        
        # Prioritize meaningful noun chunks
        meaningful_chunks = []
        for chunk in doc.noun_chunks:
            # A chunk is meaningful if it contains at least one non-stopword/non-generic noun or adjective
            if any(self.is_meaningful_token(tok) and tok.pos_ in ['NOUN', 'PROPN', 'ADJ'] for tok in chunk):
                # Clean the chunk text
                clean_chunk = ' '.join(tok.text for tok in chunk if not tok.is_stop and not tok.is_punct)
                if clean_chunk:
                    meaningful_chunks.append(clean_chunk.strip())

        # Use KeyBERT for semantic relevance but guide it with our chunks
        try:
            # Use the original text for context, but extract our pre-identified meaningful chunks
            keybert_results = self.kw_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 4), 
                stop_words='english', 
                use_mmr=True, 
                diversity=0.7, 
                top_n=top_n * 2
            )
        except Exception as e:
            print(f"[Error] KeyBERT extraction failed: {e}")
            keybert_results = []

        # Combine and refine results
        final_phrases = set(meaningful_chunks)
        for phrase, score in keybert_results:
            # Add KeyBERT results if they are sufficiently different and meaningful
            lower_phrase = phrase.lower()
            if not any(lower_phrase in seen.lower() or seen.lower() in lower_phrase for seen in final_phrases):
                # A quick check to see if the phrase from KeyBERT is just generic words
                doc_phrase = self.nlp(phrase)
                if any(self.is_meaningful_token(tok) for tok in doc_phrase):
                    final_phrases.add(phrase)

        # Score the final phrases based on a combination of length and position
        scored_results = []
        for i, phrase in enumerate(list(final_phrases)):
            # Simple scoring: longer phrases that appear earlier are better
            score = (len(phrase.split()) * 0.5) + (1 - (i / len(final_phrases)) * 0.5)
            scored_results.append({'phrase': phrase.title(), 'score': score})

        # Sort and return top N
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return scored_results[:top_n]

class SmartReplySystem:
    """
    The main class that orchestrates the entire conversational AI system.
    """
    def __init__(self, user_id: str, user_name: str, history_filename: str, prompt_style: str = 'enhanced'):
        """
        Initializes the entire system, loading all models and components.
        """
        self.user_id = user_id
        self.user_name = user_name
        self.history_filename = history_filename
        self.prompt_style = prompt_style 
        self.conversation_history: List[Dict[str, Any]] = self.load_conversation_history()
        self.groq_cloud_client = Groq(api_key=GROQ_API_KEY)
        self.models: Dict[str, Any] = {}
        self.dialogue_tracker = DialogueStateTracker(user_id)
        self.topic_labels = self._load_topic_labels_from_csv('topic_examples.csv')

        print("\n--- Loading AI Models ---")
        self.models['embedding'] = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Sentence Embedding model loaded.")

        self.relevance_scorer = RelevanceScorer(sentence_model=self.models['embedding'])
        self.keyphrase_extractor = AdvancedKeyphraseExtractor(nlp_spacy=nlp)
        
        # Initialize Web Search and RAG System
        self.web_search = WebSearchService()
        self.rag_system = MongoDBRAGSystem(user_id=self.user_id)
        print("✓ Web Search and RAG System initialized.")
        
        def _load_pipeline(task: str, model_name: str, **kwargs) -> Optional[Pipeline]:
            try:
                print(f"Loading model for {task}: {model_name}...")
                model = pipeline(task, model=model_name, **kwargs)
                print(f"✓ {task.capitalize()} model loaded.")
                return model
            except Exception as e:
                print(f"[CRITICAL] Could not load {task} model '{model_name}': {e}. System will be degraded.")
                return None
        
        self.models['sentiment'] = _load_pipeline("sentiment-analysis", "cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.models['emotion'] = _load_pipeline("text-classification", "SamLowe/roberta-base-go_emotions", return_all_scores=True)
        
        # Use a more powerful DeBERTa model for primary topic detection
        print("Loading primary topic detection model...")
        zero_shot_model = _load_pipeline("zero-shot-classification", "cross-encoder/nli-deberta-v3-base")
        if zero_shot_model is None:
            print("[CRITICAL] Primary topic model failed to load. Falling back to older model.")
            zero_shot_model = _load_pipeline("zero-shot-classification", "facebook/bart-large-mnli")

        self.models['intent_fallback'] = zero_shot_model
        self.models['topic_primary'] = zero_shot_model # Assign for clarity
        
        # --- Dynamic Learning Setup ---
        self.new_intent_file = 'new_intent_examples.csv'
        self.new_topic_file = 'new_topic_examples.csv'
        self.new_emotion_file = 'new_emotion_examples.csv'
        self._initialize_example_files()        # --- Load Fine-Tuned Models ---
        try:
            intent_model_path = './models/intent_classifier'
            if os.path.exists(intent_model_path):
                self.models['intent'] = pipeline("text-classification", model=intent_model_path)
                print("✓ Fine-tuned Intent model loaded.")
            else:
                print("[Warning] Fine-tuned intent model not found. Falling back.")
                self.models['intent'] = None
        except Exception as e:
            print(f"[Error] Failed to load fine-tuned intent model: {e}")
            self.models['intent'] = None

        try:
            topic_model_path = './models/topic_classifier'
            if os.path.exists(topic_model_path):
                self.models['topic'] = pipeline("text-classification", model=topic_model_path)
                print("✓ Fine-tuned Topic model loaded.")
            else:
                print("[Warning] Fine-tuned topic model not found. Falling back.")
                self.models['topic'] = None
        except Exception as e:
            print(f"[Error] Failed to load fine-tuned topic model: {e}")
            self.models['topic'] = None
            
        print("--- All AI Models Loaded Successfully ---\n")

    def _initialize_example_files(self):
        """Creates the new example CSV files if they don't exist."""
        for file, header in [
            (self.new_intent_file, ['text', 'intent']),
            (self.new_topic_file, ['text', 'topic']),
            (self.new_emotion_file, ['text', 'emotion'])
        ]:
            if not os.path.exists(file):
                with open(file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                print(f"✓ Created new examples file: {file}")

    def _update_examples_csv(self, filename: str, text: str, label: str):
        """Appends a new discovered example to the appropriate CSV file."""
        try:
            with open(filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([text, label])
            print(f"[Learning] New example added to {filename}: '{label}'")
        except Exception as e:
            print(f"[Error] Could not write to {filename}: {e}")

    def _load_topic_labels_from_csv(self, file_path: str) -> List[str]:
        """
        Loads unique topic labels from the 'topic' column of a CSV file.
        """
        if not os.path.exists(file_path):
            print(f"[Warning] Topic examples file not found at '{file_path}'. Using a default list.")
            return ['entertainment', 'sports', 'technology', 'news', 'health', 'statement', 'question']
        
        try:
            topics = set()
            # Try different encodings to handle various file formats
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        # Handle potential headers and different CSV formats gracefully
                        for line in f:
                            parts = line.strip().split(',')
                            if len(parts) > 1 and parts[1].lower() != 'topic': # Ignore header
                                topic = parts[1].strip()
                                if topic:
                                    topics.add(topic)
                    break  # If successful, break out of encoding loop
                except UnicodeDecodeError:
                    continue  # Try next encoding
            
            # Add some generic but important labels that might be missed
            additional_labels = {'statement', 'question', 'opinion', 'relationship', 'family'}
            topics.update(additional_labels)

            print(f"✓ Dynamically loaded {len(topics)} unique topic labels from '{file_path}'.")
            return list(topics)
        except Exception as e:
            print(f"[Error] Failed to load topics from '{file_path}': {e}. Using a default list.")
            return ['entertainment', 'sports', 'technology', 'news', 'health', 'statement', 'question']

    def load_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Loads the conversation history from a JSON file if it exists.
        """
        if os.path.exists(self.history_filename):
            with open(self.history_filename, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def save_conversation_history(self) -> None:
        """
        Saves the current conversation history to its JSON file.
        """
        with open(self.history_filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=4)
    
    def cleanup(self) -> None:
        """
        Cleanup method to properly close database connections.
        """
        try:
            if hasattr(self, 'rag_system') and self.rag_system.client:
                self.rag_system.close_connections()
            print("✓ System cleanup completed")
        except Exception as e:
            print(f"[WARNING] Cleanup failed: {e}")

    def calculate_reward(self, feedback_rating: int) -> float:
        """
        Converts a 1-5 star user rating into a numerical reward for RL.
        """
        reward_mapping = {1: 1.0, 2: 0.5, 3: 0.0, 4: -0.5, 5: -1.0}
        return reward_mapping.get(feedback_rating, 0.0)

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyzes sentiment using the primary model first, falling back to VADER.
        It also calculates a VADER compound score for macro analysis regardless.
        """
        primary_label, primary_score = None, None
        
        # 1. Prioritize the primary transformer-based model
        if self.models.get('sentiment'):
            try:
                result = self.models['sentiment'](text)[0]
                primary_label = result['label']
                primary_score = result['score']
            except Exception as e:
                print(f"[Warning] Primary sentiment model failed: {e}. Falling back to VADER.")
        
        # 2. Always calculate VADER scores for the numerical compound score (for macro-average)
        #    and for fallback labels if the primary model failed.
        sia = SentimentIntensityAnalyzer()
        vader_scores = sia.polarity_scores(text)
        compound_score = vader_scores['compound']

        # 3. If the primary model failed, determine label from VADER
        if primary_label is None:
            if compound_score >= 0.05:
                primary_label, primary_score = 'POSITIVE', vader_scores['pos']
            elif compound_score <= -0.05:
                primary_label, primary_score = 'NEGATIVE', vader_scores['neg']
            else:
                primary_label, primary_score = 'NEUTRAL', vader_scores['neu']

        return {
            'label': primary_label,
            'score': primary_score,
            'compound': compound_score
        }

    def calculate_macro_sentiment(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates the average sentiment over the course of the conversation history.
        """
        if not conversation_history:
            return {'label': 'NEUTRAL', 'score': 0.0}

        # Use the stored compound scores for a consistent numerical average
        compound_scores = [turn.get('sentiment_compound', 0.0) for turn in conversation_history if 'sentiment_compound' in turn]
        
        if not compound_scores:
            return {'label': 'NEUTRAL', 'score': 0.0}

        macro_score = np.mean(compound_scores)

        if macro_score >= 0.05:
            label = 'POSITIVE'
        elif macro_score <= -0.05:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
            
        return {'label': label, 'score': macro_score}

    def detect_emotion(self, text: str) -> str:
        """
        Detects the primary emotion in a given text and learns new ones.
        """
        if self.models.get('emotion'):
            try:
                results = self.models['emotion'](text)[0]
                top_emotion = max(results, key=lambda x: x['score'])
                if top_emotion['score'] > 0.85: # High confidence threshold
                    self._update_examples_csv(self.new_emotion_file, text, top_emotion['label'])
                return top_emotion['label']
            except Exception: pass
        return "neutral"

    def detect_intent(self, text: str) -> str:
        """
        Detects the user's intent and learns new ones.
        """
        # 1. Try fine-tuned model
        if self.models.get('intent'):
            try:
                result = self.models['intent'](text)[0]
                if result['score'] > 0.90: # High confidence
                    return result['label']
            except Exception as e:
                print(f"[Warning] Fine-tuned intent model failed: {e}. Falling back.")
        
        # 2. Fallback to zero-shot
        if self.models.get('intent_fallback'):
            try:
                candidate_labels = ['question', 'opinion', 'request', 'complaint', 'appreciation', 'agreement', 'statement', 'matrimonial inquiry']
                result = self.models['intent_fallback'](text, candidate_labels=candidate_labels)
                top_intent = result['labels'][0]
                top_score = result['scores'][0]
                if top_score > 0.90: # High confidence threshold for learning
                    self._update_examples_csv(self.new_intent_file, text, top_intent)
                return top_intent
            except Exception:
                pass
        return "statement"

    def _detect_topics_with_llm(self, text: str, top_n: int = 2) -> List[str]:
        """
        [Secondary Approach] Uses a powerful LLM to extract topics when the primary model fails.
        """
        print("[Info] Primary model confidence low. Falling back to LLM-based topic detection.")
        try:
            prompt = f"""You are a topic analysis expert. Analyze the following user message and extract the {top_n} most relevant, concise topics. The topics should be 1-3 words long.

            User Message: "{text}"

            Respond in valid JSON with a key "topics" containing a list of the extracted topic strings.
            Example: {{"topics": ["Stock Market", "Investment Advice"]}}
            """
            response = self._get_llm_json_response(prompt)
            if response and response.get("topics"):
                return response["topics"]
        except Exception as e:
            print(f"[Error] LLM-based topic detection failed: {e}")
        return [] # Return empty list on failure

    def detect_topics(self, text: str, top_n: int = 2) -> list:
        """
        Detects topics using a two-step process: a primary transformer model 
        followed by a secondary LLM-based approach as a fallback.
        """
        # 1. Primary Approach: Use the powerful DeBERTa zero-shot model
        if self.models.get('topic_primary'):
            try:
                # A comprehensive list for general-purpose topic detection
                candidate_labels = [
                    'technology', 'entertainment', 'movies', 'music', 'gaming', 'sports', 
                    'finance', 'business', 'politics', 'world news', 'health', 'fitness', 
                    'food', 'travel', 'education', 'career', 'personal development', 
                    'relationships', 'family', 'fashion', 'art', 'science', 'history',
                    'matrimonial inquiry', 'personal values', 'lifestyle choices'
                ]
                result = self.models['topic_primary'](text, candidate_labels=candidate_labels, multi_label=True)
                
                top_topics = [label for label, score in zip(result['labels'], result['scores']) if score > 0.75]
                
                if top_topics:
                    print("[Info] Topics detected using primary transformer model.")
                    return top_topics[:top_n]
            except Exception as e:
                print(f"[Warning] Primary topic model (transformer) failed: {e}.")

        # 2. Secondary Approach: Fallback to LLM-based detection
        return self._detect_topics_with_llm(text, top_n)

    def calculate_enhanced_engagement(self, analysis_result: Dict[str, Any]) -> float:
        """
        Calculates a sophisticated engagement score using a feature-engineered model.
        """
        doc = nlp(analysis_result.get('user_input', ''))
        
        word_count = len([token for token in doc if not token.is_punct])
        length_score = 0.0
        if word_count > 25: length_score = 1.0
        elif word_count > 10: length_score = 0.8
        elif word_count > 4: length_score = 0.5
        else: length_score = 0.1

        unique_words_ratio = len(set(t.text.lower() for t in doc if not t.is_punct)) / word_count if word_count > 0 else 0
        num_sentences = len(list(doc.sents))
        complexity_score = (unique_words_ratio + (num_sentences / 5.0)) / 2.0
        
        is_question = 1.0 if analysis_result.get('intent_detected') == 'question' else 0.0
        richness_score = min((len(analysis_result.get('key_phrases', [])) + len(analysis_result.get('topics_detected', []))) / 8.0, 1.0)
        
        sentiment_score = analysis_result.get('sentiment_score', 0.5)
        emotional_intensity = abs(sentiment_score - 0.5) * 2
        if analysis_result.get('sentiment_label') == 'NEGATIVE':
            emotional_intensity *= 1.2

        relevance_score = analysis_result.get('relevance_score', 0.5)

        engagement_score = (
            length_score * 0.30 +
            complexity_score * 0.20 +
            richness_score * 0.15 +
            emotional_intensity * 0.15 +
            relevance_score * 0.10 +
            is_question * 0.10
        )
        
        return min(max(engagement_score, 0.0), 1.0)

    def determine_engagement_level(self, engagement_score: float) -> str:
        """
        Classifies the engagement score into a discrete level.
        """
        if engagement_score >= 0.80: return 'high'
        elif engagement_score >= 0.55: return 'medium'
        else: return 'low'

    def generate_llm_replies(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Generates contextual reply suggestions based on the selected prompt style.
        """
        if self.prompt_style == 'enhanced':
            return self._generate_replies_enhanced(analysis_result)
        else:
            return self._generate_replies_original(analysis_result)

    def _generate_replies_enhanced(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Generates friendly, detailed, and contextually appropriate reply suggestions (new style).
        """
        engagement_level = analysis_result['engagement_level']
        intent = analysis_result.get('intent_detected')

        web_search_context = ""
        if 'web_search_results' in analysis_result:
            web_results = analysis_result['web_search_results']
            web_search_context = f"""
            
        REAL-TIME WEB SEARCH RESULTS:
        Query: {web_results.get('query', '')}
        Search Type: {web_results.get('search_type', '')}
        Results: {json.dumps(web_results.get('results', []), indent=2)}
        
        Use this real-time information to provide accurate, current, and helpful responses. If the user asked about restaurants, food courts, or locations, incorporate specific names and details from the search results.
        """
        rag_context = ""
        if 'similar_conversations' in analysis_result or 'relevant_knowledge' in analysis_result:
            rag_context = f"""
            
        RELEVANT PAST CONTEXT:
        Similar Conversations: {json.dumps(analysis_result.get('similar_conversations', []), indent=2)}
        Relevant Knowledge: {json.dumps(analysis_result.get('relevant_knowledge', []), indent=2)}
        
        Use this context to provide personalized responses based on the user's conversation history and preferences.
        """
        personalized_context = self.get_personalized_context()

        # Create a JSON-serializable version of analysis_result
        serializable_analysis = self._make_json_serializable(analysis_result)

        base_prompt = f"""You are 'Echo', a world-class conversational AI. Your persona is friendly, empathetic, and insightful. Your primary goal is to make the user feel heard and understood. You are a clever conversationalist with access to real-time information and user context.

        Current Conversation Analysis: {json.dumps(serializable_analysis, indent=2)}
        {web_search_context}
        {rag_context}
        {personalized_context}
        """
        
        # Check if we have web search results to provide specific information
        has_web_results = 'web_search_results' in analysis_result and analysis_result['web_search_results'].get('results')
        user_input = analysis_result.get('user_input', '').lower()
        
        # Handle different types of queries appropriately
        if intent == 'matrimonial inquiry':
            # Matrimonial queries should NOT use web search results
            prompt = base_prompt + f"""
            The user is making a matrimonial inquiry. Your tone should be warm, respectful, and deeply engaging. Your goal is to build trust and encourage the user to share more about their needs and aspirations.
            Generate TWO replies:
            1. A reply that acknowledges one or two specific qualities the user mentioned, and then asks a thoughtful, open-ended question about the *feeling* or *value* behind one of those qualities. (e.g., "It sounds like 'honesty' is really important to you. What does a truly honest partnership feel like in your day-to-day life?").
            2. A reply that gently and creatively asks about their own personality in the context of a relationship. (e.g., "It's wonderful that you're looking for someone with a great sense of humor. How would you describe your own style of humor when you're with someone you care about?").
            
            Respond in JSON with a key "suggestions" containing a list of TWO strings. Add a relevant, gentle emoji to each.
            """
        elif has_web_results and ('food' in user_input or 'restaurant' in user_input or 'court' in user_input):
                prompt = base_prompt + f"""
                The user is asking about food/restaurants and you have real-time web search results. You MUST provide specific restaurant names and details from the search results.
                
                CRITICAL INSTRUCTIONS:
                - Extract actual restaurant names from the web search results
                - Include specific details like ratings, cuisine types, or specialties mentioned
                - Do NOT ask questions - provide direct recommendations
                - Use the format: "Based on current information, here are the best options in [location]:"
                
                Generate ONE comprehensive reply that:
                1. Lists 3-4 specific restaurants/food places with their names from the search results
                2. Includes brief descriptions of what makes each place special
                3. Ends with "Would you like more details about any of these places?"
                
                Example format: "Based on current information, here are the best food courts in Ooty: 1. [Name] - [description], 2. [Name] - [description]..."
                
                Respond in JSON with a key "suggestions" containing a list of ONE string. Add food emojis.
                """
        elif has_web_results and ('tourist' in user_input or 'place' in user_input or 'spot' in user_input or 'visit' in user_input):
                prompt = base_prompt + f"""
                The user is asking about tourist spots/places and you have real-time web search results. You MUST provide specific place names and details from the search results.
                
                CRITICAL INSTRUCTIONS:
                - Extract actual tourist attraction names from the web search results
                - Include specific details like ratings, types of attractions, or special features mentioned
                - Do NOT ask questions - provide direct recommendations
                - Use the format: "Based on current information, here are the best tourist spots in [location]:"
                
                Generate ONE comprehensive reply that:
                1. Lists 3-4 specific tourist attractions with their names from the search results
                2. Includes brief descriptions of what makes each place special
                3. Ends with "Would you like more information about visiting any of these places?"
                
                Example format: "Based on current information, here are the best tourist spots in Ooty: 1. [Name] - [description], 2. [Name] - [description]..."
                
                Respond in JSON with a key "suggestions" containing a list of ONE string. Add travel emojis.
                """
        elif has_web_results:
            # General web search results for other queries
            prompt = base_prompt + f"""
            You have real-time web search results for the user's query. Provide specific, helpful information based on the search results.
            
            Generate TWO informative replies:
            1. A comprehensive answer using specific details from the search results.
            2. A follow-up question or offer for additional help.
            
            IMPORTANT: Use actual information from the web search results. Be specific and helpful.
            Respond in JSON with a key "suggestions" containing a list of TWO strings. Add relevant emojis.
            """
        elif intent == 'matrimonial inquiry':
            prompt = base_prompt + f"""
            The user is making a matrimonial inquiry. Your tone should be warm, respectful, and deeply engaging. Your goal is to build trust and encourage the user to share more about their needs and aspirations.
            Generate TWO replies:
            1. A reply that acknowledges one or two specific qualities the user mentioned, and then asks a thoughtful, open-ended question about the *feeling* or *value* behind one of those qualities. (e.g., "It sounds like 'honesty' is really important to you. What does a truly honest partnership feel like in your day-to-day life?").
            2. A reply that gently and creatively asks about their own personality in the context of a relationship. (e.g., "It's wonderful that you're looking for someone with a great sense of humor. How would you describe your own style of humor when you're with someone you care about?").
            
            Respond in JSON with a key "suggestions" containing a list of TWO strings. Add a relevant, gentle emoji to each.
            """
        elif engagement_level == 'high':
            prompt = base_prompt + f"""
            The user is highly engaged. Your tone is curious and deep. Generate ONE excellent, thoughtful reply that dives deeper into their last message. Ask an open-ended question to encourage them to share more.
            Respond in JSON with a key "suggestions" containing a list with ONE string. Add a relevant emoji.
            """
        elif engagement_level == 'medium':
            prompt = base_prompt + f"""
            The user is moderately engaged. Your tone is encouraging. Generate TWO varied, insightful replies:
            1. A reply that explores the LATEST message in more detail.
            2. A reply that thoughtfully connects the LATEST message to a previous one, if it makes sense. If not, offer a different perspective on the latest topic.
            Respond in JSON with a key "suggestions" containing a list of TWO strings. Add an emoji to each.
            """
        else: 
            prompt = base_prompt + f"""
            The user is disengaging. Your tone is gentle and inviting. Generate THREE creative replies to spark their interest:
            1. A simple, curious question about their LATEST message.
            2. A thoughtful question that connects two different topics from their RECENT HISTORY.
            3. A completely new, interesting question to gracefully pivot the conversation (e.g., about hobbies, recent movies, or a fun hypothetical).
            Respond in JSON with a key "suggestions" containing a list of THREE strings. Add an emoji to each.
            """
            
        try:
            response = self._get_llm_json_response(prompt, model="openai/gpt-oss-120b")
            if response and response.get("suggestions"):
                return response["suggestions"]
        except Exception as e:
            print(f"[Error] LLM Reply Generation Error: {e}")
        # Provide better fallback responses based on user input and intent
        user_input = analysis_result.get('user_input', '').lower()
        intent = analysis_result.get('intent_detected', '')
        
        if intent == 'matrimonial inquiry':
            return [
                "It sounds like you have a clear vision of what you're looking for in a life partner. What qualities matter most to you in building a strong relationship? 💕",
                "Family values seem very important to you. How do you envision balancing tradition with modern life in your future relationship? 🏡"
            ]
        elif 'tourist' in user_input or 'place' in user_input or 'spot' in user_input:
            return [
                "I'd love to help you find great tourist spots! Could you tell me which city or area you're interested in visiting? 🏞️",
                "What type of attractions do you enjoy most - historical sites, natural beauty, or cultural experiences? 🎯"
            ]
        elif 'food' in user_input or 'restaurant' in user_input:
            return [
                "I'd be happy to help you find great food places! What type of cuisine are you in the mood for? 🍽️",
                "Are you looking for fine dining, casual spots, or local street food experiences? 😋"
            ]
        else:
            return ["That's really interesting. Could you tell me a bit more about that?", "I'm curious to hear more about your thoughts on this.", "Thanks for sharing! What's on your mind now?"]

    def _generate_replies_original(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Generates contextual reply suggestions based on engagement level (original style).
        """
        intent = analysis_result.get('intent_detected')
        base_prompt = f"""You are a conversational AI. Generate insightful reply suggestions based on the analysis.
        Current Conversation Analysis: {json.dumps(analysis_result, indent=2)}
        """

        if intent == 'matrimonial inquiry':
            prompt = base_prompt + """
            The user has a matrimonial inquiry. Generate two formal and respectful replies.
            1. Acknowledge the user's detailed request and ask a clarifying question about their priorities.
            2. Summarize the key traits they are looking for and ask how they envision their future together.
            Respond in JSON with a key "suggestions" containing a list of TWO strings.
            """
        else:
            prompt = base_prompt + """
            The user is in a general conversation. Generate three casual replies to continue the chat.
            1. Ask a simple open-ended question about their last message.
            2. Offer a related thought.
            3. Pivot to a new, easy topic.
            Respond in JSON with a key "suggestions" containing a list of THREE strings.
            """
            
        try:
            response = self._get_llm_json_response(prompt, model="openai/gpt-oss-120b")
            if response and response.get("suggestions"):
                return response["suggestions"]
        except Exception as e:
            print(f"[Error] LLM Reply Generation Error: {e}")
        
        return ["That's interesting, tell me more.", "What else is on your mind?", "Thanks for sharing."]

    def perform_intelligent_web_search(self, analysis_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Performs intelligent web search based on user input analysis.
        Dynamically detects locations, food types, and search needs from user input.
        """
        user_input = analysis_result.get('user_input', '')
        intent = analysis_result.get('intent_detected', '')
        topics = analysis_result.get('topics_detected', [])
        key_phrases = analysis_result.get('key_phrases', [])
        # Don't search for matrimonial, personal, or relationship queries
        if intent == 'matrimonial inquiry' or any(topic in ['relationships', 'family', 'personal values', 'lifestyle choices'] for topic in topics):
            return None
        
        search_triggers = [
            'recommend', 'suggest', 'find', 'where', 'what', 'best', 'good', 'top',
            'restaurant', 'food', 'place', 'location', 'current', 'latest', 'news',
            'weather', 'price', 'cost', 'review', 'rating', 'near', 'around',
            'popular', 'famous', 'show me', 'tell me about', 'looking for'
        ]
        should_search = any(trigger in user_input.lower() for trigger in search_triggers)

        # Only search for factual topics like travel, food, technology, etc.
        factual_topics = ['travel', 'food', 'technology', 'entertainment', 'sports', 'health', 'education', 'business']
        if intent in ['question', 'request'] and any(topic in factual_topics for topic in topics):
            should_search = True
        
        if not should_search:
            return None
        
        try:
            doc = nlp(user_input)
            detected_locations = []
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC']: 
                    detected_locations.append(ent.text.lower())
            food_keywords = ['restaurant', 'food', 'eat', 'dining', 'cuisine', 'food court', 
                           'cafe', 'bar', 'pizza', 'pasta', 'sushi', 'burger', 'coffee',
                           'bakery', 'deli', 'bistro', 'tavern', 'grill', 'buffet']
            
            detected_food_terms = []
            for token in doc:
                if token.text.lower() in food_keywords or token.pos_ == 'NOUN':
                    detected_food_terms.append(token.text.lower())
            
            is_food_query = any(keyword in user_input.lower() for keyword in food_keywords)

            search_results = None
            
            if is_food_query and detected_locations:
                location = detected_locations[0] 
                food_terms = ' '.join(detected_food_terms[:3])
                query = f"{food_terms} restaurants {location}"
                search_results = self.web_search.search_local_businesses(query, location)
                search_type = "local_business"
            elif detected_locations and not is_food_query:
                location = detected_locations[0]
                query = f"{' '.join(key_phrases[:2])} {location}"
                search_results = self.web_search.search(query, num_results=5)
                search_type = "location_search"
            elif topics:
                query = f"{' '.join(topics)} {' '.join(key_phrases[:2])}"
                search_results = self.web_search.search(query, num_results=5)
                search_type = "topic_search"
            else:
                query = ' '.join(key_phrases[:3]) if key_phrases else user_input
                search_results = self.web_search.search(query, num_results=3)
                search_type = "general_search"
            
            if search_results:
                return {
                    'query': query,
                    'search_type': search_type,
                    'results': search_results,
                    'detected_locations': detected_locations,
                    'detected_food_terms': detected_food_terms,
                    'timestamp': datetime.now().isoformat(),
                    'user_input': user_input,
                    'triggered_by': intent
                }
            
        except Exception as e:
            print(f"[ERROR] Web search failed: {e}")
        
        return None

    def _make_json_serializable(self, obj):
        """
        Convert datetime objects and other non-serializable objects to strings.
        """
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj

    def get_personalized_context(self) -> str:
        """
        Get personalized context based on user's learning patterns.
        """
        try:
            learning_patterns = self.rag_system.get_user_learning_patterns()
            
            if not learning_patterns:
                return ""
            
            context = f"""
            
        USER PERSONALIZATION CONTEXT:
        - Total conversations: {learning_patterns.get('total_conversations', 0)}
        - Average engagement: {learning_patterns.get('average_engagement', 0):.2f}
        - Preferred intents: {learning_patterns.get('preferred_intents', {})}
        - Preferred topics: {learning_patterns.get('preferred_topics', {})}
        - Sentiment pattern: {learning_patterns.get('sentiment_pattern', {})}
        
        Use this information to tailor your responses to the user's communication style and preferences.
        """
            
            return context
            
        except Exception as e:
            print(f"[ERROR] Failed to get personalized context: {e}")
            return ""

    def _get_llm_json_response(self, prompt: str, model: str = "openai/gpt-oss-120b") -> Dict:
        """
        A robust wrapper for making calls to the Groq API and parsing JSON.
        """
        chat_completion = self.groq_cloud_client.chat.completions.create(messages=[{"role": "system", "content": "You are a helpful AI assistant. You only respond in valid JSON."}, {"role": "user", "content": prompt}], model=model, response_format={"type": "json_object"})
        return json.loads(chat_completion.choices[0].message.content)

    def process_input(self, user_input: str) -> Dict[str, Any]:
        """
        The main pipeline for processing a single user input.
        """
        print("\n========================= ANALYSIS =========================")
        analysis_result = {'timestamp': datetime.now().isoformat(), 'user_input': user_input}
        
        analysis_result['key_phrases'] = [kp['phrase'] for kp in self.keyphrase_extractor.extract_key_phrases(user_input, top_n=4)]
        micro_sentiment = self.analyze_sentiment(user_input)
        analysis_result['sentiment_label'] = micro_sentiment['label']
        analysis_result['sentiment_score'] = micro_sentiment['score']
        analysis_result['sentiment_compound'] = micro_sentiment['compound'] 
        
        macro_sentiment = self.calculate_macro_sentiment(self.conversation_history)
        analysis_result['macro_sentiment_label'] = macro_sentiment['label']
        analysis_result['macro_sentiment_score'] = macro_sentiment['score']

        analysis_result['intent_detected'] = self.detect_intent(user_input)
        analysis_result['emotion_detected'] = self.detect_emotion(user_input)
        analysis_result['topics_detected'] = self.detect_topics(user_input, top_n=2)
        
        analysis_result['relevance_score'] = self.relevance_scorer.calculate_score(analysis_result, self.conversation_history)
        analysis_result['engagement_score'] = self.calculate_enhanced_engagement(analysis_result)
        analysis_result['engagement_level'] = self.determine_engagement_level(analysis_result['engagement_score'])
        
        self.dialogue_tracker.update_state(analysis_result)
        print("\n[Dialogue State]", self.dialogue_tracker.get_state()['slots'])

        # Web Search Integration with error handling
        try:
            web_search_results = self.perform_intelligent_web_search(analysis_result)
            if web_search_results:
                analysis_result['web_search_results'] = web_search_results
                print(f"[INFO] Web search triggered: {web_search_results['search_type']} - Found {len(web_search_results.get('results', []))} results")
                # Only store if MongoDB is connected
                if self.rag_system.client:
                    self.rag_system.store_real_time_data(web_search_results)
            else:
                print("[INFO] No web search triggered for this query")
        except Exception as e:
            print(f"[WARNING] Web search failed: {e}")
        
        # RAG System Integration with error handling
        try:
            if self.rag_system.client:  # Only if MongoDB is connected
                similar_conversations = self.rag_system.retrieve_similar_conversations(user_input, limit=3)
                relevant_knowledge = self.rag_system.retrieve_relevant_knowledge(user_input, limit=2)
                
                if similar_conversations:
                    analysis_result['similar_conversations'] = similar_conversations
                if relevant_knowledge:
                    analysis_result['relevant_knowledge'] = relevant_knowledge
        except Exception as e:
            print(f"[WARNING] RAG system retrieval failed: {e}")
        
        sentiment_display = ""
        if analysis_result['engagement_level'] in ['high', 'medium']:
            sentiment_display = f"{analysis_result['sentiment_score']:.2f} (Micro - {analysis_result['sentiment_label']})"
        else:
            sentiment_display = f"{analysis_result['macro_sentiment_score']:.2f} (Macro - {analysis_result['macro_sentiment_label']})"

        print(f"1. Keyphrases: {analysis_result['key_phrases']}")
        print(f"2. Sentiment: {sentiment_display}")
        print(f"3. Intent: {analysis_result['intent_detected'].capitalize()}")
        print(f"4. Emotion: {analysis_result['emotion_detected'].capitalize()}")
        print(f"5. Topics: {analysis_result['topics_detected']}")
        print(f"6. Relevance Score (Advanced): {analysis_result['relevance_score']:.2f}")
        print(f"7. Engagement Score & Level: {analysis_result['engagement_level'].capitalize()} ({analysis_result['engagement_score']:.2f})")

        analysis_result['reply_suggestions'] = self.generate_llm_replies(analysis_result)
        print("============================================================")
        # Store conversation in RAG system for future learning
        try:
            if self.rag_system.client:  # Only if MongoDB is connected
                self.rag_system.store_conversation(analysis_result)
        except Exception as e:
            print(f"[WARNING] Failed to store conversation: {e}")
        
        self.conversation_history.append(analysis_result)
        return analysis_result

    def log_to_csv(self, data: Dict, file_path: str, header: List[str]) -> None:
        """
        A utility function to append data to a CSV file.
        """
        try:
            file_exists = os.path.exists(file_path)
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if not file_exists: writer.writeheader()
                writer.writerow(data)
        except Exception as e: print(f"[Error] Failed to log to {file_path}: {e}")

    def collect_and_log_feedback(self, suggestions: List[str]) -> str:
        """
        Manages the user feedback loop and logs data for RL training.
        """
        print("\n[System] How should I reply? (Choose a number)")
        for i, reply in enumerate(suggestions, 1):
            print(f" {i}. {reply}")
        print(f" {len(suggestions) + 1}. None of these.")
        
        chosen_reply = ""
        try:
            choice_num = int(input("> "))
            if 1 <= choice_num <= len(suggestions):
                chosen_reply = suggestions[choice_num - 1]
            else:
                chosen_reply = "None of the above"
        except (ValueError, IndexError):
            chosen_reply = "Invalid choice"

        print("\n[System] Was this a good response? (Choose an option)")
        options = {"g": "👍 Great", "o": "🙂 Okay", "b": "👎 Bad"}
        print(" | ".join(f"({k}) {v}" for k, v in options.items()))
        
        feedback_choice = ""
        while feedback_choice not in options:
            feedback_choice = input("> ").lower()

        rating_map = {"g": 1, "o": 3, "b": 5}
        rating = rating_map.get(feedback_choice, 3)

        reward = self.calculate_reward(rating)
        observation = self.conversation_history[-1]
        action = chosen_reply
        
        rl_data = {
            'timestamp': datetime.now().isoformat(), 'user_id': self.user_id,
            'observation_json': json.dumps({k: v for k, v in observation.items() if k != 'reply_suggestions'}),
            'action_taken': action, 'reward': reward
        }
        self.log_to_csv(rl_data, CSV_FILES['rl_data'], CSV_HEADERS['rl_data'])
        # Store RL data in RAG system for learning
        try:
            if self.rag_system.client:  # Only if MongoDB is connected
                self.rag_system.store_rl_data(observation, action, reward)
        except Exception as e:
            print(f"[WARNING] Failed to store RL data: {e}") 
        print(f"\n[System] Thank you! Feedback logged (Reward: {reward}).")
        self.conversation_history[-1]['selected_reply'] = chosen_reply
        return chosen_reply

    def collect_exit_feedback(self) -> None:
        """
        Collects a final star-based rating at the end of the conversation.
        """
        print("\n" + "="*20 + " Final Feedback " + "="*20)
        star_rating = ""
        while not star_rating.isdigit() or not 1 <= int(star_rating) <= 5:
            star_rating_input = input("Overall, how would you rate this conversation? (1-5 stars): ").strip()
            numeric_part = "".join(filter(str.isdigit, star_rating_input))
            if numeric_part:
                star_rating = numeric_part

        feedback_data = {
            'timestamp': datetime.now().isoformat(), 'user_id': self.user_id,
            'star_rating': star_rating, 'helpful_feedback': 'N/A',
            'suggestions_feedback': 'N/A', 'improvements_feedback': 'N/A'
        }
        self.log_to_csv(feedback_data, CSV_FILES['exit_feedback'], CSV_HEADERS['exit_feedback'])
        print("\nThank you for your feedback! It has been saved.")
        print("="*56)

