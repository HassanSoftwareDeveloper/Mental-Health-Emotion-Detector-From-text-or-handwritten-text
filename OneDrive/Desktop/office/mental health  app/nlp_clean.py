import re
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize stopwords
stop_words = set(stopwords.words("english"))
custom_stopwords = {"im", "ive", "youre", "theyre", "dont", "wont", "cant", "us", "pm", "am"}
stop_words.update(custom_stopwords)

def clean_text(text: str) -> str:
    """
    Clean and preprocess text for emotion analysis.
    Returns the cleaned text string.
    """
    if not text or not text.strip():
        return ""

    # Convert to lowercase
    text = text.lower()

    # Expand contractions
    contractions = {
        "won't": "will not", "can't": "can not", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
        "'ve": " have", "'m": " am"
    }
    for c, full in contractions.items():
        text = re.sub(r'\b' + re.escape(c) + r'\b', full, text)

    # Remove special characters, keep alphanumeric and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize text
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Remove minimal stopwords
    minimal_stopwords = {"pm", "am", "etc"}
    filtered_tokens = [word for word in tokens if word.lower() not in minimal_stopwords]

    # Lemmatize using spaCy, preserving pronouns
    doc = nlp(" ".join(filtered_tokens))
    lemmatized = [token.text if token.pos_ == "PRON" else token.lemma_ for token in doc]

    return " ".join(lemmatized)

def generate_emotion_summary(emotions: list) -> str:
    """
    Generate a summary of detected emotions for mental health analysis.
    Args:
        emotions: List of dictionaries with 'label' and 'score' keys.
    Returns:
        A string summarizing the emotional state.
    """
    if not emotions:
        return "No emotions detected in the provided text."

    # Sort emotions by score
    emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
    primary_emotion = emotions[0]['label']
    primary_score = emotions[0]['score']

    # Categorize emotions
    positive = [e for e in emotions if e['label'].lower() in ['happiness', 'joy', 'love', 'excitement']]
    negative = [e for e in emotions if e['label'].lower() in ['anger', 'fear', 'sadness', 'anxiety', 'depression']]
    neutral = [e for e in emotions if e['label'].lower() not in ['happiness', 'joy', 'love', 'excitement', 'anger', 'fear', 'sadness', 'anxiety', 'depression']]

    # Generate summary
    summary = f"The primary detected emotion is {primary_emotion} with a confidence score of {primary_score:.2f}. "
    
    if positive and max(e['score'] for e in positive) > 0.5:
        summary += "The text shows significant positive emotions, suggesting a sense of well-being. "
    if negative and max(e['score'] for e in negative) > 0.5:
        summary += "There are notable negative emotions, which may indicate stress or mental health concerns. "
    if neutral and max(e['score'] for e in neutral) > 0.5:
        summary += "Neutral emotions are also present, indicating a balanced emotional state. "

    # Add mental health advice
    if any(e['label'].lower() in ['sadness', 'anxiety', 'depression'] and e['score'] > 0.5 for e in emotions):
        summary += "Consider reaching out to a mental health professional for support."
    else:
        summary += "The emotional profile appears stable, but regular self-care is recommended."

    return summary