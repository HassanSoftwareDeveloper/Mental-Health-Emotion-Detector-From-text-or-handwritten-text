# # used for tokenization, stopword removal and other text processing task
# #raw text cleanup or preperation
# import nltk
# #allow pattern matching and cleaning text
# import re
# # task like entity recognition,lemmatization and pos tagging
# import spacy
# #stop common words
# from nltk.corpus import stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')# for lemmatization


# # speed up filtering word in the main function

# stop_words=set(stopwords.words("english"))
# # load the small english model in spaCy
# nlp=spacy.load("en_core_web_sm")





# def clean_text(text):
#     # remove special characters and digits
#     text = text.lower()
#       text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and whitespace
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces


#     #splits the cleaned text into individual words
#     tokens= nltk.word_tokenize(text)
#     # remove very short words
#     filtered= [word for word in tokens if word not in stop_words and len(word)>2]

#     return " ".join(filtered)









































import nltk
import re
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data with error handling
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)

# Call this at module level to ensure resources are available
download_nltk_resources()

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Stopwords
stop_words = set(stopwords.words("english"))
custom_stopwords = {"im", "ive", "youre", "theyre", "dont", "wont", "cant", "us", "pm", "am"}
stop_words.update(custom_stopwords)

def clean_text(text: str) -> str:
    if not text.strip():
        return ""
    
    text = text.lower()
    
    # Clean up special characters (expand if needed)
    text = re.sub(r'[^a-zA-Z0-9\s\'.,!?]', '', text)
    
    # Expand contractions
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize with error handling
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        print(f"Tokenization error: {e}")
        tokens = text.split()  # Fallback to simple whitespace tokenization
    
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatize using spaCy
    doc = nlp(" ".join(filtered_tokens))
    lemmatized_tokens = [token.lemma_ if token.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV'] else token.text for token in doc]
    
    return " ".join(lemmatized_tokens)