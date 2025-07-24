# #pipeline is a simple way to load pretrained 
# # model for task like summarization , emotion detection etc

# from transformers import pipeline

# import torch

# classifier= pipeline("text-classification", 
# # model="nlptown/bert-base-multilingual-uncased",
#                       model="j-hartmann/emotion-english-distilroberta-base",
#                       top_k=None)


# [
#   {"label": "joy", "score": 0.85},
#   {"label": "sadness", "score": 0.03},
#   {"label": "anger", "score": 0.02},
#   ...
# ]


# # to make it easy to use this in a Streamlit app or other code
# def detect_emotions(text):
#     # check if the input is empty or white space
#     if not text.strip():
#         return[{"label": "No text " , " score": 1.0}]
# # send text to the hugging face emotion classifier
#     result =classifier(text)
#     return result [0]
 



































from transformers import pipeline
import torch

# Initialize the emotion classifier
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=0 if torch.cuda.is_available() else -1
)

# Map emotions to mental health categories
MENTAL_HEALTH_MAPPING = {
    'sadness': 'depression',
    'fear': 'anxiety',
    'anger': 'stress',
    'joy': 'well-being',
    'disgust': 'negative_outlook',
    'surprise': 'uncertainty',
    'neutral': 'neutral'
}

def detect_emotions(text):
    if not text.strip():
        return [{"label": "No text", "score": 1.0}]
    
    try:
        results = classifier(text)
        emotions = results[0]
        
        # Enhance results with mental health context
        for emotion in emotions:
            emotion['mental_health_category'] = MENTAL_HEALTH_MAPPING.get(
                emotion['label'].lower(),
                'other'
            )
        
        return emotions
    except Exception as e:
        # Fallback to simple keyword matching if model fails
        return simple_emotion_detection(text)

def simple_emotion_detection(text):
    """Fallback method when transformer model fails"""
    text_lower = text.lower()
    emotions = []
    
    # Check for depression indicators
    sad_words = ['sad', 'depress', 'hopeless', 'empty', 'lonely']
    sadness_score = sum(text_lower.count(word) for word in sad_words) / 10
    
    # Check for anxiety indicators
    anxiety_words = ['anxious', 'worry', 'fear', 'scared', 'nervous']
    anxiety_score = sum(text_lower.count(word) for word in anxiety_words) / 10
    
    # Check for stress indicators
    stress_words = ['stress', 'angry', 'frustrat', 'overwhelm', 'pressure']
    stress_score = sum(text_lower.count(word) for word in stress_words) / 10
    
    emotions.extend([
        {"label": "sadness", "score": min(sadness_score, 0.99), "mental_health_category": "depression"},
        {"label": "fear", "score": min(anxiety_score, 0.99), "mental_health_category": "anxiety"},
        {"label": "anger", "score": min(stress_score, 0.99), "mental_health_category": "stress"},
        {"label": "neutral", "score": max(0, 1 - (sadness_score + anxiety_score + stress_score)), "mental_health_category": "neutral"}
    ])
    
    # Sort by score descending
    emotions.sort(key=lambda x: x['score'], reverse=True)
    return emotions[:4]  # Return top 4 emotions