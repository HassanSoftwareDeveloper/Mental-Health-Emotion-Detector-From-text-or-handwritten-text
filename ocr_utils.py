



# import cv2
# import pytesseract
# import numpy as np
# from PIL import Image
# import torch
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# import google.generativeai as genai
# import os
# import logging
# import time

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Configure Tesseract path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Initialize models with retries
# def initialize_models():
#     # Initialize TrOCR
#     trocr_processor, trocr_model = None, None
#     try:
#         trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
#         trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
#         logger.info("TrOCR model loaded successfully")
#     except Exception as e:
#         logger.error(f"Failed to load TrOCR: {str(e)}")

#     # Initialize Gemini
#     gemini_model = None
#     try:
#         genai.configure(api_key='AIzaSyAQFx60xwC75mCCSmJycx49Og6Rvo4ULGU')
#         gemini_model = genai.GenerativeModel('gemini-pro-vision')
#         logger.info("Gemini model initialized successfully")
#     except Exception as e:
#         logger.error(f"Failed to initialize Gemini: {str(e)}")

#     return trocr_processor, trocr_model, gemini_model

# trocr_processor, trocr_model, gemini_model = initialize_models()

# def enhance_handwriting_image(image):
#     """Military-grade image enhancement for handwriting"""
#     if len(image.shape) == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image
    
#     # Advanced preprocessing pipeline
#     denoised = cv2.fastNlMeansDenoising(gray, None, h=20, templateWindowSize=9, searchWindowSize=21)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12,12))
#     contrast = clahe.apply(denoised)
#     _, threshold = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     kernel = np.ones((2,2), np.uint8)
#     processed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    
#     return processed

# def extract_with_gemini(image):
#     """Perfect Gemini extraction with retries"""
#     if gemini_model is None:
#         return None
    
#     try:
#         pil_img = Image.fromarray(image).convert("RGB")
#         if max(pil_img.size) > 1600:
#             pil_img.thumbnail((1600, 1600), Image.LANCZOS)
        
#         # Ultra-specific prompt
#         prompt = """EXTRACT THIS HANDWRITTEN TEXT 100% ACCURATELY:
#         - Preserve EXACT spelling, case, and punctuation
#         - Keep all line breaks
#         - Never correct or modify the text
#         - Return ONLY the raw text, no commentary
        
#         Text must be IDENTICAL to the handwriting:"""
        
#         for _ in range(5):  # 5 retries
#             try:
#                 response = gemini_model.generate_content([prompt, pil_img])
#                 if response.text:
#                     return response.text.strip()
#                 time.sleep(1)
#             except Exception as e:
#                 logger.warning(f"Gemini attempt failed: {str(e)}")
#                 time.sleep(1)
#         return None
#     except Exception as e:
#         logger.error(f"Gemini extraction failed: {str(e)}")
#         return None

# def extract_with_trocr(image):
#     """TrOCR extraction with error handling"""
#     if trocr_processor is None or trocr_model is None:
#         return None
    
#     try:
#         pil_image = Image.fromarray(image).convert("RGB")
#         pixel_values = trocr_processor(images=pil_image, return_tensors="pt").pixel_values
#         generated_ids = trocr_model.generate(pixel_values)
#         return trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     except Exception as e:
#         logger.error(f"TrOCR extraction failed: {str(e)}")
#         return None

# def extract_with_tesseract(image, mode="standard"):
#     """Tesseract with optimized config"""
#     config = r'--oem 3 --psm 11' if mode == "handwriting" else r'--oem 3 --psm 6'
#     return pytesseract.image_to_string(image, config=config)

# def extract_text_from_image(image, mode="gemini ai (best for handwriting)", show_steps=False, debug=False):
#     """Guaranteed extraction pipeline"""
#     original_image = image.copy()
    
#     # Handwriting mode - Gemini first
#     if mode == "gemini ai (best for handwriting)":
#         gemini_text = extract_with_gemini(original_image)
#         if gemini_text and len(gemini_text.split()) >= 2:
#             return gemini_text
        
#         # Fallback to TrOCR
#         processed_img = enhance_handwriting_image(original_image)
#         trocr_text = extract_with_trocr(processed_img)
#         if trocr_text and len(trocr_text.split()) >= 2:
#             return trocr_text
        
#         # Final fallback to Tesseract
#         tess_text = extract_with_tesseract(processed_img, "handwriting")
#         return tess_text if tess_text.strip() else "No text extracted"
    
#     # Other modes (standard/enhanced)
#     processed_img = enhance_handwriting_image(original_image) if "handwriting" in mode.lower() else \
#                    preprocess_image(original_image, mode, show_steps)
    
#     if mode == "enhanced":
#         trocr_text = extract_with_trocr(processed_img)
#         if trocr_text:
#             return trocr_text
#         return extract_with_tesseract(processed_img, "enhanced")
    
#     # Standard mode
#     return extract_with_tesseract(processed_img)

# def preprocess_image(image, mode="standard", show_steps=False):
#     """General image preprocessing"""
#     if len(image.shape) == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     else:
#         gray = image
    
#     blur = cv2.GaussianBlur(gray, (3, 3), 0)
#     _, processed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     if show_steps:
#         cv2.imshow("Processed Image", processed)
#         cv2.waitKey(0)
    
#     return processed










import google.generativeai as genai
from PIL import Image
import numpy as np
import logging

# Configure the nuclear-grade extraction
genai.configure(api_key='AIzaSyAQFx60xwC75mCCSmJycx49Og6Rvo4ULGU')
model = genai.GenerativeModel('gemini-1.5-flash')

def extract_text_from_image(image, mode="gemini", show_steps=False, debug=False):
    """Military-grade text extraction that PRESERVES WORD ORDER"""
    try:
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        # Optimal size for handwriting
        image = image.resize((800, 400), Image.LANCZOS)
        
        # Nuclear prompt that forces exact preservation
        response = model.generate_content([
            "⚠️ READ THIS CAREFULLY ⚠️\n"
            "Extract this handwritten text with 100% accuracy:\n"
            "- PRESERVE THE EXACT WORD ORDER\n"
            "- Keep ALL punctuation and line breaks\n"
            "- NEVER rearrange words\n"
            "- Return ONLY the raw text exactly as written\n\n"
            "Example:\n"
            "If the handwriting says: 'I feel sad today'\n"
            "YOU MUST RETURN: 'I feel sad today'",
            image
        ], request_options={"timeout": 15})
        
        if response.text:
            # Validate we got proper text
            text = response.text.strip()
            if len(text.split()) >= 3:  # At least 3 words
                return text
                
        return "Extraction failed - try a clearer image"
        
    except Exception as e:
        logging.error(f"Extraction failed: {e}")
        return "Extraction error"