import cv2
import pytesseract
import numpy as np
from PIL import Image as PILImage
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import google.generativeai as genai
import os

# Configure OCR paths
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize TrOCR for handwriting
try:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
except Exception as e:
    print(f"Error loading TrOCR: {e}")
    processor = None
    model = None

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyAQFx60xwC75mCCSmJycx49Og6Rvo4ULGU')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro-vision')
else:
    gemini_model = None

def preprocess_image(image, mode="standard", show_steps=False):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    if mode in ["enhanced"]:
        denoised = cv2.fastNlMeansDenoising(gray, None, h=30, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        processed = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    else:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, processed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if show_steps:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        plt.imshow(gray, cmap='gray')
        plt.title("Original")
        plt.subplot(1,2,2)
        plt.imshow(processed, cmap='gray')
        plt.title("Processed")
        plt.show()

    return processed

def extract_with_tesseract(image, mode="standard"):
    config = r'--oem 3 --psm 6'
    if mode == "enhanced":
        config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()\'"'
    return pytesseract.image_to_string(image, config=config)

def extract_with_trocr(image):
    if processor is None or model is None:
        return "TrOCR model not available"
    try:
        pil_image = PILImage.fromarray(image).convert("RGB")
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        return f"TrOCR Error: {str(e)}"

def extract_with_gemini(image):
    if gemini_model is None:
        return "Gemini API not configured"
    try:
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image).convert("RGB")
        elif isinstance(image, PILImage.Image):
            image = image.convert("RGB")
        else:
            return "Unsupported image format"

        prompt = (
            "Extract all text exactly as it appears, preserving indentation, spacing, and formatting. If it's code, preserve its structure."
        )
        response = gemini_model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

def extract_text_from_image(image, mode="standard", show_steps=False, debug=False):
    try:
        if mode == "gemini ai (best for handwriting)":
            if debug: print("Using Gemini AI for text extraction")
            result = extract_with_gemini(image)
            if "Error" not in result and len(result.strip()) > 5:
                return result.strip()
            else:
                if debug: print("Gemini failed, falling back to TrOCR")
                return extract_with_trocr(image)

        processed_img = preprocess_image(image, mode, show_steps)

        if mode == "enhanced":
            if debug: print("Using TrOCR for text extraction")
            result = extract_with_trocr(image)
            if "Error" not in result and len(result.strip()) > 5:
                return result.strip()
            else:
                if debug: print("TrOCR failed, falling back to Tesseract")
                return extract_with_tesseract(processed_img, "enhanced")

        else:
            if debug: print("Using Tesseract for text extraction")
            result = extract_with_tesseract(processed_img, "standard")
            if len(result.strip()) > 3:
                return result.strip()
            else:
                if debug: print("Standard OCR failed, trying enhanced")
                return extract_text_from_image(image, "enhanced", show_steps, debug)

    except Exception as e:
        if debug: print(f"OCR Error: {str(e)}")
        return f"Text extraction failed: {str(e)}"

    return "No text could be extracted"