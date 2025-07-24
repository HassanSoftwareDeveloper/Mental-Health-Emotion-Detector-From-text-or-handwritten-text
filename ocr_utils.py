# import cv2
# # it is powerfull library for extracting text from images 
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# import numpy as np
 

# def preprocess_image(image):
#     # Convert the image to grayscale
#     #ocr work better on grayscale imgae because color is unnecessary and adds noise
    
#     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    

# # applies gaussianblur to smooth the image
# # bluring remove small noise which help with thresholding

#     blur =cv2.GaussianBlur(gray,(5,5),0)

#     #now convert the image into black and white(binary)
#     _, thresh = cv2.threshold(blur , 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return thresh


# #encapsulate the full process of image to text
# # call this function whatever you need to extract text from an image

# def extract_text_from_image(image):
#     preprocessed = preprocess_image(image)
#     #use tesseract to extract text from the cleaned image
#     # this is the core step of ocr converting image pixels to characters
#     text=pytesseract.image_to_string(preprocessed)

#     return text










































import cv2
import pytesseract
import numpy as np

# Configure Tesseract path (update with your path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image, mode="standard", show_steps=False):
    """Preprocess image for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if mode.lower() == "enhanced":
        # Enhanced processing for handwriting
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        processed = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        # Standard processing for printed text
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, processed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if show_steps:
        cv2.imshow("Processed Image", processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return processed

def extract_text_from_image(image, mode="standard", show_steps=False):
    """Extract text from image using OCR with configurable mode"""
    preprocessed = preprocess_image(image, mode, show_steps)
    
    # Configure Tesseract
    custom_config = r'--oem 3 --psm 6'
    if mode.lower() == "enhanced":
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()\'\"/\\-:; '
    
    try:
        text = pytesseract.image_to_string(preprocessed, config=custom_config)
        return text.strip() if text else "No text could be extracted"
    except Exception as e:
        print(f"OCR Error: {str(e)}")
        return "OCR processing failed"

# Test function to verify the module is working
if __name__ == "__main__":
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(test_image, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    print("Test extraction:", extract_text_from_image(test_image))














