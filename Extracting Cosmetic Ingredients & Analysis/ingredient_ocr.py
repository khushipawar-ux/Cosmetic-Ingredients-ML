import cv2
import pytesseract
import numpy as np
import re
import json
from ingredient_aliases import INGREDIENT_ALIASES
from PIL import Image

# 🔧 Configure tesseract path (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------- IMAGE PREPROCESSING ----------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return thresh


# ---------- OCR EXTRACTION ----------
def extract_text(image):
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text


# ---------- INGREDIENT SECTION EXTRACTION ----------
def extract_ingredient_block(text):
    text = text.lower()

    patterns = [
        r"ingredients[:\s]*(.*)",
        r"composition[:\s]*(.*)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)

    return ""


# ---------- INGREDIENT CLEANING ----------
def clean_ingredients(ingredient_text):
    ingredient_text = re.sub(r'[^a-zA-Z0-9, ]', '', ingredient_text)
    ingredients = [i.strip() for i in ingredient_text.split(",") if len(i.strip()) > 2]

    cleaned = []
    for ing in ingredients:
        ing_lower = ing.lower()
        normalized = INGREDIENT_ALIASES.get(ing_lower, ing.title())
        cleaned.append(normalized)

    return list(dict.fromkeys(cleaned))  # Remove duplicates


# ---------- MAIN PIPELINE ----------
def extract_ingredients_from_image(image_path):
    processed_image = preprocess_image(image_path)
    text = extract_text(processed_image)
    ingredient_block = extract_ingredient_block(text)
    ingredients = clean_ingredients(ingredient_block)

    return {
        "raw_text": text,
        "ingredients": ingredients
    }


# ---------- SAVE OUTPUT ----------
def save_output(data, output_path="output/ingredients.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ---------- RUN ----------
if __name__ == "__main__":
    image_path = "sample_images/shampoo_label.jpg"
    output = extract_ingredients_from_image(image_path)
    save_output(output)

    print("✅ Extracted Ingredients:")
    for ing in output["ingredients"]:
        print(" -", ing)
