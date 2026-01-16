# Cosmetic Ingredients ML 🧴🔬

A comprehensive machine learning project focused on cosmetic products, combining **Computer Vision (OCR)** to extract ingredients and **Predictive Modeling** to analyze product popularity.

---

## 🌟 Overview

This repository contains two main modules:
1. **Extracting Cosmetic Ingredients & Analysis**: A tool that uses Optical Character Recognition (OCR) to read and extract ingredient lists from product packaging images.
2. **Cosmetic Product Prediction**: A machine learning pipeline that predicts product popularity based on features like brand, category, price, and ratings.

---

## 📁 Project Structure

```text
Cosmetic-Ingredients-ML/
├── Extracting Cosmetic Ingredients & Analysis/   # OCR Module
│   ├── ingredient_ocr.py                         # OCR extraction logic
│   ├── ingredient_aliases.py                     # Normalized ingredient mapping
│   ├── sample_images/                            # Images for testing
│   └── requirements.txt                          # OCR dependencies
├── cosmetic_prediction/                           # ML Prediction Module
│   ├── main.py                                   # End-to-end training & prediction
│   ├── train_model.py                            # Model training script
│   ├── predict_top_products.py                   # Inference script
│   ├── cosmetic.db                               # SQLite database for storing data
│   └── requirements.txt                          # ML dependencies
└── README.md                                     # Root project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system.
  - *Note: On Windows, ensure the path in `ingredient_ocr.py` matches your Tesseract installation (default: `C:\Program Files\Tesseract-OCR\tesseract.exe`).*

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/khushipawar-ux/Cosmetic-Ingredients-ML.git
   cd Cosmetic-Ingredients-ML
   ```

2. Install dependencies for each module:
   ```bash
   # Install OCR dependencies
   pip install -r "Extracting Cosmetic Ingredients & Analysis/requirements.txt"

   # Install Prediction dependencies
   pip install -r "cosmetic_prediction/requirements.txt"
   ```

---

## 🛠️ Usage

### 1. Extracting Ingredients (OCR)
Place a product label image in the `sample_images` folder and run:
```bash
python "Extracting Cosmetic Ingredients & Analysis/ingredient_ocr.py"
```
The script will preprocess the image, extract text, and save the cleaned ingredients to `output/ingredients.json`.

### 2. Product Popularity Prediction
To run the full pipeline (load data, train model, and predict top products):
```bash
cd cosmetic_prediction
python main.py
```
This will:
- Load the CSV dataset into a SQLite database.
- Train a `RandomForestRegressor` model.
- Save the top 10 predicted products to `output/top_predicted_products.csv`.

---

## 🧪 Technologies Used

- **Computer Vision**: OpenCV, Pytesseract (Tesseract OCR Engine)
- **Machine Learning**: Scikit-Learn (Random Forest), Pandas, NumPy
- **Database**: SQLite, SQLAlchemy
- **Data Preprocessing**: Label Encoding, Feature Engineering

---

## 📝 License
This project is for educational and practice purposes. See the [LICENSE](LICENSE) file for details (if applicable).

---

## 👩‍💻 Author
**Khushi Pawar** - [GitHub Profile](https://github.com/khushipawar-ux)
