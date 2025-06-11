# 🥬 Grocery Classification API

This is the backend API for classifying grocery items and retrieving the best price from our database.

It supports both **image-based** and **text-based search** using a trained **MobileNetV3 classification model and EasyOCR**.

### 🛠️ Stack
- Python
- FastAPI
- PostgreSQL
- EasyOCR
- MobileNetV3 (trained with 99.4% train / 88.1% val accuracy)

### Model Hosting
→ [Model for MobileNet + OCR Model on Hugging Face](https://huggingface.co/yukieos/grocery_classification)
→ [API for MobileNet + OCR Model on Hugging Face](https://huggingface.co/spaces/yukieos/groceryclassifier)
→ [API for text search on Render](https://grocery-classification.onrender.com)

### 📦 Features
- POST endpoint for image input
- Text similarity search for price matching
- Score-based result ranking
