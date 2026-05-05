# 🔢 Handwritten Digit Recognizer

A Python-based handwritten digit recognition system using MLP neural network with custom image preprocessing pipeline and model persistence.

## 🛠️ Tech Stack
`Python` · `scikit-learn` · `scikit-image` · `NumPy` · `Matplotlib` · `joblib`

## ⚙️ How It Works
1. Images are preprocessed — grayscale, inverted, resized to 28×28, normalized, and flattened
2. An MLP classifier (1 hidden layer, 128 neurons, Adam optimizer) is trained on the dataset
3. Trained model is saved as `my_model.pkl` and reloaded for testing
4. Accuracy is evaluated on unseen test images

## 🚀 Setup & Run
```bash
python -m pip install scikit-learn scikit-image numpy matplotlib joblib
python driver_digit_classifier_v3.py
```

## 📁 Structure
```
submit/
├── digit_classifier_class_v3.py   # Core classifier class
├── driver_digit_classifier_v3.py  # Main entry point
├── my_model.pkl                   # Saved model
├── train_digits_dataset/          # Training images
└── test_digits_dataset/           # Test images