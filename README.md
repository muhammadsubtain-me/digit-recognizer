# Handwritten Digit Recognizer

A python based handwritten digit recognition using MLP neural network with custom image preprocessing pipeline and model persistence.

## Tech Stack
`Python` · `scikit-learn` · `scikit-image` · `NumPy` · `Matplotlib` · `joblib`

## How It Works
1. Each digit image is converted to grayscale, inverted, resized to 28×28, normalized to [0,1] and flattened into a 784-element feature vector.
2. An MLP classifier with 1 hidden layer of 128 neurons and Adam optimizer is trained
3. Trained model is saved as `my_model.pkl` and reloaded for testing
4. Accuracy is evaluated on written test images

## Setup & Run
```bash
python -m pip install scikit-learn scikit-image numpy matplotlib joblib
python driver_digit_classifier_v3.py
```

