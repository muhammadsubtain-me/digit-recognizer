
import os
from digit_classifier_class_v3 import DigitClassifier

train_images_folder = "train_digits_dataset"
test_images_folder  = "test_digits_dataset"

train_folder = os.path.join(os.getcwd(), train_images_folder)
test_folder  = os.path.join(os.getcwd(), test_images_folder)

# ── Training Phase ────────────────────────────────────────────────────────────
digit_classifier = DigitClassifier(model_layers=(128,), learn_rate=0.001, solver="adam")

digit_classifier.train(train_folder)

digit_classifier.save_model(model_name="my_model.pkl")

# show the images
digit_classifier.plot_train_images(train_folder)

# ── Testing Phase ─────────────────────────────────────────────────────────────
test_classifier = DigitClassifier()
test_classifier.load_model(model_name="my_model.pkl")
test_classifier.test_model(test_folder)
