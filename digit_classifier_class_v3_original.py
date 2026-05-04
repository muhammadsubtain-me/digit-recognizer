#
import os
import joblib
import numpy as np

from skimage import io, color, transform, img_as_float
from skimage import util

from sklearn.metrics import accuracy_score # this is needed for the testing phase
from sklearn.neural_network import MLPClassifier

from matplotlib import pyplot as plt

#
class DigitClassifier:
    def __init__(self, model_layers=(128,), learn_rate=0.01, solver="sgd"):

        self.lr = learn_rate
        self.solver = solver
        self.model_layers = model_layers

        self.model = MLPClassifier(hidden_layer_sizes=self.model_layers,
                                   max_iter=500, alpha=1e-4,
                                   solver=self.solver, verbose=True,
                                   tol=1e-4, random_state=1,
                                   learning_rate_init=self.lr)
        
        self.train_images = np.array([])
        self.train_labels = np.array([])
        self.test_images = np.array([])
        self.test_labels = np.array([])

    def load_dataset(self, folder, mode):
        images_list = []
        labels_list = []

        for file in os.listdir(folder):
            if file.endswith((".jpeg", "jpg", ".png")):
                img_file_path = os.path.join(folder, file)

                # read the image given its path
                img_orig = io.imread(img_file_path)

                # Convert the RGB image to grayscale image
                # Note: in some apps (other than MS Paint) when an image is created, an alpha channel is added
                #       to the image (transparency layer).
                #       In this case, the dimension is [:, :, 4] (R-G-B-Alpha), but we need only [:, :, 3] (R-G-B)
                #       check if the image has an alpha channel or not
                #       The "if" statement can be removed if the image has only the three channels RGB
                if img_orig.shape[2] > 3:
                    img_gray = color.rgb2gray(img_orig[:, :, :3])  # extract only the RGB channels and exclude the alpha channel
                else:
                    img_gray = color.rgb2gray(img_orig)
                
                # Invert the image    
                img_gray = util.invert(img_gray)

                # resize image to 28 X 28 if necessary
                if img_gray.shape[0] != 28 or img_gray.shape[1] != 28:
                    img_resized = transform.resize(img_gray, (28, 28))
                else:
                    img_resized = img_gray

                # normalize the image [0, 1]
                img_normalized = img_as_float(img_resized)

                # flatten the image
                img_flat = img_normalized.flatten()

                # create training features from pixels of the image
                # create labels from the name of the image file
                images_list.append(img_flat)
                label = int(file[0])
                labels_list.append(label)

        if mode == "train":
            self.train_images = np.array(images_list)
            self.train_labels = np.array(labels_list)
       
    # Method to train the MLP
    def train(self, folder):
        self.load_dataset(folder, "train")
        self.model.fit(self.train_images, self.train_labels)
    
    # This method saves the trained model so that it can be loaded later
    def save_model(self, model_name="trained_model.pkl"):
        joblib.dump(self.model, model_name)
    # ========================== Project ======================================
    # Student must provide the implementation of this method
    # This method load the trained model
    def load_model(self, model_name="trained_model.pkl"):
        pass

    # Student must provide the implementation of this method
    # Method to test the trained model using a a test dataset
    def test_model(self, folder):
        pass
    
    # ========================= Additional Methods ============================
    # The following methods are used to display a sample of the training dataset
    def get_image_list(self, folder):
        image_list = []
        for file in os.listdir(folder):
            if file.endswith((".jpeg", "jpg", ".png")):
                img_path = os.path.join(folder, file)
                image_list.append(io.imread(img_path))
        return np.array(image_list)

    def plot_train_images(self, folder):
        image_list = self.get_image_list(folder)
        num_samples = 50
        rows = 5
        cols = 10
        indexes = np.random.randint(0, image_list.shape[0], size=num_samples)
        images = image_list[indexes]

        plt.figure(figsize=(cols, rows))
        for i in range(num_samples):
            plt.subplot(rows, cols, i + 1)
            image = images[i]
            image = util.invert(image)
            plt.imshow(image)
            plt.axis('off')
        plt.show()
# =============================================================================