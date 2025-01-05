# Import necessary libraries
import cv2                          # For reading image files
from tensorflow.keras.models import load_model # For loading the pre-trained model (use tensorflow.keras instead of keras directly)
from PIL import Image               # For image processing
import numpy as np                  # For numerical operations

# Load the pre-trained model from a file
model = load_model('BrainTumor50EpochsCategorical.h5')  # Replace with the correct model file name if necessary

# Read the image from the specified file path
image = cv2.imread('Br35H-Mask-RCNN\VAL\y699.jpg')

# Convert the image from BGR to RGB
img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Resize the image to the input size expected by the model (64x64 pixels)
img = img.resize((64, 64))

# Convert the PIL Image object to a NumPy array and normalize pixel values to [0, 1]
img = np.array(img) / 255.0

# Add an extra dimension to the image array to match the input shape expected by the model
# This creates a batch dimension with size 1 (i.e., shape: (1, 64, 64, 3))
input_img = np.expand_dims(img, axis=0)

# Predict the class probabilities for the input image using the loaded model
probs = model.predict(input_img)

# Get the class with the highest probability
result = np.argmax(probs, axis=1)

# Print the predicted class
if result[0]==1:
    print("Brain Tumor Detected")
else:
    print("No Brain Tumour Detected")

print(result)
