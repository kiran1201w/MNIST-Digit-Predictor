import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# 1. Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# 2. Build the Feedforward Neural Network
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten 28x28 images into a 1D array
    Dense(128, activation='relu'),  # First hidden layer
    Dense(64, activation='relu'),   # Second hidden layer
    Dense(10, activation='softmax') # Output layer for 10 digit classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Train the model
print("Training the model...")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 4. Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {accuracy:.4f}")

# 5. Function to predict custom handwritten digit images
def predict_custom_image(image_path):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert colors for black digits on white background
    img = img.resize((28, 28))  # Resize to 28x28
    
    # Display the image
    plt.imshow(img, cmap='gray')
    plt.title("Custom Input Image")
    plt.show()
    
    # Preprocess the image
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 28, 28)  # Reshape for model input
    
    # Predict using the trained model
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions)
    print(f"Predicted Label: {predicted_label}")

# 6. Input your own image file
if __name__ == "__main__":
    print("Model training complete.")
    print("To test your own image, provide the image file path.")
    image_path = input("Enter the path to your handwritten digit image: ").strip()
    try:
        predict_custom_image(image_path)
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the image is in a valid format and path is correct.")
