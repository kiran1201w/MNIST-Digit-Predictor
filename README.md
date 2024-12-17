# 🖊️ MNIST Handwritten Digit Recognition

This project is a simple implementation of a feedforward neural network using TensorFlow/Keras to recognize handwritten digits (0-9) from the MNIST dataset. The project also allows users to input their own digit images for prediction.

---

## 📚 Table of Contents
- [📖 Overview](#overview)
- [✨ Features](#features)
- [⚙️ Requirements](#requirements)
- [🚀 Setup Instructions](#setup-instructions)
- [🧠 How It Works](#how-it-works)
- [✍️ Custom Input](#custom-input)
- [🛠️ Troubleshooting](#troubleshooting)

---

## 📖 Overview
The MNIST dataset is a collection of 28x28 pixel grayscale images of handwritten digits. This project trains a neural network to classify these images into their respective digit labels (0-9).

---

## ✨ Features
- **🧩 Neural Network Architecture**: 
   - Input: Flattened 28x28 image.
   - Two hidden layers with ReLU activation.
   - Output: 10 classes with softmax activation.
- **📊 Trainable on MNIST**: Trains on the MNIST dataset.
- **🖼️ Custom Image Prediction**: Allows users to input their own digit image for prediction.
- **📈 Model Accuracy Display**: Displays test accuracy after training.

---

## ⚙️ Requirements
To run the project, install the following dependencies:

- 🐍 Python 3.8+ (3.9 or higher recommended)
- 🧠 TensorFlow 2.x (tested on TensorFlow 2.9.1)
- 🔢 NumPy
- 📊 Matplotlib
- 🖼️ Pillow (PIL)

Install dependencies using pip:
```bash
pip install tensorflow numpy matplotlib pillow
```

---

## 🚀 Setup Instructions
1. **🔗 Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/mnist-digit-predictor.git
   cd mnist-digit-predictor
   ```
2. **▶️ Run the Script**:
   Execute the following command to train the model and make predictions:
   ```bash
   python ml.py
   ```
3. **🖌️ Custom Image Input**:
   Prepare a handwritten digit image (28x28 pixels or larger) with a white background and black digit.

---

## 🧠 How It Works
1. **🏋️ Training**:
   - The MNIST dataset is loaded and normalized (pixel values scaled to [0, 1]).
   - A feedforward neural network is trained using the Adam optimizer and sparse categorical cross-entropy loss.
2. **🔍 Prediction**:
   - The user inputs the path to a custom handwritten digit image.
   - The image is preprocessed:
      - Converted to grayscale.
      - Resized to 28x28 pixels.
      - Pixel values are normalized.
   - The trained model predicts the digit label.
3. **📤 Output**:
   - The predicted label is displayed alongside the input image.

---

## ✍️ Custom Input
To test your own handwritten digit image:
1. Create a black digit on a white background (use any image editing tool).
2. Save it in `.png` or `.jpg` format.
3. Provide the image path when prompted in the script.

**Example**:
```
Enter the path to your handwritten digit image: /path/to/image.png
Predicted Label: 5
```

---

## 🛠️ Troubleshooting
### ⚠️ Common Issues:
1. **🐍 TensorFlow Compatibility Error**:
   - Ensure TensorFlow version is 2.9.1 or compatible with your Python version.
   - Use Python 3.8 or higher.

   Install the correct version:
   ```bash
   pip install tensorflow==2.9.1
   ```
2. **🖼️ Custom Image Errors**:
   - Ensure the image is grayscale and has a clear digit with minimal noise.
   - Resize the image to approximately 28x28 pixels for better predictions.

3. **⚙️ oneDNN Optimizations Warning**:
   - To suppress TensorFlow warnings, set the environment variable:
     ```bash
     export TF_ENABLE_ONEDNN_OPTS=0  # For Linux/macOS
     set TF_ENABLE_ONEDNN_OPTS=0     # For Windows
     ```

---

## 👤 Author
**Your Name**  
GitHub: [@kiran1201w](https://github.com/kiran1201w)

---

## 📝 License
This project is licensed under the MIT License.
