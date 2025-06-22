# MNIST Handwritten Digit Classifier 🧠🖊️

A beginner-friendly deep learning project that uses a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) from the MNIST dataset.

## 🛠 Features

- Built with TensorFlow and Keras
- 98%+ accuracy on test data
- Predict digits from custom images
- Ready-to-use `predict_digit.py` script

## 📦 Installation

```bash
git clone https://github.com/your-username/mnist-digit-classifier
cd mnist-digit-classifier
pip install -r requirements.txt
```

## 🚀 Training the Model

Run the Jupyter notebook:

```bash
jupyter notebook mnist_classification.ipynb
```

## 🔍 Predicting Digits

Write any digit using `draw_digit.py`:

```bash
python draw_digit.py
```

Save 28×28 grayscale image as `digit.png` and run:

```bash
python predict_digit.py digit.png
```

## 📈 Results

- Accuracy: ~98.5% on MNIST test set
- Model saved as `mnist_cnn.h5`

## 🧠 Concepts Used

- CNNs (Conv2D, MaxPooling)
- One-hot encoding
- Normalization
- Model evaluation and saving

## 🌐 Web App with Drawing Interface

In addition to the CLI script, this project includes a Streamlit-based web app `app.py` that allows users to:

- 📂 Upload a custom 28×28 grayscale digit image
- ✍️ Draw a digit directly in the browser using a canvas
- 🧠 Instantly view predictions from the trained model

## ▶️ Running the Web App

Run the app:

```bash
streamlit run app.py
```

## 🖼 Interface Features

- Upload Mode: Upload a `.png`, `.jpg`, or `.jpeg` image (resized to 28×28).
- Draw Mode: Use the mouse or touchscreen to draw a digit on a 280×280 canvas.
- Automatically resizes and processes the drawing for prediction.
- Displays the predicted digit and the input image for confirmation.

## 🧩 Powered by:

- `streamlit-drawable-canvas` for real-time drawing
- TensorFlow/Keras model trained on the MNIST dataset
