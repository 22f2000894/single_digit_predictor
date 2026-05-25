# Single Digit Classifier (PyTorch CNN + Gradio)

**Convolutional Neural Network (CNN)** trained on the MNIST dataset using **PyTorch**, and  web UI deployment built with **Gradio**. 

---

## 🚀 Live Demo on Hugging Face Spaces
You can test your own handwritten images or draw directly on the browser using the live application here:
🔗 **https://huggingface.co/spaces/22f2000894/single_digit_predictor**

---

## ✨ Features

* **Auto-Cropping & Bounding Boxes**: Automatically isolates and extracts the handwritten digit from raw photos with giant borders or off-center placements.
* **Otsu's Adaptive Thresholding**: Cleans up grey shadows, phone lens artifacts, and varying paper illumination conditions.
* **Dynamic Background Inversion**: Smart background checking translates black ink on white paper into standard white ink on a black matrix background seamlessly.
* **Interactive Side-by-Side UI**: Displays the original photo, the internal 28x28 processed matrix the model actually evaluates, and top prediction probabilities side-by-side.

---

## 🏗️ Model Architecture

**Convolutional Neural Network (CNN)** has:

1. **Convolution Block 1**: `Conv2d` (1 -> 32 channels, 3x3 kernel) + `BatchNorm2d` + `ReLU` + `MaxPool2d` (shrinks input to 14x14)
2. **Convolution Block 2**: `Conv2d` (32 -> 64 channels, 3x3 kernel) + `BatchNorm2d` + `ReLU` + `MaxPool2d` (shrinks input to 7x7)
3. **Classification Dense Head**: `Flatten` (64 * 7 * 7 = 3,136 features) -> `Linear` (128 units) -> `ReLU` -> `Dropout` (0.4 rate to prevent overfitting) -> `Linear` (10 output logits).

---

## (Data Augmentation)
To make this model robust against imperfect real-world camera inputs, training was upgraded with dynamic **Torchvision Transformations**:
* **Random Rotation (15°)**: Emulates tilted handwriting.
* **Random Affine Shifting & Scaling**: Emulates varied phone camera scaling zoom and off-center margins.

---

## 💻 How to Run Locally

### 1. Clone the Space or Repository
```bash
git clone <your-repository-url>
cd single_digit_predictor
```

### 2. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Application
Ensure your trained model weight file (`mnist_cnn.pth`) is present in the root directory, then execute:
```bash
python app.py
```
