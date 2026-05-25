import os
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn

# model uses cpu for inferences 
device = torch.device("cpu")
# cnn model class
class CNNDigitRecognition(nn.Module):
    def __init__(self):
        super(CNNDigitRecognition, self).__init__()
        # convolation
        # block 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # flatten
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.relu2(self.bn1(self.conv2(x))))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# model and model path
model = CNNDigitRecognition()
MODEL_PATH = "mnist_cnn.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# prediction function for gradio
def predict_digit(input_image):
    if input_image is None:
        return None, None, "Please upload or draw an image."

    # Converting input image (RGB NumPy array) to Grayscale
    if len(input_image.shape) == 3:
        img_gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = input_image

    # Dynamic Thresholding to eliminate paper shadows
    _, thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Inverting background colors if canvas area is predominantly white
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    # Detecting bounding regions and crop to isolate the digit layout
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        pad = int(max(w, h) * 0.15)
        h_img, w_img = img_gray.shape
        y_start, y_end = max(0, y - pad), min(h_img, y + h + pad)
        x_start, x_end = max(0, x - pad), min(w_img, x + w + pad)
        cropped = thresh[y_start:y_end, x_start:x_end]
    else:
        cropped = thresh

    # Scaling down matrix dimensions to fit standard MNIST input requirements
    img_resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalising to match original dataset statistics
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_normalized - 0.1307) / 0.3081

    # Converting format layout into 4D tensor matrix array structure
    img_tensor = (
        torch.tensor(img_normalized).unsqueeze(0).unsqueeze(0).to(device)
    )

    # Run inference metrics calculation
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1).flatten()

    # Format probability outputs for Gradio's Label engine interface component
    predictions_dict = {str(i): float(probabilities[i]) for i in range(10)}

    # RETURN has THREE ELEMENTS: [Original RGB Image, Processed 28x28 Matrix, Predictions Label]
    return input_image, img_resized, predictions_dict


with gr.Blocks(title="Handwritten Digit Predictor") as digit_predictor:
    gr.Markdown("# Handwritten Digit Classifier CNN")
    gr.Markdown(
        "Upload a photo of any handwritten digit or use your mouse/finger to draw. "
    )

    with gr.Row():
        # Input Section on Left
        with gr.Column():
            input_component = gr.Image(
                label="Draw or Upload Image Matrix", type="numpy"
            )
            submit_btn = gr.Button("Predict Digit", variant="primary")

        # Output Section on Right
        with gr.Column():
            with gr.Row():
                out_original = gr.Image(label="Your Input Visual Target")
                out_processed = gr.Image(
                    label="Internal CNN Matrix Vision (28x28)", cmap="gray"
                )
            out_label = gr.Label(num_top_classes=3, label="Classifier Output Scores")

    # Connect components to execution trigger event logic loops
    submit_btn.click(
        fn=predict_digit,
        inputs=input_component,
        outputs=[out_original, out_processed, out_label],
    )

# Launching the engine interface
if __name__ == "__main__":
    digit_predictor.launch()
