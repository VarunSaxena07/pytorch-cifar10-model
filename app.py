import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import CNN

# CIFAR10 classes
classes = [
'airplane','automobile','bird','cat','deer',
'dog','frog','horse','ship','truck'
]

# Load model
model = CNN()
model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))
model.eval()

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🧠",
    layout="centered"
)
st.sidebar.title("About this Project")

st.sidebar.info(
"""
This is a **beginner-friendly CNN project** trained on the **CIFAR-10 dataset**.

The goal of this project is to demonstrate the **complete deep learning workflow**:

• Building a Convolutional Neural Network  
• Training and validation in PyTorch  
• Saving the best model  
• Creating an interactive web app with Streamlit  

⚠️ Since CIFAR-10 images are very small (32×32), the model may not perform well on high-resolution real-world images.

This project is intended **for learning and demonstration purposes**.
"""
)
st.sidebar.subheader("**Model Details**")

st.sidebar.write("""
Architecture: Custom CNN  
Framework: PyTorch  
Dataset: CIFAR-10  
Classes: 10  
Training Epochs: 10
""")
st.title("🧠 CIFAR-10 Image Classifier")
st.write("Upload an image and the CNN will predict the class.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)

        # Convert logits → probabilities
        probs = F.softmax(outputs, dim=1)

        # Get prediction
        confidence, predicted = torch.max(probs, 1)

    prediction = classes[predicted.item()]
    confidence_score = confidence.item() * 100

    st.success(f"Prediction: **{prediction}**")
    st.write(f"Confidence: **{confidence_score:.2f}%**")