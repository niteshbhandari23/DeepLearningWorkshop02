import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.title("Cats vs Dogs vs Pandas Classifier")
st.write("Upload an image of a cat, dog, or panda, and the model will classify it.")

# Define class names
class_names = ['cat', 'dog', 'panda']

# Load the model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, len(class_names))
    )
    # Ensure you have 'best_model.pth' trained and saved in your directory.
    # We use map_location=torch.device('cpu') so it runs anywhere
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.warning("Model weights 'best_model.pth' not found. Please train the model first.")
        return None
    model.eval()
    return model

model = load_model()

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    if model is not None:
        st.write("Classifying...")
        # Preprocess the image
        input_tensor = transform(image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        
        # Predict
        with torch.no_grad():
            output = model(input_batch)
            _, predicted_idx = torch.max(output, 1)
            predicted_class = class_names[predicted_idx.item()]
            
        st.success(f"Prediction: **{predicted_class.capitalize()}**")
