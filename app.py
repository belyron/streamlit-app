import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import streamlit as st
import os

# Step 1: Define the EnhancedModel class
class EnhancedModel(nn.Module):
    def __init__(self, n_classes: int):
        super(EnhancedModel, self).__init__()
        
        # Load the pretrained ResNet18
        self.resnet = resnet18(pretrained=True)

        # Replace the original fc layer with a new one for classification
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, n_classes)

    def forward(self, x):
        return self.resnet(x)

# Step 2: Define the model loading function
def load_model(n_classes, device):
    model = EnhancedModel(n_classes=n_classes)
    model.to(device)
    model_path = 'enhanced_model.pth'
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        state_dict.pop('resnet.fc.weight', None)  # Remove fc layer weights to avoid mismatch
        state_dict.pop('resnet.fc.bias', None)
        model.load_state_dict(state_dict, strict=False)
        model.eval()  # Set the model to evaluation mode
    else:
        st.error(f"Ошибка: файл '{model_path}' не найден. Проверьте путь и наличие файла.")
    return model

# Step 3: Define image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match ResNet18 input size
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize(           # Normalize with ImageNet values
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Step 4: Define the Streamlit app
def main():
    st.title('Enhanced Model Deployment with Streamlit')
    st.write('Upload an image to predict its class')

    # Step 5: Add an image uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Step 6: Display the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Step 7: Load the model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(n_classes=6, device=device)  # Change n_classes to your actual number of classes

        # Step 8: Preprocess the image
        input_tensor = preprocess_image(image).to(device)

        # Step 9: Make a prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class = torch.max(output, 1)
        
        # Step 10: Display the result
        label_name_to_label_id = {'in': 0, 'pi': 1, 'cr': 2, 'pa': 3, 'sc': 4, 'ro': 5}
        label_id_to_name = {v: k for k, v in label_name_to_label_id.items()}
        st.write(f'Predicted Class: {label_id_to_name[predicted_class.item()]}')

# Step 11: Run the Streamlit app
if __name__ == '__main__':
    main()
