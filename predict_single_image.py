import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
from models.resnet_model import get_resnet_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class labels in the same order as your training folders
class_names = ['bleached', 'healthy']

# Load model
model = get_resnet_model().to(device)
model.load_state_dict(torch.load('resnet_coral_classifier.pth', map_location=device))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob > 0.5 else 0
        confidence = prob if prediction == 1 else 1 - prob
        print(f"\n🔍 Prediction: {class_names[prediction].upper()} ✅")
        print(f"📊 Confidence: {confidence*100:.2f}%\n")

# Main execution
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("❌ Usage: python predict_single_image.py path_to_image")
    else:
        predict_image(sys.argv[1])
