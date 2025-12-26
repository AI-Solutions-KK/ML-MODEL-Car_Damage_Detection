import torch
from PIL import Image
from torchvision import transforms

from app.model_def import CarClassifierResNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'Front Breakage',
    'Front Crushed',
    'Front Normal',
    'Rear Breakage',
    'Rear Crushed',
    'Rear Normal'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load model ONCE
model = CarClassifierResNet(num_classes=len(CLASS_NAMES))
state_dict = torch.load("models/saved_model.pth", map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

def predict_damage(pil_image: Image.Image) -> str:
    tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        idx = int(torch.argmax(outputs, dim=1))

    return CLASS_NAMES[idx]
