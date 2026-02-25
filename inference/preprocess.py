from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def preprocess(file, device):
    img = Image.open(file).convert("RGB")
    return transform(img).unsqueeze(0).to(device)
