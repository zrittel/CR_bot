import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class DigitCNN35x27(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 6, 64)
        self.fc2 = nn.Linear(64, 11)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DigitRecognizer:
    def __init__(self, model_path="digit_model_35x27.pth", device=None):
        self.device = device or torch.device("cpu")
        self.model = DigitCNN35x27().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((35, 27)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def predict(self, img_path_or_pil):
        if isinstance(img_path_or_pil, str):
            img = Image.open(img_path_or_pil)
        else:
            img = img_path_or_pil

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            pred = output.argmax(1).item()
            confidence = F.softmax(output, dim=1)[0, pred].item()

        return pred + 1, confidence
