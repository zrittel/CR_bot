import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from scr.ai.card_recognizer_model import CardCNN

CARD_CLASSES = {
    0: "None",
    1: "archers",
    2: "giant",
    3: "goblin_cage",
    4: "goblins",
    5: "knight",
    6: "mini_PEKKA",
    7: "minions",
    8: "musketeer",
}


class CardRecognizer:
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device("cpu")
        self.model = CardCNN(num_classes=9).to(self.device)

        if model_path is None:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            model_path = os.path.join(project_root, "data", "models", "card_model.pth")

        print(f"🔍 Загружаю модель карт: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель карт не найдена: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def predict(self, img_path_or_pil):
        """Предсказывает название карты"""
        if isinstance(img_path_or_pil, str):
            img = Image.open(img_path_or_pil).convert("RGB")
        else:
            img = (
                img_path_or_pil.convert("RGB")
                if img_path_or_pil.mode != "RGB"
                else img_path_or_pil
            )

        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            pred_idx = output.argmax(1).item()
            confidence = F.softmax(output, dim=1)[0, pred_idx].item()

        card_name = CARD_CLASSES[pred_idx]
        return card_name, confidence

    def predict_batch(self, img_paths_or_pils):
        """Предсказывает для нескольких карт сразу"""
        results = []
        for img in img_paths_or_pils:
            card_name, confidence = self.predict(img)
            results.append((card_name, confidence))
        return results
