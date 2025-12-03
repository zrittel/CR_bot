import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


class CardDataset(Dataset):
    def __init__(self, root):
        self.samples = []
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        for label, card_name in CARD_CLASSES.items():
            d = os.path.join(root, card_name)
            if not os.path.exists(d):
                print(f"⚠️  Папка {card_name} не найдена")
                continue

            for fname in os.listdir(d):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(d, fname), label))

            print(
                f"✓ {card_name}: {len([s for s in self.samples if s[1] == label])} изображений"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    dataset = CardDataset("data/training_data/data_cards")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    print(f"Dataset size: {len(dataset)} изображений\n")

    if len(dataset) == 0:
        print("Ошибка: датасет пуст!")
        return

    model = CardCNN(num_classes=9).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    print("=== Обучение карт (50 эпох) ===")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(imgs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Валидация
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs, labels = imgs.to(device), labels.to(device)
                output = model(imgs)
                correct += (output.argmax(1) == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:2d}/{epochs} | Loss: {total_loss / len(dataloader):.4f} | Acc: {accuracy:.2%}"
            )

    # Сохранение
    os.makedirs("data/models", exist_ok=True)
    torch.save(model.state_dict(), "data/models/card_model.pth")
    print("\n✓ Модель сохранена: data/models/card_model.pth")


if __name__ == "__main__":
    train()
