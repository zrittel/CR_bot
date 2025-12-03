import os
import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Метки классов (0-10 + None)
DIGIT_CLASSES = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "None",
}


class DigitsDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.samples = []
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((35, 27)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        # Загружаем папки 0-10 и None
        for label in range(12):  # 0-11 (12 классов)
            if label == 11:
                folder_name = "None"
            else:
                folder_name = str(label)

            d = os.path.join(root, folder_name)
            if not os.path.exists(d):
                print(f"⚠️  Папка {folder_name} не найдена")
                continue

            count = 0
            for fname in os.listdir(d):
                if fname.lower().endswith(".png"):
                    self.samples.append((os.path.join(d, fname), label))
                    count += 1

            if count > 0:
                print(f"✓ {folder_name}: {count} изображений")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path)
        img = self.transform(img)
        return img, label


class DigitCNN35x27(nn.Module):
    def __init__(self, num_classes=12):  # 12 классов (0-10 + None)
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 6, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(dataloader, model, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)
    accuracy = correct / total if total > 0 else 0
    return accuracy


def predict_single(model, img_tensor, device):
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(device))
        pred = output.argmax(1).item()
        confidence = F.softmax(output, dim=1)[0, pred].item()
    return pred, confidence


def main():
    device = torch.device("cpu")
    print(f"Device: {device}\n")

    dataset = DigitsDataset("data/training_data/data_digits")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Dataset size: {len(dataset)} images\n")

    if len(dataset) == 0:
        print("❌ Ошибка: датасет пуст!")
        return

    # 12 классов (0-10 + None)
    model = DigitCNN35x27(num_classes=12).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    print("=== Обучение (100 эпох, 12 классов) ===")
    for epoch in range(epochs):
        loss = train_epoch(dataloader, model, loss_fn, optimizer, device)
        acc = evaluate(dataloader, model, device)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {loss:.4f} | Acc: {acc:.2%}")

    # Сохранение модели
    os.makedirs("data/models", exist_ok=True)
    torch.save(model.state_dict(), "data/models/digit_model_35x27.pth")
    print("\n✓ Модель сохранена: data/models/digit_model_35x27.pth\n")

    final_acc = evaluate(dataloader, model, device)
    print(f"Финальная точность: {final_acc:.2%}\n")

    print("=== Предсказания ===")
    model.eval()
    correct_count = 0
    for idx, (img, label) in enumerate(dataset):
        pred, conf = predict_single(model, img, device)
        real_label = DIGIT_CLASSES.get(label, "Unknown")
        pred_label = DIGIT_CLASSES.get(pred, "Unknown")
        is_correct = pred == label
        status = "✓" if is_correct else "✗"
        if is_correct:
            correct_count += 1
        print(
            f"{status} Изображение {idx}: реальная={real_label}, предсказание={pred_label}, уверенность={conf:.2%}"
        )

    print(
        f"\nИтого: {correct_count}/{len(dataset)} правильно ({correct_count / len(dataset):.2%})"
    )


if __name__ == "__main__":
    main()

