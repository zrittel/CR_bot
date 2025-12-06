"""
=============================================================================================

                                        version 2

=============================================================================================
"""

import os

os.environ["ROCBLAS_LAYER"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"


import sys
from pathlib import Path

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adbutils import adb
import subprocess
from datetime import datetime
from PIL import Image

from scr.ai.elic_recognizer import DigitRecognizer
from scr.ai.card_recognizer import CardRecognizer
from ultralytics import YOLO

import numpy as np
import time
import signal


class CRActions:
    def __init__():
        pass

    def tap(self, x, y):
        self.device.shell(f"input tap {x} {y}")
        return None

    def swipe(self, x0, y0, x, y, duration=100):
        self.device.shell(f"input swipe {x0} {y0} {x} {y} {duration}")
        return None


class CRRecognizer:
    def _extract_card_img(self):
        cards = []
        img = Image.open(self.SCREENSHOT_PATH)
        for id, (x, y) in enumerate(self.CARD_POSITIONS):
            w, h = self.CARD_SIZE
            card = img.crop((x, y, x + w, y + h))
            path = Path(f"temp/cards_img/card_{id}.png")
            card.save(path)
            cards.append((card, path))
        return cards

    def get_cards(self):
        cards_img = self._extract_card_img()
        result = []
        for i, (_, path) in enumerate(cards_img):
            name, conf = self.card_recognizer.predict(str(path))
            result.append(
                {"index": i, "name": name, "confidence": conf, "path": str(path)}
            )
        return result

    # ==================================================================================

    def _extract_elixir_img(self, screenshot: Image.Image) -> Image.Image:
        x1, y1, x2, y2 = self.ELIC_CROP
        elic = screenshot.crop((x1, y1, x2, y2))
        elic.save("temp/elic/elic_screenshot.png")
        return elic

    def get_elixir(self):
        screenshot = self.get_raw_screenshot()
        elic = self._extract_elixir_image(screenshot)
        _, label, conf = self.digit_recognizer.predict(elic)
        mana = int(label) if label != "None" else 0
        return {"elixir": mana, "confidence": conf}

    # ==================================================================================

    def _extract_arena_image(self) -> Image.Image:
        screenshot = Image.open(self.SCREENSHOT_PATH)
        x1, y1, x2, y2 = self.ARENA_CROP
        arena = screenshot.crop((x1, y1, x2, y2))
        arena.save("temp/arena_img/arena_screenshot.png")
        return arena

    def get_arena_objects(self):
        arena = self._extract_arena_image(self.SCREENSHOT_PATH)

        results = self.arena_detector(arena, verbose=False, imgsz=800)
        objects = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                conf = float(box.conf[0].cpu())
                cls = int(box.cls[0].cpu())

                objects.append(
                    {
                        "type": self.arena_detector.names[cls],
                        "confidence": conf,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    }
                )
        return objects

    def get_arena_state(self):
        """Удобный сбор всего состояния разом."""
        screenshot = self.get_raw_screenshot()

        # Карты
        card_imgs = self._extract_card_img()
        cards = []
        for i, (_, path) in enumerate(card_imgs):
            name, conf = self.card_recognizer.predict(str(path))
            cards.append({"index": i, "name": name, "confidence": conf})

        # Эликсир
        elic = self._extract_elixir_img(screenshot)
        _, label, conf_e = self.digit_recognizer.predict(elic)
        mana = int(label) if label != "None" else 0

        # Объекты
        arena = self._extract_arena_image()
        results = self.arena_detector(arena, verbose=False, imgsz=800)
        objects = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                conf = float(box.conf[0].cpu())
                cls = int(box.cls[0].cpu())
                objects.append(
                    {
                        "type": self.arena_detector.names[cls],
                        "confidence": conf,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                    }
                )

        return {
            "elixir": {"elixir": mana, "confidence": conf_e},
            "cards": cards,
            "objects": objects,
        }


class CRBot(CRActions, CRRecognizer):
    """main class to control CR"""

    # const
    ADB_DEVICE = "192.168.240.112:5555"
    PACKAGE = "com.supercell.clashroyale"
    ACTIVITY = "com.supercell.titan.GameApp"

    # UI coords
    CARD_CROP = (129, 850, 129 + 4 * 108 + 1, 850 + 125)  # x, y, x+w, y+h
    CARD_POSITIONS = [(129 + i * 108, 850) for i in range(4)]
    CARD_SIZE = (101, 125)

    ELIC_CROP = (154, 977, 154 + 35, 977 + 27)
    ARENA_CROP = (0, 0, 576, 800)

    SCREENSHOT_PATH = "temp/screenshot/battle_screenshot.png"
    CARDS_DIR = "temp/cards_img"
    ELIC_DIR = "temp/elic"
    ARENA_DIR = "temp/arena_img"

    def __init__(self, model_path="runs/detect/train11/weights/best.pt"):
        """Bot initialization"""
        self.device = adb.device(self.ADB_DEVICE)
        self.digit_recognizer = DigitRecognizer()
        self.card_recognizer = CardRecognizer()
        self.arena_detector = YOLO(model_path)

        # Ceck and create dir
        self._setup_directories()

        print("✓ CRBot initializated")

    def _setup_directories(self):
        """creating dir"""
        for directory in [
            Path(self.SCREENSHOT_PATH).parent,
            Path(self.CARDS_DIR),
            Path(self.ELIC_DIR),
            Path(self.ARENA_DIR),
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def start_CR(self):
        self.device.shell(f"am start -n {self.PACKAGE}/{self.ACTIVITY}")
        print("CR starting | waiting 10s")
        time.sleep(10)

    def get_raw_screenshot(self) -> Image.Image:
        self.device.shell("screencap /sdcard/screenshot.png")
        subprocess.run(
            [
                "adb",
                "-s",
                "192.168.240.112:5555",
                "pull",
                "/sdcard/screenshot.png",
                self.SCREENSHOT_PATH,
            ]
        )
        return Image.open(self.SCREENSHOT_PATH)


"""
================================================

                    TESTS

================================================
"""

test_bot = CRBot()
# print(base_bot.get_arena_state())


def signal_handler(sig, frame):
    print("\n🛑 Цикл остановлен пользователем (Ctrl+C)")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

print("🔄 Запуск цикла (Ctrl+C для остановки)...")
while True:
    time.sleep(1)
    print(test_bot.get_arena_state())


"""
=============================================================================================

                                        version 1

=============================================================================================
"""

# import os
#
# os.environ["ROCBLAS_LAYER"] = "0"
# os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
#
# import sys
# from pathlib import Path
#
# # Добавляем корень проекта в PYTHONPATH
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))
#
# from adbutils import adb
# import subprocess
# from datetime import datetime
# from PIL import Image
#
# from scr.ai.elic_recognizer import DigitRecognizer
# from scr.ai.card_recognizer import CardRecognizer
# from ultralytics import YOLO
#
# import numpy as np
# import time
# import signal
# import sys
#
# recognizer = DigitRecognizer()
# card_recognizer = CardRecognizer()
# model = YOLO("runs/detect/train11/weights/best.pt")
#
#
# d = adb.device("192.168.240.112:5555")
# # pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
#
# # Точный запуск по Activity
# # Альтернатива monkey
# # d.shell('monkey -p com.supercell.clashroyale 1')
#
#
# class CR_activites:
#     def __init__():
#         return None
#
#     def start_CR():
#         d.shell("am start -n com.supercell.clashroyale/com.supercell.titan.GameApp")
#         return None
#
#     def get_screenshot():
#         # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         file_name = "temp/screenshot/battle_screenshot.png"
#
#         d.shell("screencap /sdcard/screenshot.png")
#         subprocess.run(
#             [
#                 "adb",
#                 "-s",
#                 "192.168.240.112:5555",
#                 "pull",
#                 "/sdcard/screenshot.png",
#                 file_name,
#             ]
#         )
#         return None
#
#     def tap(x, y):
#         d.shell(f"input tap {x} {y}")
#         return None
#
#     def swipe(x0, y0, x, y, duration=100):
#         d.shell(f"input swipe {x0} {y0} {x} {y} {duration}")
#         return None
#
#     def get_cards_images():
#         cards_screenshot = Image.open("temp/screenshot/battle_screenshot.png")
#
#         cards = [0] * 4
#         x = 129
#         y = 850
#
#         for i in range(0, 4):
#             cards[i] = cards_screenshot.crop(
#                 (x + i * 108, y, x + i * 108 + 101, y + 125)
#             )
#             cards[i].save(f"temp/cards_img/card_{i}.png")
#
#         return None
#
#     def get_cards():
#         CR_activites.get_cards_images()
#         cards = []
#         for i in range(0, 4):
#             card_name, confidence = card_recognizer.predict(
#                 f"temp/cards_img/card_{i}.png"
#             )
#             cards.append(card_name)
#         print(cards) return None
#
#     def get_elic_image():
#         elic_screenshot = Image.open("temp/screenshot/battle_screenshot.png")
#         elic = elic_screenshot.crop((154, 977, 154 + 35, 977 + 27))
#         elic.save("temp/elic/elic_screenshot.png")
#
#         # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         # elic.save(f"data/training_data/data_digits/raw/elic_{timestamp}.png")
#         return None
#
#     def get_elic_count():
#         CR_activites.get_elic_image()
#         elic_img = Image.open("temp/elic/elic_screenshot.png")
#         digit, label, conf = recognizer.predict(elic_img)
#         return label
#
#     def get_arena_screenshot():
#         arena_screenshot = Image.open("temp/screenshot/battle_screenshot.png")
#         arena_screenshot = arena_screenshot.crop((41, 7, 41 + 494, 7 + 784))
#         arena_screenshot.save(f"temp/arena_img/arena_screenshot.png")
#         return arena_screenshot
#
#     def get_arena_status():
#         arena_screenshot = CR_activites.get_arena_screenshot()
#         results = self.arena_detector(screenshot, verbose=False, imgsz=784)
#         objects = []
#
#         for r in results:
#             if r.boxes is not None:
#                 for box in r.boxes:
#                     x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
#                     conf = box.conf[0].item()
#                     cls = int(box.cls[0].item())
#
#                     objects.append(
#                         {
#                             "type": self.arena_detector.names[cls],
#                             "confidence": conf,
#                             "bbox": [int(x1), int(y1), int(x2), int(y2)],
#                             "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
#                         }
#                     )
#
#         return objectsk
#
