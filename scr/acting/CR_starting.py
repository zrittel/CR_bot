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
import numpy as np
import time
import signal
import sys

recognizer = DigitRecognizer()
card_recognizer = CardRecognizer()

d = adb.device("192.168.240.112:5555")
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Точный запуск по Activity
# Альтернатива monkey
# d.shell('monkey -p com.supercell.clashroyale 1')


class CR_activites:
    def __init__():
        return None

    def start_CR():
        d.shell("am start -n com.supercell.clashroyale/com.supercell.titan.GameApp")
        return None

    def get_screenshot():
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = "temp/screenshot/battle_screenshot.png"

        d.shell("screencap /sdcard/screenshot.png")
        subprocess.run(
            [
                "adb",
                "-s",
                "192.168.240.112:5555",
                "pull",
                "/sdcard/screenshot.png",
                file_name,
            ]
        )
        return None

    def tap(x, y):
        d.shell(f"input tap {x} {y}")
        return None

    def swipe(x0, y0, x, y, duration=100):
        d.shell(f"input swipe {x0} {y0} {x} {y} {duration}")
        return None

    def get_cards_images():
        cards_screenshot = Image.open("temp/screenshot/battle_screenshot.png")

        cards = [0] * 4
        x = 129
        y = 850

        for i in range(0, 4):
            cards[i] = cards_screenshot.crop(
                (x + i * 108, y, x + i * 108 + 101, y + 125)
            )
            cards[i].save(f"temp/cards_img/card_{i}.png")

        return None

    def get_cards():
        CR_activites.get_cards_images()
        cards = []
        for i in range(0, 4):
            card_name, confidence = card_recognizer.predict(
                f"temp/cards_img/card_{i}.png"
            )
            cards.append(card_name)
        print(cards)
        return None

    def get_elic_image():
        elic_screenshot = Image.open("temp/screenshot/battle_screenshot.png")
        elic = elic_screenshot.crop((154, 977, 154 + 35, 977 + 27))
        elic.save("temp/elic/elic_screenshot.png")

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # elic.save(f"data/training_data/data_digits/raw/elic_{timestamp}.png")
        return None

    def get_elic_count():
        CR_activites.get_elic_image()
        elic_img = Image.open("temp/elic/elic_screenshot.png")
        digit, label, conf = recognizer.predict(elic_img)
        return label


"""
================================================

                    TESTS

================================================
"""

# print(CR_activites.get_screenshot())


# print(CR_activites.get_cards_images("20251202_153320.png"))


# print(CR_activites.get_elic_image("20251130_145126"))

# print(CR_activites.start_CR())

# print(CR_activites.get_elic_codsunt("20251130_145126"))


def signal_handler(sig, frame):
    print("\n🛑 Цикл остановлен пользователем (Ctrl+C)")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

print("🔄 Запуск цикла (Ctrl+C для остановки)...")
while True:
    time.sleep(0.2)
    CR_activites.get_screenshot()
    CR_activites.get_elic_image()
    elic = CR_activites.get_elic_count()
    print(f"Elic: {elic}")
    print(CR_activites.get_cards())
    # if elic != -1 and elic is not None:
    #     CR_activites.get_elic_image()
    #

# CR_activites.get_elic_image("20251201_185447")

"""

    TEST 2 

"""
# from PIL import Image
#
# img = Image.open("CR_screenshots/20251130_135921")
# x1, y1, x2, y2 = 100, 100, 300, 300
# cropped = img.crop((x1, y1, x2, y2))
# cropped.save("crop_result.png")  # JPEG без прозрачности
# cropped.show()
#
