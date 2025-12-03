from adbutils import adb
import subprocess
from datetime import datetime
from PIL import Image
from elic_recognizer import DigitRecognizer
import numpy as np
import os
import time


recognizer = DigitRecognizer("digit_model_35x27.pth")

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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"temp/screenshot/{timestamp}.png"

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
        return f"{timestamp}.png"

    def tap(x, y):
        d.shell(f"input tap {x} {y}")
        return None

    def swipe(x0, y0, x, y, duration=100):
        d.shell(f"input swipe {x0} {y0} {x} {y} {duration}")
        return None

    def get_cards_images(file_name):
        cards_screenshot = Image.open(f"temp/screenshot/{file_name}")

        cards = [0] * 4
        x = 129
        y = 850

        for i in range(0, 4):
            cards[i] = cards_screenshot.crop(
                (x + i * 108, y, x + i * 108 + 101, y + 125)
            )
            cards[i].save(f"temp/cards_img/card_{i}.png")

        return None

    def get_elic_image(file_name):
        elic_screenshot = Image.open(f"temp/screenshot/{file_name}")
        elic = elic_screenshot.crop((154, 977, 154 + 35, 977 + 27))
        elic.save("temp/elic/elic_screenshot.png")

        return None

    def get_elic_count(file_name):
        CR_activites.get_elic_image(file_name)
        elic_img = Image.open("temp/elic/elic_screenshot.png")
        digit, conf = recognizer.predict(elic_img)
        return digit - 1 if conf >= 0.7 else None


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


for _ in range(60):
    time.sleep(0.2)
    print(CR_activites.get_elic_count(CR_activites.get_screenshot()))

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
