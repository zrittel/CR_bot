import os

os.environ["ROCBLAS_LAYER"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

from ultralytics import YOLO
from PIL import Image

model = YOLO("runs/detect/train11/weights/best.pt")

# Загрузи изображение
img = Image.open(
    "data/training_data/data_arena/images/val/arena_screenshot_20251204_121530.png"
)

# Детекция + автоматическое рисование bbox
results = model(img, save=True, show=True)

results[0].show()  # покажет окно с bbox
# results[0].save('result_with_bbox.jpg')  # сохранит с bbox

