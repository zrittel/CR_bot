from ultralytics import YOLO


class GameBot:
    def __init__(self):
        self.arena_detector = YOLO("runs/detect/arena_final/weights/best.pt")

    def analyze_arena(self, screenshot):
        results = self.arena_detector(screenshot, verbose=False, imgsz=784)
        objects = []

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    objects.append(
                        {
                            "type": self.arena_detector.names[cls],
                            "confidence": conf,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                        }
                    )

        return objects


print(
    GameBot.analyze_arena(
        "data/training_data/data_arena/images/val/arena_screenshot_20251204_121530.png"
    )
)
