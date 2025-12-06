import os

os.environ["ROCBLAS_LAYER"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adbutils import adb
import subprocess
from PIL import Image
from ultralytics import YOLO
import time
from collections import Counter

from scr.ai.elic_recognizer import DigitRecognizer
from scr.ai.card_recognizer import CardRecognizer

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import signal


# ==================== PRINTER ====================

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from collections import Counter
import time
import sys
import signal


class GameStatePrinter:
    def __init__(self):
        self.console = Console()

    def print_game_state(self, state):
        """Красиво выводит состояние игры (один раз)"""
        # Очищаем экран
        self.console.clear()

        # Главная панель с одной таблицей
        panel = Panel(
            self._build_game_table(state),
            title="[bold cyan]🎮 CLASH ROYALE BOT[/]",
            border_style="bright_blue",
            padding=(1, 2),
            expand=False,
        )
        self.console.print(panel)

    def _build_game_table(self, state):
        """Создаёт таблицу со ВСЕМ состоянием"""
        table = Table(box=box.ROUNDED, padding=(0, 1))
        table.add_column("", style="")
        table.add_column("Информация", style="")
        table.add_column("Значение", style="", justify="right")

        # Мана
        table.add_row(
            "💎",
            "Эликсир",
            f"[bold green]{state['elixir']['elixir']}[/] ({state['elixir']['confidence']:.1%})",
        )

        # Карты - все в одну строку
        cards_text = " | ".join(
            [
                f"[cyan]{c['name']}[/][green]({c['confidence']:.0%})[/]"
                for c in state["cards"]
            ]
        )
        table.add_row("🎴", "Карты в руке", cards_text)

        # Разделяем объекты на союзных и вражеских
        ally_objects = [obj for obj in state["objects"] if obj["type"].startswith("A_")]
        enemy_objects = [
            obj for obj in state["objects"] if obj["type"].startswith("E_")
        ]

        # Союзные
        if ally_objects:
            ally_text = self._format_objects_row(ally_objects)
            table.add_row(
                "🛡️ ", "Союзные войска", f"{ally_text} [dim]({len(ally_objects)})[/]"
            )

        # Вражеские
        if enemy_objects:
            enemy_text = self._format_objects_row(enemy_objects)
            table.add_row(
                "⚔️ ",
                "Враждебные войска",
                f"{enemy_text} [dim]({len(enemy_objects)})[/]",
            )

        # Если арена пуста
        if not ally_objects and not enemy_objects:
            table.add_row("📍", "Арена", "[dim]Пусто[/]")

        return table

    def _format_objects_row(self, objects):
        """Форматирует группу объектов в одну строку"""
        formatted = []
        for obj in objects:
            # Выбираем цвет в зависимости от типа
            if obj["type"].startswith("A_"):
                color = "green"  # Союзные - зелёные
            elif obj["type"].startswith("E_"):
                color = "red"  # Враги - красные
            else:
                color = "yellow"

            # Сокращённое название юнита
            short_name = self._shorten_unit_name(obj["type"])
            formatted.append(f"[{color}]{short_name}[/][dim]{obj['confidence']:.0%}[/]")

        return " | ".join(formatted)

    def _shorten_unit_name(self, full_name):
        """Сокращает название юнита"""
        short = full_name
        for prefix in ["A_U_", "E_U_", "A_B_", "E_B_", "A_", "E_"]:
            if short.startswith(prefix):
                short = short[len(prefix) :]
                break

        short = short.replace("-", " ").replace("_", " ")
        return short.title()

    def print_arena_details(self, objects):
        """Подробный вывод объектов на арене с разделением"""
        if not objects:
            self.console.print("[yellow]⚠️  Объектов не обнаружено[/]")
            return

        # Разделяем на союзных и враждебных
        ally_objects = [obj for obj in objects if obj["type"].startswith("A_")]
        enemy_objects = [obj for obj in objects if obj["type"].startswith("E_")]

        # Таблица союзных
        if ally_objects:
            self.console.print("[bold green]🛡️  СОЮЗНЫЕ ВОЙСКА[/]")
            ally_table = self._create_objects_table(ally_objects, "green")
            self.console.print(ally_table)
            self.console.print()

        # Таблица враждебных
        if enemy_objects:
            self.console.print("[bold red]⚔️  ВРАЖДЕБНЫЕ ВОЙСКА[/]")
            enemy_table = self._create_objects_table(enemy_objects, "red")
            self.console.print(enemy_table)
            self.console.print()

    def _create_objects_table(self, objects, color):
        """Создаёт таблицу для группы объектов"""
        table = Table(
            title=f"Всего: {len(objects)} юнитов",
            show_header=True,
            header_style=f"bold {color}",
            box=box.ROUNDED,
        )
        table.add_column("Тип", style=color, width=20)
        table.add_column("Уверенность", justify="right", width=15)
        table.add_column("Позиция", justify="center", width=20)

        for obj in objects:
            short_name = self._shorten_unit_name(obj["type"])
            center = f"({obj['center'][0]}, {obj['center'][1]})"
            table.add_row(short_name, f"{obj['confidence']:.1%}", center)

        return table


# ==================== ACTIONS ====================


class CRActions:
    """Класс для выполнения действий в игре"""

    def tap(self, x: int, y: int):
        """Тап по координатам"""
        self.device.shell(f"input tap {x} {y}")

    def swipe(self, x0: int, y0: int, x1: int, y1: int, duration: int = 500):
        """Свайп от точки A к точке B"""
        self.device.shell(f"input swipe {x0} {y0} {x1} {y1} {duration}")

    def long_press(self, x: int, y: int, duration: int = 1000):
        """Долгое нажатие"""
        self.device.shell(f"input swipe {x} {y} {x} {y} {duration}")


# ==================== RECOGNIZER ====================


class CRRecognizer:
    """Класс для распознавания состояния игры"""

    def _extract_card_img(self):
        """Вырезает изображения карт из скриншота"""
        cards = []
        img = Image.open(self.SCREENSHOT_PATH)
        for idx, (x, y) in enumerate(self.CARD_POSITIONS):
            w, h = self.CARD_SIZE
            card = img.crop((x, y, x + w, y + h))
            path = Path(f"{self.CARDS_DIR}/card_{idx}.png")
            card.save(path)
            cards.append((card, path))
        return cards

    def get_cards(self):
        """Получает список карт в руке"""
        cards_img = self._extract_card_img()
        result = []
        for i, (_, path) in enumerate(cards_img):
            name, conf = self.card_recognizer.predict(str(path))
            result.append(
                {"index": i, "name": name, "confidence": conf, "path": str(path)}
            )
        return result

    # ==================== ELIXIR ====================

    def _extract_elixir_img(self, screenshot: Image.Image) -> Image.Image:
        """Вырезает изображение эликсира"""
        x1, y1, x2, y2 = self.ELIC_CROP
        elic = screenshot.crop((x1, y1, x2, y2))
        elic.save(f"{self.ELIC_DIR}/elic_screenshot.png")
        return elic

    def get_elixir(self):
        """Получает количество эликсира"""
        screenshot = self.get_raw_screenshot()
        elic = self._extract_elixir_img(screenshot)
        _, label, conf = self.digit_recognizer.predict(elic)
        elixir = int(label) if label != "None" else 0
        return {"elixir": elixir, "confidence": conf}

    # ==================== ARENA ====================

    def _extract_arena_image(self, screenshot: Image.Image = None) -> Image.Image:
        """Вырезает изображение арены"""
        if screenshot is None:
            screenshot = Image.open(self.SCREENSHOT_PATH)

        x1, y1, x2, y2 = self.ARENA_CROP
        arena = screenshot.crop((x1, y1, x2, y2))
        arena.save(f"{self.ARENA_DIR}/arena_screenshot.png")
        return arena

    def get_arena_objects(self):
        """Получает список объектов на арене"""
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

        return objects

    def get_arena_state(self):
        """Получает полное состояние игры одним запросом"""
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
        elixir = int(label) if label != "None" else 0

        # Объекты
        arena = self._extract_arena_image(screenshot)
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
            "elixir": {"elixir": elixir, "confidence": conf_e},
            "cards": cards,
            "objects": objects,
        }


# ==================== MAIN BOT ====================


class CRBot(CRActions, CRRecognizer):
    """Главный класс бота Clash Royale"""

    # Константы ADB
    ADB_DEVICE = "192.168.240.112:5555"
    PACKAGE = "com.supercell.clashroyale"
    ACTIVITY = "com.supercell.titan.GameApp"

    # Координаты UI элементов
    CARD_POSITIONS = [(129 + i * 108, 850) for i in range(4)]
    CARD_SIZE = (101, 125)
    ELIC_CROP = (154, 977, 154 + 35, 977 + 27)
    ARENA_CROP = (0, 0, 576, 800)

    # Пути
    SCREENSHOT_PATH = "temp/screenshot/battle_screenshot.png"
    CARDS_DIR = "temp/cards_img"
    ELIC_DIR = "temp/elic"
    ARENA_DIR = "temp/arena_img"

    def __init__(self, model_path="runs/detect/train11/weights/best.pt"):
        """Инициализация бота"""
        # ADB подключение
        self.device = adb.device(self.ADB_DEVICE)

        # Распознаватели
        self.digit_recognizer = DigitRecognizer()
        self.card_recognizer = CardRecognizer()
        self.arena_detector = YOLO(model_path)

        # Принтер состояния
        self.printer = GameStatePrinter()

        # Создание директорий
        self._setup_directories()

        print("✓ CRBot инициализирован успешно")

    def _setup_directories(self):
        """Создание необходимых директорий"""
        for directory in [
            Path(self.SCREENSHOT_PATH).parent,
            Path(self.CARDS_DIR),
            Path(self.ELIC_DIR),
            Path(self.ARENA_DIR),
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def start_game(self):
        """Запуск игры"""
        print("🎮 Запуск Clash Royale...")
        self.device.shell(f"am start -n {self.PACKAGE}/{self.ACTIVITY}")
        print("⏳ Ожидание запуска... (10 сек)")
        time.sleep(10)
        print("✓ Игра запущена")

    def get_raw_screenshot(self) -> Image.Image:
        """Получение скриншота"""
        self.device.shell("screencap /sdcard/screenshot.png")
        subprocess.run(
            [
                "adb",
                "-s",
                self.ADB_DEVICE,
                "pull",
                "/sdcard/screenshot.png",
                self.SCREENSHOT_PATH,
            ],
            capture_output=True,
        )
        return Image.open(self.SCREENSHOT_PATH)

    # ==================== Удобные методы вывода ====================

    def print_game_state(self):
        """Красиво выводит состояние игры"""
        state = self.get_arena_state()
        self.printer.print_game_state(state)

    def print_arena_details(self):
        """Выводит подробную информацию об объектах"""
        state = self.get_arena_state()
        self.printer.print_arena_details(state["objects"])


# ==================== ИСПОЛЬЗОВАНИЕ ====================

if __name__ == "__main__":
    bot = CRBot(model_path="runs/detect/train11/weights/best.pt")
    printer = GameStatePrinter()

    def signal_handler(sig, frame):
        print("\n🛑 Цикл остановлен пользователем (Ctrl+C)")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("🔄 Запуск цикла (Ctrl+C для остановки)...")
    while True:
        state = bot.get_arena_state()
        printer.print_game_state(state)  # ← Выводит ОДНу таблицу
        time.sleep(0.02)  # Обновление каждые 2 секунды
