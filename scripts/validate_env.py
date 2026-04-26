#!/usr/bin/env python3
"""
validate_env.py — проверка окружения перед запуском local-knowledge-builder.
Выводит отчёт, возвращает код 0 (всё ок) или 1 (есть блокирующие ошибки).
"""
import sys
import os
import platform
import shutil
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# === КОНФИГУРАЦИЯ ===
MIN_RAM_GB = 16
MIN_VRAM_GB = 10
MIN_DISK_FREE_GB = 20
REQUIRED_DIRS = ["./data", "./models", "./logs", "./config.yaml"]


def log(msg: str, status: str = "info"):
    if RICH_AVAILABLE:
        console = Console()
        icons = {"ok": "✅", "warn": "⚠️", "err": "❌", "info": "ℹ️"}
        colors = {"ok": "green", "warn": "yellow", "err": "red", "info": "blue"}
        console.print(f"[{colors[status]}]{icons[status]} {msg}[/{colors[status]}]")
    else:
        prefix = {"ok": "[OK]", "warn": "[WARN]", "err": "[ERR]", "info": "[INFO]"}
        print(f"{prefix[status]} {msg}")


def check_python() -> bool:
    py_ver = sys.version_info
    is_64 = sys.maxsize > 2**32
    if py_ver.major == 3 and py_ver.minor >= 10 and is_64:
        log(f"Python {py_ver.major}.{py_ver.minor} (64-bit)", "ok")
        return True
    log(f"Python {py_ver.major}.{py_ver.minor} — нужен 3.10+ 64-bit", "err")
    return False


def check_platform() -> bool:
    sys_name = platform.system()
    if sys_name == "Windows":
        ver = platform.version()
        if ver.startswith("10.0"):
            log(f"Windows 10/11 detected", "ok")
            return True
        log(f"Windows {ver} — не тестировалось", "warn")
        return True
    elif sys_name == "Linux" and "microsoft" in platform.release().lower():
        log("WSL2 detected — убедитесь, что GPU проброшен", "warn")
        return True
    log(f"Платформа {sys_name} не поддерживается", "err")
    return False


def check_ram() -> bool:
    try:
        import psutil
        total_ram = psutil.virtual_memory().total / (1024**3)
        if total_ram >= MIN_RAM_GB:
            log(f"RAM: {total_ram:.1f} ГБ (≥{MIN_RAM_GB} ГБ)", "ok")
            return True
        log(f"RAM: {total_ram:.1f} ГБ — мало", "err")
        return False
    except ImportError:
        log("psutil не установлен — пропуск проверки RAM", "warn")
        return True  # Не блокируем


def check_vram() -> bool:
    """Проверка VRAM: сначала pynvml, потом torch.cuda, потом пропуск."""
    # Попытка 1: pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_vram = info.total / (1024**3)
        free_vram = info.free / (1024**3)
        pynvml.nvmlShutdown()
        if total_vram >= MIN_VRAM_GB:
            log(f"VRAM (NVML): {total_vram:.1f} ГБ всего, {free_vram:.1f} ГБ свободно", "ok")
            return True
        log(f"VRAM (NVML): {total_vram:.1f} ГБ — мало для контекста >8K", "warn")
        return True  # Не блокируем, но предупреждаем
    except ImportError:
        pass  # pynvml нет, пробуем torch
    except Exception as e:
        log(f"NVML проверка не удалась: {type(e).__name__}", "warn")
    
    # Попытка 2: torch.cuda
    try:
        import torch
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            log(f"VRAM (torch): {total_vram:.1f} ГБ доступно", "ok")
            if total_vram >= MIN_VRAM_GB:
                return True
            log(f"VRAM: {total_vram:.1f} ГБ — мало, но пробуем", "warn")
            return True
    except ImportError:
        pass
    except Exception as e:
        log(f"torch.cuda проверка не удалась: {e}", "warn")
    
    # Фолбэк: просто констатируем наличие CUDA в llama-cpp
    log("VRAM: не удалось проверить автоматически — полагаемся на llama-cpp runtime", "warn")
    return True  # Не блокируем


def check_disk() -> bool:
    project_root = Path(__file__).resolve().parent.parent
    total, used, free = shutil.disk_usage(project_root)
    free_gb = free / (1024**3)
    if free_gb >= MIN_DISK_FREE_GB:
        log(f"Диск: {free_gb:.1f} ГБ свободно", "ok")
        return True
    log(f"Диск: {free_gb:.1f} ГБ — мало", "err")
    return False


def check_paths() -> bool:
    """Исправленная проверка путей."""
    project_root = Path(__file__).resolve().parent.parent
    
    # Проверка длинных путей: ИСПРАВЛЕНО — скобки вокруг "a" * 200
    try:
        test_path = project_root / ("a" * 200)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("x")
        test_path.unlink()
        log("Длинные пути: поддерживаются", "ok")
    except OSError as e:
        log(f"Длинные пути: НЕ поддерживаются — включите LongPathsEnabled в реестре ({e})", "err")
        return False
    
    # Проверка прав на запись
    all_writable = True
    for item in REQUIRED_DIRS:
        path = project_root / item.rstrip("/")
        if item.endswith(".yaml"):
            path = path.parent
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            log(f"Нет прав на запись в {item}: {e}", "err")
            all_writable = False
    if all_writable:
        log("Права на запись: ОК", "ok")
    return all_writable


def check_llama_cpp() -> bool:
    try:
        import llama_cpp
        # Простая проверка: если модуль импортируется — ок
        log("llama-cpp-python: установлен", "ok")
        return True
    except ImportError:
        log("llama-cpp-python: НЕ установлен", "err")
        return False
    except Exception as e:
        log(f"llama-cpp-python: ошибка: {e}", "err")
        return False


def check_optional_deps():
    optional = [
        ("trafilatura", "Парсинг"),
        ("chromadb", "Векторное хранилище"),
        ("sentence_transformers", "Эмбеддинги"),
        ("pydantic", "Валидация"),
    ]
    for module, desc in optional:
        try:
            __import__(module)
            log(f"{module}: OK ({desc})", "ok")
        except ImportError:
            log(f"{module}: отсутствует ({desc})", "warn")


def main():
    if RICH_AVAILABLE:
        console = Console()
        console.print(Panel.fit("🔍 local-knowledge-builder: проверка окружения", style="bold cyan"))
    else:
        log("=== local-knowledge-builder: проверка окружения ===", "info")

    checks = [
        ("Python", check_python),
        ("Платформа", check_platform),
        ("Диск", check_disk),
        ("Пути/права", check_paths),
        ("llama-cpp", check_llama_cpp),
        ("RAM", check_ram),
        ("VRAM", check_vram),  # Теперь не блокирующая
    ]

    results = []
    for name, func in checks:
        try:
            res = func()
            results.append((name, res))
        except Exception as e:
            log(f"{name}: исключение: {e}", "err")
            results.append((name, False))

    check_optional_deps()

    failed = [name for name, res in results if res is False]
    
    if RICH_AVAILABLE:
        console.print("\n" + "="*50)
    else:
        print("\n" + "="*50)
    
    if not failed:
        log("✅ Все критические проверки пройдены.", "ok")
        return 0
    else:
        log(f"❌ Блокирующие ошибки: {', '.join(failed)}", "err")
        return 1


if __name__ == "__main__":
    sys.exit(main())