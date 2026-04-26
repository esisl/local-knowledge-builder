# Проверка CUDA, RAM, paths перед запуском
#!/usr/bin/env python3
"""
validate_env.py — проверка окружения перед запуском local-knowledge-builder.
Выводит отчёт в rich-формате, возвращает код 0 (всё ок) или 1 (есть блокирующие ошибки).
"""
import sys
import os
import platform
import shutil
import ctypes
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# === КОНФИГУРАЦИЯ ПРОВЕРОК ===
MIN_RAM_GB = 16
MIN_VRAM_GB = 10  # 3060 имеет 12 ГБ, но часть занята ОС
MIN_DISK_FREE_GB = 20
MAX_PATH_LENGTH = 240  # Запас до 260 (Windows MAX_PATH)
REQUIRED_DIRS = ["./data", "./models", "./logs", "./config.yaml"]


def log(msg: str, status: str = "info"):
    """Универсальный логгер: rich если есть, иначе print."""
    if RICH_AVAILABLE:
        console = Console()
        icons = {"ok": "✅", "warn": "⚠️", "err": "❌", "info": "ℹ️"}
        colors = {"ok": "green", "warn": "yellow", "err": "red", "info": "blue"}
        console.print(f"[{colors[status]}]{icons[status]} {msg}[/{colors[status]}]")
    else:
        prefix = {"ok": "[OK]", "warn": "[WARN]", "err": "[ERR]", "info": "[INFO]"}
        print(f"{prefix[status]} {msg}")


def check_python() -> bool:
    """Python 3.10+ и 64-бит."""
    py_ver = sys.version_info
    is_64 = sys.maxsize > 2**32
    if py_ver.major == 3 and py_ver.minor >= 10 and is_64:
        log(f"Python {py_ver.major}.{py_ver.minor} (64-bit)", "ok")
        return True
    log(f"Python {py_ver.major}.{py_ver.minor} {'32-bit' if not is_64 else ''} — нужен 3.10+ 64-bit", "err")
    return False


def check_platform() -> bool:
    """Только Windows 10+ или WSL2."""
    sys_name = platform.system()
    if sys_name == "Windows":
        ver = platform.version()
        # Windows 10 = 10.0.xxxxx
        if ver.startswith("10.0"):
            log(f"Windows 10/11 detected", "ok")
            return True
        log(f"Windows {ver} — не тестировалось, возможны проблемы", "warn")
        return True  # Не блокируем, но предупреждаем
    elif sys_name == "Linux" and "microsoft" in platform.release().lower():
        log("WSL2 detected — убедитесь, что проброшен GPU (nvidia-smi в WSL)", "warn")
        return True
    log(f"Платформа {sys_name} не поддерживается", "err")
    return False


def check_ram() -> bool:
    """Проверка системной ОЗУ."""
    total_ram = psutil.virtual_memory().total / (1024**3) if HAS_PSUTIL else 0
    if total_ram >= MIN_RAM_GB:
        log(f"RAM: {total_ram:.1f} ГБ (≥{MIN_RAM_GB} ГБ)", "ok")
        return True
    log(f"RAM: {total_ram:.1f} ГБ — мало, возможны свопы и падения", "err")
    return False


def check_vram() -> Optional[bool]:
    """Проверка VRAM через pynvml. Возвращает None, если библиотека не установлена."""
    if not HAS_PYNVML:
        log("pynvml не установлен — пропуск проверки VRAM (установите nvidia-ml-py3)", "warn")
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_vram = info.total / (1024**3)
        free_vram = info.free / (1024**3)
        pynvml.nvmlShutdown()
        
        if total_vram >= MIN_VRAM_GB:
            log(f"VRAM: {total_vram:.1f} ГБ всего, {free_vram:.1f} ГБ свободно", "ok")
            return True
        log(f"VRAM: {total_vram:.1f} ГБ — мало для инференса с контекстом >8K", "err")
        return False
    except Exception as e:
        log(f"Ошибка при опросе GPU: {e}", "warn")
        return None


def check_disk() -> bool:
    """Свободное место на диске проекта."""
    project_root = Path(__file__).resolve().parent.parent
    total, used, free = shutil.disk_usage(project_root)
    free_gb = free / (1024**3)
    if free_gb >= MIN_DISK_FREE_GB:
        log(f"Диск: {free_gb:.1f} ГБ свободно (≥{MIN_DISK_FREE_GB} ГБ)", "ok")
        return True
    log(f"Диск: {free_gb:.1f} ГБ — мало для моделей + индексы + кэш", "err")
    return False


def check_paths() -> bool:
    """Проверка длины путей и прав на запись."""
    project_root = Path(__file__).resolve().parent.parent
    # Проверка MAX_PATH
    test_path = project_root / "a" * 200
    try:
        test_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_path, "w") as f:
            f.write("x")
        test_path.unlink()
        log(f"Длинные пути: поддерживаются", "ok")
    except OSError:
        log(f"Длинные пути: НЕ поддерживаются — включите LongPathsEnabled в реестре", "err")
        return False
    
    # Проверка прав на запись в REQUIRED_DIRS
    all_writable = True
    for item in REQUIRED_DIRS:
        path = project_root / item.rstrip("/")
        if item.endswith(".yaml"):
            # Файл: проверяем родительскую директорию
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
        log("Права на запись в рабочие директории: ОК", "ok")
    return all_writable


def check_llama_cpp() -> bool:
    """Проверка, что llama-cpp-python установлен и видит CUDA."""
    try:
        import llama_cpp
        # Проверка сборки: наличие CUDA в backend
        backend = getattr(llama_cpp, "LLAMA_BACKEND_CUDA", None)
        if backend is not None:
            log("llama-cpp-python: установлен с поддержкой CUDA", "ok")
            return True
        # Если нет явного флага, пробуем инициализировать контекст
        from llama_cpp import llama_backend_init
        llama_backend_init()
        log("llama-cpp-python: установлен (проверка CUDA — runtime)", "warn")
        return True
    except ImportError:
        log("llama-cpp-python: НЕ установлен", "err")
        return False
    except Exception as e:
        log(f"llama-cpp-python: ошибка инициализации: {e}", "err")
        return False


def check_optional_deps():
    """Проверка опциональных, но полезных зависимостей."""
    optional = [
        ("trafilatura", "Парсинг веб-страниц"),
        ("chromadb", "Векторное хранилище"),
        ("sentence_transformers", "Эмбеддинги"),
        ("pydantic", "Валидация данных"),
    ]
    for module, desc in optional:
        try:
            __import__(module)
            log(f"{module}: установлен ({desc})", "ok")
        except ImportError:
            log(f"{module}: отсутствует ({desc}) — установите позже", "warn")


def main():
    """Главная точка входа."""
    # Динамический импорт опциональных библиотек, чтобы скрипт не падал без них
    global psutil, HAS_PSUTIL, pynvml, HAS_PYNVML
    try:
        import psutil
        HAS_PSUTIL = True
    except ImportError:
        HAS_PSUTIL = False
        log("psutil не установлен — пропуск проверки RAM", "warn")
    
    try:
        import pynvml
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        HAS_PYNVML = True
    except ImportError:
        HAS_PYNVML = False

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
    ]
    
    # RAM и VRAM — только если есть зависимости
    if HAS_PSUTIL:
        checks.insert(2, ("RAM", check_ram))
    if HAS_PYNVML:
        checks.insert(3, ("VRAM", check_vram))

    results = []
    for name, func in checks:
        try:
            res = func()
            results.append((name, res))
        except Exception as e:
            log(f"{name}: исключение при проверке: {e}", "err")
            results.append((name, False))

    # Опциональные зависимости — не блокируют запуск
    check_optional_deps()

    # Итог
    failed = [name for name, res in results if res is False]
    warned = [name for name, res in results if res is None]
    
    if RICH_AVAILABLE:
        console.print("\n" + "="*50)
    else:
        print("\n" + "="*50)
    
    if not failed:
        log("✅ Все критические проверки пройдены. Можно запускать пайплайн.", "ok")
        if warned:
            log(f"⚠️  Предупреждения по: {', '.join(warned)}", "warn")
        return 0
    else:
        log(f"❌ Блокирующие ошибки: {', '.join(failed)}. Исправьте перед запуском.", "err")
        return 1


if __name__ == "__main__":
    sys.exit(main())