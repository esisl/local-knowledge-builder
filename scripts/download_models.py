#!/usr/bin/env python3
"""
download_models.py — загрузка моделей для local-knowledge-builder.
Использует huggingface_hub с fallback на прямые ссылки.
"""
import sys
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# === КОНФИГУРАЦИЯ МОДЕЛЕЙ ===
# Начинаем с лёгкой 0.5B для отладки пайплайна
# Чтобы переключиться на 7B, замените DEFAULT_MODEL на "7B"
DEFAULT_MODEL = "0.5B"  # или "7B"

MODELS_CONFIG = {
    "0.5B": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",  # Исправлено: точка вместо подчёркивания + доступное квантование
        "subfolder": "llm",
        "expected_sha256": None,
        "description": "Лёгкая модель для отладки (~380 МБ)"
    },
    "7B": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF", 
        "file": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "subfolder": "llm",
        "expected_sha256": None,
        "description": "Полноценная модель для продакшна (~4.3 ГБ)"
    }
}

# Эмбеддер (универсальный, не зависит от LLM)
EMBEDDING_CONFIG = {
    "repo": "BAAI/bge-small-en-v1.5",
    "file": "onnx/model.onnx",  # Правильный путь внутри репо
    "subfolder": "embeddings/bge-small-en-v1.5",
    "description": "Эмбеддер для векторизации (~130 МБ)"
}

# Прямые ссылки как аварийный fallback
FALLBACK_URLS = {
    "qwen2.5-0_5b-instruct-q8_0.gguf": 
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0_5b-instruct-q8_0.gguf",
    "qwen2.5-7b-instruct-q4_k_m.gguf":
        "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
}


def log(msg: str, status: str = "info"):
    if HAS_RICH:
        console = Console()
        icons = {"ok": "✅", "err": "❌", "info": "ℹ️", "dl": "⬇️", "warn": "⚠️"}
        colors = {"ok": "green", "err": "red", "info": "blue", "dl": "yellow", "warn": "yellow"}
        console.print(f"[{colors[status]}]{icons[status]} {msg}[/{colors[status]}]")
    else:
        prefix = {"ok": "[OK]", "err": "[ERR]", "info": "[INFO]", "dl": "[DL]", "warn": "[WARN]"}
        print(f"{prefix[status]} {msg}")


def calculate_sha256(filepath: Path) -> Optional[str]:
    """Вычисляет SHA256 хеш файла для проверки целостности."""
    try:
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return None


def download_via_hf(repo: str, filename: str, target_dir: Path) -> Optional[Path]:
    """Попытка скачать через huggingface_hub."""
    try:
        # Проверяем, существует ли файл в репо (чтобы избежать 404)
        available_files = list_repo_files(repo)
        if filename not in available_files:
            log(f"Файл {filename} не найден в репозитории {repo}", "err")
            # Показываем доступные файлы для отладки
            gguf_files = [f for f in available_files if f.endswith('.gguf')][:5]
            if gguf_files:
                log(f"Доступные .gguf файлы: {', '.join(gguf_files)}", "info")
            return None
        
        downloaded_path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        return Path(downloaded_path)
    except Exception as e:
        log(f"hf_hub_download ошибка: {type(e).__name__}: {e}", "warn")
        return None


def download_via_fallback(url: str, target_path: Path) -> bool:
    """Скачивание через requests как запасной вариант."""
    try:
        import requests
        from tqdm import tqdm
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"Прямая загрузка: {url}", "dl")
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total = int(response.headers.get('content-length', 0))
        with open(target_path, 'wb') as f, tqdm(
            desc=target_path.name,
            total=total,
            unit='B',
            unit_scale=True,
            disable=not HAS_RICH
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        # Проверка размера файла
        if target_path.stat().st_size < 1024 * 1024:  # Менее 1 МБ — явно ошибка
            log("Скачан слишком маленький файл — возможно, это ошибка", "err")
            target_path.unlink()
            return False
            
        log(f"Сохранено: {target_path} ({target_path.stat().st_size / 1024**2:.1f} МБ)", "ok")
        return True
    except Exception as e:
        log(f"Fallback ошибка: {e}", "err")
        return False


def download_model(config: dict, local_root: Path) -> bool:
    """Универсальная функция загрузки модели."""
    repo = config["repo"]
    filename = config["file"]
    subfolder = config["subfolder"]
    expected_hash = config.get("expected_sha256")
    
    target_dir = local_root / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    
    # Если файл уже есть и хеш совпадает — пропускаем
    if target_path.exists():
        if expected_hash:
            actual_hash = calculate_sha256(target_path)
            if actual_hash == expected_hash:
                log(f"Файл уже есть и хеш совпадает: {filename}", "ok")
                return True
            else:
                log(f"Хеш не совпадает, перезагружаем: {filename}", "warn")
                target_path.unlink()
        else:
            log(f"Файл уже существует: {filename} (пропускаем)", "info")
            return True
    
    log(f"Загрузка: {filename} ({config['description']})", "dl")
    
    # Попытка 1: huggingface_hub
    result = download_via_hf(repo, filename, target_dir)
    if result and result.exists():
        log(f"✅ Загружено через HF: {result.name}", "ok")
        return True
    
    # Попытка 2: fallback URL
    if filename in FALLBACK_URLS:
        if download_via_fallback(FALLBACK_URLS[filename], target_path):
            return True
    
    log(f"❌ Не удалось загрузить {filename}", "err")
    return False


def main():
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if HAS_RICH:
        Console().print(f"📦 local-knowledge-builder: загрузка моделей", style="bold cyan")
        Console().print(f"🎯 Используем модель: {DEFAULT_MODEL}", style="yellow")
    else:
        print(f"=== Загрузка моделей (режим: {DEFAULT_MODEL}) ===")
    
    # Загружаем LLM
    llm_config = MODELS_CONFIG[DEFAULT_MODEL]
    llm_ok = download_model(llm_config, models_dir)
    
    # Загружаем эмбеддер (опционально для этапа 0)
    embed_ok = download_model(EMBEDDING_CONFIG, models_dir)
    
    # Итог
    if llm_ok:
        log(f"✅ LLM готова: {llm_config['file']}", "ok")
        if not embed_ok:
            log("⚠️  Эмбеддер не загружен (не критично для этапа 0)", "warn")
        return 0
    else:
        log("❌ Критическая ошибка: LLM не загружена", "err")
        log("💡 Проверьте интернет, или попробуйте вручную скачать по ссылке:", "info")
        log(f"   {FALLBACK_URLS.get(llm_config['file'], 'N/A')}", "info")
        return 1


if __name__ == "__main__":
    # Проверяем наличие tqdm для красивого прогресса
    try:
        from tqdm import tqdm
    except ImportError:
        log("tqdm не установлен — прогресс будет упрощён", "warn")
    
    sys.exit(main())