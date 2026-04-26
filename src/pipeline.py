#!/usr/bin/env python3
"""
pipeline.py — оркестратор пайплайна local-knowledge-builder.
Этап 0: базовый инференс без RAG.
"""
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml
from pydantic import BaseModel, Field

from src.search import search_queries, generate_search_queries, SearchResult
from src.crawler import crawl_batch, CrawledPage

# === КОНФИГ-МОДЕЛИ (валидация через pydantic) ===
class LLMConfig(BaseModel):
    name: str = Field(..., description="Имя файла .gguf")
    n_ctx: int = Field(8192, ge=512, le=32768, description="Размер контекста")
    n_gpu_layers: int = Field(25, ge=0, le=100, description="Слои на GPU")
    cache_prompt: bool = Field(True, description="Кэшировать промпт для сессии")

class ProjectConfig(BaseModel):
    project_name: str
    paths: Dict[str, str]
    llm: LLMConfig
    
    @property
    def models_dir(self) -> Path:
        return Path(self.paths["models"])
    
    @property
    def llm_path(self) -> Path:
        return self.models_dir / "llm" / self.llm.name


# === RAG-ENGINE (заглушка для этапа 0) ===
class SimpleLLM:
    """Обёртка над llama-cpp-python для базового инференса."""
    
    def __init__(self, config: LLMConfig, model_path: Path):
        from llama_cpp import Llama
        
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}\nЗапустите: python scripts/download_models.py")
        
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            verbose=False,
            # Важные флаги для 3060 12GB:
            offload_kqv=True,  # Выгружать KQV на GPU
        )
        self.cache_prompt = config.cache_prompt
        self.history: List[Dict[str, str]] = []
    
    def ask(self, question: str, system_prompt: Optional[str] = None) -> str:
        """Простой запрос-ответ с поддержкой истории."""
        # Формируем промпт в формате Qwen Chat
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.history)
        messages.append({"role": "user", "content": question})
        
        # Генерируем ответ
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.2,  # Низкая температура для фактологических ответов
            max_tokens=1024,
            stream=False,
        )
        
        answer = response["choices"][0]["message"]["content"]
        
        # Сохраняем в историю (если включено кэширование)
        if self.cache_prompt:
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})
        
        return answer.strip()
    
    def save_session(self, path: Path):
        """Сохраняет историю сессии."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
    
    def load_session(self, path: Path):
        """Загружает историю сессии."""
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.history = json.load(f)


# === CLI ENTRY POINT ===
def load_config(config_path: Path) -> ProjectConfig:
    """Загружает и валидирует config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ProjectConfig(**raw)

def run_stage0(config: ProjectConfig):
    """Этап 0: интерактивный инференс + доступ к другим этапам."""
    print(f"\n🚀 {config.project_name} — консоль управления")
    print(f"Модель: {config.llm.name}")
    print("Команды: 'exit', 'session save/load <path>', 'collect <topic>', 'help'\n")
    
    llm = SimpleLLM(config.llm, config.llm_path)
    system_prompt = "Ты — полезный ассистент. Отвечай кратко и по делу."
    
    while True:
        try:
            user_input = input("👤 Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Завершение работы.")
            break
        
        if not user_input:
            continue
        
        # === Команды управления ===
        if user_input.lower() in ("exit", "quit", "выход"):
            print("👋 Завершение работы.")
            break
        
        if user_input.lower() == "help":
            print("""
📋 Доступные команды:
  exit / quit          — выход
  session save <path>  — сохранить историю сессии
  session load <path>  — загрузить историю сессии
  collect <тема>       — запустить сбор данных (Этап 1)
  help                 — эта справка
            """)
            continue
        
        if user_input.startswith("session "):
            parts = user_input.split(maxsplit=2)
            if len(parts) < 2:
                print("Используйте: session save <path> или session load <path>")
                continue
            cmd, arg = parts[1], parts[2] if len(parts) > 2 else "session.json"
            session_path = Path(arg)
            if cmd == "save":
                llm.save_session(session_path)
                print(f"💾 Сессия сохранена в {session_path}")
            elif cmd == "load":
                llm.load_session(session_path)
                print(f"📥 Сессия загружена из {session_path}")
            else:
                print("Неизвестная команда сессии. Используйте save/load.")
            continue
        
        if user_input.startswith("collect "):
            topic = user_input[len("collect "):].strip()
            if not topic:
                print("Используйте: collect <тема>, например: collect Erlang concurrency")
                continue
            run_stage1(config, topic)
            continue
        
        # === Обычный запрос к модели ===
        print("🤖 Модель: ", end="", flush=True)
        try:
            answer = llm.ask(user_input, system_prompt)
            print(answer)
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")

def run_stage1(config: ProjectConfig, topic: str, max_urls: int = 20):
    """Этап 1: сбор данных по теме с подробным логированием."""
    from pathlib import Path
    
    print(f"\n🔍 {config.project_name} — этап 1: сбор данных")
    print(f"Тема: {topic}")
    
    queries = generate_search_queries(topic)
    print(f"📋 Поисковые запросы ({len(queries)}):")
    for i, q in enumerate(queries, 1):
        print(f"   {i}. {q}")
    
    print(f"\n🔎 Поиск URL...")
    try:
        search_results = search_queries(queries, max_results_per_query=5)
        print(f"✅ Найдено {len(search_results)} уникальных URL:")
        for i, r in enumerate(search_results, 1):
            print(f"   {i}. {r.url}")
    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")
        print("💡 Убедитесь, что установлен пакет: pip install ddgs")
        return
    
    if not search_results:
        print("⚠️  Поисковая выдача пуста. Попробуйте другую тему.")
        return

    raw_dir = Path(config.paths["data"]) / "raw"
    print(f"\n🕷️  Парсинг страниц (сохранение в {raw_dir})...")
    
    urls_to_crawl = [r.url for r in search_results[:max_urls]]
    crawl_results = crawl_batch(urls_to_crawl, raw_dir, skip_existing=True)
    
    # Подробный отчёт
    saved = [r for r in crawl_results if r["status"] == "saved"]
    skipped = [r for r in crawl_results if r["status"] == "skipped"]
    errors = [r for r in crawl_results if r["status"] == "error"]
    
    print(f"\n📊 Итог:")
    print(f"   ✅ Сохранено: {len(saved)}")
    for r in saved: print(f"      📄 {r['file']}")
    
    print(f"   ⏭️  Пропущено: {len(skipped)}")
    for r in skipped: print(f"      ⏩ {r['url']}")
    
    print(f"   ❌ Ошибки: {len(errors)}")
    for r in errors: print(f"      🚫 {r['url']}\n         → {r['error']}")
    
    if saved:
        print(f"\n💡 Следующий шаг: дедупликация и чанкинг (Этап 2)")

def main():
    """Точка входа."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "config.yaml"
    
    if not config_path.exists():
        print(f"❌ Конфиг не найден: {config_path}")
        print("Создайте config.yaml по образцу из документации.")
        return 1
    
    try:
        config = load_config(config_path)
        run_stage0(config)
        return 0
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        if "CUDA" in str(e) or "memory" in str(e).lower():
            print("💡 Совет: уменьшите n_gpu_layers в config.yaml до 20 или n_ctx до 4096")
        return 1


if __name__ == "__main__":
    sys.exit(main())