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

from src.cleaner import filter_by_relevance, load_markdown_file
from src.chunker import process_directory, CleanedChunk

from src.embedder import embed_chunks_jsonl
from src.indexer import save_to_chromadb, export_db

from src.rag_engine import RAGEngine, RAGResult

# === КОНФИГ-МОДЕЛИ (валидация через pydantic) ===
class LLMConfig(BaseModel):
    name: str = Field(..., description="Имя файла .gguf")
    n_ctx: int = Field(8192, ge=512, le=32768, description="Размер контекста")
    n_gpu_layers: int = Field(25, ge=0, le=100, description="Слои на GPU")
    cache_prompt: bool = Field(True, description="Кэшировать промпт для сессии")

# === НОВОЕ: конфигурация RAG ===
class RAGConfig(BaseModel):
    top_k: int = Field(4, ge=1, le=20, description="Количество чанков для ретривала")
    min_score: float = Field(0.65, ge=0.0, le=1.0, description="Порог релевантности")
    fallback_message: str = Field("В базе нет релевантных данных. Уточните запрос.", description="Ответ при пустом ретривале")

class ProjectConfig(BaseModel):
    project_name: str
    paths: Dict[str, str]
    llm: LLMConfig
    rag: RAGConfig = Field(default_factory=RAGConfig)  # <-- ДОБАВЛЕНО
    
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
  process <тема>       — дедупликация + чанкинг (Этап 2)
  index <тема>         — векторизация + индексация + экспорт (Этап 3)
  query <вопрос>       — выполнить RAG-запрос по последней проиндексированной теме             
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

        if user_input.startswith("process "):
            topic = user_input[len("process "):].strip()
            if not topic:
                print("Используйте: process <тема>, например: process Erlang concurrency")
                continue
            run_stage2(config, topic)
            continue

        if user_input.startswith("index "):
            topic = user_input[len("index "):].strip()
            if not topic:
                print("Используйте: index <тема>")
                continue
            run_stage3(config, topic)
            continue

        if user_input.startswith("query "):
            # Для простоты: тема берётся из последнего успешного index/collect
            # В продакшене — хранить текущую тему в состоянии
            topic = "Erlang concurrency"  # Заглушка, потом вынесем в config
            question = user_input[len("query "):].strip()
            if not question:
                print("Используйте: query <вопрос>")
                continue
            run_stage4(config, topic, question)
            continue

        if user_input.startswith("build "):
            topic = user_input[len("build "):].strip()
            if not topic:
                print("Используйте: build <тема>")
                continue
            run_build(config, topic)
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

def run_stage2(config: ProjectConfig, topic: Optional[str] = None):
    """Этап 2: дедупликация + чанкинг."""
    from pathlib import Path
    
    print(f"\n🔧 {config.project_name} — этап 2: очистка и чанкинг")
    
    raw_dir = Path(config.paths["data"]) / "raw"
    chunks_dir = Path(config.paths["data"]) / "chunks"
    
    if not raw_dir.exists():
        print(f"❌ Папка с сырыми данными не найдена: {raw_dir}")
        print("💡 Сначала выполните: collect <тема>")
        return
    
    # Параметры из конфига или дефолты
    max_tokens = getattr(config, 'chunk_max_tokens', 700)
    overlap = getattr(config, 'chunk_overlap', 100)
    dedup_threshold = getattr(config, 'dedup_threshold', 0.85)
    
    # Запускаем пайплайн
    stats = process_directory(
        raw_dir=raw_dir,
        output_dir=chunks_dir,
        max_tokens=max_tokens,
        overlap=overlap,
        dedup_threshold=dedup_threshold
    )
    
    # Опциональная фильтрация по релевантности
    if topic:
        print(f"\n🎯 Фильтрация по теме: '{topic}'")
        chunks_file = chunks_dir / "chunks.jsonl"
        if chunks_file.exists():
            # Загружаем чанки
            chunks = []
            with open(chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    chunks.append(json.loads(line))
            
            # Фильтруем
            from src.cleaner import filter_by_relevance
            filtered = filter_by_relevance(chunks, topic, min_score=0.6)
            
            # Перезаписываем
            with open(chunks_file, 'w', encoding='utf-8') as f:
                for c in filtered:
                    f.write(json.dumps(c, ensure_ascii=False) + '\n')
            
            print(f"✅ После фильтрации: {len(filtered)} чанков")
    
    print(f"\n💡 Следующий шаг: векторизация и индексация (Этап 3)")
    print(f"   Команда: python -m src.pipeline → index <тема>")

def run_stage3(config: ProjectConfig, topic: str):
    """Этап 3: векторизация + индексация + экспорт."""
    from pathlib import Path

    print(f"\n🔮 {config.project_name} — этап 3: векторизация и индексация")

    chunks_file = Path(config.paths["data"]) / "chunks" / "chunks.jsonl"
    if not chunks_file.exists():
        print(f"❌ Файл чанков не найден: {chunks_file}")
        print("💡 Сначала выполните: process <тема>")
        return

    # 1. Векторизация
    try:
        chunks_with_emb = embed_chunks_jsonl(chunks_file, batch_size=32)
    except ImportError as e:
        print(f"❌ {e}")
        return

    if not chunks_with_emb:
        print("⚠️  Нечего индексировать.")
        return

    # 2. Индексация
    safe_topic = topic.replace(" ", "_").lower()
    db_dir = Path(config.paths["data"]) / "vectors" / f"{safe_topic}_db"
    save_to_chromadb(chunks_with_emb, db_dir, collection_name=safe_topic)

    # 3. Экспорт
    export_dir = Path(config.paths["data"]) / "exports"
    archive, sha = export_db(db_dir, export_dir, collection_name=safe_topic)
    print(f"💾 Архив готов для переноса на изолированную машину: {archive}")
    print(f"🔒 Контрольная сумма: {sha}")

    print("\n🎉 RAG-база готова! Следующий шаг: тестирование запросов (Этап 4)")

def run_stage4(config: ProjectConfig, topic: str, query: str):
    """Этап 4: выполнение RAG-запроса."""
    from pathlib import Path
    
    print(f"\n🔍 {config.project_name} — RAG-запрос")
    
    # Пути к базе и модели
    safe_topic = topic.replace(" ", "_").lower()
    db_path = Path(config.paths["data"]) / "vectors" / f"{safe_topic}_db"
    llm_path = Path(config.paths["models"]) / "llm" / config.llm.name
    
    if not db_path.exists():
        print(f"❌ База не найдена: {db_path}")
        print("💡 Сначала выполните: index <тема>")
        return
    
    if not llm_path.exists():
        print(f"❌ Модель не найдена: {llm_path}")
        print("💡 Запустите: python scripts/download_models.py")
        return
    
    # Инициализация движка
    rag_config = {
        "collection_name": safe_topic,
        "n_ctx": config.llm.n_ctx,
        "n_gpu_layers": config.llm.n_gpu_layers,
        # ✅ Прямой доступ к полям Pydantic-модели
        "top_k": config.rag.top_k,
        "min_score": config.rag.min_score,
        "fallback_message": config.rag.fallback_message
    }
    
    print(f"🧠 Загрузка RAG-движка...")
    engine = RAGEngine(db_path, llm_path, rag_config)
    
    # Выполнение запроса
    print(f"❓ Вопрос: {query}")
    result = engine.ask(query)
    
    # Вывод ответа
    print(f"\n🤖 Ответ:\n{result.answer}")
    
    # Источники
    if result.sources:
        print(f"\n📚 Использовано источников: {result.context_used}")
        for i, src in enumerate(result.sources, 1):
            print(f"   [{i}] {src['title'][:60]}...")
            print(f"       📁 {src['source']} | 🎯 {src['score']:.2f}")
    else:
        print(f"\n⚠️  Источники не найдены (порог релевантности: {rag_config['min_score']})")

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
    
def run_build(config: ProjectConfig, topic: str):
    """Полный пайплайн: collect → process → index."""
    print(f"\n🏗️  {config.project_name} — полная сборка базы: '{topic}'")
    
    # Этап 1
    run_stage1(config, topic, max_urls=50)
    
    # Этап 2
    run_stage2(config, topic)
    
    # Этап 3
    run_stage3(config, topic)
    
    print(f"\n✅ Сборка завершена! База готова для запросов.")
    print(f"💡 Используйте: query <ваш вопрос>")


if __name__ == "__main__":
    sys.exit(main())