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
    """Этап 0: интерактивный инференс без RAG."""
    print(f"\n🚀 {config.project_name} — этап 0: базовый инференс")
    print(f"Модель: {config.llm.name}")
    print(f"Контекст: {config.llm.n_ctx} токенов, GPU-слои: {config.llm.n_gpu_layers}")
    print("Введите 'exit' для выхода, 'session save/load <path>' для управления сессией.\n")
    
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
        
        # Команды сессии
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
        
        if user_input.lower() in ("exit", "quit", "выход"):
            print("👋 Завершение работы.")
            break
        
        # Запрос к модели
        print("🤖 Модель: ", end="", flush=True)
        try:
            answer = llm.ask(user_input, system_prompt)
            print(answer)
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")
            print("Попробуйте уменьшить n_ctx или n_gpu_layers в config.yaml")


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