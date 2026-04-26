#!/usr/bin/env python3
"""
search.py — обёртка над duckduckgo-search для получения релевантных URL.
Возвращает список с метаданными для последующего парсинга.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time

try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

@dataclass
class SearchResult:
    """Структурированный результат поиска."""
    title: str
    url: str
    snippet: str
    source: str  # domain
    rank: int    # позиция в выдаче
    fetched_at: str  # ISO timestamp
    
    def to_dict(self) -> dict:
        return asdict(self)


def search_queries(queries: List[str], max_results_per_query: int = 10, 
                   delay_sec: float = 1.0) -> List[SearchResult]:
    """
    Ищет по списку запросов, объединяет результаты, удаляет дубли по URL.
    
    Args:
        queries: список поисковых запросов
        max_results_per_query: лимит результатов на запрос
        delay_sec: задержка между запросами (чтобы не забанили)
    
    Returns:
        Список SearchResult без дубликатов URL
    """
    if not HAS_DDGS:
        raise ImportError("Установите: pip install duckduckgo-search")
    
    all_results: Dict[str, SearchResult] = {}  # url -> result (дедупликация)
    
    with DDGS() as ddgs:
        for i, query in enumerate(queries):
            try:
                # DuckDuckGo API
                results = ddgs.text(query, max_results=max_results_per_query)
                
                for rank, res in enumerate(results, start=1):
                    url = res.get("href", "")
                    if not url or not url.startswith("http"):
                        continue
                    
                    # Пропускаем уже увиденные URL
                    if url in all_results:
                        continue
                    
                    # Парсим домен для метаданных
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    
                    result = SearchResult(
                        title=res.get("title", "")[:200],  # Обрезаем слишком длинные
                        url=url,
                        snippet=res.get("body", "")[:300],
                        source=domain,
                        rank=rank + i * max_results_per_query,  # Глобальный ранк
                        fetched_at=datetime.now().isoformat()
                    )
                    all_results[url] = result
                
                # Анти-бот задержка
                if i < len(queries) - 1:
                    time.sleep(delay_sec)
                    
            except Exception as e:
                print(f"⚠️  Ошибка при поиске '{query}': {e}")
                continue
    
    return list(all_results.values())


def generate_search_queries(topic: str, llm_hint: Optional[str] = None) -> List[str]:
    """
    Генерирует список поисковых запросов из темы.
    
    Args:
        topic: основная тема (например, "Erlang concurrency")
        llm_hint: опциональная подсказка от LLM для расширения
    
    Returns:
        Список из 3-7 целевых запросов
    """
    # Базовые шаблоны — можно расширять
    templates = [
        f"{topic} official documentation",
        f"{topic} tutorial for beginners",
        f"{topic} best practices",
        f"{topic} common pitfalls",
        f"{topic} performance optimization",
        f"{topic} vs alternatives",
    ]
    
    # Если есть подсказка от LLM — добавляем её запросы
    if llm_hint and isinstance(llm_hint, str):
        # Простой парсинг: одна строка = один запрос, максимум 3
        extra = [q.strip() for q in llm_hint.split("\n") if q.strip()][:3]
        templates.extend(extra)
    
    # Возвращаем уникальные, обрезанные до 100 символов
    seen = set()
    queries = []
    for t in templates:
        q = t[:100].strip()
        if q and q not in seen:
            seen.add(q)
            queries.append(q)
    
    return queries[:7]  # Лимит на количество запросов