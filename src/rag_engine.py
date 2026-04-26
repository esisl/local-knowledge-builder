#!/usr/bin/env python3
"""
rag_engine.py — RAG-инференс: ретривал + генерация + цитирование.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import chromadb
from llama_cpp import Llama

@dataclass
class RAGResult:
    """Результат RAG-запроса."""
    answer: str
    sources: List[Dict]  # [{url, title, score}, ...]
    query: str
    context_used: int  # сколько чанков реально использовано
    
    def to_dict(self) -> dict:
        return asdict(self)


class RAGEngine:
    """Основной класс RAG-движка."""
    
    def __init__(self, db_path: Path, llm_path: Path, config: Dict):
        """
        Args:
            db_path: путь к папке ChromaDB
            llm_path: путь к .gguf модели
            config: словарь с параметрами
        """
        # 1. Инициализация ChromaDB
        self.client = chromadb.PersistentClient(path=str(db_path))
        collection_name = config.get("collection_name", "rag_db")
        self.collection = self.client.get_collection(name=collection_name)
        
        # 2. Инициализация LLM
        self.llm = Llama(
            model_path=str(llm_path),
            n_ctx=config.get("n_ctx", 8192),
            n_gpu_layers=config.get("n_gpu_layers", 25),
            verbose=False,
            offload_kqv=True,
        )
        
        # 3. Эмбеддер (ОБЯЗАТЕЛЬНО тот же, что использовался при индексации)
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        
        # 4. Параметры RAG
        self.top_k = config.get("top_k", 4)
        self.min_score = config.get("min_score", 0.65)
        self.fallback_message = config.get("fallback_message", "В базе нет релевантных данных.")

    def retrieve(self, query: str) -> List[Tuple[Dict, float]]:
        # 1. Векторизуем запрос ТОЙ ЖЕ моделью
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0].tolist()
        
        # 2. Ищем по векторам, а не по тексту
        results = self.collection.query(
            query_embeddings=[query_embedding],  # <-- Исправлено
            n_results=self.top_k * 2,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'] or not results['documents'][0]:
            return []
        
        chunks = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            # ChromaDB по умолчанию использует L2-дистанцию для кастомных эмбеддингов.
            # Для нормализованных векторов: cosine_sim = 1 - (l2_dist / 2)
            score = 1.0 - (dist / 2.0) if dist is not None else 0.0
            
            if score >= self.min_score:
                chunks.append({
                    'text': doc,
                    'metadata': meta or {},
                    'score': round(score, 3)
                })
        
        chunks.sort(key=lambda x: x['score'], reverse=True)
        return [(c, c['score']) for c in chunks[:self.top_k]]

    def build_prompt(self, query: str, retrieved: List[Tuple[Dict, float]]) -> str:
        """Формирует промпт в стиле Qwen Chat с контекстом."""
        # Формируем блок контекста
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved, 1):
            meta = chunk.get('metadata', {})
            source = meta.get('source_file', meta.get('url', 'unknown'))
            header = meta.get('chunk_header', '')
            
            context_parts.append(
                f"[{i}] Источник: {source} (релевантность: {score:.2f})\n"
                f"{header}\n{chunk['text']}\n"
            )
        
        context_block = "\n---\n".join(context_parts) if context_parts else "Контекст не найден."
        
        # Системный промпт
        system = """Ты — ассистент, отвечающий на вопросы по предоставленному контексту.
Правила:
1. Отвечай только на основе приведённых источников.
2. Если информации недостаточно — скажи об этом.
3. В конце ответа перечисли использованные источники в формате [1], [2] и т.д.
4. Будь краток, но точен."""

        # Пользовательский запрос
        user_prompt = f"""Контекст:
{context_block}

Вопрос: {query}

Ответ:"""
        
        # Формат сообщений для Qwen Chat
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt}
        ]
        
        return messages
    
    def ask(self, query: str, history: Optional[List[Dict]] = None) -> RAGResult:
        """Полный цикл: ретривал → промпт → генерация → результат."""
        # 1. Ретривал
        retrieved = self.retrieve(query)
        
        if not retrieved:
            return RAGResult(
                answer=self.fallback_message,
                sources=[],
                query=query,
                context_used=0
            )
        
        # 2. Формирование промпта
        messages = self.build_prompt(query, retrieved)
        if history:
            # Вставляем историю перед последним user-сообщением
            messages = messages[:1] + history + messages[1:]
        
        # 3. Генерация
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
            stream=False,
        )
        
        answer = response["choices"][0]["message"]["content"].strip()
        
        # 4. Формирование источников для вывода
        sources = [
            {
                "title": chunk['metadata'].get('chunk_header', '')[:100],
                "source": chunk['metadata'].get('source_file', chunk['metadata'].get('url', 'unknown')),
                "score": score
            }
            for chunk, score in retrieved
        ]
        
        return RAGResult(
            answer=answer,
            sources=sources,
            query=query,
            context_used=len(retrieved)
        )