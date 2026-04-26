#!/usr/bin/env python3
"""
embedder.py — векторизация чанков на CPU.
Использует BAAI/bge-small-en-v1.5 (лёгкая, точная, не конкурирует за VRAM).
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False

def load_model(model_name: str = "BAAI/bge-small-en-v1.5") -> SentenceTransformer:
    if not HAS_ST:
        raise ImportError("Установите: pip install sentence-transformers")
    print(f"🧠 Загрузка эмбеддер-модели: {model_name} (CPU)")
    # device="cpu" гарантирует, что модель не займёт VRAM, нужный для LLM
    return SentenceTransformer(model_name, device="cpu")

def embed_chunks_jsonl(chunks_path: Path, batch_size: int = 32) -> List[Dict]:
    """Загружает JSONL, векторизует тексты, добавляет поле 'embedding'."""
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    if not chunks:
        print("⚠️  Файл чанков пуст")
        return []

    model = load_model()
    texts = [c['text'] for c in chunks]

    print("🔢 Векторизация чанков...")
    # convert_to_tensor=False возвращает numpy array, который легко сериализуется
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)

    # Добавляем векторы к чанкам
    for i, chunk in enumerate(chunks):
        chunk['embedding'] = embeddings[i].tolist()

    print(f"✅ Векторизовано {len(chunks)} чанков")
    return chunks