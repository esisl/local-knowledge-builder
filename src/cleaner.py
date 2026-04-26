#!/usr/bin/env python3
"""
cleaner.py — дедупликация и фильтрация сырых данных.
Использует MinHash для быстрого поиска дублей и эмбеддинги для оценки релевантности.
"""
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
import json
import hashlib

# === MinHash для дедупликации ===
try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False
    # Заглушки, чтобы код не падал при импорте
    class MinHash: pass
    class MinHashLSH: pass

# === Эмбеддинги для фильтрации по теме ===
try:
    from sentence_transformers import SentenceTransformer
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    # Заглушка
    class SentenceTransformer: pass

@dataclass
class CleanedChunk:
    """Готовый к индексации чанк."""
    id: str  # уникальный хеш
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)
    source_file: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)


def compute_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Создаёт MinHash-сигнатуру текста для сравнения."""
    m = MinHash(num_perm=num_perm)
    # Токенизация: простые слова, нижний регистр
    tokens = [t.lower() for t in text.split() if len(t) > 3]
    for token in tokens:
        m.update(token.encode('utf8'))
    return m


def deduplicate_files(raw_dir: Path, threshold: float = 0.85) -> List[Path]:
    """
    Удаляет дубликаты файлов по контенту.
    
    Args:
        raw_dir: папка с .md файлами
        threshold: порог сходства (0.85 = 85% совпадения = дубль)
    
    Returns:
        Список уникальных файлов (пути Path)
    """
    if not HAS_DATASKETCH:
        # Фолбэк: просто возвращаем все файлы, если datasketch не установлен
        return list(raw_dir.glob("*.md"))
    
    files = list(raw_dir.glob("*.md"))
    if not files:
        return []
    
    # LSH для быстрого поиска похожих
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    file_hashes: Dict[str, MinHash] = {}
    unique_files: List[Path] = []
    
    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Читаем только тело, пропускаем YAML-фронтматтер
            content = f.read()
            if content.startswith('---'):
                parts = content.split('---', 2)
                content = parts[2] if len(parts) > 2 else content
        
        # Считаем хеш
        m = compute_minhash(content)
        file_id = filepath.name
        
        # Проверяем на дубли
        duplicates = lsh.query(m)
        if not duplicates:
            # Новый уникальный файл
            lsh.insert(file_id, m)
            file_hashes[file_id] = m
            unique_files.append(filepath)
        else:
            # Дубль: сравниваем с ближайшим, чтобы убедиться
            best_match = duplicates[0]
            similarity = file_hashes[best_match].jaccard(m)
            if similarity < threshold:
                # Ложное срабатывание — добавляем
                lsh.insert(file_id, m)
                file_hashes[file_id] = m
                unique_files.append(filepath)
            # else: пропускаем как дубль
    
    print(f"🗑️  Дедупликация: {len(files)} → {len(unique_files)} уникальных файлов")
    return unique_files


def filter_by_relevance(chunks: List[Dict], topic: str, 
                        min_score: float = 0.6) -> List[Dict]:
    """
    Отфильтровывает чанки, нерелевантные теме.
    
    Использует BGE-small эмбеддинги (на CPU).
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        print("⚠️  sentence-transformers не установлен — пропуск фильтрации по релевантности")
        return chunks
    
    try:
        # Загружаем модель на CPU (не конкурируем за VRAM с LLM)
        device = "cpu"
        model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
        
        # Эмбеддинг темы
        topic_embedding = model.encode([topic], show_progress_bar=False)[0]
        
        # Пакетное кодирование чанков
        texts = [c['text'] for c in chunks]
        chunk_embeddings = model.encode(texts, show_progress_bar=False, batch_size=16)
        
        # Косинусное сходство
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity([topic_embedding], chunk_embeddings)[0]
        
        # Фильтрация
        filtered = [c for c, s in zip(chunks, scores) if s >= min_score]
        print(f"🎯 Фильтрация по релевантности: {len(chunks)} → {len(filtered)} чанков (порог {min_score})")
        return filtered
        
    except Exception as e:
        print(f"⚠️  Ошибка фильтрации по релевантности: {e}")
        return chunks  # Возвращаем всё, если не удалось отфильтровать


def load_markdown_file(filepath: Path) -> Optional[Dict]:
    """Загружает .md файл, парсит фронтматтер и контент."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {}
        body = content
        
        # Парсим YAML-фронтматтер (простой парсер)
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                body = parts[2].strip()
                for line in frontmatter.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
        
        return {
            'filepath': str(filepath),
            'filename': filepath.name,
            'metadata': metadata,
            'content': body,
            'word_count': len(body.split())
        }
    except Exception as e:
        print(f"⚠️  Ошибка чтения {filepath}: {e}")
        return None