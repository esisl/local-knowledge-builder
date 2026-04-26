#!/usr/bin/env python3
"""
indexer.py — сохранение в ChromaDB и упаковка архива для переноса.
"""
import json
import tarfile
import hashlib
from pathlib import Path
from typing import List, Dict
import chromadb

def save_to_chromadb(chunks_with_embeddings: List[Dict], db_path: Path, collection_name: str = "rag_db"):
    """Создает или перезаписывает коллекцию в ChromaDB."""
    print(f"💾 Сохранение в ChromaDB: {db_path}")

    client = chromadb.PersistentClient(path=str(db_path))

    # Для прототипа пересоздаем коллекцию (в проде нужна инкрементальная логика)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=collection_name)

    ids = [c['id'] for c in chunks_with_embeddings]
    documents = [c['text'] for c in chunks_with_embeddings]
    metadatas = [
        {k: str(v)[:150] for k, v in c.get('metadata', {}).items() if v}
        for c in chunks_with_embeddings
    ]
    embeddings = [c['embedding'] for c in chunks_with_embeddings]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    print(f"📚 Индекс создан. Документов: {collection.count()}")

def export_db(db_path: Path, output_dir: Path, collection_name: str = "rag_db"):
    """Упаковывает БД в .tar.gz с контрольной суммой."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{collection_name}_export.tar.gz"

    print(f"📦 Экспорт в {archive_path}...")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(db_path, arcname=db_path.name)
        
        # Добавляем config.yaml, если он есть в корне
        config_path = db_path.parent.parent / "config.yaml"
        if config_path.exists():
            tar.add(config_path, arcname=config_path.name)

    with open(archive_path, 'rb') as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
        
    print(f"✅ Архив создан. SHA256: {sha256[:16]}...")
    return archive_path, sha256