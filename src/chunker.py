#!/usr/bin/env python3
"""
chunker.py — разбиение текста на чанки для RAG.
Сохраняет заголовки, код и метаданные.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
import re
import hashlib
from src.cleaner import CleanedChunk

# === Простой маркдаун-парсер для сохранения структуры ===
def split_by_headers(text: str, max_tokens: int = 700, overlap: int = 100) -> List[Dict]:
    """
    Разбивает текст по заголовкам, затем дробит крупные блоки.
    
    Args:
        text: markdown-текст
        max_tokens: целевой размер чанка в токенах (приблизительно)
        overlap: перекрытие между чанками (токены)
    
    Returns:
        Список {text, header_path, start_pos, end_pos}
    """
    # Простая эвристика: 1 токен ≈ 4 символа для оценки
    max_chars = max_tokens * 4
    overlap_chars = overlap * 4
    
    # Находим заголовки (#, ##, ### и т.д.)
    header_pattern = r'^(#{1,6})\s+(.+)$'
    lines = text.split('\n')
    
    sections = []
    current_header = "Document"
    current_content = []
    
    for line in lines:
        match = re.match(header_pattern, line.strip())
        if match:
            # Сохраняем предыдущий раздел
            if current_content:
                sections.append({
                    'header': current_header,
                    'content': '\n'.join(current_content)
                })
            # Новый заголовок
            level = len(match.group(1))
            title = match.group(2).strip()
            current_header = f"{'#' * level} {title}"
            current_content = [line]
        else:
            current_content.append(line)
    
    # Последний раздел
    if current_content:
        sections.append({
            'header': current_header,
            'content': '\n'.join(current_content)
        })
    
    # Теперь дробим крупные разделы
    chunks = []
    for section in sections:
        content = section['content']
        header = section['header']
        
        if len(content) <= max_chars:
            # Целиком в один чанк
            chunks.append({
                'text': f"{header}\n\n{content}".strip(),
                'header_path': header,
                'word_count': len(content.split())
            })
        else:
            # Дробим по предложениям/абзацам
            # Простой подход: разбиваем по двойным переносам
            paragraphs = re.split(r'\n\s*\n', content)
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para)
                if current_size + para_size > max_chars and current_chunk:
                    # Сохраняем текущий чанк
                    joined = '\n\n'.join(current_chunk)
                    chunk_text = f"{header}\n\n{joined}".strip()
                    chunks.append({
                        'text': chunk_text,
                        'header_path': header,
                        'word_count': len(chunk_text.split())
                    })
                    # Переносим часть в следующий чанк (overlap)
                    if overlap_chars > 0 and len(current_chunk) > 1:
                        current_chunk = current_chunk[-1:]  # Оставляем последний абзац
                        current_size = len(current_chunk[0])
                    else:
                        current_chunk = []
                        current_size = 0
                
                current_chunk.append(para)
                current_size += para_size
            
            # Последний чанк раздела
            if current_chunk:
                joined = '\n\n'.join(current_chunk)
                chunk_text = f"{header}\n\n{joined}".strip()
                chunks.append({
                    'text': chunk_text,
                    'header_path': header,
                    'word_count': len(chunk_text.split())
                })
    
    return chunks


def create_chunks_from_file(filepath: Path, 
                           max_tokens: int = 700, 
                           overlap: int = 100) -> List[CleanedChunk]:
    """
    Создаёт чанки из одного .md файла.
    
    Returns:
        Список CleanedChunk с уникальными ID и метаданными.
    """
    from src.cleaner import load_markdown_file
    
    doc = load_markdown_file(filepath)
    if not doc:
        return []
    
    # Разбиваем текст
    raw_chunks = split_by_headers(doc['content'], max_tokens, overlap)
    
    chunks = []
    for i, chunk in enumerate(raw_chunks):
        # Уникальный ID: хеш от контента + источника
        chunk_id = hashlib.sha256(
            f"{doc['filename']}:{i}:{chunk['text'][:100]}".encode('utf-8')
        ).hexdigest()[:16]
        
        # Метаданные чанка
        metadata = {
            **doc['metadata'],
            'chunk_header': chunk['header_path'],
            'source_file': doc['filename'],
            'chunk_index': i,
            'total_chunks': len(raw_chunks),
            'word_count': chunk['word_count']
        }
        
        chunks.append(CleanedChunk(
            id=chunk_id,
            text=chunk['text'],
            metadata=metadata,
            source_file=doc['filename'],
            chunk_index=i,
            total_chunks=len(raw_chunks)
        ))
    
    return chunks


def process_directory(raw_dir: Path, output_dir: Path,
                     max_tokens: int = 700, overlap: int = 100,
                     dedup_threshold: float = 0.85) -> Dict[str, int]:
    """
    Полный пайплайн: загрузка → дедупликация → чанкинг → сохранение.
    
    Returns:
        Статистика обработки.
    """
    from src.cleaner import deduplicate_files
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Дедупликация файлов
    unique_files = deduplicate_files(raw_dir, dedup_threshold)
    
    # 2. Чанкинг каждого файла
    all_chunks: List[CleanedChunk] = []
    for filepath in unique_files:
        file_chunks = create_chunks_from_file(filepath, max_tokens, overlap)
        all_chunks.extend(file_chunks)
    
    # 3. Сохранение в JSONL
    output_file = output_dir / "chunks.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')
    
    stats = {
        'files_processed': len(unique_files),
        'chunks_created': len(all_chunks),
        'output_file': str(output_file)
    }
    
    print(f"📦 Создано {len(all_chunks)} чанков из {len(unique_files)} файлов")
    print(f"💾 Сохранено в: {output_file}")
    
    return stats