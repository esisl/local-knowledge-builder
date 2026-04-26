#!/usr/bin/env python3
"""
crawler.py — загрузка и очистка веб-страниц.
Использует trafilatura для извлечения основного контента.
"""
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, asdict, field
from datetime import datetime
import re

try:
    import trafilatura
    from trafilatura.settings import use_config
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

@dataclass
class CrawledPage:
    """Результат парсинга одной страницы."""
    url: str
    title: str
    content: str  # чистый markdown/text
    word_count: int
    fetched_at: str
    metadata: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save_markdown(self, output_dir: Path, filename: Optional[str] = None):
        """Сохраняет контент в .md файл с метаданными в YAML-фронтматтере."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            # Генерируем имя из URL
            from urllib.parse import urlparse
            domain = urlparse(self.url).netloc.replace(".", "_")
            safe_title = re.sub(r'[^\w\-]', '_', self.title[:50]).strip('_')
            filename = f"{domain}_{safe_title}_{self.fetched_at[:10]}.md"
        
        filepath = output_dir / filename
        
        # Формируем фронтматтер
        frontmatter = [
            "---",
            f"url: {self.url}",
            f"title: {self.title}",
            f"fetched_at: {self.fetched_at}",
            f"word_count: {self.word_count}",
        ]
        for key, value in self.metadata.items():
            frontmatter.append(f"{key}: {value}")
        frontmatter.append("---")
        
        # Записываем файл
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(frontmatter) + "\n\n")
            f.write(self.content)
        
        return filepath


def crawl_url(url: str, timeout_sec: int = 30) -> CrawledPage:
    """
    Загружает и парсит одну страницу.
    
    Возвращает CrawledPage с error=None при успехе, или с описанием ошибки.
    """
    if not HAS_TRAFILATURA:
        return CrawledPage(
            url=url, title="", content="", word_count=0,
            fetched_at=datetime.now().isoformat(),
            error="trafilatura not installed"
        )
    
    try:
        # Настраиваем trafilatura на агрессивную очистку
        config = use_config()
        config.set("DEFAULT", "extensive_cleanup", "true")
        config.set("DEFAULT", "favor_precision", "true")
        
        # Загружаем страницу
        downloaded = trafilatura.fetch_url(url, timeout=timeout_sec)
        if not downloaded:
            return CrawledPage(
                url=url, title="", content="", word_count=0,
                fetched_at=datetime.now().isoformat(),
                error="Failed to fetch URL"
            )
        
        # Парсим контент
        content = trafilatura.extract(
            downloaded,
            config=config,
            output_format="markdown",  # или "txt" для чистого текста
            include_comments=False,
            include_tables=True,
            include_formatting=True,
        )
        
        if not content or len(content.strip()) < 50:
            return CrawledPage(
                url=url, title="", content="", word_count=0,
                fetched_at=datetime.now().isoformat(),
                error="No meaningful content extracted"
            )
        
        # Извлекаем метаданные
        metadata = trafilatura.extract_metadata(downloaded, config=config)
        meta_dict = {
            k: str(v)[:200] for k, v in metadata.__dict__.items() 
            if v and isinstance(v, (str, int))
        } if metadata else {}
        
        # Чистим заголовок
        title = metadata.title if metadata and metadata.title else "Untitled"
        title = re.sub(r'\s+', ' ', title.strip())[:200]
        
        return CrawledPage(
            url=url,
            title=title,
            content=content.strip(),
            word_count=len(content.split()),
            fetched_at=datetime.now().isoformat(),
            metadata=meta_dict
        )
        
    except Exception as e:
        return CrawledPage(
            url=url, title="", content="", word_count=0,
            fetched_at=datetime.now().isoformat(),
            error=f"{type(e).__name__}: {str(e)[:100]}"
        )


def crawl_batch(urls: list[str], output_dir: Path, skip_existing: bool = True) -> list[dict]:
    """
    Парсит список URL, сохраняет каждый в .md.
    Returns: list[dict] с полями {url, status, file/error}
    """
    from urllib.parse import urlparse
    results = []
    
    for url in urls:
        # Проверка на уже скачанные (эвристика по домену)
        if skip_existing:
            domain = urlparse(url).netloc.replace(".", "_")
            if list(output_dir.glob(f"{domain}_*.md")):
                results.append({"url": url, "status": "skipped"})
                continue
        
        page = crawl_url(url)
        if page.error:
            results.append({"url": url, "status": "error", "error": page.error})
            continue
        
        filepath = page.save_markdown(output_dir)
        results.append({"url": url, "status": "saved", "file": filepath.name})
        
    return results
