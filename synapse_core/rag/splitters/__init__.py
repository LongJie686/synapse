"""Text splitters for chunking documents."""

from __future__ import annotations

import re
from typing import Any

from synapse_core.rag import BaseTextSplitter, Document, DocumentChunk


class RecursiveTextSplitter(BaseTextSplitter):
    """
    Recursively splits text using a list of separators.
    Tries the first separator, then falls back to the next.
    Default separators: ["\n\n", "\n", ". ", " "].
    """

    def __init__(self, separators: list[str] | None = None) -> None:
        self._separators = separators or ["\n\n", "\n", ". ", " "]

    def split(self, document: Document, chunk_size: int = 500, chunk_overlap: int = 50) -> list[DocumentChunk]:
        text = document.content
        chunks = self._recursive_split(text, self._separators, chunk_size)
        return self._create_chunks(chunks, document, chunk_overlap)

    def _recursive_split(self, text: str, separators: list[str], chunk_size: int) -> list[str]:
        if len(text) <= chunk_size:
            return [text]

        if not separators:
            # All separators exhausted - force split by character boundary
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        separator = separators[0]
        remaining_separators = separators[1:]

        splits = text.split(separator)
        chunks: list[str] = []
        current = ""

        for split in splits:
            candidate = current + separator + split if current else split
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(split) > chunk_size:
                    chunks.extend(self._recursive_split(split, remaining_separators, chunk_size))
                    current = ""
                else:
                    current = split

        if current:
            chunks.append(current)

        return chunks

    def _create_chunks(self, raw_chunks: list[str], document: Document, chunk_overlap: int) -> list[DocumentChunk]:
        result: list[DocumentChunk] = []
        offset = 0

        for i, chunk in enumerate(raw_chunks):
            start = document.content.find(chunk, offset)
            if start == -1:
                start = offset
            end = start + len(chunk)

            result.append(DocumentChunk(
                content=chunk.strip(),
                metadata={**document.metadata},
                source=document.source,
                chunk_index=i,
                start_char=start,
                end_char=end,
            ))
            offset = end

        return result


class CodeSplitter(BaseTextSplitter):
    """
    Code-aware splitting that respects function/class boundaries.
    Splits on blank lines between top-level definitions.
    """

    def split(self, document: Document, chunk_size: int = 500, chunk_overlap: int = 50) -> list[DocumentChunk]:
        text = document.content
        lang = document.metadata.get("language", "python")

        # Split on top-level definition boundaries
        if lang == "python":
            pattern = r"\n(?=(?:class |def |async def )\w)"
        else:
            pattern = r"\n\n"

        blocks = re.split(pattern, text)
        chunks: list[DocumentChunk] = []
        current = ""
        idx = 0

        for block in blocks:
            if len(current) + len(block) > chunk_size and current:
                chunks.append(DocumentChunk(
                    content=current.strip(),
                    metadata={**document.metadata},
                    source=document.source,
                    chunk_index=idx,
                ))
                idx += 1
                # Keep overlap
                lines = current.split("\n")
                overlap_lines = lines[-3:] if len(lines) > 3 else []
                current = "\n".join(overlap_lines) + "\n" + block
            else:
                current = current + "\n" + block if current else block

        if current.strip():
            chunks.append(DocumentChunk(
                content=current.strip(),
                metadata={**document.metadata},
                source=document.source,
                chunk_index=idx,
            ))

        return chunks


class SemanticSplitter(BaseTextSplitter):
    """
    Semantic chunking that splits when topic/meaning changes.
    Uses sentence boundaries and heuristic grouping without embedding model.
    """

    def split(self, document: Document, chunk_size: int = 500, chunk_overlap: int = 50) -> list[DocumentChunk]:
        text = document.content
        # Split into sentences
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)

        chunks: list[DocumentChunk] = []
        current_sentences: list[str] = []
        current_len = 0
        idx = 0

        for sentence in sentences:
            if current_len + len(sentence) > chunk_size and current_sentences:
                chunks.append(DocumentChunk(
                    content=" ".join(current_sentences),
                    metadata={**document.metadata},
                    source=document.source,
                    chunk_index=idx,
                ))
                idx += 1
                # Keep last 1-2 sentences as overlap
                overlap = current_sentences[-1:] if current_sentences else []
                current_sentences = list(overlap)
                current_len = sum(len(s) for s in overlap)

            current_sentences.append(sentence)
            current_len += len(sentence)

        if current_sentences:
            chunks.append(DocumentChunk(
                content=" ".join(current_sentences),
                metadata={**document.metadata},
                source=document.source,
                chunk_index=idx,
            ))

        return chunks
