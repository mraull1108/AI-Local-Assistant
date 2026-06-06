"""RAG over Ariel's Obsidian vault.

Indexes Markdown notes with nomic-embed-text (via Ollama) into a persistent
ChromaDB collection on disk. Re-indexes only files whose content hash changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import chromadb

VAULT_PATH = Path.home() / "Obsidian" / "IT" / "Jarvis"
CACHE_DIR = Path.home() / ".cache" / "jarvis" / "chroma"
COLLECTION = "obsidian-jarvis"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


def _embed(text: str) -> list[float]:
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/embeddings",
        data=json.dumps({"model": EMBED_MODEL, "prompt": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["embedding"]


def _chunk(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE])
        if start + CHUNK_SIZE >= len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


class VaultRAG:
    def __init__(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CACHE_DIR))
        self.coll = self.client.get_or_create_collection(name=COLLECTION)

    def _existing_hashes(self) -> dict[str, str]:
        try:
            data = self.coll.get(include=["metadatas"])
        except Exception:
            return {}
        result: dict[str, str] = {}
        for meta in data.get("metadatas") or []:
            path = meta.get("path")
            h = meta.get("hash")
            if path and h:
                result[path] = h
        return result

    def index(self, force: bool = False) -> dict:
        """Index/reindex changed files. Returns stats dict."""
        if not VAULT_PATH.exists():
            return {"error": f"vault no encontrado en {VAULT_PATH}"}

        files = list(VAULT_PATH.rglob("*.md"))
        existing = {} if force else self._existing_hashes()
        stats = {"checked": 0, "indexed": 0, "skipped": 0}

        for fp in files:
            stats["checked"] += 1
            rel = str(fp.relative_to(VAULT_PATH))
            try:
                content = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            h = hashlib.md5(content.encode("utf-8", errors="ignore")).hexdigest()

            if existing.get(rel) == h:
                stats["skipped"] += 1
                continue

            # Drop old chunks for this file (if any)
            try:
                self.coll.delete(where={"path": rel})
            except Exception:
                pass

            chunks = _chunk(content)
            ids, embeddings, docs, metas = [], [], [], []
            for i, chunk in enumerate(chunks):
                try:
                    emb = _embed(chunk)
                except Exception as e:
                    print(f"[vault_rag] embed failed for {rel}#{i}: {e}", flush=True)
                    continue
                ids.append(f"{rel}#{i}")
                embeddings.append(emb)
                docs.append(chunk)
                metas.append({"path": rel, "hash": h, "chunk_idx": i})

            if ids:
                self.coll.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
                stats["indexed"] += 1

        return stats

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not query.strip():
            return []
        emb = _embed(query)
        result = self.coll.query(query_embeddings=[emb], n_results=k)
        out = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                {
                    "path": meta.get("path", "?"),
                    "text": doc,
                    "distance": float(dist),
                }
            )
        return out


_SINGLETON: VaultRAG | None = None


def get_rag() -> VaultRAG:
    """Lazy-loaded singleton; first call may trigger initial indexing."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = VaultRAG()
        # If the collection is empty, do a one-time bulk index.
        if _SINGLETON.coll.count() == 0:
            print("[vault_rag] indexing vault for first time...", flush=True)
            stats = _SINGLETON.index()
            print(f"[vault_rag] {stats}", flush=True)
    return _SINGLETON


if __name__ == "__main__":
    import sys

    rag = get_rag()
    if len(sys.argv) > 1 and sys.argv[1] == "reindex":
        print(rag.index(force=True))
    elif len(sys.argv) > 1 and sys.argv[1] == "search":
        query = " ".join(sys.argv[2:])
        for hit in rag.search(query):
            print(f"\n— {hit['path']}  (dist={hit['distance']:.3f})")
            print(hit["text"][:300])
    else:
        print(f"collection has {rag.coll.count()} chunks")
        print("usage: vault_rag.py reindex | search <query>")
