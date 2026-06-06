"""Personal memory for Jarvis — what it knows about Ariel.

A single markdown file at ~/.config/jarvis/memory.md, organized in sections.
The daemon reads it once at startup and keeps a live copy in RAM; whenever the
`remember` tool is invoked, both the file and the in-memory copy are updated
so subsequent LLM turns see the new fact.

Why markdown and not JSON: humans should be able to read and edit this file
directly with any editor without breaking it. Sections are just `## Heading`.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from threading import Lock

MEMORY_FILE = Path.home() / ".config" / "jarvis" / "memory.md"

DEFAULT_TEMPLATE = """# Memoria personal de Ariel

> Hechos que Jarvis ha aprendido sobre Ariel. Se carga en cada conversación.
> Puedes editar este archivo directamente; respeta los encabezados `##`.

## Personas

## Proyectos

## Preferencias

## Otros
"""

_LOCK = Lock()
_CACHE: str | None = None


def _ensure() -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text(DEFAULT_TEMPLATE, encoding="utf-8")


def load() -> str:
    """Read memory file fresh from disk. Use at daemon startup."""
    global _CACHE
    _ensure()
    _CACHE = MEMORY_FILE.read_text(encoding="utf-8")
    return _CACHE


def current() -> str:
    """Return cached memory (loaded once, updated in-place by add/forget)."""
    if _CACHE is None:
        return load()
    return _CACHE


def add(fact: str, category: str = "Otros") -> str:
    """Append a fact under the given section. Creates the section if missing."""
    with _LOCK:
        _ensure()
        content = MEMORY_FILE.read_text(encoding="utf-8")
        category = category.strip() or "Otros"
        # Allow synonyms in Spanish
        category = {
            "persona": "Personas", "personas": "Personas", "contacto": "Personas",
            "contactos": "Personas",
            "proyecto": "Proyectos", "proyectos": "Proyectos",
            "preferencia": "Preferencias", "preferencias": "Preferencias",
            "general": "Otros", "otros": "Otros", "otro": "Otros",
        }.get(category.lower(), category)

        line = f"- {fact.strip()} _(añadido {date.today().isoformat()})_"

        header = f"## {category}"
        if header in content:
            # Insert at end of section (before next ## or EOF)
            pattern = re.compile(
                rf"(^{re.escape(header)}\s*\n(?:.*\n)*?)(?=^## |\Z)",
                re.MULTILINE,
            )
            content = pattern.sub(lambda m: m.group(1).rstrip() + "\n" + line + "\n\n", content, count=1)
        else:
            # New section at end
            content = content.rstrip() + f"\n\n{header}\n\n{line}\n"

        MEMORY_FILE.write_text(content, encoding="utf-8")
        global _CACHE
        _CACHE = content
    return f"recordado en '{category}': {fact.strip()}"


def forget(pattern: str) -> str:
    """Remove all lines matching the pattern (case-insensitive substring)."""
    with _LOCK:
        _ensure()
        content = MEMORY_FILE.read_text(encoding="utf-8")
        needle = pattern.strip().lower()
        if not needle:
            return "patrón vacío"
        new_lines = []
        removed = 0
        for line in content.splitlines():
            if line.startswith("- ") and needle in line.lower():
                removed += 1
                continue
            new_lines.append(line)
        content = "\n".join(new_lines) + ("\n" if not content.endswith("\n") else "")
        MEMORY_FILE.write_text(content, encoding="utf-8")
        global _CACHE
        _CACHE = content
    return f"olvidados {removed} hechos que mencionaban '{pattern}'"


def search(query: str) -> str:
    """Substring search across memory lines. For exact recall use."""
    _ensure()
    content = MEMORY_FILE.read_text(encoding="utf-8")
    needle = query.strip().lower()
    hits = [ln for ln in content.splitlines() if ln.startswith("- ") and needle in ln.lower()]
    if not hits:
        return f"no encuentro nada sobre '{query}' en mi memoria"
    return "\n".join(hits[:10])
