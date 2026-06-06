"""Tools the LLM can call to control the system."""
import json
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path


def _run(cmd: list[str], timeout: int = 5) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() or r.stderr.strip() or "ok"
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout}s"
    except FileNotFoundError:
        return f"command not found: {cmd[0]}"


def _http_get_json(url: str, timeout: float = 4.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "jarvis/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _browse(url: str, as_app: bool = True) -> str:
    """Open url in the user's preferred browser, as a PWA-style window if possible."""
    for cmd in ("chromium", "google-chrome-stable", "google-chrome", "brave-browser"):
        if shutil.which(cmd):
            arg = f"--app={url}" if as_app else url
            _spawn(f'uwsm-app -- {cmd} "{arg}"')
            return f"abriendo {url}"
    if shutil.which("firefox"):
        _spawn(f'uwsm-app -- firefox "{url}"')
        return f"abriendo {url}"
    _spawn(f'xdg-open "{url}"')
    return f"abriendo {url} con xdg-open"


def _spawn(command: str) -> None:
    """Spawn a shell command detached from jarvis, via Hyprland dispatcher."""
    _run(["hyprctl", "dispatch", f'hl.dsp.exec_cmd([[{command}]])'])


# Apps that are not local binaries — open as a web app in the default browser.
APP_ALIASES: dict[str, str] = {
    "whatsapp": "https://web.whatsapp.com",
    "telegram-web": "https://web.telegram.org",
    "gmail": "https://mail.google.com",
    "correo": "https://mail.google.com",
    "youtube": "https://www.youtube.com",
    "claude": "https://claude.ai",
    "chatgpt": "https://chat.openai.com",
    "calendar": "https://calendar.google.com",
    "calendario": "https://calendar.google.com",
    "google maps": "https://maps.google.com",
    "maps": "https://maps.google.com",
    "drive": "https://drive.google.com",
}


def open_app(name: str) -> str:
    """Launch a graphical application by name.

    Resolution order:
    1. If name matches a known web-app alias (WhatsApp, Gmail, ...) → open as PWA.
    2. If a local binary with that name exists → launch it via uwsm-app.
    3. Try gtk-launch with name.desktop (handles apps that aren't on PATH).
    4. Give up with a clear error.
    """
    key = name.strip().lower()
    if key in APP_ALIASES:
        return _browse(APP_ALIASES[key])

    binary = key.split()[0]
    if shutil.which(binary):
        _spawn(f"uwsm-app -- {key}")
        return f"abriendo {name}"

    # Try .desktop entry via gtk-launch
    if shutil.which("gtk-launch"):
        for candidate in (key, key.replace(" ", "-"), key.replace(" ", "_")):
            r = subprocess.run(
                ["gtk-launch", candidate], capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0:
                return f"abriendo {name}"

    return f"no encuentro la app '{name}'. ¿Está instalada o tiene otro nombre?"


def open_url(url: str) -> str:
    """Open a URL in the user's default browser."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return _browse(url, as_app=False)


def close_window() -> str:
    """Close the currently focused window."""
    return _run(["hyprctl", "dispatch", "hl.dsp.window.close()"])


def switch_workspace(number: int) -> str:
    """Switch to workspace number (1-10)."""
    if not 1 <= number <= 10:
        return f"invalid workspace {number}, must be 1-10"
    return _run(["hyprctl", "dispatch", f'hl.dsp.focus({{ workspace = "{number}" }})'])


def list_windows() -> str:
    """List all open windows with their titles and workspaces."""
    out = _run(["hyprctl", "clients", "-j"], timeout=3)
    return out[:2000]


def take_screenshot() -> str:
    """Take a screenshot of the current screen."""
    return _run(["omarchy-cmd-screenshot"], timeout=10)


def toggle_nightlight() -> str:
    """Toggle blue light filter on/off."""
    return _run(["omarchy", "toggle", "nightlight"], timeout=5)


def run_shell(command: str) -> str:
    """Run an arbitrary shell command. Use with caution.

    Args:
        command: full shell command to execute
    """
    try:
        r = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True, timeout=10,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:1500] if out else "ok (no output)"
    except subprocess.TimeoutExpired:
        return "command timed out"


def set_volume(percent: int) -> str:
    """Set system volume to an absolute percentage (0-100)."""
    if not 0 <= percent <= 100:
        return f"invalid percent {percent}, must be 0-100"
    _run(["pamixer", "--set-volume", str(percent)])
    return f"volumen al {percent}%"


def adjust_volume(delta: int) -> str:
    """Increase or decrease volume by delta percentage points. Negative = lower."""
    flag = "-i" if delta >= 0 else "-d"
    _run(["pamixer", flag, str(abs(delta))])
    new = _run(["pamixer", "--get-volume"])
    return f"volumen ahora al {new}%"


def mute_toggle() -> str:
    """Toggle audio mute on/off."""
    _run(["pamixer", "-t"])
    muted = _run(["pamixer", "--get-mute"])
    return "silenciado" if muted == "true" else "sonido restaurado"


def set_brightness(percent: int) -> str:
    """Set screen brightness to an absolute percentage (1-100)."""
    if not 1 <= percent <= 100:
        return f"invalid percent {percent}, must be 1-100"
    _run(["brightnessctl", "set", f"{percent}%"])
    return f"brillo al {percent}%"


def adjust_brightness(delta: int) -> str:
    """Increase or decrease brightness by delta percentage points. Negative = lower."""
    arg = f"+{delta}%" if delta >= 0 else f"{abs(delta)}%-"
    _run(["brightnessctl", "set", arg])
    return f"brillo ajustado en {delta:+d}%"


def media_control(action: str) -> str:
    """Control the active media player. action in: play, pause, play_pause, next, previous, stop, status."""
    mapping = {
        "play": ["play"],
        "pause": ["pause"],
        "play_pause": ["play-pause"],
        "toggle": ["play-pause"],
        "next": ["next"],
        "previous": ["previous"],
        "prev": ["previous"],
        "stop": ["stop"],
        "status": ["status"],
    }
    args = mapping.get(action.lower())
    if args is None:
        return f"acción desconocida: {action}"
    out = _run(["playerctl", *args])
    if action.lower() in ("status",):
        return f"estado del reproductor: {out}"
    if "No players found" in out:
        return "no hay ningún reproductor activo"
    # Try to add current track info for confirmation
    title = _run(["playerctl", "metadata", "--format", "{{title}} — {{artist}}"])
    if title and "No players found" not in title:
        return f"{action} ok ({title})"
    return f"{action} ok"


def get_weather(location: str) -> str:
    """Get current weather for a city/place name. Uses Open-Meteo (no API key).

    Two HTTP calls: geocoding (name → lat/lon) then forecast.
    """
    try:
        geo = _http_get_json(
            "https://geocoding-api.open-meteo.com/v1/search?"
            + urllib.parse.urlencode({"name": location, "count": 1, "language": "es"})
        )
        results = geo.get("results") or []
        if not results:
            return f"no encuentro la ubicación '{location}'"
        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        label = f"{place.get('name', location)}, {place.get('country_code', '')}".strip(", ")
        fc = _http_get_json(
            "https://api.open-meteo.com/v1/forecast?"
            + urllib.parse.urlencode({
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "timezone": "auto",
            })
        )
        cur = fc.get("current") or {}
        codes = {
            0: "despejado", 1: "casi despejado", 2: "parcialmente nublado", 3: "nublado",
            45: "niebla", 48: "niebla con escarcha",
            51: "llovizna ligera", 53: "llovizna moderada", 55: "llovizna densa",
            61: "lluvia ligera", 63: "lluvia moderada", 65: "lluvia fuerte",
            71: "nieve ligera", 73: "nieve moderada", 75: "nieve fuerte",
            80: "chubascos ligeros", 81: "chubascos moderados", 82: "chubascos fuertes",
            95: "tormenta", 96: "tormenta con granizo ligero", 99: "tormenta con granizo fuerte",
        }
        desc = codes.get(cur.get("weather_code"), f"código {cur.get('weather_code')}")
        return (
            f"En {label}: {desc}, {cur.get('temperature_2m')}°C, "
            f"humedad {cur.get('relative_humidity_2m')}%, "
            f"viento {cur.get('wind_speed_10m')} km/h."
        )
    except Exception as e:
        return f"error obteniendo el tiempo: {e}"


def _wikipedia_title(query: str, lang: str) -> str | None:
    """Find the best Wikipedia title for a query. Tries opensearch (matches title) then full-text search."""
    base = f"https://{lang}.wikipedia.org/w/api.php"
    try:
        opens = _http_get_json(
            base + "?" + urllib.parse.urlencode({
                "action": "opensearch", "search": query, "limit": 1, "format": "json",
            })
        )
        titles = opens[1] if isinstance(opens, list) and len(opens) > 1 else []
        if titles:
            return titles[0]
    except Exception:
        pass
    try:
        full = _http_get_json(
            base + "?" + urllib.parse.urlencode({
                "action": "query", "list": "search", "srsearch": query,
                "srlimit": 1, "format": "json", "srprop": "",
            })
        )
        hits = full.get("query", {}).get("search", [])
        if hits:
            return hits[0]["title"]
    except Exception:
        pass
    return None


def _wikipedia_summary(query: str, lang: str) -> str | None:
    """Return the Wikipedia summary for the best title match in the given language."""
    title = _wikipedia_title(query, lang)
    if not title:
        return None
    try:
        slug = urllib.parse.quote(title.replace(" ", "_"))
        summary = _http_get_json(f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{slug}")
        extract = summary.get("extract")
        if not extract:
            return None
        return f"{extract[:700]} (Wikipedia: {title})"
    except Exception:
        return None


VAULT_ROOT = Path.home() / "Obsidian" / "IT" / "Jarvis"


def _vault_path(path: str) -> Path:
    """Resolve a path relative to the vault, blocking traversal outside it."""
    p = Path(os.path.expanduser(path))
    if not p.is_absolute():
        p = VAULT_ROOT / p
    p = p.resolve()
    if VAULT_ROOT.resolve() not in p.parents and p.resolve() != VAULT_ROOT.resolve():
        raise ValueError(f"path fuera del vault: {p}")
    return p


def read_note(path: str) -> str:
    """Read a markdown note from the vault. Path can be absolute or relative to the vault root."""
    try:
        p = _vault_path(path)
    except ValueError as e:
        return str(e)
    if not p.exists():
        return f"no existe: {p}"
    if not p.is_file():
        return f"no es archivo: {p}"
    try:
        return p.read_text(encoding="utf-8", errors="ignore")[:8000]
    except OSError as e:
        return f"error leyendo: {e}"


def write_note(path: str, content: str) -> str:
    """Write content to a note in the vault. Creates the file (and parents) or overwrites."""
    try:
        p = _vault_path(path)
    except ValueError as e:
        return str(e)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"escrito {len(content)} caracteres en {p.relative_to(VAULT_ROOT)}"
    except OSError as e:
        return f"error escribiendo: {e}"


def append_to_note(path: str, text: str) -> str:
    """Append text to the end of a note. Creates the file if it doesn't exist."""
    try:
        p = _vault_path(path)
    except ValueError as e:
        return str(e)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            if p.stat().st_size > 0:
                f.write("\n")
            f.write(text)
        return f"añadidos {len(text)} caracteres a {p.relative_to(VAULT_ROOT)}"
    except OSError as e:
        return f"error añadiendo: {e}"


def list_vault(folder: str = "") -> str:
    """List markdown files in the vault. Pass a subfolder to scope, or empty for the whole vault."""
    try:
        base = _vault_path(folder) if folder else VAULT_ROOT
    except ValueError as e:
        return str(e)
    if not base.exists():
        return f"no existe: {base}"
    files = sorted(p.relative_to(VAULT_ROOT) for p in base.rglob("*.md"))
    if not files:
        return "vault vacío"
    return "\n".join(str(f) for f in files[:80])


def vault_search(query: str, k: int = 4) -> str:
    """Semantic search over Ariel's Obsidian vault.

    Returns top k matching chunks with their file paths. Use when the user
    asks about something they may have written down, or when context from
    their notes would help answer.
    """
    try:
        from vault_rag import get_rag
    except ImportError as e:
        return f"vault_rag no disponible: {e}"
    try:
        hits = get_rag().search(query, k=max(1, min(int(k), 8)))
    except Exception as e:
        return f"error en vault_search: {e}"
    if not hits:
        return "sin resultados en el vault"
    parts = []
    for h in hits:
        parts.append(f"[{h['path']}]\n{h['text'][:400]}")
    return "\n\n---\n\n".join(parts)


def vault_reindex() -> str:
    """Force re-indexing of the Obsidian vault. Use after big changes."""
    try:
        from vault_rag import get_rag
    except ImportError as e:
        return f"vault_rag no disponible: {e}"
    try:
        stats = get_rag().index(force=True)
    except Exception as e:
        return f"error reindexando: {e}"
    return f"vault re-indexado: {stats}"


def remember(fact: str, category: str = "Otros") -> str:
    """Save a personal fact about Ariel to permanent memory.

    Use whenever Ariel reveals something about himself, his people, projects, or
    preferences. Examples: 'mi novia se llama Stefy', 'soy alérgico al gluten',
    'trabajo en una empresa de fintech', 'me gusta el café americano sin azúcar'.

    Categories: 'Personas' (people in his life), 'Proyectos', 'Preferencias',
    'Otros' (default). Be concise — one fact per call.
    """
    from personal_memory import add
    return add(fact, category)


def forget_fact(pattern: str) -> str:
    """Remove from memory all facts matching the pattern (substring match)."""
    from personal_memory import forget
    return forget(pattern)


def recall(query: str) -> str:
    """Search the personal memory for facts matching the query."""
    from personal_memory import search
    return search(query)


def type_text(text: str) -> str:
    """Type text into the currently focused window (Wayland)."""
    if not shutil.which("wtype"):
        return "wtype no está instalado. Instálalo con: sudo pacman -S wtype"
    try:
        r = subprocess.run(["wtype", text], capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return "wtype timeout"
    if r.returncode != 0:
        return f"wtype falló: {r.stderr.strip()}"
    return f"escrito ({len(text)} caracteres)"


def press_key(combo: str) -> str:
    """Press a key combination in the focused window. Use modifiers + key, e.g.
    'ctrl+c', 'super+l', 'alt+tab', 'return', 'escape', 'pageup'.

    wtype syntax: -M for modifier down, -m for modifier up, -k for key press.
    """
    if not shutil.which("wtype"):
        return "wtype no está instalado. Instálalo con: sudo pacman -S wtype"

    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    if not parts:
        return f"combo vacío: {combo}"
    mods = parts[:-1]
    key = parts[-1]

    # wtype modifier names. Aliases for what humans say:
    mod_map = {"ctrl": "ctrl", "control": "ctrl", "shift": "shift",
               "alt": "alt", "super": "logo", "win": "logo", "meta": "logo"}
    args = ["wtype"]
    for m in mods:
        if m not in mod_map:
            return f"modificador desconocido: {m}"
        args.extend(["-M", mod_map[m]])
    args.extend(["-k", key])
    for m in reversed(mods):
        args.extend(["-m", mod_map[m]])
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return "wtype timeout"
    if r.returncode != 0:
        return f"wtype falló: {r.stderr.strip()}"
    return f"pulsado {combo}"


def claude_code(prompt: str, working_dir: str = "") -> str:
    """Delegate a complex task to Claude Code in headless mode.

    Use this when the request needs more than a few tool calls — generating
    files (PDFs, .py scripts, .md notes), multi-step research, code generation,
    or anything that needs reasoning beyond a single shot.

    Claude Code can read/write files, run commands, search the web. Be explicit
    about WHERE files should go and WHAT format the user wants.

    Args:
        prompt: full task description for Claude. Be specific about output
                location and format. Example: "Investiga sobre quantum computing
                y guarda un resumen en ~/Obsidian/IT/Jarvis/Investigaciones/
                Quantum.md con frontmatter y wikilinks a notas existentes".
        working_dir: optional cwd for Claude (defaults to ~/). Useful when the
                     task is scoped to a directory (e.g. an Obsidian vault).
    """
    if not shutil.which("claude"):
        return "Claude Code no está disponible en este sistema."

    cwd = os.path.expanduser(working_dir.strip() or "~")

    args = [
        "claude",
        "-p", prompt,
        # Daemon can't answer interactive permission prompts — must skip.
        "--dangerously-skip-permissions",
        # Safety net: hard cap per invocation. Tunable if a task needs more.
        "--max-budget-usd", "1.00",
    ]
    try:
        r = subprocess.run(
            args, capture_output=True, text=True, cwd=cwd, timeout=300,
        )
    except subprocess.TimeoutExpired:
        return "Claude Code tardó más de 5 minutos. La tarea era demasiado grande o se quedó colgado."

    output = (r.stdout or "").strip()
    if r.returncode != 0:
        err = (r.stderr or "").strip()[:600]
        return f"Claude Code falló (exit {r.returncode}): {err or output[:200]}"

    if not output:
        return "Claude Code terminó sin output. Probablemente hizo el trabajo (mira los archivos)."

    # Keep the response speakable: trim to ~1500 chars. Full output went to disk
    # via whatever files Claude created.
    if len(output) > 1500:
        return output[:1400] + " […salida truncada]"
    return output


def play_spotify(query: str) -> str:
    """Open Spotify and search for an artist/song/album. Will launch Spotify if not running.

    Note: Spotify's free tier (without OAuth) cannot autoplay a specific track
    from a query. This opens the search results inside Spotify; the user (or a
    follow-up media_control('play')) starts playback.
    """
    if not shutil.which("spotify"):
        return "Spotify no está instalado. Instálalo con `pacman -S spotify-launcher` o desde la web."

    # Launch Spotify if not already running
    already_running = subprocess.run(["pgrep", "-x", "spotify"], capture_output=True).returncode == 0
    if not already_running:
        _spawn("uwsm-app -- spotify")
        # Wait for Spotify to register on MPRIS (up to ~6s)
        for _ in range(30):
            time.sleep(0.2)
            r = subprocess.run(["playerctl", "-l"], capture_output=True, text=True)
            if "spotify" in r.stdout:
                break

    # Tell Spotify to open the search URI. MPRIS OpenUri triggers a navigation
    # to the results page. Quoting is important — the URI must be one token.
    uri = "spotify:search:" + urllib.parse.quote(query)
    r = subprocess.run(
        ["playerctl", "-p", "spotify", "open", uri],
        capture_output=True, text=True, timeout=5,
    )
    if r.returncode != 0:
        return f"abrí Spotify pero falló la búsqueda: {r.stderr.strip() or r.stdout.strip()}"

    # Trigger play. Best-effort: Spotify free may need a click on a specific track.
    subprocess.run(["playerctl", "-p", "spotify", "play"], capture_output=True, timeout=3)
    return f"buscando '{query}' en Spotify"


def web_search(query: str) -> str:
    """Search the web for factual information. Tries Wikipedia (es then en), falls back to DuckDuckGo."""
    for lang in ("es", "en"):
        hit = _wikipedia_summary(query, lang)
        if hit:
            return hit
    try:
        data = _http_get_json(
            "https://api.duckduckgo.com/?"
            + urllib.parse.urlencode({
                "q": query, "format": "json", "no_html": 1, "skip_disambig": 1,
            })
        )
        if data.get("AbstractText"):
            return f"{data['AbstractText']} (fuente: {data.get('AbstractSource', 'DuckDuckGo')})"
        if data.get("Answer"):
            return str(data["Answer"])
    except Exception:
        pass
    return f"no he encontrado información sobre '{query}'"


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Launch a graphical application by name. Supports local binaries (firefox, chromium, spotify, alacritty, nautilus, gimp, ...) AND well-known web apps that open as a PWA in Chromium: whatsapp, telegram-web, gmail, youtube, claude, chatgpt, google maps, calendar, drive. Use this for any 'abre X' / 'open X' request.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "App name. Examples: 'spotify', 'whatsapp', 'firefox', 'gmail'"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a specific URL in the default browser. Use for 'abre la página X' or when the user gives a URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_spotify",
            "description": "Open Spotify and search for an artist, song, or album. Use for 'pon a X', 'reproduce X en Spotify', 'busca X en Spotify'. Launches Spotify if not running.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Artist, song or album name. Examples: 'Drake', 'Bohemian Rhapsody', 'Random Access Memories'"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_window",
            "description": "Close the currently focused window",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_workspace",
            "description": "Switch to a workspace by number (1 through 10)",
            "parameters": {
                "type": "object",
                "properties": {"number": {"type": "integer", "minimum": 1, "maximum": 10}},
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_windows",
            "description": "List all currently open windows and their workspaces",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_screenshot",
            "description": "Take a screenshot of the screen",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "toggle_nightlight",
            "description": "Toggle the blue light filter (night light) on or off",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": "ACTUALLY change the system audio volume to an absolute percentage. Call this whenever the user asks to set volume to a specific value (e.g. 'pon el volumen al 30%').",
            "parameters": {
                "type": "object",
                "properties": {"percent": {"type": "integer", "minimum": 0, "maximum": 100}},
                "required": ["percent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_volume",
            "description": "ACTUALLY raise or lower the system audio volume by a relative delta in percentage points. Call this for requests like 'sube el volumen', 'baja el volumen un poco', 'sube 10%'. Use negative for lower.",
            "parameters": {
                "type": "object",
                "properties": {"delta": {"type": "integer"}},
                "required": ["delta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mute_toggle",
            "description": "ACTUALLY toggle audio mute on/off. Call this when the user says 'silencia', 'mute', 'quita el sonido'.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_brightness",
            "description": "ACTUALLY change the screen brightness to an absolute percentage. Call this when the user asks for a specific brightness value.",
            "parameters": {
                "type": "object",
                "properties": {"percent": {"type": "integer", "minimum": 1, "maximum": 100}},
                "required": ["percent"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_brightness",
            "description": "ACTUALLY raise or lower screen brightness by a relative delta in percentage points. Call this for 'sube/baja el brillo'. Use negative for lower.",
            "parameters": {
                "type": "object",
                "properties": {"delta": {"type": "integer"}},
                "required": ["delta"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "media_control",
            "description": "ACTUALLY control media playback (Spotify, browser audio, etc.). Call this whenever the user says 'pausa', 'reproduce', 'siguiente canción', 'canción anterior', 'para la música'. Valid actions: play, pause, play_pause, next, previous, stop, status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play", "pause", "play_pause", "next", "previous", "stop", "status"],
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city or place. Use when the user asks about the weather, temperature, or conditions somewhere.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City or place name, e.g. 'Tokio', 'Madrid', 'Nueva York'"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for factual information. Use for questions you don't already know or that require current information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": "Read the full contents of a markdown note from Ariel's Obsidian vault. Path can be absolute or relative to the vault root (e.g. 'Conceptos/Tool Calling.md').",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": "Create or overwrite a note in the vault. Path relative to the vault. Use for short notes (< 8000 chars). For complex/long content use claude_code instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "description": "Full markdown content including frontmatter if needed"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_to_note",
            "description": "Append text to the end of an existing note (creates if missing). Use for 'añade esto a la nota X' or quick captures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["path", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_vault",
            "description": "List markdown files in the vault or a subfolder. Use to discover what notes exist before writing/connecting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {"type": "string", "description": "Optional subfolder (e.g. 'Conceptos'). Empty = whole vault."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vault_search",
            "description": "Semantic search over Ariel's Obsidian vault (notas en ~/Obsidian/IT/Jarvis). Use this BEFORE web_search when the question is about something Ariel might have written: his project notes, concepts he's studying, gotchas from his work. Returns the most relevant chunks with their paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language query. Can be a question or keywords."},
                    "k": {"type": "integer", "description": "Number of chunks to return (1-8). Default 4."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vault_reindex",
            "description": "Force re-indexing of the Obsidian vault. Use ONLY when the user explicitly asks ('reindexa el vault', 'actualiza el índice') or after a big bulk change.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Save a personal fact about Ariel to permanent memory. Call this WHENEVER Ariel reveals personal information: names of people in his life ('mi novia se llama Stefy', 'mi entrenador es Pablo'), preferences ('me gusta el café sin azúcar'), projects, allergies, work, etc. One fact per call. Be concise. This is NOT for general knowledge — only for facts about ARIEL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "One concise fact, e.g. 'Stefy es la novia de Ariel'"},
                    "category": {"type": "string", "enum": ["Personas", "Proyectos", "Preferencias", "Otros"], "description": "Section to store under"},
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_fact",
            "description": "Remove all facts from memory whose text contains the given pattern. Use when Ariel says 'olvida que X', 'borra lo de Y'.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Search personal memory. Usually you don't need to call this because memory is already in your system prompt; use only if Ariel asks 'qué sabes sobre X' or to double-check.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text into the currently focused window. Use for 'escribe X en el editor', 'pega esto en el campo activo'. The user must have the target window focused.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press a key combination in the focused window. Examples: 'ctrl+c', 'super+l', 'alt+tab', 'return', 'escape', 'pageup'. Modifiers: ctrl, shift, alt, super.",
            "parameters": {
                "type": "object",
                "properties": {"combo": {"type": "string", "description": "Key combo like 'ctrl+s' or just 'return'"}},
                "required": ["combo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "claude_code",
            "description": "Delegate a COMPLEX task to Claude Code (a more capable AI agent). Use for: generating files (PDFs, Python scripts, markdown notes), multi-step web research that ends in a file, code generation, organizing folders, reading and reasoning about a directory of files, anything beyond a few tool calls. Always be EXPLICIT about output paths and format. Example prompts: 'Investiga X y guárdalo en ~/Obsidian/IT/Jarvis/X.md', 'Crea ~/scripts/foo.py que haga Y', 'Genera un PDF en ~/Desktop/resumen.pdf con el contenido de ~/notas.md'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Full task description in Spanish or English. Be specific about output location and format."},
                    "working_dir": {"type": "string", "description": "Optional working directory. Use the relevant folder when scoped (e.g. '~/Obsidian/IT/Jarvis' for vault tasks)."},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run an arbitrary shell command. Only use when no specialized tool fits.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


DISPATCH = {
    "open_app": open_app,
    "open_url": open_url,
    "play_spotify": play_spotify,
    "close_window": close_window,
    "switch_workspace": switch_workspace,
    "list_windows": list_windows,
    "take_screenshot": take_screenshot,
    "toggle_nightlight": toggle_nightlight,
    "set_volume": set_volume,
    "adjust_volume": adjust_volume,
    "mute_toggle": mute_toggle,
    "set_brightness": set_brightness,
    "adjust_brightness": adjust_brightness,
    "media_control": media_control,
    "get_weather": get_weather,
    "web_search": web_search,
    "read_note": read_note,
    "write_note": write_note,
    "append_to_note": append_to_note,
    "list_vault": list_vault,
    "vault_search": vault_search,
    "vault_reindex": vault_reindex,
    "remember": remember,
    "forget_fact": forget_fact,
    "recall": recall,
    "type_text": type_text,
    "press_key": press_key,
    "claude_code": claude_code,
    "run_shell": run_shell,
}


def call(name: str, args: dict) -> str:
    fn = DISPATCH.get(name)
    if fn is None:
        return f"unknown tool: {name}"
    try:
        return str(fn(**args))
    except TypeError as e:
        return f"bad args for {name}: {e}"
