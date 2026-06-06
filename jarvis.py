"""Jarvis: local voice-controlled AI assistant.

Modes:
    uv run jarvis.py               # interactive loop (Enter to record)
    uv run jarvis.py --once        # one cycle then exit
    uv run jarvis.py --text "..."  # skip STT, use given text
    uv run jarvis.py --daemon      # long-running: load once, listen on FIFO
    uv run jarvis.py --trigger     # write to FIFO so a running daemon does one cycle
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make CUDA libs from pip wheels (nvidia-cublas-cu12, nvidia-cudnn-cu12) visible
# to ctranslate2 / faster-whisper. Must happen BEFORE importing those libs.
_VENV_SITE = Path(__file__).parent / ".venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
_LIBS = ":".join(str(_VENV_SITE / "nvidia" / sub / "lib") for sub in ("cublas", "cudnn", "cuda_nvrtc"))
os.environ["LD_LIBRARY_PATH"] = _LIBS + ":" + os.environ.get("LD_LIBRARY_PATH", "")

# Force HuggingFace hub to NOT hit the network when loading models.
# Without this, faster-whisper does a HEAD check on every startup; if the
# network is down (or just slow) the daemon hangs in "Cargando Whisper..."
# until the request times out (~2 min).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import queue
import re
import subprocess
import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig
import ollama

import personal_memory
import tools

ROOT = Path(__file__).parent
VOICE_PATH = ROOT / "voices" / "es_ES-davefx-medium.onnx"
FIFO_PATH = ROOT / ".trigger.fifo"

WHISPER_MODEL = "small"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
SAMPLE_RATE = 16000
MAX_RECORD_S = 20
# Seconds of silence after speech that ends recording. 1.2 s era demasiado
# corto: cortaba al hacer pausas naturales entre cláusulas. 2.0 s deja
# margen para "Hola Jarvis... [pausa pensando]... abre WhatsApp" sin
# disparar el cut-off prematuro.
SILENCE_S = 2.0
SILENCE_RMS = 0.015
# Piper speech speed: 1.0 = default voice speed, >1.0 = slower, <1.0 = faster.
# 1.15 sounds noticeably calmer in es_ES-davefx-medium without becoming sluggish.
TTS_LENGTH_SCALE = 1.15
_SYN_CONFIG = SynthesisConfig(length_scale=TTS_LENGTH_SCALE)

SYSTEM_PROMPT_TEMPLATE = """Eres Jarvis, un asistente de voz que CONTROLA un PC Linux (Omarchy/Hyprland) para Ariel.

REGLA #1 — la más importante:
Las acciones reales del sistema solo ocurren si EMITES una tool_call estructurada. Si solo dices "vale, subo el volumen" sin emitir tool_call, NO PASA NADA en el sistema — estás mintiendo. Llama SIEMPRE a la tool apropiada para acciones del sistema.

Si el usuario pide MÚLTIPLES acciones en un mismo turno (ej: "abre WhatsApp y dime el clima"), emite las tool_calls UNA POR UNA. No combines todo en una respuesta de texto. No describas la acción ANTES de ejecutarla — primero el tool_call, después el texto de confirmación.

Mapeo rápido:
- "sube/baja/pon volumen", "silencia" → adjust_volume / set_volume / mute_toggle
- "sube/baja/pon brillo" → adjust_brightness / set_brightness
- "pausa", "reproduce", "siguiente canción", "anterior" → media_control
- "abre X", "cierra ventana", "cambia al workspace N" → open_app / close_window / switch_workspace
- "abre la página/web Y" → open_url
- "pon a <artista> en spotify", "reproduce <canción>" → play_spotify
- "qué tiempo hace en X" → get_weather (pasa solo el nombre del lugar)
- preguntas sobre lo que ARIEL ha escrito/anotado/estudiado (proyecto Jarvis, conceptos personales, gotchas suyas) → vault_search PRIMERO. Si no devuelve nada útil, fallback a web_search.
- preguntas factuales generales (Wikipedia-style) → web_search (formula la query en términos clave)
- "investiga X y guárdalo en...", "genera un PDF/script/nota con Y", "escribe un archivo que Z", "ordena la carpeta W", "lee y resume las notas de V" → claude_code (pasa working_dir con el directorio relevante si la tarea es sobre un directorio concreto, por ejemplo "~/Obsidian/IT/Jarvis" para tareas sobre el vault). Sé explícito en el prompt sobre la ruta de salida y el formato.

Estilo:
- Responde SIEMPRE en español, breve y natural — esto se reproduce por voz.
- Después de ejecutar una tool, confirma brevemente lo hecho.
- Sin listas, sin código, sin markdown — solo prosa.

MEMORIA — REGLAS CRÍTICAS:
1) Lo que sabes sobre Ariel está abajo. Úsalo SIEMPRE en tus respuestas sin preguntar de nuevo.
2) Si Ariel revela información nueva sobre sí mismo (nombres de gente, lugares, gustos, hechos personales), llama a la tool `remember` con esa información. La tool es la ÚNICA manera de guardar en memoria: si solo dices "lo recuerdo" sin invocar la tool, NADA se guarda.
3) Si Ariel pregunta algo que YA está en la memoria abajo, responde directo desde ahí. NO llames a `recall` ni anuncies que vas a "buscar" — la memoria YA está en este prompt.

CONOCIMIENTO ACTUAL SOBRE ARIEL:
{memory}
"""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(memory=personal_memory.current())


def notify(text: str, urgency: str = "low") -> None:
    """Send a desktop notification via mako/notify-send if available."""
    try:
        subprocess.run(
            ["notify-send", "-u", urgency, "-t", "3000", "-a", "Jarvis", "Jarvis", text],
            check=False, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def record_with_vad() -> np.ndarray:
    block = int(SAMPLE_RATE * 0.1)
    audio = []
    silent_blocks = 0
    speech_started = False
    start = time.monotonic()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=block) as stream:
        while time.monotonic() - start < MAX_RECORD_S:
            data, _ = stream.read(block)
            audio.append(data.copy())
            rms = float(np.sqrt(np.mean(data**2)))
            if rms > SILENCE_RMS:
                speech_started = True
                silent_blocks = 0
            elif speech_started:
                silent_blocks += 1
                if silent_blocks * 0.1 >= SILENCE_S:
                    break
    return np.concatenate(audio).flatten() if audio else np.zeros(0, dtype=np.float32)


def transcribe(model: WhisperModel, audio: np.ndarray) -> str:
    if audio.size == 0:
        return ""
    segments, _ = model.transcribe(audio, language="es", vad_filter=True, beam_size=1)
    return "".join(s.text for s in segments).strip()


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_TTS_END = object()


def _split_sentences(text: str) -> list[str]:
    """Coarse sentence splitter for TTS streaming. Short fragments get glued
    to the previous sentence so we don't ship 2-syllable chunks to Piper."""
    parts = _SENTENCE_SPLIT.split(text.strip())
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if out and len(p) < 8:
            out[-1] = (out[-1] + " " + p).strip()
        else:
            out.append(p)
    return out or [text.strip()]


def speak(voice: PiperVoice, text: str) -> None:
    """Synthesize sentence-by-sentence in a background thread and play in order.

    Cuts time-to-first-audio: the user hears the start of the response as soon
    as Piper finishes the FIRST sentence, while later sentences are still
    being synthesized in parallel.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return

    audio_q: queue.Queue = queue.Queue(maxsize=8)

    def produce() -> None:
        try:
            for s in sentences:
                chunks: list[np.ndarray] = []
                sr: int | None = None
                for chunk in voice.synthesize(s, syn_config=_SYN_CONFIG):
                    chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
                    sr = chunk.sample_rate
                if chunks and sr is not None:
                    audio_q.put((np.concatenate(chunks), sr))
        finally:
            audio_q.put(_TTS_END)

    threading.Thread(target=produce, daemon=True).start()
    while True:
        item = audio_q.get()
        if item is _TTS_END:
            return
        audio, sr = item
        sd.play(audio, samplerate=sr)
        sd.wait()


_TOOL_NAMES = {t["function"]["name"] for t in tools.TOOLS_SCHEMA}


def _recover_simulated_tool_call(content: str) -> tuple[str, dict] | None:
    """Qwen 2.5 7B sometimes emits a tool call as raw text instead of structured
    tool_calls. Try to recover one across known text formats."""
    if not content:
        return None

    # Strip common wrappers: [tool_call ...], <tool_call>...</tool_call>
    stripped = content
    stripped = re.sub(r'\[\s*tool[_ ]?call\s*', '[', stripped, flags=re.IGNORECASE)
    stripped = re.sub(r'</?tool[_ ]?call[^>]*>', '', stripped, flags=re.IGNORECASE)

    patterns = [
        # name({"arg": value, ...})    or   name(arg="value", ...)
        r'\b([a-z_]+)\s*\(\s*(\{[^()]*\})\s*\)',
        # [name, {"arg": value, ...}]   or   ["$name", {...}]
        r'\[\s*"?\$?([a-z_]+)"?\s*,\s*(\{[^()]*\})\s*\]',
        # [name({"arg": value})]
        r'\[\s*([a-z_]+)\s*\(\s*(\{[^()]*\})\s*\)\s*\]',
    ]
    for pat in patterns:
        m = re.search(pat, stripped, re.DOTALL)
        if m and m.group(1) in _TOOL_NAMES:
            try:
                return m.group(1), json.loads(m.group(2))
            except json.JSONDecodeError:
                continue

    # Pattern: name(arg1="value1", arg2="value2") — keyword args, not JSON
    m = re.search(r'\b([a-z_]+)\s*\(\s*([a-z_]+\s*=\s*"[^"]*"(?:\s*,\s*[a-z_]+\s*=\s*"[^"]*")*)\s*\)', stripped)
    if m and m.group(1) in _TOOL_NAMES:
        args = {}
        for kv in re.finditer(r'([a-z_]+)\s*=\s*"([^"]*)"', m.group(2)):
            args[kv.group(1)] = kv.group(2)
        return m.group(1), args

    return None


_ACTION_CLAIM_PATTERNS = (
    re.compile(r"\bvoy a (?:abr|envi|sub|baj|silenc|paus|reprod|cerr|cambi|lanz|escrib|busc|cre[ao]|pong|reproduz)", re.IGNORECASE),
    re.compile(r"\b(?:he|estoy|estaba|estaré|estare) (?:abr|envi|sub|baj|silenc|paus|reprod|cerr|cambi|lanz|escrib|busc|cre[ao]|pong)", re.IGNORECASE),
    re.compile(r"\b(?:abriendo|enviando|subiendo|bajando|silenciando|pausando|reproduciendo|cerrando|cambiando|lanzando|escribiendo|buscando|abrindo|creando|poniendo)\b", re.IGNORECASE),
    re.compile(r"\b(?:abrí|envié|subí|bajé|silencié|pausé|reproduje|cerré|cambié|lancé|escribí|busqué|puse|creé)\b", re.IGNORECASE),
    re.compile(r"\bhe (?:abierto|enviado|subido|bajado|silenciado|pausado|reproducido|cerrado|cambiado|lanzado|escrito|buscado|puesto|creado)\b", re.IGNORECASE),
    # Present tense, 1st person ("abro WhatsApp", "envío el mensaje")
    re.compile(r"\b(?:abro|envío|envio|subo|bajo|silencio|pauso|reproduzco|cierro|cambio|lanzo|escribo|busco|pongo|creo)\b", re.IGNORECASE),
)


def _looks_like_action_claim(text: str) -> bool:
    """True if the assistant's text describes performing an action. Used to
    detect hallucinations where the model claims it acted but emitted no
    tool_call. Cheap heuristic — a few false positives are fine, they just
    trigger one extra round-trip."""
    return any(p.search(text) for p in _ACTION_CLAIM_PATTERNS)


_NUDGE_MSG = (
    "Acabas de describir una acción pero NO emitiste un tool_call. "
    "Las acciones solo ocurren si emites un tool_call estructurado. "
    "Emite ahora el tool_call correspondiente (open_app, get_weather, "
    "media_control, etc.), SIN volver a describir la acción en texto."
)


def chat_with_tools(client: ollama.Client, history: list[dict]) -> str:
    initial_len = len(history)
    nudged = False
    for _ in range(5):
        resp = client.chat(
            model=OLLAMA_MODEL,
            messages=history,
            tools=tools.TOOLS_SCHEMA,
            # temperature=0 makes tool-call emission as consistent as possible.
            options={"temperature": 0.0},
        )
        msg = resp["message"]
        history.append(msg)
        tool_calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()

        # Defensive recovery: if no structured tool_calls came but the content
        # looks like a tool call written as text, treat it as one.
        if not tool_calls:
            recovered = _recover_simulated_tool_call(content)
            if recovered:
                fn, args = recovered
                result = tools.call(fn, args)
                print(f"  🔧 (recovered) {fn}({args}) -> {result[:80]}", flush=True)
                history.append({"role": "tool", "content": result, "name": fn})
                continue

            # Hallucinated action: the model claims it did something but
            # emitted no tool_call. Drop the lying message from history,
            # nudge once with a user-role correction, and retry. If we've
            # already nudged in this turn, give up and return — but still
            # drop the offending message so it doesn't poison future turns.
            if _looks_like_action_claim(content):
                history.pop()
                if not nudged:
                    nudged = True
                    history.append({"role": "user", "content": _NUDGE_MSG})
                    print("  ⚠️  alucinación detectada, reintentando con nudge", flush=True)
                    continue
                print("  ⚠️  alucinación persiste tras nudge, devuelvo texto sin guardar", flush=True)
                return content

            return content

        for tc in tool_calls:
            fn = tc["function"]["name"]
            args = tc["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            result = tools.call(fn, args)
            print(f"  🔧 {fn}({args}) -> {result[:80]}", flush=True)
            history.append({"role": "tool", "content": result, "name": fn})
    # Hit the iteration cap. The assistant+tool messages we accumulated this
    # turn are a string of failures; keeping them in history poisons future
    # turns (the model "learns" that tool calls don't work and starts
    # hallucinating actions). Drop everything we added this turn.
    del history[initial_len:]
    return "He tenido problemas ejecutando esa acción, ¿lo intentamos de otra forma?"


RESET_PHRASES = (
    "olvida lo anterior", "olvídalo", "olvidalo", "resetea contexto",
    "borra el contexto", "empieza de cero", "reinicia conversación",
    "reinicia conversacion", "nueva conversación", "nueva conversacion",
)
MAX_HISTORY_MSGS = 24  # turns get compacted after this

# Wake word config. openwakeword's hey_jarvis is a pre-trained model; we use
# the ONNX runtime since onnxruntime is already loaded transitively via
# piper-tts. 0.5 is the library default; bump if false positives bite, lower
# if it misses the cue.
WAKE_WORD_NAME = "hey_jarvis"
WAKE_WORD_THRESHOLD = 0.5
WAKE_SR = 16000
WAKE_CHUNK = 1280  # 80 ms at 16 kHz — the canonical openwakeword frame


class Jarvis:
    """Holds preloaded models and a persistent conversation history."""

    def __init__(self, with_stt: bool = True) -> None:
        print("Cargando Piper...", flush=True)
        self.voice = PiperVoice.load(str(VOICE_PATH))
        print("Conectando a Ollama...", flush=True)
        self.client = ollama.Client()
        self.whisper: WhisperModel | None = None
        if with_stt:
            print("Cargando Whisper...", flush=True)
            self.whisper = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        # Load personal memory once; tools update it in-place.
        personal_memory.load()
        # Conversation history persists across turns until a reset phrase.
        self.history: list[dict] = []
        # Wake word state. `wake_paused` is set while one_cycle owns the mic
        # so the listener thread releases its InputStream first (PortAudio
        # would otherwise fail with "device busy"). The model is lazy-loaded
        # only when start_wake_word() is invoked.
        self.wake_paused = threading.Event()
        self._oww = None
        self._wake_thread: threading.Thread | None = None

    def _trim_history(self) -> None:
        """Keep history bounded without splitting a tool_call/tool_result pair.

        A naive tail-slice can leave an orphan `tool` message at the start
        (whose preceding assistant tool_call was dropped). Ollama rejects that
        and the daemon errors out, so advance the trim point until the first
        kept message is a clean user/assistant turn.
        """
        if len(self.history) <= MAX_HISTORY_MSGS:
            return
        trimmed = self.history[-MAX_HISTORY_MSGS:]
        while trimmed and trimmed[0].get("role") == "tool":
            trimmed = trimmed[1:]
        self.history = trimmed

    def _maybe_auto_remember(self, user_text: str, history: list[dict]) -> None:
        """Fallback: if the user revealed personal info and the LLM didn't
        emit `remember`, run a second mini-call to extract and save."""
        # Skip if remember was already invoked this turn
        for msg in history:
            if isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("name") == "remember":
                return
        cues = ("mi ", "mis ", "soy ", "me llamo", "me gusta", "me gustan",
                "prefiero", "trabajo en", "vivo en", "tengo un", "tengo una",
                "mi novia", "mi novio", "mi amigo", "mi amiga", "mi madre",
                "mi padre", "mi hermano", "mi hermana", "mi jefe", "mi pareja",
                "soy alérgico", "soy alergico")
        low = user_text.lower()
        if not any(c in low for c in cues):
            return
        prompt = (
            "Del texto, extrae UN hecho personal sobre el usuario. Responde SOLO un JSON con la forma "
            '{"fact": "...", "category": "Personas|Proyectos|Preferencias|Otros"} '
            'o {} si no hay hecho personal claro. El hecho debe ser una frase corta y autosuficiente '
            '(ej: "Stefy es la novia de Ariel", no "Stefy"). '
            f'Texto: "{user_text}"'
        )
        try:
            resp = self.client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
            )
            content = (resp["message"].get("content") or "").strip()
            m = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if not m:
                return
            data = json.loads(m.group(0))
            fact = data.get("fact", "").strip()
            if not fact:
                return
            result = tools.call("remember", {"fact": fact, "category": data.get("category", "Otros")})
            print(f"  💾 (auto-remember) {result}", flush=True)
        except Exception as e:
            print(f"[auto-remember falló: {e}]", flush=True)

    def handle_text(self, user_text: str) -> None:
        if not user_text:
            print("(vacío)", flush=True)
            return
        print(f"👤 {user_text}", flush=True)

        lowered = user_text.strip().lower().rstrip(".!?¿¡")
        if any(p in lowered for p in RESET_PHRASES):
            self.history = []
            print("🤖 (contexto borrado)", flush=True)
            speak(self.voice, "Vale, empezamos de cero.")
            return

        # Rebuild system prompt each turn so updated personal memory is visible
        system_msg = {"role": "system", "content": build_system_prompt()}
        history = [system_msg] + self.history + [{"role": "user", "content": user_text}]

        reply = chat_with_tools(self.client, history)
        self._maybe_auto_remember(user_text, history)
        # history was mutated by chat_with_tools — keep everything after the system msg
        self.history = history[1:]
        self._trim_history()

        print(f"🤖 {reply}", flush=True)
        if reply:
            speak(self.voice, reply)

    def one_cycle(self) -> None:
        assert self.whisper is not None
        # Reclaim the mic from the wake-word thread (if any) before recording.
        # The 150 ms sleep gives the listener time to exit its sd.RawInputStream
        # context — otherwise PortAudio errors with "Device unavailable".
        self.wake_paused.set()
        time.sleep(0.15)
        try:
            notify("🎙️ Escuchando...")
            audio = record_with_vad()
            notify("Procesando...")
            text = transcribe(self.whisper, audio)
            if not text:
                notify("No se entendió nada", urgency="normal")
                print("(silencio o no se entendió)", flush=True)
                return
            self.handle_text(text)
        finally:
            # Half-second grace period so the TTS tail doesn't bleed into the
            # wake-word listener and re-trigger on Jarvis's own voice.
            time.sleep(0.5)
            self.wake_paused.clear()

    def start_wake_word(self) -> bool:
        """Spin up the background wake-word listener. Returns False if the
        model couldn't be loaded (logged) so the daemon can keep running with
        only the SUPER+H trigger path."""
        if self._wake_thread is not None:
            return True
        try:
            from openwakeword.model import Model
            print("Cargando wake-word (hey_jarvis)...", flush=True)
            self._oww = Model(
                wakeword_models=[WAKE_WORD_NAME],
                inference_framework="onnx",
            )
        except Exception as e:
            print(f"[wake-word] no se pudo cargar el modelo: {e}", flush=True)
            self._oww = None
            return False
        self._wake_thread = threading.Thread(
            target=self._wake_word_loop, name="wake-word", daemon=True,
        )
        self._wake_thread.start()
        return True

    def _wake_word_loop(self) -> None:
        """Continuously read mic at 16 kHz int16, run openwakeword on each
        80 ms chunk, fire the FIFO trigger when 'hey_jarvis' crosses the
        threshold. Releases the audio device whenever wake_paused is set."""
        assert self._oww is not None
        while True:
            # External pause (one_cycle is running) — wait, don't hold the mic.
            while self.wake_paused.is_set():
                time.sleep(0.1)
            try:
                with sd.RawInputStream(
                    samplerate=WAKE_SR, channels=1, dtype="int16",
                    blocksize=WAKE_CHUNK,
                ) as stream:
                    while not self.wake_paused.is_set():
                        data, _overflow = stream.read(WAKE_CHUNK)
                        audio = np.frombuffer(bytes(data), dtype=np.int16)
                        scores = self._oww.predict(audio)
                        score = float(scores.get(WAKE_WORD_NAME, 0.0))
                        if score > WAKE_WORD_THRESHOLD:
                            print(f"  🔔 {WAKE_WORD_NAME} (score={score:.2f})", flush=True)
                            # Pause self until one_cycle finishes — prevents
                            # double-fire while we're still reading the same
                            # phrase, and frees the mic for VAD.
                            self.wake_paused.set()
                            # Reset openwakeword's internal buffer so the
                            # next listen starts clean (otherwise lingering
                            # frames from this detection can re-trigger).
                            self._oww.reset()
                            send_trigger("go")
                            break
            except Exception as e:
                print(f"[wake-word] {e}", flush=True)
                time.sleep(1.0)


def run_daemon() -> int:
    """Long-running daemon: load models once, do a cycle each time someone writes to FIFO."""
    if FIFO_PATH.exists() and not FIFO_PATH.is_fifo():
        FIFO_PATH.unlink()
    if not FIFO_PATH.exists():
        os.mkfifo(FIFO_PATH, 0o600)

    j = Jarvis(with_stt=True)
    if j.start_wake_word():
        print("Wake-word activo: di 'Hey Jarvis' o pulsa SUPER+H", flush=True)
    else:
        print("Wake-word no disponible — solo SUPER+H", flush=True)
    notify("Jarvis listo")
    print(f"Daemon listo. Trigger: {FIFO_PATH}", flush=True)
    while True:
        # Opening O_RDONLY blocks until someone writes; we re-open each cycle to drain.
        with open(FIFO_PATH, "r") as f:
            cmd = f.read().strip()
        if cmd == "quit":
            print("recibido quit, saliendo", flush=True)
            return 0
        try:
            j.one_cycle()
        except Exception as e:  # don't kill the daemon on a single bad turn
            print(f"error: {e}", flush=True)
            notify(f"Error: {e}", urgency="critical")


def send_trigger(cmd: str = "go") -> int:
    """Tell a running daemon to do one cycle (or quit)."""
    if not FIFO_PATH.exists():
        print(f"FIFO no existe: {FIFO_PATH}. ¿Daemon corriendo?", file=sys.stderr)
        notify("Daemon no está corriendo", urgency="critical")
        return 1
    # Open with O_WRONLY|O_NONBLOCK so we don't hang if no reader is ready.
    fd = os.open(FIFO_PATH, os.O_WRONLY | os.O_NONBLOCK)
    try:
        os.write(fd, (cmd + "\n").encode())
    finally:
        os.close(fd)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    p.add_argument("--text", type=str)
    p.add_argument("--daemon", action="store_true")
    p.add_argument("--trigger", action="store_true")
    p.add_argument("--quit", action="store_true", help="Tell daemon to exit")
    args = p.parse_args()

    if args.trigger:
        return send_trigger("go")
    if args.quit:
        return send_trigger("quit")
    if args.daemon:
        return run_daemon()

    # Interactive / one-shot / text modes (legacy paths)
    j = Jarvis(with_stt=not args.text)

    if args.text:
        j.handle_text(args.text)
        return 0

    while True:
        try:
            input("\n[Enter para hablar, Ctrl-C para salir] ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        j.one_cycle()
        if args.once:
            return 0


if __name__ == "__main__":
    sys.exit(main())
