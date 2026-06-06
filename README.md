# Jarvis

Local voice-controlled AI assistant for Omarchy/Hyprland.

## Stack

- **LLM**: Ollama + Qwen 2.5 7B Instruct (Q4_K_M) — runs on RTX 4060, ~4.8 GB VRAM
- **STT**: faster-whisper `small` model on GPU
- **TTS**: Piper with `es_ES-davefx-medium` voice
- **System control**: hyprctl + omarchy via tool calling

## Architecture

```
SUPER+H ──> jarvis.py --trigger ──┐
                                  │  (writes "go" to FIFO)
                                  ▼
                          jarvis.py --daemon ◄── systemd user service
                                  │
                                  ▼
                          mic ──> Whisper ──> Ollama (with tools) ──> Piper ──> speakers
```

The daemon loads all models once on boot, then sleeps. Pressing SUPER+H
triggers one record→transcribe→reason→speak cycle (~2-3s latency).

## Setup

Requirements: Linux, [Ollama](https://ollama.com), [uv](https://docs.astral.sh/uv/), and (optional but recommended) an NVIDIA GPU.

```bash
# 1. Python dependencies
uv sync

# 2. Pull the models
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text          # used by the Obsidian vault RAG

# 3. Download a Piper voice into voices/ (not bundled in this repo)
#    e.g. es_ES-davefx-medium — see https://github.com/rhasspy/piper/blob/master/VOICES.md
mkdir -p voices   # place es_ES-davefx-medium.onnx and .onnx.json here

# 4. (optional) enable the systemd user service for SUPER+H
systemctl --user enable --now jarvis
```

The personal memory file (`~/.config/jarvis/memory.md`) and the vault RAG index
(`~/.cache/jarvis/chroma/`) are created automatically and never leave your machine.

## Usage

### Daily

Just press **SUPER + H**. Speak. Stop. Jarvis responds.

### Manual commands

```bash
# Trigger the running daemon (same as SUPER+H)
uv run python jarvis.py --trigger

# Skip mic, send a text question (useful for debugging)
uv run python jarvis.py --text "cámbiame al workspace 3"

# Interactive loop (press Enter to record each turn)
uv run python jarvis.py

# Tell the daemon to exit
uv run python jarvis.py --quit
```

### Service management

```bash
systemctl --user status jarvis        # check
systemctl --user restart jarvis       # restart (e.g. after code change)
systemctl --user stop jarvis          # stop
systemctl --user disable jarvis       # don't auto-start on login
journalctl --user -u jarvis -f        # tail logs
```

## Tools the LLM can call

Defined in `tools.py`. The model decides when to call them based on
natural-language requests.

| Tool | Example phrase |
|------|----------------|
| `open_app` | "abre firefox" |
| `close_window` | "cierra esta ventana" |
| `switch_workspace` | "cámbiame al workspace 3" |
| `list_windows` | "qué ventanas tengo abiertas" |
| `take_screenshot` | "haz una captura" |
| `toggle_nightlight` | "activa la luz nocturna" |
| `run_shell` | catch-all for arbitrary shell |

Add a new tool: implement a function in `tools.py`, append to `TOOLS_SCHEMA`
and `DISPATCH`. Restart the daemon.

## Troubleshooting

**"FIFO no existe" when pressing SUPER+H**
The daemon isn't running. `systemctl --user start jarvis`.

**Whisper fails with `libcublas.so.12` not found**
The bundled CUDA wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) aren't
visible. `jarvis.py` sets `LD_LIBRARY_PATH` automatically — check the
`_VENV_SITE` path at the top of the file matches your Python version.

**No microphone**
Currently jarvis records from the default PipeWire/PulseAudio source. Check
`pactl info | grep "Default Source"` and switch with
`pactl set-default-source <name>`.

**The model talks too long / uses markdown**
Edit `SYSTEM_PROMPT` in `jarvis.py`.

**Latency feels high**
First request after idle warms up CUDA contexts (~1s extra). After that,
cold model load is avoided because the daemon keeps everything resident.

## File layout

```
~/jarvis/
├── jarvis.py             # entrypoint (daemon, trigger, text, interactive)
├── tools.py              # functions the LLM can call + JSON schema
├── personal_memory.py    # markdown-backed memory about Ariel
├── vault_rag.py          # ChromaDB RAG over the Obsidian vault
├── voices/               # Piper voice files (.onnx + .json)
├── pyproject.toml        # deps managed by uv
└── .trigger.fifo         # IPC pipe (created at daemon start)

~/.config/hypr/bindings.lua            # SUPER+H binding added here
~/.config/systemd/user/jarvis.service  # autostart unit
~/.config/jarvis/memory.md             # personal memory file (auto-created)
~/.cache/jarvis/chroma/                # vault RAG index
```

## Future work

- Wake word ("Hey Jarvis") via openWakeWord
- PDF generation tool (pandoc + weasyprint)
- Spotify autoplay via OAuth (currently search-only)
- Cancel/interrupt while Jarvis is speaking

## License

MIT © 2026 Ariel Alejandro Hidalgo Rodríguez. See [LICENSE](LICENSE).

A personal project, built to learn — shared in case it's useful to someone else.
