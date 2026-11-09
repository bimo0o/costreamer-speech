"""Build-time downloads: avoid re-downloading paid ingress on every rental."""
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
from piper.download_voices import download_voice

models = Path('/models')
models.mkdir(exist_ok=True)
snapshot_download('Systran/faster-whisper-small', local_dir=models/'whisper-small')
download_voice('ru_RU-denis-medium', models)
hf_hub_download('rhasspy/piper-voices', 'ru/ru_RU/denis/medium/MODEL_CARD', local_dir=models/'licenses')
