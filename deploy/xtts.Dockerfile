ARG XTTS_BASE=ghcr.io/bimo0o/mantella-xtts-skyrim:latest
FROM ${XTTS_BASE}
USER root
ENV PYTHONUNBUFFERED=1 TTS_ENGINE=xtts STT_MODEL=/models/whisper-small OMP_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv openssl ca-certificates && rm -rf /var/lib/apt/lists/*
# Keep the original XTTS/PyTorch environment intact; STT and gateway are isolated.
RUN python3 -m venv /opt/speech-venv
COPY speech_server/requirements.txt /tmp/speech-requirements.txt
RUN /opt/speech-venv/bin/pip install --no-cache-dir -r /tmp/speech-requirements.txt nvidia-cublas-cu12==12.4.5.8 nvidia-cudnn-cu12==9.1.1.17
COPY speech_server /opt/costreamer/speech_server
RUN /opt/speech-venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-small',local_dir='/models/whisper-small')"
COPY deploy/start-xtts.py /opt/costreamer/start-xtts.py
EXPOSE 8443
WORKDIR /app
ENTRYPOINT ["/opt/speech-venv/bin/python", "/opt/costreamer/start-xtts.py"]
