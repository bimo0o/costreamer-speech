"""Run private Mantella XTTS and authenticated TLS STT/TTS gateway together."""
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
import httpx

root=Path('/data')
for name in ('latents/ru','speakers/ru','output','tls'):
    (root/name).mkdir(parents=True,exist_ok=True)
source=Path('/app/latent_speaker_folder/ru')
if not source.exists():source=Path('/app/latent_speaker_folder/en')
for file in source.glob('*.json'):
    shutil.copyfile(file,root/'latents/ru'/(file.stem.lower()+'.json'))
certificate=root/'tls/cert.pem'
key=root/'tls/key.pem'
if os.getenv('SPEECH_TLS_CERT_B64') and os.getenv('SPEECH_TLS_KEY_B64'):
    import base64
    certificate.write_bytes(base64.b64decode(os.environ.pop('SPEECH_TLS_CERT_B64'),validate=True))
    key.write_bytes(base64.b64decode(os.environ.pop('SPEECH_TLS_KEY_B64'),validate=True))
    key.chmod(0o600)
if not certificate.exists():
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','365','-subj','/CN=costreamer-speech','-addext','subjectAltName=DNS:localhost,IP:127.0.0.1','-keyout',str(key),'-out',str(certificate)],check=True)
    key.chmod(0o600)
print(certificate.read_text(),flush=True)
xtts=subprocess.Popen(['/usr/bin/env','python3','-m','xtts_api_server','-hs','127.0.0.1','-p','8020','-lsf','/data/latents','-sf','/data/speakers','-o','/data/output','-mf','/app/xtts_models','-d','cuda'])
gateway=None
def shutdown(*args):
    for child in (gateway,xtts):
        if child and child.poll() is None:child.terminate()
signal.signal(signal.SIGTERM,shutdown)
signal.signal(signal.SIGINT,shutdown)
try:
    for _ in range(360):
        if xtts.poll() is not None:raise RuntimeError('XTTS exited; inspect its log')
        try:
            response=httpx.get('http://127.0.0.1:8020/get_tts_settings',timeout=1)
            if response.is_success:break
        except httpx.HTTPError:pass
        time.sleep(1)
    else:raise RuntimeError('XTTS startup timed out')
    env={**os.environ,'PYTHONPATH':'/opt/costreamer'}
    import site
    libraries=[]
    for packages in site.getsitepackages():
        for library in ('cublas','cudnn'):
            libraries.append(str(Path(packages)/'nvidia'/library/'lib'))
    env['LD_LIBRARY_PATH']=':'.join(libraries+[os.getenv('LD_LIBRARY_PATH','')])
    gateway=subprocess.Popen([sys.executable,'-m','uvicorn','speech_server.app:app','--host','0.0.0.0','--port','8443','--ssl-keyfile',str(key),'--ssl-certfile',str(certificate),'--ws-max-size','16384','--limit-concurrency','8'],env=env)
    while xtts.poll() is None and gateway.poll() is None:
        # Mantella may write streamed responses to disk; expire them after 5 minutes.
        for audio in (root/'output').glob('*.wav'):
            if time.time()-audio.stat().st_mtime>300:audio.unlink(missing_ok=True)
        time.sleep(2)
finally:
    shutdown()
