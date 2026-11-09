"""GPU STT and CPU TTS only. No OpenRouter/Vast credentials on rented hosts."""
import asyncio
import contextlib
import hmac
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

TOKEN = os.getenv('SPEECH_TOKEN', '')
ENGINE = os.getenv('TTS_ENGINE','xtts')
VOICE = os.getenv('TTS_VOICE', 'femalenord')
runtime = {}


@asynccontextmanager
async def lifespan(app):
    if len(TOKEN) < 32:
        raise RuntimeError('SPEECH_TOKEN must contain at least 32 characters')
    from faster_whisper import WhisperModel
    runtime['stt'] = WhisperModel(os.getenv('STT_MODEL', '/models/whisper-small'), device='cuda', compute_type='int8_float16', cpu_threads=2, num_workers=1)
    if ENGINE=='xtts':
        from .xtts import XTTS
        runtime['xtts'] = XTTS()
    else:
        from piper import PiperVoice
        runtime['tts'] = PiperVoice.load('/models/ru_RU-denis-medium.onnx', use_cuda=False)
    runtime['executor'] = ThreadPoolExecutor(max_workers=1, thread_name_prefix='stt')
    runtime['tts_lock'] = threading.Lock()
    runtime['busy'] = False
    runtime['last_activity'] = time.monotonic()
    yield
    runtime['executor'].shutdown(wait=True, cancel_futures=True)
    if 'xtts' in runtime:
        runtime['xtts'].http.close()


app = FastAPI(title='CoStreamer speech worker', lifespan=lifespan)


def authorized(header):
    return bool(TOKEN) and hmac.compare_digest(header or '', 'Bearer '+TOKEN)


@app.middleware('http')
async def auth(request: Request, next):
    if not authorized(request.headers.get('authorization')):
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail':'Unauthorized'}, status_code=401)
    if request.headers.get('content-length') and int(request.headers['content-length']) > (13*1024*1024 if request.url.path=='/voices/upload' else 20000):
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail':'Payload too large'}, status_code=413)
    return await next(request)


@app.get('/health')
async def health():
    if ENGINE=='xtts':
        def check():
            response=runtime['xtts'].http.get('/get_tts_settings');response.raise_for_status()
        await asyncio.to_thread(check)
    return {'status':'ready', 'stt':'faster-whisper/small/int8_float16', 'tts':ENGINE, 'rate':24000}


@app.get('/voices')
async def voices():
    return {'voices':runtime['xtts'].voices() if ENGINE=='xtts' else [{'voice_id':'ru_RU-denis-medium','name':'Денис · Piper'}], 'expression_supported':False,'cloning_supported':ENGINE=='xtts'}



@app.post('/voices/upload')
async def upload_voice(request: Request):
    if ENGINE!='xtts':
        raise HTTPException(400,'Загрузка образцов доступна только для XTTS')
    data=bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data)>12*1024*1024:
            raise HTTPException(413,'Образец больше 12 МБ')
    try:
        return await asyncio.to_thread(runtime['xtts'].upload,request.query_params.get('name',''),bytes(data))
    except ValueError as e:
        raise HTTPException(400,str(e)) from None

class Speech(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    voice: str = VOICE
    speed: float = Field(default=1, ge=.7, le=1.2)


@app.post('/tts')
async def tts(body: Speech):
    if ENGINE=='xtts':
        voice=body.voice or VOICE
        if voice not in {v['voice_id'] for v in runtime['xtts'].voices()}:
            raise HTTPException(400,'Выберите доступный голос XTTS')
        return StreamingResponse(runtime['xtts'].stream(body.text,voice,body.speed),media_type='application/octet-stream',headers={'X-Sample-Rate':'24000','Cache-Control':'no-store'})
    if body.voice not in ('', 'ru_RU-denis-medium'):
        raise HTTPException(400, 'Unknown voice; refresh voice list')
    # StreamingResponse runs the synchronous iterator in a thread, never on the event loop.
    def chunks():
        import numpy as np
        from scipy.signal import resample_poly
        from math import gcd
        from piper import SynthesisConfig
        with runtime['tts_lock']:
            for chunk in runtime['tts'].synthesize(body.text, syn_config=SynthesisConfig(length_scale=1/body.speed)):
                audio = chunk.audio_float_array
                if chunk.sample_rate != 24000:
                    divisor = gcd(chunk.sample_rate, 24000)
                    audio = resample_poly(audio, 24000//divisor, chunk.sample_rate//divisor)
                data = (np.clip(audio, -1, 1)*32767).astype('<i2').tobytes()
                for i in range(0, len(data), 8192):
                    yield data[i:i+8192]
    return StreamingResponse(chunks(), media_type='application/octet-stream', headers={'X-Sample-Rate':'24000','Cache-Control':'no-store'})


def transcribe(pcm):
    import numpy as np
    samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32)/32768
    segments, _ = runtime['stt'].transcribe(
        samples, language='ru', beam_size=1, condition_on_previous_text=False,
        vad_filter=True, vad_parameters={'min_silence_duration_ms': 450},
        no_speech_threshold=0.6, log_prob_threshold=-0.8,
        compression_ratio_threshold=2.4,
    )
    return ' '.join(s.text.strip() for s in segments).strip()


@app.websocket('/stt')
async def stt(ws: WebSocket):
    if not authorized(ws.headers.get('authorization')) or runtime.get('busy'):
        await ws.close(code=1008)
        return
    import webrtcvad
    from .segmentation import Segmenter
    runtime['busy'] = True  # one streamer per economical instance
    await ws.accept()
    vad = webrtcvad.Vad(2)
    segmenter = Segmenter(lambda pcm:vad.is_speech(pcm,16000))
    queue = asyncio.Queue(maxsize=3)
    utterance = 0
    async def infer():
        while True:
            final, pcm, identity = await queue.get()
            started = time.monotonic()
            text = await asyncio.get_running_loop().run_in_executor(runtime['executor'],transcribe,pcm)
            # Whisper can return an empty segment for silence/noise. Do not leak
            # those as chat events; they otherwise look like endless empty turns.
            if not text:
                continue
            await ws.send_json({'type':'transcript','text':text,'final':final,'utterance':identity,'latency_ms':round((time.monotonic()-started)*1000)})
    task = asyncio.create_task(infer())
    window, count = time.monotonic(), 0
    try:
        await ws.send_json({'type':'ready','rate':16000})
        while True:
            packet = await asyncio.wait_for(ws.receive(), timeout=30)
            if task.done():
                task.result()
            if packet['type']=='websocket.disconnect':
                break
            if time.monotonic()-window > 1:
                window, count = time.monotonic(), 0
            count += 1
            if count > 100:
                raise ValueError('Audio rate exceeded')
            events = []
            if packet.get('bytes') is not None:
                data = packet['bytes']
                if len(data)>8192 or len(data)%2:
                    raise ValueError('Invalid PCM packet')
                events = segmenter.feed(data)
            elif packet.get('text') == 'finalize':
                pcm = segmenter.finish()
                if pcm:
                    events = [('final',pcm)]
            elif packet.get('text') == 'ping':
                continue
            for kind, pcm in events:
                if kind=='start':
                    utterance += 1
                    await ws.send_json({'type':'speech_start','utterance':utterance})
                elif kind=='partial':
                    if queue.empty():
                        queue.put_nowait((False,pcm,utterance))
                else:
                    await asyncio.wait_for(queue.put((True,pcm,utterance)),timeout=3)
    except asyncio.CancelledError:
        raise
    except Exception:
        with contextlib.suppress(Exception):
            await ws.close(code=1011)
    finally:
        task.cancel()
        await asyncio.gather(task,return_exceptions=True)
        runtime['busy'] = False
