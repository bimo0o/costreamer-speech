"""Private adapter to the user's Mantella XTTS image, never an exposed proxy."""
import io
import os
import re
import struct
import threading
import wave
from pathlib import Path
import httpx

VOICE_NAME = re.compile(r'^[A-Za-z0-9_-]{1,64}(?:\.wav)?$')


def validate_wav(data):
    if not 44 <= len(data) <= 12*1024*1024:
        raise ValueError('WAV должен быть не больше 12 МБ')
    try:
        with wave.open(io.BytesIO(data),'rb') as wav:
            seconds = wav.getnframes()/wav.getframerate()
            if wav.getnchannels()!=1 or wav.getsampwidth()!=2 or wav.getframerate() not in (16000,22050,24000,44100,48000) or not 3<=seconds<=30:
                raise ValueError('Нужен mono PCM WAV 16 bit, от 3 до 30 секунд')
            if len(wav.readframes(wav.getnframes())) != wav.getnframes()*2:
                raise ValueError('WAV повреждён')
    except (wave.Error,EOFError,ZeroDivisionError):
        raise ValueError('Не удалось прочитать PCM WAV') from None


class WavPCM:
    """Remove a possibly split RIFF header; reject an unexpected audio format."""
    def __init__(self):
        self.buffer = bytearray()
        self.ready = False
        self.valid_format = False

    def feed(self,data):
        if self.ready:
            return data
        self.buffer.extend(data)
        if len(self.buffer)<12:
            return b''
        if self.buffer[:4]!=b'RIFF' or self.buffer[8:12]!=b'WAVE':
            raise ValueError('XTTS вернул неверный WAV-поток')
        offset = 12
        while len(self.buffer)>=offset+8:
            kind = self.buffer[offset:offset+4]
            length = struct.unpack_from('<I',self.buffer,offset+4)[0]
            if kind==b'data':
                if not self.valid_format:
                    raise ValueError('XTTS должен возвращать PCM mono 24000 Гц 16 bit')
                self.ready = True
                result = bytes(self.buffer[offset+8:])
                self.buffer.clear()
                return result
            if length>65536 or offset+8+length>65536:
                raise ValueError('Слишком большой WAV-заголовок')
            if len(self.buffer)<offset+8+length:
                return b''
            if kind==b'fmt ' and length>=16:
                format,channels,rate,_,_,bits = struct.unpack_from('<HHIIHH',self.buffer,offset+8)
                self.valid_format = (format,channels,rate,bits)==(1,1,24000,16)
            offset += 8+length+(length%2)
        return b''


class XTTS:
    def __init__(self):
        self.http = httpx.Client(base_url='http://127.0.0.1:8020',timeout=90)
        self.lock = threading.Lock()
        self.latents = Path(os.getenv('XTTS_LATENTS','/data/latents/ru'))
        self.samples = Path(os.getenv('XTTS_SAMPLES','/data/speakers/ru'))
        self.samples.mkdir(parents=True,exist_ok=True)

    def voices(self):
        names = {p.stem for p in self.latents.glob('*.json')}
        names.update(p.name for p in self.samples.glob('*.wav'))
        return [{'voice_id':name,'name':name+(' · свой образец' if name.endswith('.wav') else ' · Mantella XTTS')} for name in sorted(names)]

    def upload(self,name,data):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',name):
            raise ValueError('Имя: латинские буквы, цифры, _ или -, до 64 символов')
        validate_wav(data)
        path = self.samples/(name.lower()+'.wav')
        if path.exists():
            raise ValueError('Такой голос уже есть; выберите другое имя')
        with path.open('xb') as file:
            file.write(data)
        return {'voice_id':path.name,'name':name+' · свой образец'}

    def stream(self,text,voice,speed):
        if not VOICE_NAME.fullmatch(voice) or voice not in {v['voice_id'] for v in self.voices()}:
            raise ValueError('Выберите голос из списка сервера')
        with self.lock:
            settings = self.http.get('/get_tts_settings')
            settings.raise_for_status()
            values = settings.json()
            values['speed'] = speed
            response = self.http.post('/set_tts_settings',json=values)
            response.raise_for_status()
            # Old Mantella builds have /tts_to_audio only; current ones can stream.
            with self.http.stream('POST','/tts_stream',json={'text':text,'speaker_wav':voice,'language':'ru'}) as r:
                if r.status_code not in (404,405):
                    r.raise_for_status()
                    parser = WavPCM()
                    for chunk in r.iter_bytes():
                        pcm = parser.feed(chunk)
                        if pcm:
                            yield pcm
                    if not parser.ready:
                        raise ValueError('XTTS вернул неполный WAV')
                    return
            with self.http.stream('POST','/tts_to_audio/',json={'text':text,'speaker_wav':voice,'language':'ru'}) as r:
                r.raise_for_status()
                parser = WavPCM()
                for chunk in r.iter_bytes():
                    pcm = parser.feed(chunk)
                    if pcm:
                        yield pcm
                if not parser.ready:
                    raise ValueError('XTTS вернул неполный WAV')
