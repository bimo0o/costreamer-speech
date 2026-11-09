"""Bounded 16 kHz PCM endpointing, independent of inference libraries."""
from collections import deque


class Segmenter:
    frame_bytes = 640  # 20 ms, signed int16 mono at 16 kHz

    def __init__(self, is_speech, silence_frames=20, max_frames=400):
        self.is_speech = is_speech
        self.silence_frames, self.max_frames = silence_frames, max_frames
        self.pending = bytearray()
        self.preroll = deque(maxlen=10)
        self.frames = []
        self.silent = self.voiced = 0
        self.active = False

    def feed(self, data):
        self.pending.extend(data)
        events = []
        while len(self.pending) >= self.frame_bytes:
            frame = bytes(self.pending[:self.frame_bytes])
            del self.pending[:self.frame_bytes]
            speech = self.is_speech(frame)
            if not self.active:
                self.preroll.append(frame)
                self.voiced = self.voiced+1 if speech else 0
                if self.voiced >= 3:
                    self.active = True
                    self.frames = list(self.preroll)
                    self.preroll.clear()
                    events.append(('start', b''))
                continue
            self.frames.append(frame)
            self.silent = 0 if speech else self.silent+1
            if self.silent >= self.silence_frames or len(self.frames) >= self.max_frames:
                events.append(('final', self.finish()))
            elif len(self.frames) % 75 == 0:
                events.append(('partial', b''.join(self.frames)))
        return events

    def finish(self):
        pcm = b''.join(self.frames) if self.active else b''
        self.frames = []
        self.active = False
        self.silent = self.voiced = 0
        self.preroll.clear()
        return pcm
