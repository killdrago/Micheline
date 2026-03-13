# micheline/voice/tts_pyttsx3.py
# Fallback TTS (offline) basé sur pyttsx3, avec découpe automatique des longs textes.

from __future__ import annotations
import threading
from typing import Optional, List

try:
    import pyttsx3
except Exception:
    pyttsx3 = None


class Pyttsx3TTS:
    def __init__(self, rate: int = 175, volume: float = 1.0, voice_hint: str = "female:fr"):
        self._ok = pyttsx3 is not None
        self._engine = None
        self.rate = int(rate)
        self.volume = float(volume)
        self.voice_hint = (voice_hint or "").lower()

        if self._ok:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)
                self._engine.setProperty("volume", self.volume)
                voice_id = self._select_voice(self.voice_hint)
                if voice_id is not None:
                    self._engine.setProperty("voice", voice_id)
            except Exception:
                self._ok = False
                self._engine = None

    def available(self) -> bool:
        return self._ok and (self._engine is not None)

    def stop(self):
        try:
            if self._engine:
                self._engine.stop()
        except Exception:
            pass

    def speak(self, text: str):
        if not text or not self.available():
            return
        # Petits textes: chemin simple
        if len(text) <= 600:
            t = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
            t.start()
        else:
            # Longs textes: découpe en segments parlables
            chunks = self._split_for_tts(text, chunk_chars=600)
            t = threading.Thread(target=self._speak_chunks_blocking, args=(chunks,), daemon=True)
            t.start()

    # -------------------- internes --------------------

    def _select_voice(self, hint: str) -> Optional[str]:
        try:
            voices = self._engine.getProperty("voices") or []
            def score(v):
                name = (v.name or "").lower()
                lang = ",".join(getattr(v, "languages", []) or []).lower()
                vid  = (v.id or "").lower()
                s = 0
                if "female" in name or "fem" in name or "femme" in name: s += 2
                if "fr" in lang or "fr" in name or "fr" in vid: s += 3
                return s
            best, best_s = None, -1
            for v in voices:
                sc = score(v)
                if sc > best_s:
                    best, best_s = v, sc
            return best.id if best is not None else None
        except Exception:
            return None

    def _speak_blocking(self, text: str):
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            pass

    def _split_for_tts(self, text: str, chunk_chars: int = 600) -> List[str]:
        """
        Découpe le texte long en segments parlables pour pyttsx3.
        Essaye de couper sur la ponctuation forte ou un espace.
        """
        import re
        s = (text or "").strip()
        if len(s) <= chunk_chars:
            return [s]
        parts: List[str] = []
        pos = 0
        L = len(s)
        while pos < L:
            end = min(L, pos + chunk_chars)
            segment = s[pos:end]
            # Cherche une fin de phrase dans le segment
            m = list(re.finditer(r'[\.!\?\n]', segment))
            if m:
                cut = m[-1].end()
                parts.append(segment[:cut].strip())
                pos += cut
            else:
                # Sinon coupe au dernier espace raisonnable
                space = segment.rfind(' ')
                if space > 200:
                    parts.append(segment[:space].strip())
                    pos += space
                else:
                    parts.append(segment.strip())
                    pos = end
        return [p for p in parts if p]

    def _speak_chunks_blocking(self, chunks: List[str]):
        try:
            for c in chunks:
                if not c:
                    continue
                self._engine.say(c)
            self._engine.runAndWait()
        except Exception:
            pass