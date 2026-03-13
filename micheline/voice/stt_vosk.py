# micheline/voice/stt_vosk.py
# Service STT (Vosk) avec écoute micro en thread, callbacks partiel/final.
# Respecte vad_silence_ms: n'émet on_final qu'après vad_silence_ms ms de silence.
# Sample rate auto-sûr: force 16000 si valeur aberrante.

from __future__ import annotations
import os
import json
import threading
import queue
import time
from typing import Callable, Optional

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from vosk import Model, KaldiRecognizer
except Exception:
    Model = None
    KaldiRecognizer = None


class STTVoskService:
    def __init__(
        self,
        model_dir: str,
        sample_rate: int = 16000,
        vad_silence_ms: int = 800,
        device: Optional[int] = None,
    ):
        """
        model_dir: dossier du modèle Vosk (ex: micheline/models/stt/vosk/fr)
        sample_rate: 16000 recommandé
        vad_silence_ms: délai de silence avant d'émettre on_final
        device: index du device micro (None = défaut)
        """
        self.model_dir = model_dir
        # Auto-sûr: si invalide, force 16000
        try:
            sr = int(sample_rate)
            if sr < 8000 or sr > 192000:
                sr = 16000
        except Exception:
            sr = 16000
        self.sample_rate = sr

        self.vad_silence_ms = int(vad_silence_ms)
        self.device = device

        self._model = None
        self._rec = None
        self._stream = None
        self._audio_q: "queue.Queue[bytes]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._running = False

        self._on_partial: Optional[Callable[[str], None]] = None
        self._on_final: Optional[Callable[[str], None]] = None

        # Buffer des finals + horodatage du dernier audio
        self._final_buffer = []
        self._last_audio_ts = 0.0

    def available(self) -> bool:
        return (sd is not None) and (Model is not None) and os.path.isdir(self.model_dir)

    def start(self, on_final: Callable[[str], None], on_partial: Optional[Callable[[str], None]] = None) -> bool:
        """
        Lance l'écoute. on_partial(text) pour les partiels, on_final(text) pour les segments finalisés.
        Retourne True si démarré, False sinon.
        """
        if self._running:
            return True
        if not self.available():
            return False

        try:
            if self._model is None:
                self._model = Model(self.model_dir)
            self._rec = KaldiRecognizer(self._model, self.sample_rate)
        except Exception:
            return False

        self._on_partial = on_partial
        self._on_final = on_final
        self._stop_evt.clear()
        self._final_buffer = []
        self._last_audio_ts = time.time()

        try:
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate // 2),
                device=self.device,
                dtype="int16",
                channels=1,
                callback=self._on_audio_chunk,
            )
            self._stream.start()
        except Exception:
            self._stream = None
            return False

        self._worker = threading.Thread(target=self._run_loop, daemon=True)
        self._worker.start()
        self._running = True
        return True

    def stop(self):
        """Arrête l'écoute et ferme le stream proprement (flush du buffer final)."""
        if not self._running:
            return
        self._stop_evt.set()
        try:
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=1.5)
        except Exception:
            pass
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        # Flush du buffer s'il reste du texte
        try:
            if self._final_buffer and self._on_final:
                joined = " ".join(self._final_buffer).strip()
                if joined:
                    self._on_final(joined)
        except Exception:
            pass
        self._stream = None
        self._worker = None
        self._running = False
        self._final_buffer = []

    # -------------------- internes --------------------

    def _on_audio_chunk(self, indata, frames, time_info, status):  # noqa: A002
        if status:
            # ignore warnings (over/underflows)
            pass
        try:
            self._audio_q.put(bytes(indata))
            self._last_audio_ts = time.time()
        except Exception:
            pass

    def _maybe_flush_final(self):
        """Émet on_final si on a du texte en buffer ET suffisamment de silence écoulé."""
        if not self._final_buffer:
            return
        silence_ms = (time.time() - self._last_audio_ts) * 1000.0
        if silence_ms >= max(0, self.vad_silence_ms):
            try:
                if self._on_final:
                    joined = " ".join(self._final_buffer).strip()
                    if joined:
                        self._on_final(joined)
            except Exception:
                pass
            finally:
                self._final_buffer = []

    def _run_loop(self):
        """Thread de traitement: alimente le recognizer et déclenche les callbacks."""
        while not self._stop_evt.is_set():
            try:
                data = self._audio_q.get(timeout=0.2)
            except queue.Empty:
                # Pas de nouveau son -> teste le silence pour flush
                self._maybe_flush_final()
                continue

            try:
                if self._rec.AcceptWaveform(data):
                    # Résultat final d'un sous-segment -> on bufferise seulement
                    try:
                        j = json.loads(self._rec.Result() or "{}")
                        text = (j.get("text") or "").strip()
                    except Exception:
                        text = ""
                    if text:
                        self._final_buffer.append(text)
                else:
                    # Résultat partiel (continu) -> on passe à l'app
                    try:
                        j = json.loads(self._rec.PartialResult() or "{}")
                        p = (j.get("partial") or "").strip()
                    except Exception:
                        p = ""
                    if p and self._on_partial:
                        try:
                            self._on_partial(p)
                        except Exception:
                            pass

                # Après traitement, si pas de nouveau son depuis un moment, flush
                self._maybe_flush_final()

            except Exception:
                # En cas d'erreur, on continue (stream vivant)
                pass