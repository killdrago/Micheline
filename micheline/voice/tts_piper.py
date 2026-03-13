# micheline/voice/tts_piper.py
# TTS Piper: utilise le binaire "piper" + un modèle voix .onnx (+ .onnx.json)
# Fallback prévu dans main via pyttsx3 si binaire absent ou voix introuvable.

from __future__ import annotations
import os
import shutil
import subprocess
import tempfile
import threading
import wave
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None


class PiperTTS:
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        preferred_rate: Optional[int] = None,
        device: Optional[int] = None,
    ):
        """
        model_path: chemin .onnx
        config_path: chemin .onnx.json (optionnel)
        preferred_rate: non utilisé (taux auto via WAV)
        device: device audio (None=par défaut)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.preferred_rate = preferred_rate
        self.device = device
        self._has_piper = shutil.which("piper") is not None

    def available(self) -> bool:
        """Retourne True si Piper + modèle et backend audio sont disponibles."""
        return self._has_piper and os.path.isfile(self.model_path) and (sd is not None)

    def speak(self, text: str):
        """Synthétise et lit le texte (thread non bloquant)."""
        if not text or not self.available():
            return
        t = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
        t.start()

    # -------------------- internes --------------------

    def _speak_blocking(self, text: str):
        """Chemin bloquant: synthèse via Piper -> lecture WAV via sounddevice."""
        wav_path = None
        try:
            # Fichier temporaire de sortie audio
            with tempfile.NamedTemporaryFile(prefix="micheline_tts_", suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name

            # Commande Piper
            cmd = ["piper", "-m", self.model_path, "-f", wav_path]
            if self.config_path and os.path.isfile(self.config_path):
                cmd += ["-c", self.config_path]

            # Lance Piper et pousse le texte sur stdin
            p = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                universal_newlines=True,
            )
            try:
                p.stdin.write((text or "").strip() + "\n")
                p.stdin.flush()
                p.stdin.close()
            except Exception:
                pass

            # Timeout configurable via config.PIPER_WAIT_TIMEOUT (fallback 180s)
            try:
                from importlib import import_module
                cfg = import_module("config")
                timeout_sec = int(getattr(cfg, "PIPER_WAIT_TIMEOUT", 180))
            except Exception:
                timeout_sec = 180

            try:
                p.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                # Si Piper met trop de temps, on tuer le process proprement
                try:
                    p.kill()
                except Exception:
                    pass

            # Lecture audio (sounddevice)
            if sd is None:
                return
            self._play_wav(wav_path)

        except Exception:
            # On ignore les erreurs pour ne pas casser l'UI
            pass
        finally:
            # Nettoyage du WAV temporaire
            try:
                if wav_path and os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass

    def _play_wav(self, wav_path: str):
        """Lit le WAV en mémoire avec sounddevice (lecture bloquante)."""
        try:
            with wave.open(wav_path, "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                n_frames = wf.getnframes()
                raw = wf.readframes(n_frames)

            data = np.frombuffer(raw, dtype=np.int16)
            if n_channels > 1:
                data = data.reshape(-1, n_channels)
            data = data.astype(np.float32) / 32768.0

            sd.play(data, samplerate=sr, device=self.device, blocking=True)
            sd.stop()
        except Exception:
            pass