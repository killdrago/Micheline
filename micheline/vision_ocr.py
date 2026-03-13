# vision_ocr.py - OCR local (Paddle/Tesseract) centralisé via config
# - Largeur max, langue et longueur max via config

from __future__ import annotations
import os
from typing import Dict, Any
from PIL import Image, ImageOps, ImageFilter
import config

# Détection des backends OCR disponibles
_PADDLE_OK = False
_PYTESS_OK = False
_paddle_ocr = None

try:
    from paddleocr import PaddleOCR  # type: ignore
    _PADDLE_OK = True
except Exception:
    _PADDLE_OK = False

try:
    import pytesseract  # type: ignore
    _PYTESS_OK = True
except Exception:
    _PYTESS_OK = False


def _load_image(path: str) -> Image.Image:
    """
    Charge et pré-traite l'image pour l'OCR.
    - Grayscale
    - Upscale (<= OCR_IMAGE_MAX_WIDTH px de large)
    - Binarisation OTSU
    - Suppression des lignes de grille (tableur) via morpho (horizontal/vertical)
    - Lissage léger + fermeture pour raccrocher les caractères
    - Deskew (rotation légère) si possible
    """
    try:
        import numpy as np
        import cv2
        cv_ok = True
    except Exception:
        cv_ok = False

    img = Image.open(path)
    if img.mode != "L":
        img = ImageOps.grayscale(img)

    target_w = int(getattr(config, "OCR_IMAGE_MAX_WIDTH", 2800))

    if not cv_ok:
        # Fallback simple si OpenCV indispo
        if img.width < target_w:
            ratio = target_w / float(img.width)
            img = img.resize((target_w, max(1, int(img.height * ratio))), Image.LANCZOS)
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=180, threshold=3))
        return img

    import numpy as np
    import cv2

    arr = np.array(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Upscale si nécessaire
    if arr.shape[1] < target_w:
        scale = min(3.0, target_w / float(arr.shape[1]))
        arr = cv2.resize(arr, (int(arr.shape[1] * scale), int(arr.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)

    # Denoise léger + OTSU
    blur = cv2.GaussianBlur(arr, (3, 3), 0)
    _, bin_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Inversion pour morpho
    inv = 255 - bin_img

    # Suppression des lignes horizontales/verticales (grilles)
    h_ks = max(25, arr.shape[1] // 80)   # adaptatif
    v_ks = max(25, arr.shape[0] // 80)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (h_ks, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_ks))
    try:
        lines_h = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_h, iterations=1)
        lines_v = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel_v, iterations=1)
        lines = cv2.bitwise_or(lines_h, lines_v)
        no_lines_inv = cv2.bitwise_and(inv, cv2.bitwise_not(lines))
    except Exception:
        no_lines_inv = inv

    # Revenir en "texte noir sur fond blanc"
    cleaned = 255 - no_lines_inv

    # Fermeture légère pour raccrocher les caractères
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Deskew (rotation légère)
    try:
        coords = np.column_stack(np.where(cleaned < 128))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            if abs(angle) > 0.5 and abs(angle) < 15:
                (h, w) = cleaned.shape
                M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
                cleaned = cv2.warpAffine(cleaned, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass

    out = Image.fromarray(cleaned)
    out = ImageOps.autocontrast(out, cutoff=1)
    out = out.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=2))
    return out

def _ensure_paddle(lang: str = "fr"):
    """
    Initialise PaddleOCR de façon compatible.
    """
    global _paddle_ocr
    if not _PADDLE_OK:
        return None
    if _paddle_ocr is not None:
        return _paddle_ocr

    try:
        import inspect
        params = set()
        try:
            sig = inspect.signature(PaddleOCR.__init__)
            params = set(sig.parameters.keys())
        except Exception:
            pass

        if "show_log" in params:
            try:
                _paddle_ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
            except Exception:
                _paddle_ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        else:
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    except Exception:
        try:
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
        except Exception:
            _paddle_ocr = None

    return _paddle_ocr
    
def _ocr_with_paddle(img: Image.Image, lang: str = "fr") -> Dict[str, Any]:
    import numpy as np
    ocr = _ensure_paddle(lang=lang)
    if ocr is None:
        return {"text": "", "backend": "none", "confidence": None, "n_chars": 0, "truncated": False}

    result = ocr.ocr(np.array(img), cls=True)
    lines, confidences = [], []
    try:
        for page in result or []:
            if not page:
                continue
            for det in page:
                text = det[1][0]
                conf = float(det[1][1]) if isinstance(det[1][1], (int, float)) else None
                if text and text.strip():
                    lines.append(text.strip())
                    if conf is not None:
                        confidences.append(conf)
    except Exception:
        pass

    text = "\n".join(lines).strip()
    conf_avg = float(sum(confidences) / max(1, len(confidences))) if confidences else None
    truncated = False
    max_len = int(getattr(config, "OCR_MAX_CHARS", 12000))
    if len(text) > max_len:
        text = text[:max_len] + "\n[...]"
        truncated = True

    return {"text": text, "backend": "paddleocr", "confidence": conf_avg, "n_chars": len(text), "truncated": truncated}

def _ocr_with_tesseract(img: Image.Image, lang: str = "fra+eng") -> Dict[str, Any]:
    """
    OCR Tesseract robuste (tiling + psm variants).
    """
    if not _PYTESS_OK:
        return {"text": "", "backend": "none", "confidence": None, "n_chars": 0, "truncated": False}

    try:
        import shutil
        import pytesseract
        if not shutil.which("tesseract") and os.name == "nt":
            win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(win_path):
                pytesseract.pytesseract.tesseract_cmd = win_path
    except Exception:
        pass

    im = img.copy()
    try:
        w, h = im.size
        if w < 1800:
            scale = min(3.0, 1800.0 / max(1, float(w)))
            im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        im = ImageOps.autocontrast(im, cutoff=1)
        im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=180, threshold=3))
    except Exception:
        pass

    try:
        import pytesseract
    except Exception:
        return {"text": "", "backend": "none", "confidence": None, "n_chars": 0, "truncated": False}

    def run_psm(src_img: Image.Image, lang_code: str, psm: int) -> str:
        try:
            cfg = f"--oem 3 --psm {psm}"
            return pytesseract.image_to_string(src_img, lang=lang_code, config=cfg) or ""
        except Exception:
            return ""

    langs_to_try = [lang]
    if "eng" not in lang:
        langs_to_try.append("eng")

    best_text = ""
    best_tag = "pytesseract"

    for lg in langs_to_try:
        for psm in (6, 4, 11, 13):
            t = run_psm(im, lg, psm).strip()
            if len(t) > len(best_text):
                best_text = t
                best_tag = f"pytesseract-psm{psm}"
        if len(best_text) < 50:
            try:
                W, H = im.size
                cols = 3
                step = max(1, W // cols)
                parts = []
                for i in range(cols):
                    left = i * step
                    right = W if i == cols - 1 else (i + 1) * step
                    crop = im.crop((left, 0, right, H))
                    for psm in (6, 4):
                        tt = run_psm(crop, lg, psm).strip()
                        if tt:
                            parts.append(tt)
                combined = "\n".join(parts).strip()
                if len(combined) > len(best_text):
                    best_text = combined
                    best_tag = "pytesseract-tiles"
            except Exception:
                pass
        if len(best_text) >= 50:
            break

    best_text = best_text.strip()
    truncated = False
    max_len = int(getattr(config, "OCR_MAX_CHARS", 12000))
    if len(best_text) > max_len:
        best_text = best_text[:max_len] + "\n[...]"
        truncated = True

    return {"text": best_text, "backend": best_tag if best_text else "pytesseract", "confidence": None, "n_chars": len(best_text), "truncated": truncated}

def extract_text(image_path: str, lang_primary: str = None) -> Dict[str, Any]:
    """
    Extrait du texte depuis un screenshot local.
    - Si PaddleOCR dispo -> Paddle (lang=config.OCR_LANG_PRIMARY)
    - Sinon -> pytesseract (lang='fra+eng')
    """
    if not os.path.isfile(image_path):
        return {"text": "", "backend": "none", "confidence": None, "n_chars": 0, "truncated": False}

    lang_primary = (lang_primary or getattr(config, "OCR_LANG_PRIMARY", "fr")).strip()
    img = _load_image(image_path)

    # 1) PaddleOCR si dispo
    if _PADDLE_OK:
        out = _ocr_with_paddle(img, lang=lang_primary)
        if out.get("text"):
            return out

    # 2) Fallback pytesseract si dispo
    if _PYTESS_OK:
        return _ocr_with_tesseract(img, lang="fra+eng")

    return {"text": "", "backend": "none", "confidence": None, "n_chars": 0, "truncated": False}