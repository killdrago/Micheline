# setup.py - Phase 4: RAG + Fine-Tuning deps (Corrigé)
# - Ajout de la dépendance manquante `langchain-text-splitters`.
# - Ajout de l'installation des deps Fine‑Tuning (torch/transformers/peft/datasets/accelerate [+ bitsandbytes Linux GPU]).
# - Le reste est identique (voix, portable, etc.).

import subprocess
import sys
import pkg_resources
import os
import zipfile
import tarfile
import shutil
import time
import platform
from pathlib import Path

# ========================= Utils généraux =========================

def which(cmd):
    try:
        return shutil.which(cmd)
    except Exception:
        return None

def run_stream(cmd, env=None, cwd=None):
    try:
        p = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, encoding='utf-8', errors='replace',
            env=env, cwd=cwd
        )
        for line in iter(p.stdout.readline, ''):
            sys.stdout.write(line); sys.stdout.flush()
        p.wait()
        return p.returncode == 0
    except Exception as e:
        joined = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        print(f"*** ERREUR exec: {joined} -> {e}")
        return False

def install_with_progress(package_or_cmd, is_pip=True):
    if is_pip:
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package_or_cmd]
        title = f"pip install {package_or_cmd}"
    else:
        cmd = package_or_cmd
        title = f"exec: {' '.join(cmd)}"
    print(f"\n--- {title} ---")
    ok = run_stream(cmd)
    print(f"--- {'OK' if ok else 'ECHEC'}: {title} ---")
    return ok

def ensure_dir(path):
    try: os.makedirs(path, exist_ok=True)
    except Exception as e: print(f"[WARN] mkdir {path}: {e}")

def has_pkg(pkg_name: str) -> bool:
    try:
        pkg_resources.get_distribution(pkg_name); return True
    except Exception:
        return False

def check_and_fix_numpy():
    try:
        numpy_version = pkg_resources.get_distribution("numpy").version
        if pkg_resources.parse_version(numpy_version) >= pkg_resources.parse_version("2.0.0"):
            print("Correction NumPy -> 1.26.4 ...")
            run_stream([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"])
            run_stream([sys.executable, "-m", "pip", "install", "numpy==1.26.4"])
    except Exception:
        pass

def detect_platform_tag():
    m = platform.machine().lower()
    if os.name == "nt":
        return "windows-amd64"
    if sys.platform == "darwin":
        return "macos-aarch64" if m in ("arm64","aarch64") else "macos-x86_64"
    return "linux-aarch64" if m in ("arm64","aarch64") else "linux-x86_64"

def download_with_progress(url, dest_path, chunk_size=1024*1024):
    try:
        import requests
    except Exception:
        install_with_progress("requests"); import requests
    try:
        ensure_dir(os.path.dirname(dest_path))
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            done = 0; t0 = time.time()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk); done += len(chunk)
                        if total > 0:
                            bar = int(50 * done / total)
                            speed = done / max(1, (time.time()-t0)) / (1024*1024)
                            sys.stdout.write(f"\rDL [{'#'*bar}{'.'*(50-bar)}] {done/1e6:.1f}/{total/1e6:.1f} MB @ {speed:.2f} MB/s"); sys.stdout.flush()
        print()
        return True
    except Exception as e:
        print(f"\n[ERREUR] download {url}: {e}")
        return False

def extract_any(archive_path, dest_dir):
    try:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zf: zf.extractall(dest_dir)
        elif archive_path.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, 'r:gz') as tf: tf.extractall(dest_dir)
        elif archive_path.endswith(".tar"):
            with tarfile.open(archive_path, 'r') as tf: tf.extractall(dest_dir)
        else:
            print(f"[WARN] Format inconnu pour {archive_path}"); return False
        return True
    except Exception as e:
        print(f"[ERREUR] Extraction {archive_path}: {e}")
        return False

# ========================= Squelette projet =========================

def create_micheline_skeleton(base_dir="micheline"):
    print("\n--- Squelette de dossiers 'micheline' ---")
    dirs = [
        os.path.join(base_dir, "models", "llm"),
        os.path.join(base_dir, "models", "stt", "vosk", "fr"),
        os.path.join(base_dir, "models", "stt", "whisper"),
        os.path.join(base_dir, "models", "tts", "piper"),
        os.path.join(base_dir, "models", "embeddings"),
        os.path.join(base_dir, "memory", "db"),
        os.path.join(base_dir, "memory", "vector"),
        os.path.join(base_dir, "rag", "index", "faiss"),
        os.path.join(base_dir, "rag", "corpus", "raw_cache"),
        os.path.join(base_dir, "rag", "corpus", "clean"),
        os.path.join(base_dir, "logs"),
        os.path.join(base_dir, "cache"),
        os.path.join(base_dir, "configs"),
    ]
    for d in dirs: ensure_dir(d)

# ========================= Build Tools & OCR =========================

def is_msvc_present():
    if os.name != "nt": return True
    try:
        proc = subprocess.run(["where", "cl"], capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False

def ensure_visual_cpp_build_tools():
    if os.name != "nt":
        print("--- Outils C++: non-Windows, ignoré ---"); return True
    if is_msvc_present():
        print("--- MSVC présent ---"); return True
    print("\n=== Installation Visual Studio Build Tools (C++) ===")
    dl_dir = os.path.join("micheline", "cache"); ensure_dir(dl_dir)
    bootstrapper = os.path.join(dl_dir, "vs_BuildTools.exe")
    url = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
    if not download_with_progress(url, bootstrapper):
        print("[WARN] Téléchargement Build Tools impossible. Installe manuellement.")
        return False
    cmd = [bootstrapper, "--add", "Microsoft.VisualStudio.Workload.VCTools", "--includeRecommended", "--passive", "--norestart", "--wait"]
    ok = run_stream(cmd)
    if not ok: print("[WARN] MSVC install a échoué.")
    return is_msvc_present()

def install_llama_cpp_optional():
    print("\n=== Tentative d'installation llama-cpp-python ===")
    for tool in ["setuptools","wheel","scikit-build-core","ninja","cmake","typing-extensions","diskcache","jinja2"]:
        install_with_progress(tool)
    for cmd in [[sys.executable,"-m","pip","install","--upgrade","--prefer-binary","llama-cpp-python"], [sys.executable,"-m","pip","install","--upgrade","--prefer-binary","llama-cpp-python==0.3.16"]]:
        if run_stream(cmd): print("=== llama-cpp installé ==="); return True
    print("[WARN] llama-cpp non installé (optionnel)."); return False

def has_tesseract_binary() -> bool:
    if which("tesseract"): return True
    if os.name == "nt": return os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    return False

def ensure_tesseract_windows() -> bool:
    if os.name != "nt": return True
    if has_tesseract_binary(): print("--- Tesseract présent ---"); return True
    print("\n=== Installation Tesseract (winget/choco) ===")
    if which("winget"):
        if install_with_progress(["winget","install","--id=Tesseract-OCR.Project.Tesseract-OCR","-e", "--accept-package-agreements","--accept-source-agreements"], is_pip=False) and has_tesseract_binary():
            return True
    if which("choco"):
        if install_with_progress(["choco","install","tesseract","-y"], is_pip=False) and has_tesseract_binary():
            return True
    print("[WARN] Tesseract non installé (pas d'admin). Fallback OCR restera opérationnel.")
    return has_tesseract_binary()

def ensure_tesseract_languages(langs=("eng","fra")) -> bool:
    if os.name == "nt":
        default_target = r"C:\Program Files\Tesseract-OCR\tessdata"
        local_target = os.path.join("micheline","cache","tesseract","tessdata")
        target = default_target
        try:
            ensure_dir(default_target)
            test_path = os.path.join(default_target, "_wtest.tmp")
            with open(test_path, "w") as f: f.write("ok")
            os.remove(test_path)
        except Exception:
            target = local_target
            ensure_dir(target)
            os.environ["TESSDATA_PREFIX"] = os.path.dirname(target)
            print(f"[INFO] TESSDATA_PREFIX={os.environ['TESSDATA_PREFIX']}")
    else:
        target = os.path.join("micheline","cache","tesseract","tessdata"); ensure_dir(target)
        os.environ["TESSDATA_PREFIX"] = os.path.dirname(target)
    base_best = "https://github.com/tesseract-ocr/tessdata_best/raw/main"
    base_fast = "https://github.com/tesseract-ocr/tessdata_fast/raw/main"
    ok = True
    for lang in langs:
        dest = os.path.join(target, f"{lang}.traineddata")
        if os.path.exists(dest): continue
        print(f"--- DL Tesseract lang: {lang} ---")
        if not download_with_progress(f"{base_best}/{lang}.traineddata", dest):
            print(f"[INFO] Fallback fast {lang}")
            if not download_with_progress(f"{base_fast}/{lang}.traineddata", dest):
                print(f"[WARN] Impossible d'obtenir {lang}.traineddata"); ok = False
    return ok

def ensure_ocr_dependencies() -> bool:
    print("\n--- Dépendances OCR ---")
    for pkg in ["Pillow","opencv-python-headless"]:
        if not has_pkg(pkg): install_with_progress(pkg)
    paddle_ok = has_pkg("paddleocr") or install_with_progress("paddleocr")
    if not paddle_ok and not has_pkg("paddlepaddle"):
        install_with_progress("paddlepaddle==2.5.0")
        paddle_ok = has_pkg("paddleocr") or install_with_progress("paddleocr")
    tess_ok = has_pkg("pytesseract") or install_with_progress("pytesseract")
    if os.name == "nt":
        ensure_tesseract_windows(); ensure_tesseract_languages(("eng", "fra"))
    ready = bool(paddle_ok or tess_ok)
    print(f"--- OCR prêt: {ready} ---")
    return ready

# ========================= STT & TTS =========================

def ensure_vosk_fr_model(base_dir="micheline"):
    target = os.path.join(base_dir, "models", "stt", "vosk", "fr")
    if os.path.isdir(target) and os.listdir(target): print("--- Vosk FR déjà présent ---"); return True
    print("\n--- Installation Vosk FR ---")
    url = "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip"
    zip_dest = os.path.join(base_dir, "models", "stt", "vosk", "vosk-model-small-fr-0.22.zip")
    if not download_with_progress(url, zip_dest) or not extract_any(zip_dest, os.path.dirname(zip_dest)):
        print("[WARN] Installation Vosk FR échouée."); return False
    try:
        parent = os.path.dirname(zip_dest)
        candidates = [d for d in os.listdir(parent) if d.startswith("vosk-model-") and os.path.isdir(os.path.join(parent, d))]
        if candidates:
            src = os.path.join(parent, candidates[0])
            ensure_dir(target)
            for item in os.listdir(src):
                shutil.move(os.path.join(src, item), os.path.join(target, item))
            try: os.remove(zip_dest); shutil.rmtree(src, ignore_errors=True)
            except Exception: pass
            print(f"--- Vosk FR installé -> {target} ---")
            return True
    except Exception as e: print(f"[WARN] Finaliser Vosk: {e}")
    return False

def ensure_git():
    if which("git"): print("--- Git OK ---"); return True
    print("[WARN] git manquant. whisper.cpp peut échouer."); return False

def ensure_cmake():
    if which("cmake"): print("--- CMake OK ---"); return True
    print("[WARN] cmake manquant. whisper.cpp ne sera pas compilé."); return False

def ensure_whisper_cpp(base_dir="micheline", model_name=None):
    model_name = model_name or os.getenv("WHISPER_MODEL","ggml-small.bin")
    print("\n=== whisper.cpp (STT) ===")
    ensure_git(); ensure_cmake()
    if os.name == "nt": ensure_visual_cpp_build_tools()
    repo_dir = os.path.join(base_dir, "cache", "whisper.cpp")
    build_dir = os.path.join(repo_dir, "build")
    bin_name = "main.exe" if os.name == "nt" else "main"
    out_bin = os.path.join(base_dir, "models", "stt", "whisper", bin_name)
    model_path = os.path.join(base_dir, "models", "stt", "whisper", model_name)
    if not os.path.isdir(repo_dir):
        if not which("git"): print("[WARN] git absent -> skip whisper.cpp"); return False
        if not run_stream(["git","clone","--depth","1","https://github.com/ggerganov/whisper.cpp", repo_dir]):
            print("[WARN] clone whisper.cpp échoué."); return False
    else:
        run_stream(["git","pull"], cwd=repo_dir)
    if not which("cmake"): print("[WARN] cmake absent -> skip build"); return False
    if not os.path.exists(out_bin):
        ensure_dir(build_dir)
        cmake_flags = ["-DGGML_METAL=ON"] if sys.platform == "darwin" else []
        if sys.platform.startswith("linux"):
            run_stream(["sudo","-n","apt-get","install","-y","libopenblas-dev"])
            cmake_flags += ["-DGGML_BLAS=ON","-DGGML_BLAS_VENDOR=OpenBLAS"]
        if not run_stream(["cmake","-S",repo_dir,"-B",build_dir] + cmake_flags) or not run_stream(["cmake","--build",build_dir,"--config","Release","-j"]):
            print("[WARN] Build whisper.cpp échoué."); return False
        found = next((p for p in Path(build_dir).rglob(bin_name) if p.is_file()), None)
        if not found: print("[WARN] Binaire whisper.cpp introuvable après build."); return False
        ensure_dir(os.path.dirname(out_bin))
        try: shutil.copy2(found, out_bin); os.chmod(out_bin, 0o755)
        except Exception as e: print(f"[WARN] copy binary: {e}")
    if not os.path.exists(model_path):
        url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{model_name}?download=true"
        if not download_with_progress(url, model_path):
            print("[WARN] Téléchargement modèle whisper échoué."); return False
    ok = os.path.exists(out_bin) and os.path.exists(model_path)
    print(f"--- whisper.cpp prêt: {ok} (bin={out_bin}) ---")
    return ok

def smoke_test_piper(piper_bin: str, voice_onnx: str, out_wav: str = None, espeak_env_dir: str = None) -> bool:
    try:
        import tempfile
        out_wav = out_wav or os.path.join(tempfile.gettempdir(), "piper_test.wav")
        try:
            if os.path.exists(out_wav): os.remove(out_wav)
        except Exception: pass
        env = None
        if espeak_env_dir:
            env = os.environ.copy()
            env["PATH"] = espeak_env_dir + os.pathsep + env.get("PATH","")
            data_dir = os.path.join(espeak_env_dir, "espeak-ng-data")
            if os.path.isdir(data_dir): env["ESPEAK_DATA_PATH"] = data_dir
        p = subprocess.Popen([piper_bin, "-m", voice_onnx, "-f", out_wav], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=False, env=env)
        text = "Bonjour, ceci est un test de synthèse vocale locale avec Piper."
        p.communicate(input=text.encode("utf-8"), timeout=60)
        return (p.returncode == 0 and os.path.exists(out_wav) and os.path.getsize(out_wav) > 800)
    except Exception: return False

def ensure_piper_and_voice(base_dir="micheline"):
    print("\n=== Piper (TTS) — mode portable (FR+EN+IT+DE) ===")
    plat = detect_platform_tag()
    piper_dir = os.path.join(base_dir, "models", "tts", "piper")
    ensure_dir(piper_dir)
    piper_bin = os.path.join(piper_dir, "piper.exe" if os.name == "nt" else "piper")

    if not os.path.exists(piper_bin):
        print("[WARN] Téléchargement binaire Piper échoué."); return False

    es_dir = None
    if os.name == "nt":
        if os.path.exists(os.path.join(piper_dir, "espeak-ng.dll")) and os.path.isdir(os.path.join(piper_dir, "espeak-ng-data")):
            es_dir = piper_dir
            print(f"--- espeak-ng portable détecté -> {es_dir} ---")
        else:
            print("[WARN] espeak-ng.dll + espeak-ng-data manquent. Place-les dans:", piper_dir); return False

    default_pack = ["fr_FR-upmc-medium", "en_US-lessac-medium", "it_IT-riccardo-medium", "de_DE-thorsten-medium"]
    env_list = os.getenv("PIPER_VOICES", "").strip()
    voices_to_get = [v.strip() for v in env_list.split(",") if v.strip()] if env_list else default_pack
    voices_dir = os.path.join(piper_dir, "voices"); ensure_dir(voices_dir)

    def have_voice(vname: str) -> bool:
        return (os.path.exists(os.path.join(voices_dir, f"{vname}.onnx")) and os.path.exists(os.path.join(voices_dir, f"{vname}.onnx.json")))
    def hf_voice_path(vname: str) -> str:
        parts = vname.replace('_','-').split('-')
        if len(parts) == 4:
            lang, region, speaker, quality = parts
            return f"{lang}/{lang}_{region}/{speaker}/{quality}"
        return ""
    def dl_voice(vname: str) -> bool:
        base = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
        path = hf_voice_path(vname)
        if not path: print(f"[WARN] Nom de voix inconnu: {vname}"); return False
        onnx_url = f"{base}/{path}/{vname}.onnx"; json_url = f"{base}/{path}/{vname}.onnx.json"
        onnx_path = os.path.join(voices_dir, f"{vname}.onnx"); json_path = os.path.join(voices_dir, f"{vname}.onnx.json")
        print(f"--- DL voix: {vname} ---"); ok1 = download_with_progress(onnx_url, onnx_path); ok2 = download_with_progress(json_url, json_path)
        if not (ok1 and ok2):
            try:
                if os.path.exists(onnx_path): os.remove(onnx_path)
                if os.path.exists(json_path): os.remove(json_path)
            except Exception: pass
            return False
        return True

    for v in voices_to_get:
        if not have_voice(v): dl_voice(v)

    test_voice = next((v for v in default_pack if have_voice(v)), None) or next((p.stem for p in Path(voices_dir).glob("*.onnx")), None)
    if not test_voice: print("[WARN] Pas de voix trouvée pour le test."); return False
    onnx = os.path.join(voices_dir, f"{test_voice}.onnx")
    if smoke_test_piper(piper_bin, onnx, espeak_env_dir=es_dir):
        print(f"--- Piper prêt (test avec): {test_voice} ---"); return True
    print("[WARN] Piper n’a pas réussi le test audio."); return False

# ========================= Ollama, GGUF, Embeddings =========================
def ensure_ollama_installed_or_upgraded() -> bool:
    if which("ollama"):
        print("--- Ollama présent ---")
        if os.name == "nt" and which("winget"):
            install_with_progress(["winget","upgrade","Ollama.Ollama","-e", "--accept-package-agreements","--accept-source-agreements"], is_pip=False)
        return True
    print("\n=== Installation d'Ollama ===")
    if os.name == "nt":
        if which("winget") and install_with_progress(["winget","install","--id=Ollama.Ollama","-e", "--accept-package-agreements","--accept-source-agreements"], is_pip=False): return True
        if which("choco") and install_with_progress(["choco","install","ollama","-y"], is_pip=False): return True
        print("Installe manuellement: https://ollama.com/download"); return False
    else:
        print("Linux/macOS: installe via https://ollama.com/download"); return which("ollama") is not None

def ensure_ollama_running(max_wait=40) -> bool:
    try: import requests
    except Exception: install_with_progress("requests"); import requests
    def ping():
        try:
            r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
            return r.status_code == 200
        except Exception: return False
    if ping(): print("--- Ollama déjà en ligne ---"); return True
    print("--- Démarrage 'ollama serve' ---")
    flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0
    try: subprocess.Popen(["ollama","serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, creationflags=flags)
    except Exception as e: print(f"[ERREUR] ollama serve: {e}"); return False
    for _ in range(max_wait):
        if ping(): print("--- Ollama en ligne ---"); return True
        time.sleep(1)
    print("[WARN] Ollama ne répond pas."); return False

def ensure_vlm_model_auto() -> str:
    candidates = ["qwen2-vl:7b-instruct", "qwen2.5-vl:7b-instruct", "llava:13b", "moondream:1.8b"]
    if not which("ollama"): return ""
    print("\n=== Pull modèle Vision (fallback list) ===")
    for m in candidates:
        print(f"pull {m} ...")
        if run_stream(["ollama","pull",m]):
            print(f"=== Modèle Vision prêt: {m} ==="); return m
    print("[WARN] Aucun VLM téléchargé."); return ""

def ensure_llm_gguf(base_dir="micheline"):
    ensure_dir(os.path.join(base_dir, "models", "llm"))
    preferred = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    dest = os.path.join(base_dir, "models", "llm", preferred)
    if os.path.exists(dest): print(f"--- GGUF déjà présent: {dest} ---"); return True
    token = os.getenv("HF_TOKEN","").strip()
    if not token: print("[INFO] Pas de HF_TOKEN. Place manuellement le GGUF si besoin."); return False
    if not has_pkg("huggingface_hub"): install_with_progress("huggingface_hub")
    from huggingface_hub import hf_hub_download
    print("--- Téléchargement GGUF via HF Hub ---")
    try:
        path = hf_hub_download(repo_id="QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF", filename=preferred, token=token, local_dir=os.path.join(base_dir, "models", "llm"))
        ok = os.path.exists(path)
        print(f"--- GGUF {'OK' if ok else 'ECHEC'}: {path} ---")
        return ok
    except Exception as e: print(f"[WARN] GGUF via HF: {e}"); return False

def ensure_rag_packages():
    print("\n--- Dépendances RAG ---")
    pkgs = [
        "trafilatura", "beautifulsoup4", "lxml", "pypdf", "langchain-text-splitters",
        "sentence-transformers", "faiss-cpu"
    ]
    for p in pkgs:
        if not has_pkg(p): install_with_progress(p)

# ========================= Fine-Tuning (transformers/peft/datasets) =========================
def _install_torch_cpu() -> bool:
    # Installe la version CPU (stable) de torch
    cpu_index = "https://download.pytorch.org/whl/cpu"
    print("\n--- Installation torch (CPU) ---")
    ok = run_stream([sys.executable, "-m", "pip", "install", "--upgrade", "torch", "--index-url", cpu_index])
    if not ok:
        print("[WARN] torch CPU via index officiel a échoué, tentative pip simple...")
        ok = install_with_progress("torch")
    return has_pkg("torch")

def _install_torch_gpu_cu121() -> bool:
    # Essaie l’index cu121 de PyTorch (NVIDIA). À utiliser seulement si MICHELINE_TORCH_GPU=1
    cu_index = "https://download.pytorch.org/whl/cu121"
    print("\n--- Installation torch (CUDA 12.1) ---")
    ok = run_stream([sys.executable, "-m", "pip", "install", "--upgrade", "torch", "--index-url", cu_index])
    return has_pkg("torch")

def ensure_ft_packages() -> bool:
    """
    Installe les dépendances Fine‑Tuning:
      - torch (CPU par défaut; GPU si MICHELINE_TORCH_GPU=1)
      - transformers, peft, datasets, accelerate, transformers_stream_generator
      - bitsandbytes (Linux + GPU seulement)
    Retourne True si l’ensemble minimal est présent.
    """
    print("\n--- Dépendances Fine‑Tuning (transformers/peft/datasets) ---")

    # 1) torch
    want_gpu = os.getenv("MICHELINE_TORCH_GPU", "0").strip() == "1"
    if not has_pkg("torch"):
        if want_gpu:
            ok = _install_torch_gpu_cu121()
            if not ok:
                print("[WARN] torch GPU (cu121) indisponible. Fallback vers CPU.")
                ok = _install_torch_cpu()
        else:
            ok = _install_torch_cpu()
        if not ok:
            print("[WARN] torch n'a pas pu être installé.")
    else:
        print("--- torch OK ---")

    # 2) Paquets FT essentiels (y compris la dépendance pour Qwen)
    essential_ft_packages = [
        "transformers", 
        "peft", 
        "datasets", 
        "accelerate", 
        "transformers_stream_generator" # Ajouté ici
    ]
    for pkg in essential_ft_packages:
        if not has_pkg(pkg):
            install_with_progress(pkg)
        else:
            print(f"--- {pkg} OK ---")

    # 3) bitsandbytes (optionnel, GPU, Linux uniquement)
    if want_gpu and sys.platform.startswith("linux") and not has_pkg("bitsandbytes"):
        print("\n--- Tentative bitsandbytes (optionnel) ---")
        install_with_progress("bitsandbytes")
    elif want_gpu and os.name == "nt":
        print("[INFO] bitsandbytes: non supporté officiellement sous Windows. Ignoré.")

    ready = all(has_pkg(p) for p in ["torch"] + essential_ft_packages)
    print(f"--- Fine‑Tuning prêt: {ready} ---")
    return ready

def ensure_embeddings_model(base_dir="micheline", model_id="sentence-transformers/all-MiniLM-L6-v2"):
    print(f"\n--- Téléchargement embeddings: {model_id} ---")
    target_dir = os.path.join(base_dir, "models", "embeddings", "all-MiniLM-L6-v2")
    if os.path.isdir(target_dir) and os.listdir(target_dir): print(f"--- Embeddings déjà présents -> {target_dir} ---"); return True
    if not has_pkg("huggingface_hub"): install_with_progress("huggingface_hub")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=target_dir, local_dir_use_symlinks=False, ignore_patterns=["*.safetensors.index.json"])
        print(f"--- Embeddings téléchargés -> {target_dir} ---"); return True
    except Exception as e: print(f"[WARN] snapshot_download échoué: {e}"); return False

def smoke_test_embeddings(base_dir="micheline"):
    try:
        from sentence_transformers import SentenceTransformer
        model_dir = os.path.join(base_dir, "models", "embeddings", "all-MiniLM-L6-v2")
        m = SentenceTransformer(model_dir if os.path.isdir(model_dir) else "sentence-transformers/all-MiniLM-L6-v2")
        _ = m.encode(["Bonjour le monde"], normalize_embeddings=True)
        print("--- Test embeddings OK ---"); return True
    except Exception as e: print(f"[WARN] Test embeddings: {e}"); return False

# ========================= main() =========================

def main():
    print("--- Setup Micheline (full local, simplifié) ---")
    check_and_fix_numpy()

    core_packages = [
        "pandas","tensorflow","tf-keras","requests","python-dotenv","nltk","pandas_ta",
        "MetaTrader5","finta","scipy","scikit-learn","numba","mplfinance","arch",
        "vosk","sounddevice","pyttsx3","Pillow","opencv-python-headless","huggingface_hub"
    ]
    for p in core_packages:
        if not has_pkg(p): install_with_progress(p)
    try:
        import nltk
        print("\n--- NLTK: vader_lexicon ---")
        nltk.download('vader_lexicon', quiet=True)
    except Exception as e: print(f"[INFO] NLTK ignoré: {e}")

    ensure_rag_packages()
    try: ocr_ok = ensure_ocr_dependencies()
    except Exception as e: print(f"[WARN] OCR: {e}"); ocr_ok = False

    # Dépendances Fine‑Tuning
    try:
        ft_ok = ensure_ft_packages()
    except Exception as e:
        print(f"[WARN] Fine‑Tuning deps: {e}")
        ft_ok = False

    create_micheline_skeleton(base_dir="micheline")

    stt_vosk_ok = False
    try: stt_vosk_ok = ensure_vosk_fr_model(base_dir="micheline")
    except Exception as e: print(f"[WARN] Vosk: {e}")
    stt_whisper_ok = False
    try: stt_whisper_ok = ensure_whisper_cpp(base_dir="micheline")
    except Exception as e: print(f"[WARN] whisper.cpp: {e}")
    piper_ok = False
    try: piper_ok = ensure_piper_and_voice(base_dir="micheline")
    except Exception as e: print(f"[WARN] Piper: {e}")

    msvc_ok = ensure_visual_cpp_build_tools() if os.name == "nt" else True
    llama_ok = install_llama_cpp_optional() if msvc_ok else False

    try: ensure_llm_gguf(base_dir="micheline")
    except Exception as e: print(f"[INFO] GGUF: {e}")

    emb_ok = ensure_embeddings_model(base_dir="micheline")
    smoke_test_embeddings(base_dir="micheline")

    ollama_ok = ensure_ollama_installed_or_upgraded()
    vlm_model = ""
    if ollama_ok and ensure_ollama_running(max_wait=40):
        vlm_model = ensure_vlm_model_auto()

    print("\n==========================================================")
    print("Récapitulatif:")
    print(f"- llama-cpp-python: {'OK' if llama_ok else 'optionnel/non installé'}")
    print(f"- OCR (Paddle/Tesseract): {'OK' if ocr_ok else 'NON'}")
    print(f"- Fine‑tuning (torch/transformers/peft/datasets): {'OK' if ft_ok else 'NON/optionnel'}")
    print(f"- Ollama: {'OK' if ollama_ok else 'NON'}")
    print(f"- Modèle VLM: {vlm_model if vlm_model else 'NON'}")
    print(f"- STT whisper.cpp: {'OK' if stt_whisper_ok else 'NON'}")
    print(f"- STT Vosk FR: {'OK' if stt_vosk_ok else 'NON'}")
    print(f"- TTS Piper (voix): {'OK' if piper_ok else 'NON'}")
    print(f"- Embeddings (MiniLM-L6): {'OK' if emb_ok else 'NON'}")
    print("----------------------------------------------------------")
    if not ollama_ok or not vlm_model:
        print("Ollama/VLM: si besoin, installe manuellement et:\n  ollama pull qwen2-vl:7b-instruct")
    print("==========================================================")

    input("\nAppuyez sur Entrée pour fermer cette fenêtre.")

if __name__ == "__main__":
    main()