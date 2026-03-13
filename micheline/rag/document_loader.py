# micheline/rag/document_loader.py
# Charge et nettoie le contenu depuis diverses sources: URL, PDF, TXT, etc.
# Ajout: fallback BeautifulSoup quand trafilatura ne renvoie rien (mur cookies/JS)

import os
from pathlib import Path
from typing import List, Dict, Optional
import requests
import trafilatura
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

# NEW
import re
from bs4 import BeautifulSoup  # fallback extraction (installer via setup.py)
# --------

class Document:
    """Structure simple pour un document chargé, avant découpage."""
    def __init__(self, content: str, metadata: Dict):
        self.page_content = content
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"Document(metadata={self.metadata}, content_len={len(self.page_content)})"

def _clean_text_lines(text: str) -> str:
    """Compacte les lignes (supprime les vides répétées)."""
    lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()

def _bs4_fallback(html: str) -> str:
    """
    Fallback HTML -> texte brut quand trafilatura échoue.
    - Retire scripts/styles/noscript/header/footer/nav/aside/form/svg
    - Retire les bannières cookies/consent les plus communes
    - Supprime les éléments interactifs et les liens javascript pour la sécurité.
    """
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup, Comment
    except ImportError:
        return ""

    soup = BeautifulSoup(html, "lxml")

    # Supprimer les tags non-visuels ou de navigation/interaction
    tags_to_remove = ["script", "style", "noscript", "header", "footer", "nav", "aside", "form", "svg", "button", "input", "textarea", "select", "option", "label", "iframe"]
    for tag in soup(tags_to_remove):
        tag.decompose()

    # Supprimer les commentaires HTML
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Supprimer les bannières de cookies/consentement courantes
    cookie_re = re.compile(r"(cookie|consent|didomi|gdpr|privacy|banner)", re.I)
    for el in soup.find_all(attrs={"id": cookie_re}):
        el.decompose()
    for el in soup.find_all(class_=cookie_re):
        el.decompose()
    for el in soup.find_all(attrs={"role": "dialog"}):
        el.decompose()
        
    # Nettoyer les attributs potentiellement dangereux des balises restantes
    for tag in soup.find_all(True):
        for attr in list(tag.attrs.keys()):
            if attr.lower().startswith('on'): # onclick, onmouseover, etc.
                del tag[attr]
            elif attr.lower() == 'href' and tag[attr].lower().startswith('javascript:'):
                del tag[attr]

    # Extraire le texte en préservant mieux les sauts de ligne
    text = soup.get_text(separator='\n', strip=True)
    return _clean_text_lines(text)
    
# micheline/rag/document_loader.py - Remplacez par cette version

def _load_from_url(url: str) -> Optional[Document]:
    """Extrait le contenu textuel principal d'une page web (trafilatura -> fallback BS4)."""
    try:
        # --- DÉBUT DE LA MODIFICATION ---
        # On utilise des en-têtes plus complets pour mieux simuler un navigateur
        # et éviter les blocages sur des sites comme MSN.
        headers = {
            'User-Agent': getattr(config, 'RAG_HTTP_USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1', # Do Not Track
            'Upgrade-Insecure-Requests': '1',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        }
        
        # Le timeout vient aussi de la config, avec une valeur par défaut de 15s.
        timeout = getattr(config, 'RAG_HTTP_TIMEOUT', 15)

        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        # --- FIN DE LA MODIFICATION ---

        r.raise_for_status()

        # 1) Tentative principale: trafilatura
        # On utilise r.text car Trafilatura peut le gérer et c'est mieux pour l'encodage
        content = trafilatura.extract(
            r.text,
            include_comments=False,
            include_tables=bool(getattr(config, 'RAG_INCLUDE_TABLES', True))
        )

        # 2) Fallback BeautifulSoup si trafilatura échoue/renvoie vide
        if not content or not content.strip():
            print(f"[RAG] Trafilatura n'a rien extrait, tentative avec BeautifulSoup sur {url}")
            content = _bs4_fallback(r.text)

        if not content or not content.strip():
            print(f"[RAG] Aucun texte extrait via trafilatura/BS4 pour: {url}")
            return None

        metadata = {'source_type': 'url', 'source': url}
        return Document(content, metadata)

    except Exception as e:
        print(f"[ERROR] Échec du chargement de l'URL {url}: {e}")
        return None
        
def _load_from_pdf(file_path: str) -> Optional[Document]:
    """Extrait le texte d'un fichier PDF."""
    try:
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
        content = _clean_text_lines(content)
        if not content:
            return None
        metadata = {'source_type': 'pdf', 'source': file_path}
        return Document(content, metadata)
    except Exception as e:
        print(f"[ERROR] Échec de la lecture du PDF {file_path}: {e}")
        return None

def _load_from_txt(file_path: str) -> Optional[Document]:
    """Charge un fichier texte simple (UTF-8)."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        content = _clean_text_lines(content)
        if not content:
            return None
        metadata = {'source_type': 'txt', 'source': file_path}
        return Document(content, metadata)
    except Exception as e:
        print(f"[ERROR] Échec de la lecture du TXT {file_path}: {e}")
        return None

def load_source(source_path: str) -> List[Document]:
    """
    Charge une source de données, qu'il s'agisse d'une URL, d'un fichier ou d'un dossier.
    Retourne une liste de "Documents" (un par fichier trouvé).
    """
    if source_path.startswith("http://") or source_path.startswith("https://"):
        doc = _load_from_url(source_path)
        return [doc] if doc else []

    path = Path(source_path)
    if not path.exists():
        print(f"[ERROR] Le chemin '{source_path}' n'existe pas.")
        return []

    TEXT_EXTS = set((config.RAG_TEXT_EXTS or []))
    documents = []
    if path.is_file():
        if path.suffix.lower() == ".pdf":
            doc = _load_from_pdf(str(path))
            if doc: documents.append(doc)
        elif path.suffix.lower() in TEXT_EXTS:
            doc = _load_from_txt(str(path))
            if doc: documents.append(doc)
        else:
            print(f"[WARN] Format de fichier non supporté pour l'instant: {path.suffix}")
    
    elif path.is_dir():
        print(f"Scan du dossier: {path}...")
        for p in path.rglob("*"):
            if p.is_file():
                if p.suffix.lower() == ".pdf":
                    doc = _load_from_pdf(str(p))
                    if doc: documents.append(doc)
                elif p.suffix.lower() in TEXT_EXTS:
                    doc = _load_from_txt(str(p))
                    if doc: documents.append(doc)
    
    return documents

def split_documents(documents: List[Document], chunk_size: int = None, chunk_overlap: int = None) -> List[Document]:
    """
    Découpe une liste de documents en plus petits morceaux (chunks).
    """
    chunk_size = int(chunk_size or config.RAG_CHUNK_SIZE)
    chunk_overlap = int(chunk_overlap or config.RAG_CHUNK_OVERLAP)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc.page_content or "")
        for chunk_content in chunks:
            chunk_content = _clean_text_lines(chunk_content)
            if not chunk_content:
                continue
            chunk_doc = Document(content=chunk_content, metadata=doc.metadata.copy())
            all_chunks.append(chunk_doc)
            
    print(f"Découpage terminé: {len(documents)} document(s) -> {len(all_chunks)} chunks.")
    return all_chunks