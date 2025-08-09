import os
import re
import csv
import json
import time
import uuid
import shutil
import chardet
import requests
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml
import pandas as pd
import streamlit as st

# PDF
from pdfminer.high_level import extract_text as pdf_extract_text
# DOCX
import docx
# Email
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup

# OCR imports (optional but recommended)
try:
    from PIL import Image
except Exception:
    Image = None
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
# Optional HEIC support
try:
    import pillow_heif as heif
    if heif:
        heif.register_heif_opener()
except Exception:
    heif = None

APP_TITLE = "CQC Evidence Classifier"
DEFAULT_DECISIONS_LOG = "decisions.csv"
SUPPORTED_EXTS = {
    ".txt", ".pdf", ".docx", ".csv", ".xlsx", ".eml",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic"
}

# --- Caches ---
OCR_CACHE_DIR = Path(".ocr_cache"); OCR_CACHE_DIR.mkdir(exist_ok=True)
LLM_CACHE_DIR = Path(".llm_cache"); LLM_CACHE_DIR.mkdir(exist_ok=True)

def _sha256(b: bytes) -> str:
    import hashlib as _h; h=_h.sha256(); h.update(b or b""); return h.hexdigest()

def build_ocr_cache_key(file_bytes: bytes, params: Dict[str, Any]) -> str:
    payload = {"len": len(file_bytes or b""), "sha": _sha256(file_bytes), **params}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

def ocr_cache_read(key: str) -> str | None:
    p = OCR_CACHE_DIR / f"{key}.txt"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None
    return None

def ocr_cache_write(key: str, text: str) -> None:
    try:
        (OCR_CACHE_DIR / f"{key}.txt").write_text(text, encoding="utf-8")
    except Exception:
        pass

def build_llm_cache_key(text: str, taxonomy: Dict[str, Any], model: str) -> str:
    payload = {
        "taxonomy_version": taxonomy.get("metadata", {}).get("version"),
        "model": model,
        "text_head": (text or "")[:10000],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

def llm_cache_read(key: str) -> Dict[str, Any] | None:
    p = LLM_CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def llm_cache_write(key: str, data: Dict[str, Any]) -> None:
    try:
        (LLM_CACHE_DIR / f"{key}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

# --- Extractors ---
def read_file_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def detect_encoding(b: bytes) -> str:
    try:
        import chardet as _c
        guess = _c.detect(b)
        return guess.get("encoding") or "utf-8"
    except Exception:
        return "utf-8"

def extract_text_from_txt(path: Path) -> str:
    b = read_file_bytes(path)
    enc = detect_encoding(b)
    return b.decode(enc, errors="replace")

def extract_text_from_pdf(path: Path) -> str:
    try:
        return pdf_extract_text(str(path))
    except Exception as e:
        return f"[PDF extraction error: {e}]"

def extract_text_from_docx(path: Path) -> str:
    try:
        d = docx.Document(str(path))
        return "\n".join(p.text for p in d.paragraphs)
    except Exception as e:
        return f"[DOCX extraction error: {e}]"

def extract_text_from_csv(path: Path) -> str:
    try:
        df = pd.read_csv(path)
        return df.to_csv(index=False)
    except Exception as e:
        return f"[CSV read error: {e}]"

def extract_text_from_xlsx(path: Path) -> str:
    try:
        dfs = pd.read_excel(path, sheet_name=None)
        parts = []
        for name, df in dfs.items():
            parts.append(f"\n--- Sheet: {name} ---\n")
            parts.append(df.to_csv(index=False))
        return "\n".join(parts)
    except Exception as e:
        return f"[XLSX read error: {e}]"

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.extract()
    return soup.get_text("\n", strip=True)

def extract_text_from_eml(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (email_text, attachments_meta)."""
    try:
        b = read_file_bytes(path)
        msg = BytesParser(policy=policy.default).parsebytes(b)
        headers = {
            "From": str(msg.get("From", "")),
            "To": str(msg.get("To", "")),
            "Cc": str(msg.get("Cc", "")),
            "Date": str(msg.get("Date", "")),
            "Subject": str(msg.get("Subject", "")),
        }
        header_text = "\n".join(f"{k}: {v}" for k, v in headers.items() if v)

        body_parts = []
        attachments = []
        if msg.is_multipart():
            for part in msg.walk():
                cd = part.get_content_disposition()
                ct = part.get_content_type()
                if cd == "attachment":
                    filename = part.get_filename() or f"attachment_{uuid.uuid4().hex}"
                    payload = part.get_payload(decode=True) or b""
                    tmpdir = Path(".eml_attachments"); tmpdir.mkdir(exist_ok=True)
                    temp_path = tmpdir / filename
                    with open(temp_path, "wb") as f:
                        f.write(payload)
                    attachments.append({"filename": filename, "temp_path": str(temp_path)})
                elif ct == "text/plain":
                    body_parts.append(part.get_content())
                elif ct == "text/html":
                    body_parts.append(html_to_text(part.get_content()))
        else:
            ct = msg.get_content_type()
            if ct == "text/plain":
                body_parts.append(msg.get_content())
            elif ct == "text/html":
                body_parts.append(html_to_text(msg.get_content()))

        full_text = header_text + "\n\n" + "\n\n".join(body_parts).strip()
        return full_text, attachments
    except Exception as e:
        return f"[EML parse error: {e}]", []

def _ocr_available() -> bool:
    return pytesseract is not None and Image is not None

def _ocr_pdf_available() -> bool:
    return _ocr_available() and convert_from_path is not None

def ocr_image_file(path: Path, lang: str = "eng") -> str:
    if not _ocr_available():
        return "[OCR unavailable: install Tesseract & Pillow]"
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img, lang=lang or "eng")
    except Exception as e:
        return f"[Image OCR error: {e}]"

def ocr_pdf_file(path: Path, lang: str = "eng", dpi: int = 300, max_pages: int = 0) -> str:
    if not _ocr_pdf_available():
        return "[PDF OCR unavailable: install Tesseract & Poppler (pdf2image)]"
    try:
        images = convert_from_path(str(path), dpi=int(dpi))
        texts = []
        for i, img in enumerate(images, start=1):
            if max_pages and i > max_pages:
                break
            try:
                texts.append(pytesseract.image_to_string(img, lang=lang or "eng"))
            except Exception as e:
                texts.append(f"[Page {i} OCR error: {e}]")
        return "\n\n\f\n\n".join(texts)
    except Exception as e:
        return f"[PDF OCR error: {e}]"

def extract_text(path: Path, ocr_cfg: Dict[str, Any] | None = None) -> Tuple[str, List[Dict[str, Any]]]:
    """
    ocr_cfg keys:
      enable (bool), pdf_mode ('auto'|'always'|'never'), lang (str), dpi (int), max_pages (int), auto_min_chars (int)
    """
    ext = path.suffix.lower()
    ocr_cfg = ocr_cfg or {}
    enable_ocr = bool(ocr_cfg.get("enable", False))
    pdf_mode = (ocr_cfg.get("pdf_mode") or "auto").lower()
    ocr_lang = ocr_cfg.get("lang") or "eng"
    ocr_dpi = int(ocr_cfg.get("dpi") or 300)
    ocr_max_pages = int(ocr_cfg.get("max_pages") or 0)
    auto_min_chars = int(ocr_cfg.get("auto_min_chars") or 40)

    if ext == ".txt":
        return extract_text_from_txt(path), []
    if ext == ".pdf":
        txt = extract_text_from_pdf(path)
        needs_ocr = False
        if enable_ocr:
            if pdf_mode == "always":
                needs_ocr = True
            elif pdf_mode == "auto":
                if not txt or txt.strip() == "" or len(txt.strip()) < auto_min_chars or txt.startswith("[PDF extraction error"):
                    needs_ocr = True
        if needs_ocr:
            b = read_file_bytes(path)
            key = build_ocr_cache_key(b, {"kind":"pdf","lang":ocr_lang,"dpi":ocr_dpi,"max_pages":ocr_max_pages})
            cached = ocr_cache_read(key)
            if cached:
                return cached, []
            ocr_txt = ocr_pdf_file(path, lang=ocr_lang, dpi=ocr_dpi, max_pages=ocr_max_pages)
            ocr_cache_write(key, ocr_txt)
            return ocr_txt, []
        return txt, []
    if ext == ".docx":
        return extract_text_from_docx(path), []
    if ext == ".csv":
        return extract_text_from_csv(path), []
    if ext == ".xlsx":
        return extract_text_from_xlsx(path), []
    if ext == ".eml":
        return extract_text_from_eml(path)
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic"}:
        if enable_ocr:
            b = read_file_bytes(path)
            key = build_ocr_cache_key(b, {"kind":"image","lang":ocr_lang})
            cached = ocr_cache_read(key)
            if cached:
                return cached, []
            text = ocr_image_file(path, lang=ocr_lang)
            ocr_cache_write(key, text)
            return text, []
        else:
            return "[OCR disabled – enable in sidebar to process images]", []
    return f"[Unsupported file type: {ext}]", []

# --- Taxonomy & Paths ---
@st.cache_data(show_spinner=False)
def load_taxonomy(taxonomy_path: str) -> Dict[str, Any]:
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def list_quality_statement_options(taxonomy: Dict[str, Any]) -> List[Dict[str, str]]:
    return taxonomy.get("quality_statements", [])

def list_evidence_categories(taxonomy: Dict[str, Any]) -> List[str]:
    return taxonomy.get("evidence_categories", [])

def _sanitize_path_component(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("/", "-").replace("\\", "-")
    for ch in '<>:"|?*':
        s = s.replace(ch, "")
    s = " ".join(s.split())
    return s[:100]

def propose_storage_paths(taxonomy: Dict[str, Any], qs_ids: List[str], categories: List[str]) -> List[str]:
    qs_map = {q["id"]: q for q in taxonomy.get("quality_statements", [])}
    templates = taxonomy.get("path_templates", {})
    paths = []
    for qid in qs_ids:
        q = qs_map.get(qid)
        if not q:
            continue
        domain = q.get("domain", "Misc")
        title = q.get("title", qid)
        tmpl = templates.get(domain, "{domain}/{qs_title}/{category}")
        for cat in categories:
            path = (
                tmpl.replace("{domain}", _sanitize_path_component(domain))
                    .replace("{qs_id}", _sanitize_path_component(qid))
                    .replace("{qs_title}", _sanitize_path_component(title))
                    .replace("{qs_id_and_title}", _sanitize_path_component(f"{qid} – {title}"))
                    .replace("{category}", _sanitize_path_component(cat))
            )
            paths.append(path)
    return sorted(set(paths))

# --- LLM Provider ---
class LLMProvider:
    def __init__(self, provider: str, model: str, api_key: str = "", timeout: int = 60, max_retries: int = 3):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def classify(self, text: str, taxonomy: Dict[str, Any], max_chars: int = 6000) -> Dict[str, Any]:
        quality_statements = taxonomy.get("quality_statements", [])
        evidence_categories = taxonomy.get("evidence_categories", [])
        qs_brief = [{"id": q.get("id"), "domain": q.get("domain"), "title": q.get("title")} for q in quality_statements]
        system_prompt = (
            "You are a compliance assistant for a CQC-regulated care service. "
            "Given an evidence item (text extracted from a document or email), propose one or more relevant CQC "
            "Single Assessment Framework Quality Statements and one or more Evidence Categories. "
            "Return ONLY a JSON object matching the schema. Be concise with rationale."
        )
        user_prompt = {
            "schema": {
                "type": "object",
                "properties": {
                    "quality_statements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "domain": {"type": "string"},
                                "confidence": {"type": "number"},
                                "rationale": {"type": "string"},
                            },
                            "required": ["id", "confidence"],
                        },
                    },
                    "evidence_categories": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
                "required": ["quality_statements", "evidence_categories"],
            },
            "quality_statements_options": qs_brief,
            "evidence_categories_options": evidence_categories,
            "evidence_text_first_6000_chars": (text or "")[:max_chars],
        }
        if self.provider == "openai":
            return self._classify_openai(system_prompt, user_prompt)
        elif self.provider == "ollama":
            return self._classify_ollama(system_prompt, user_prompt)
        else:
            return {"error": "Unsupported provider"}

    def _classify_openai(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key or os.getenv('OPENAI_API_KEY','')}", "Content-Type": "application/json"}
        data = {
            "model": self.model or "gpt-4o-mini",
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            "temperature": 0.1,
        }
        attempt = 0
        last_err = None
        while attempt <= max(0, int(self.max_retries)):
            try:
                resp = requests.post(url, headers=headers, json=data, timeout=self.timeout)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else min(10.0, (2 ** attempt))
                    time.sleep(wait)
                    attempt += 1
                    last_err = requests.HTTPError(f"HTTP {resp.status_code}")
                    continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"error": "Model did not return valid JSON", "raw": content}
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                time.sleep(min(10.0, (2 ** attempt)))
                attempt += 1
            except requests.HTTPError as e:
                return {"error": f"HTTP error: {e}", "status": getattr(e.response, 'status_code', None), "raw": getattr(e.response, 'text', '')}
        return {"error": f"Request failed after retries: {last_err}", "status": getattr(getattr(last_err, 'response', None), 'status_code', None)}

    def _classify_ollama(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            f"SYSTEM:\n{system_prompt}\n\n"
            f"USER (JSON payload follows):\n{json.dumps(user_payload)}\n\n"
            "Reply with ONLY valid JSON."
        )
        url = "http://localhost:11434/api/generate"
        data = {"model": self.model or "llama3:8b-instruct", "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
        resp = requests.post(url, json=data, timeout=120)
        resp.raise_for_status()
        content = resp.json().get("response", "{}")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Model did not return valid JSON", "raw": content}

# --- UI ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Settings")
    input_dir = Path(st.text_input("Input folder (files to classify)", value=str(Path.cwd() / "input")))
    output_dir = Path(st.text_input("Output base folder (root for filing)", value=str(Path.cwd() / "classified")))

    taxonomy_path = st.text_input("Taxonomy file (YAML)", value=str(Path.cwd() / "cqc_taxonomy.yaml"))
    if st.button("Reload taxonomy"):
        st.cache_data.clear()
    provider = st.selectbox("LLM provider", ["openai", "ollama"], index=0)
    model = st.text_input("Model", value=("gpt-4o-mini" if provider == "openai" else "llama3:8b-instruct"))
    api_key = st.text_input("OpenAI API Key (if provider=openai)", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    request_timeout = st.number_input("API timeout (sec)", min_value=10, max_value=120, value=60, step=5)
    max_retries = st.number_input("Max retries on 429/5xx", min_value=0, max_value=10, value=3, step=1)
    use_cache = st.checkbox("Use cached results (by content)", value=True)
    cooldown_secs = st.number_input("Cooldown between calls (sec)", min_value=0, max_value=120, value=10, step=5)
    max_chars = st.number_input("Max chars sent to LLM", min_value=1000, max_value=12000, value=4000, step=500)

    st.markdown("**OCR settings**")
    enable_ocr = st.checkbox("Enable OCR for images & scanned PDFs", value=True)
    ocr_pdf_mode = st.selectbox("OCR PDFs", ["Auto (if no text)", "Always", "Never"], index=0)
    ocr_lang = st.text_input("OCR languages (Tesseract codes)", value="eng")
    ocr_dpi = st.number_input("OCR render DPI (PDF)", min_value=150, max_value=600, value=300, step=50)
    ocr_max_pages = st.number_input("OCR max pages (0 = all)", min_value=0, max_value=200, value=0, step=1)
    auto_min_chars = st.number_input("Auto-OCR if extracted chars <", min_value=0, max_value=2000, value=40, step=10)
    if enable_ocr:
        if pytesseract is None or Image is None:
            st.warning("OCR not available: install Tesseract & Pillow.")
        if convert_from_path is None:
            st.info("For scanned PDFs, install Poppler (poppler-utils) for pdf2image.")

    move_or_copy = st.radio("On approval, file by…", ["Copy", "Move"], index=0)
    st.caption("Nothing is filed without your approval. Every decision is logged.")

# Load taxonomy
try:
    taxonomy = load_taxonomy(taxonomy_path)
except Exception as e:
    st.error(f"Failed to load taxonomy: {e}")
    st.stop()

qs_options = list_quality_statement_options(taxonomy)
qs_map = {q["id"]: q for q in qs_options}
qs_id_list = [q["id"] for q in qs_options]
cat_options = list_evidence_categories(taxonomy)

# Discover files
input_dir.mkdir(parents=True, exist_ok=True)
files = sorted([p for p in input_dir.glob("**/*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
st.subheader("Files to review")
st.write(f"Found **{len(files)}** file(s) in `{input_dir}`.")

colA, colB = st.columns([2, 3])
with colA:
    if files:
        file_selected = st.selectbox("Pick a file", files, format_func=lambda p: str(p.relative_to(input_dir)))
    else:
        file_selected = None
        st.info("Drop some files into the input folder to begin (supported: .pdf .docx .xlsx .csv .txt .eml .png .jpg .jpeg .tif .tiff .bmp .webp .heic).")

with colB:
    text, attachments = ("", [])
    if file_selected:
        ocr_cfg = {
            "enable": enable_ocr,
            "pdf_mode": {"Auto (if no text)":"auto","Always":"always","Never":"never"}[ocr_pdf_mode],
            "lang": ocr_lang,
            "dpi": int(ocr_dpi),
            "max_pages": int(ocr_max_pages),
            "auto_min_chars": int(auto_min_chars),
        }
        text, attachments = extract_text(file_selected, ocr_cfg=ocr_cfg)
        st.markdown(f"**Preview: {file_selected.name}**")
        st.code(text[:8000] if text else "[No text extracted]", language="markdown")
        if attachments:
            st.markdown("**Attachments detected:**")
            for att in attachments:
                st.write(f"- {att['filename']} → {att['temp_path']}")

# --- Batch queue (propose only) ---
st.subheader("Batch queue (propose only)")
if "batch_queue" not in st.session_state:
    st.session_state["batch_queue"] = []
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = {}

batch_select = st.multiselect("Select files to add to queue", files, format_func=lambda p: str(p.relative_to(input_dir)))
if st.button("Add to queue"):
    for p in batch_select:
        if p not in st.session_state["batch_queue"]:
            st.session_state["batch_queue"].append(p)

if st.session_state["batch_queue"]:
    st.write("**Current queue:**")
    for p in st.session_state["batch_queue"]:
        st.write(f"- {p.relative_to(input_dir)}")
    if st.button("Run queue (propose only)"):
        progress = st.progress(0)
        total = len(st.session_state["batch_queue"])
        for i, fpath in enumerate(list(st.session_state["batch_queue"])):
            try:
                text_i, _ = extract_text(
                    fpath,
                    ocr_cfg={
                        "enable": enable_ocr,
                        "pdf_mode": {"Auto (if no text)":"auto","Always":"always","Never":"never"}[ocr_pdf_mode],
                        "lang": ocr_lang,
                        "dpi": int(ocr_dpi),
                        "max_pages": int(ocr_max_pages),
                        "auto_min_chars": int(auto_min_chars),
                    },
                )
                key_i = build_llm_cache_key(text_i or "", taxonomy, model)
                cached_i = llm_cache_read(key_i) if use_cache else None
                if cached_i:
                    res = cached_i
                else:
                    prov = LLMProvider(provider=provider, model=model, api_key=api_key, timeout=int(request_timeout), max_retries=int(max_retries))
                    res = prov.classify(text_i or "", taxonomy, max_chars=int(max_chars))
                    if use_cache and res and not res.get("error"):
                        llm_cache_write(key_i, res)
                st.session_state["batch_results"][str(fpath)] = res
            except Exception as e:
                st.session_state["batch_results"][str(fpath)] = {"error": str(e)}
            progress.progress(int(((i + 1) / total) * 100))
            if i < total - 1 and cooldown_secs:
                time.sleep(float(cooldown_secs))
        st.success("Batch complete. Open each file above to review and approve (nothing has been filed).")
        import pandas as _pd
        summary_rows = []
        for f, res in st.session_state["batch_results"].items():
            if isinstance(res, dict) and not res.get("error"):
                qs_ids = ",".join([q.get("id","") for q in res.get("quality_statements", [])])
                cats = ",".join(res.get("evidence_categories", []))
                summary_rows.append({"file": Path(f).name, "qs": qs_ids, "categories": cats})
            else:
                err = res.get("error","") if isinstance(res, dict) else "(error)"
                summary_rows.append({"file": Path(f).name, "qs": "(error)", "categories": err})
        if summary_rows:
            st.dataframe(_pd.DataFrame(summary_rows))

# --- Single-file propose/approve ---
prov = LLMProvider(provider=provider, model=model, api_key=api_key, timeout=int(request_timeout), max_retries=int(max_retries))

if "next_ok_at" not in st.session_state:
    st.session_state["next_ok_at"] = 0.0

if st.button("Run LLM on this file"):
    if not file_selected:
        st.warning("Pick a file first.")
    else:
        key = build_llm_cache_key(text or "", taxonomy, model)
        cached = llm_cache_read(key) if use_cache else None
        if cached:
            st.info("Loaded cached classification for this content.")
            st.session_state["llm_result"] = cached
        else:
            now = time.time()
            if now < st.session_state["next_ok_at"]:
                wait = int(st.session_state["next_ok_at"] - now)
                st.warning(f"Cooling down. Try again in {wait}s to avoid rate limits.")
            else:
                st.session_state["next_ok_at"] = now + float(cooldown_secs)
                with st.spinner("Asking the model…"):
                    result = prov.classify(text or "", taxonomy, max_chars=int(max_chars))
                st.session_state["llm_result"] = result
                if use_cache and result and not result.get("error"):
                    llm_cache_write(key, result)

result = st.session_state.get("llm_result")

if result:
    if result.get("error"):
        st.error(f"LLM error: {result['error']}")
        with st.expander("Raw output"):
            st.code(result.get("raw", ""))
    else:
        sugg_qs = result.get("quality_statements", [])
        st.markdown("**Model’s suggested Quality Statements:**")
        for q in sugg_qs:
            qid = q.get("id")
            domain = q.get("domain") or qs_map.get(qid, {}).get("domain", "?")
            title = q.get("title") or qs_map.get(qid, {}).get("title", "")
            st.write(
                f"- **{qid}** ({domain}) – {title}"
                f"  | confidence: {q.get('confidence','?')}"
                f"\n\n  rationale: {q.get('rationale','')}"
            )

        default_ids = [q.get("id") for q in sugg_qs if q.get("id") in qs_map]
        selected_qs = st.multiselect(
            "Confirm Quality Statements",
            options=qs_id_list,
            default=default_ids,
            format_func=lambda qid: f"[{qs_map[qid]['domain']}] {qid} – {qs_map[qid]['title']}" if qid in qs_map else qid,
        )

        sugg_cats = [c for c in result.get("evidence_categories", []) if c in cat_options]
        selected_cats = st.multiselect(
            "Confirm Evidence Categories (multi-select)",
            options=cat_options,
            default=sugg_cats or cat_options[:1],
        )

        st.markdown("**Proposed storage paths:**")
        paths = propose_storage_paths(taxonomy, selected_qs, selected_cats)
        for p in paths:
            st.write(f"- {p}")

        notes = st.text_area("Reviewer notes (optional)", value=result.get("notes", ""))
        signed_off = st.checkbox("I confirm the above classification and approve filing.")
        reviewer = st.text_input("Your name for the audit log", value=os.getenv("USER", "Reviewer"))

        def write_decision_log(row: Dict[str, Any]):
            file_exists = Path(DEFAULT_DECISIONS_LOG).exists()
            with open(DEFAULT_DECISIONS_LOG, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp","file","provider","model","quality_statements",
                        "evidence_categories","paths","reviewer","notes","action",
                    ],
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        col1, col2 = st.columns([1, 1])
        with col1:
            approve = st.button("Approve & File")
        with col2:
            reject = st.button("Reject (do not file)")

        if approve:
            if not signed_off:
                st.warning("Please tick the sign-off checkbox before approving.")
            else:
                first_dest_file = None
                original_path = file_selected
                original_name = file_selected.name

                for idx, rel in enumerate(paths):
                    dest_dir = output_dir / rel
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = dest_dir / original_name

                    if idx == 0:
                        if move_or_copy == "Move":
                            shutil.move(str(original_path), str(dest_file))
                            first_dest_file = dest_file
                        else:
                            shutil.copy2(str(original_path), str(dest_file))
                            first_dest_file = dest_file
                    else:
                        src = first_dest_file if move_or_copy == "Move" else original_path
                        if src and Path(src).exists():
                            shutil.copy2(str(src), str(dest_file))

                write_decision_log({
                    "timestamp": pd.Timestamp.utcnow().isoformat(),
                    "file": str(original_path),
                    "provider": provider,
                    "model": model,
                    "quality_statements": json.dumps(selected_qs),
                    "evidence_categories": json.dumps(selected_cats),
                    "paths": json.dumps(paths),
                    "reviewer": reviewer,
                    "notes": notes,
                    "action": "approved",
                })
                st.success("Filed to all selected locations.")
        if reject:
            write_decision_log({
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "file": str(file_selected) if file_selected else "",
                "provider": provider,
                "model": model,
                "quality_statements": json.dumps(selected_qs if result else []),
                "evidence_categories": json.dumps(selected_cats if result else []),
                "paths": json.dumps(paths if result else []),
                "reviewer": reviewer,
                "notes": notes,
                "action": "rejected",
            })
            st.info("Not filed. Decision recorded.")

# --- Convenience: allow `python app.py` to launch Streamlit ---
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
        if get_script_run_ctx() is None:
            import sys, subprocess
            print("Launching Streamlit server...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0]])
    except Exception:
        pass