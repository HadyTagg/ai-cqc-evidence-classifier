import os
import io
import csv
import json
import time
import uuid
import shutil
import base64
import zipfile
import hashlib
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml
import pandas as pd
import streamlit as st

# Parsing libraries
import chardet
from bs4 import BeautifulSoup
from email import policy
from email.parser import BytesParser

# Documents
from pdfminer.high_level import extract_text as pdf_extract_text
import docx

# Imaging / OCR
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
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

# ---------------------
# Caches (OCR + LLM)
# ---------------------
OCR_CACHE_DIR = Path('.ocr_cache'); OCR_CACHE_DIR.mkdir(exist_ok=True)
LLM_CACHE_DIR = Path('.llm_cache'); LLM_CACHE_DIR.mkdir(exist_ok=True)

def _sha256(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b or b""); return h.hexdigest()

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

def build_llm_cache_key(text: str, taxonomy: Dict[str, Any], model: str, mode: str) -> str:
    payload = {
        "taxonomy_version": taxonomy.get("metadata", {}).get("version"),
        "model": model,
        "mode": mode,
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

# ---------------------
# Extraction helpers (TEXT path)
# ---------------------

def read_file_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def detect_encoding(b: bytes) -> str:
    try:
        guess = chardet.detect(b)
        return guess.get("encoding") or "utf-8"
    except Exception:
        return "utf-8"

def extract_text_from_txt(path: Path) -> str:
    b = read_file_bytes(path)
    return b.decode(detect_encoding(b), errors="replace")

def extract_text_from_pdf(path: Path) -> str:
    try:
        return pdf_extract_text(str(path))
    except Exception as e:
        return f"[PDF extraction error: {e}]"

def extract_text_from_docx(path: Path) -> str:
    try:
        d = docx.Document(str(path))
        return "
".join(p.text for p in d.paragraphs)
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
            parts.append(f"
--- Sheet: {name} ---
")
            parts.append(df.to_csv(index=False))
        return "
".join(parts)
    except Exception as e:
        return f"[XLSX read error: {e}]"

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.extract()
    return soup.get_text("
", strip=True)

def extract_text_from_eml(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
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
        header_text = "
".join(f"{k}: {v}" for k, v in headers.items() if v)
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
        full_text = header_text + "

" + "

".join(body_parts).strip()
        return full_text, attachments
    except Exception as e:
        return f"[EML parse error: {e}]", []

# ---------------------
# Rasterization helpers (IMAGE path)
# ---------------------

def text_to_image(text: str, width: int = 1200, padding: int = 20) -> Image.Image:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow not available for text rasterization")
    text = (text or "").replace("
", "")
    # Basic wrapping
    lines = []
    for para in text.split("
"):
        if not para:
            lines.append("")
            continue
        # crude wrap by char count
        wrap = 110
        while len(para) > wrap:
            cut = para.rfind(" ", 0, wrap)
            if cut == -1:
                cut = wrap
            lines.append(para[:cut])
            para = para[cut:].lstrip()
        lines.append(para)
    font = ImageFont.load_default()
    # Estimate height
    line_h = 16
    height = padding * 2 + max(200, line_h * (len(lines) + 1))
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = padding
    for ln in lines:
        draw.text((padding, y), ln, fill="black", font=font)
        y += line_h
        if y > height - padding:
            break
    return img

def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG"); b = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b}"

def rasterize_to_images(path: Path, dpi: int = 200, max_pages: int = 2) -> List[Image.Image]:
    ext = path.suffix.lower()
    images: List[Image.Image] = []
    # Raw images
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic"}:
        if Image is None:
            raise RuntimeError("Pillow not available to load images")
        images = [Image.open(path)]
        return images
    # PDFs
    if ext == ".pdf" and convert_from_path is not None:
        imgs = convert_from_path(str(path), dpi=int(dpi))
        if max_pages:
            imgs = imgs[:max_pages]
        return imgs
    # DOCX -> try LibreOffice to PDF, then pdf2image; else fallback to text rasterization
    if ext == ".docx":
        outdir = Path(".docx_pdf"); outdir.mkdir(exist_ok=True)
        pdf_out = outdir / (path.stem + ".pdf")
        try:
            subprocess.run([
                "soffice", "--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(path)
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if pdf_out.exists() and convert_from_path is not None:
                imgs = convert_from_path(str(pdf_out), dpi=int(dpi))
                if max_pages:
                    imgs = imgs[:max_pages]
                return imgs
        except Exception:
            pass
        # fallback
        txt = extract_text_from_docx(path)
        return [text_to_image(txt)]
    # EML/CSV/XLSX/TXT -> rasterize extracted text
    if ext == ".eml":
        txt, _ = extract_text_from_eml(path)
        return [text_to_image(txt)]
    if ext == ".txt":
        return [text_to_image(extract_text_from_txt(path))]
    if ext == ".csv":
        return [text_to_image(extract_text_from_csv(path))]
    if ext == ".xlsx":
        return [text_to_image(extract_text_from_xlsx(path))]
    # Unknown -> best effort
    return [text_to_image(f"[Unsupported for rasterization: {ext}]
{path}")]

# ---------------------
# Taxonomy & Paths
# ---------------------
@st.cache_data(show_spinner=False)
def load_taxonomy(taxonomy_path: str) -> Dict[str, Any]:
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def list_quality_statement_options(taxonomy: Dict[str, Any]) -> List[Dict[str, str]]:
    return taxonomy.get("quality_statements", [])

def list_evidence_categories(taxonomy: Dict[str, Any]) -> List[str]:
    return taxonomy.get("evidence_categories", [])

def _sanitize(s: str) -> str:
    s = (s or '').strip().replace('/', '-').replace('\\
', '-')
    for ch in '<>:"|?*':
        s = s.replace(ch, '')
    return ' '.join(s.split())[:100]

def propose_storage_paths(taxonomy: Dict[str, Any], qs_ids: List[str], categories: List[str]) -> List[str]:
    qs_map = {q["id"]: q for q in taxonomy.get("quality_statements", [])}
    templates = taxonomy.get("path_templates", {})
    paths = []
    for qid in qs_ids:
        q = qs_map.get(qid)
        if not q:
            continue
        domain = q.get("domain", "Misc"); title = q.get("title", qid)
        tmpl = templates.get(domain, "{domain}/{qs_title}/{category}")
        for cat in categories:
            path = (tmpl
                .replace("{domain}", _sanitize(domain))
                .replace("{qs_id}", _sanitize(qid))
                .replace("{qs_title}", _sanitize(title))
                .replace("{qs_id_and_title}", _sanitize(f"{qid} – {title}"))
                .replace("{category}", _sanitize(cat)))
            paths.append(path)
    return sorted(set(paths))

# ---------------------
# LLM Provider (text + vision)
# ---------------------
class LLMProvider:
    def __init__(self, provider: str, model: str, api_key: str = "", timeout: int = 60, max_retries: int = 3):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def classify_text(self, text: str, taxonomy: Dict[str, Any], max_chars: int = 6000) -> Dict[str, Any]:
        qs = taxonomy.get("quality_statements", [])
        cats = taxonomy.get("evidence_categories", [])
        qs_brief = [{"id": q.get("id"), "domain": q.get("domain"), "title": q.get("title")} for q in qs]
        system_prompt = (
            "You are a compliance assistant for a CQC-regulated care service. "
            "Given an evidence item (text extracted from a document or email), propose one or more relevant CQC "
            "Single Assessment Framework Quality Statements and one or more Evidence Categories. "
            "Return ONLY a JSON object matching the schema. Be concise with rationale."
        )
        user_payload = {
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
            "evidence_categories_options": cats,
            "evidence_text_first_6000_chars": (text or "")[:max_chars],
        }
        return self._chat_json(system_prompt, user_payload)

    def classify_images(self, images: List[Image.Image], taxonomy: Dict[str, Any], max_images: int = 4) -> Dict[str, Any]:
        qs = taxonomy.get("quality_statements", [])
        cats = taxonomy.get("evidence_categories", [])
        qs_brief = [{"id": q.get("id"), "domain": q.get("domain"), "title": q.get("title")} for q in qs]
        system_prompt = (
            "You are a compliance assistant for a CQC-regulated care service. "
            "You will be given one or more images of an evidence item (a scanned document, email, report, or photo). "
            "Propose one or more relevant CQC Quality Statements and Evidence Categories that this item supports. "
            "Return ONLY a JSON object per the schema."
        )
        content = [{"type": "text", "text": json.dumps({
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
            "evidence_categories_options": cats,
        })}]
        for img in images[:max_images]:
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_url(img)}})
        return self._chat_json(system_prompt, None, content_override=content)

    # --- transport ---
    def _chat_json(self, system_prompt: str, user_payload: Dict[str, Any] | None, content_override: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        if self.provider != "openai":
            return {"error": "Vision classification currently implemented for OpenAI Chat Completions only."}
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key or os.getenv('OPENAI_API_KEY','')}", "Content-Type": "application/json"}
        if content_override is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_override},
            ]
        data = {"model": self.model or "gpt-4o-mini", "messages": messages, "temperature": 0.1, "response_format": {"type": "json_object"}}
        import requests, random
        attempt, last_err = 0, None
        while attempt <= 3:
            try:
                resp = requests.post(url, headers=headers, json=data, timeout=90)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    import time
                    ra = resp.headers.get("Retry-After"); wait = float(ra) if ra else min(10.0, 2 ** attempt + random.random())
                    time.sleep(wait); attempt += 1; last_err = requests.HTTPError(f"HTTP {resp.status_code}"); continue
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"error": "Model did not return valid JSON", "raw": content}
            except Exception as e:
                last_err = e; attempt += 1
        return {"error": f"Request failed after retries: {last_err}"}

# ---------------------
# UI
# ---------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Settings")
    input_dir = Path(st.text_input("Input folder (files to classify)", value=str(Path.cwd() / "input")))
    output_dir = Path(st.text_input("Output base folder (root for filing)", value=str(Path.cwd() / "classified")))

    taxonomy_path = st.text_input("Taxonomy file (YAML)", value=str(Path.cwd() / "cqc_taxonomy.yaml"))
    if st.button("Reload taxonomy"): st.cache_data.clear()

    provider = st.selectbox("LLM provider", ["openai"], index=0)
    model = st.text_input("Model", value="gpt-4o-mini")
    api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")

    request_timeout = st.number_input("API timeout (sec)", min_value=10, max_value=120, value=60, step=5)
    max_retries = st.number_input("Max retries on 429/5xx", min_value=0, max_value=10, value=3, step=1)
    use_cache = st.checkbox("Use cached results (by content)", value=True)
    cooldown_secs = st.number_input("Cooldown between calls (sec)", min_value=0, max_value=120, value=10, step=5)
    max_chars = st.number_input("Max chars sent to LLM (text mode)", min_value=1000, max_value=12000, value=4000, step=500)

    st.markdown("**Classification mode**")
    mode = st.radio("Choose mode", ["Text/OCR", "Image-only (Vision)"] , index=1)

    st.markdown("**OCR / Rasterization**")
    ocr_lang = st.text_input("OCR languages (Tesseract codes)", value="eng")
    ocr_dpi = st.number_input("OCR/Rasterize DPI", min_value=100, max_value=600, value=200, step=50)
    ocr_max_pages = st.number_input("Max pages/images to preview/classify", min_value=1, max_value=10, value=2, step=1)

    move_or_copy = st.radio("On approval, file by…", ["Copy", "Move"], index=0)
    st.caption("Nothing is filed without your approval. Every decision is logged.")

# Load taxonomy
try:
    taxonomy = load_taxonomy(taxonomy_path)
except Exception as e:
    st.error(f"Failed to load taxonomy: {e}"); st.stop()

qs_options = list_quality_statement_options(taxonomy)
qs_map = {q["id"]: q for q in qs_options}
qs_id_list = [q["id"] for q in qs_options]
cat_options = list_evidence_categories(taxonomy)

# Discover files
input_dir.mkdir(parents=True, exist_ok=True)
files = sorted([p for p in input_dir.glob("**/*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
st.subheader("Files to review")
st.write(f"Found **{len(files)}** file(s) in `{input_dir}`.")

# Provider
prov = LLMProvider(provider=provider, model=model, api_key=api_key, timeout=int(request_timeout), max_retries=int(max_retries))

colA, colB = st.columns([2, 3])
with colA:
    file_selected = st.selectbox("Pick a file", files, format_func=lambda p: str(p.relative_to(input_dir))) if files else None
    if not files:
        st.info("Drop some files into the input folder to begin (supported: .pdf .docx .xlsx .csv .txt .eml .png .jpg .jpeg .tif .tiff .bmp .webp .heic).")

with colB:
    text = ""; images = []
    if file_selected:
        if mode == "Image-only (Vision)":
            try:
                images = rasterize_to_images(file_selected, dpi=int(ocr_dpi), max_pages=int(ocr_max_pages))
                st.markdown(f"**Preview (images): {file_selected.name}**")
                st.image(images, caption=[f"image {i+1}" for i in range(len(images))], use_column_width=True)
            except Exception as e:
                st.error(f"Rasterization error: {e}")
        else:
            # Text path: use native extraction + (optional) light OCR via pdf2image+pytesseract on first pages
            ext = file_selected.suffix.lower()
            if ext == ".pdf":
                text = extract_text_from_pdf(file_selected)
                if (not text or len(text.strip()) < 40) and convert_from_path is not None and pytesseract is not None:
                    # fallback quick OCR of first pages
                    try:
                        imgs = convert_from_path(str(file_selected), dpi=int(ocr_dpi))[:int(ocr_max_pages)]
                        ocr_texts = [pytesseract.image_to_string(im, lang=ocr_lang or "eng") for im in imgs]
                        text = "

".join([text] + ocr_texts)
                    except Exception:
                        pass
            elif ext == ".docx":
                text = extract_text_from_docx(file_selected)
            elif ext == ".csv":
                text = extract_text_from_csv(file_selected)
            elif ext == ".xlsx":
                text = extract_text_from_xlsx(file_selected)
            elif ext == ".eml":
                text, _ = extract_text_from_eml(file_selected)
            elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic"} and pytesseract is not None:
                try:
                    text = pytesseract.image_to_string(Image.open(file_selected), lang=ocr_lang or "eng")
                except Exception as e:
                    text = f"[Image OCR error: {e}]"
            elif ext == ".txt":
                text = extract_text_from_txt(file_selected)
            else:
                text = f"[Unsupported file type: {ext}]"
            st.markdown(f"**Preview (text): {file_selected.name}**")
            st.code(text[:8000] if text else "[No text extracted]", language="markdown")

# Run classification
if 'next_ok_at' not in st.session_state:
    st.session_state['next_ok_at'] = 0.0

if st.button("Run LLM on this file"):
    if not file_selected:
        st.warning("Pick a file first.")
    else:
        now = time.time()
        if now < st.session_state['next_ok_at']:
            wait = int(st.session_state['next_ok_at'] - now)
            st.warning(f"Cooling down. Try again in {wait}s to avoid rate limits.")
        else:
            st.session_state['next_ok_at'] = now +  float(1)
            with st.spinner("Asking the model…"):
                if mode == "Image-only (Vision)":
                    key = build_llm_cache_key("[image-mode]" + str(file_selected), taxonomy, model, mode)
                    cached = llm_cache_read(key) if use_cache else None
                    if cached:
                        result = cached
                    else:
                        result = prov.classify_images(images or rasterize_to_images(file_selected, dpi=int(ocr_dpi), max_pages=int(ocr_max_pages)), taxonomy)
                        if use_cache and result and not result.get('error'):
                            llm_cache_write(key, result)
                else:
                    key = build_llm_cache_key(text or "", taxonomy, model, mode)
                    cached = llm_cache_read(key) if use_cache else None
                    if cached:
                        result = cached
                    else:
                        result = prov.classify_text(text or "", taxonomy, max_chars=int(max_chars))
                        if use_cache and result and not result.get('error'):
                            llm_cache_write(key, result)
            st.session_state['llm_result'] = result

result = st.session_state.get('llm_result')
if result:
    if result.get('error'):
        st.error(f"LLM error: {result['error']}")
        with st.expander("Raw output"):
            st.code(result.get('raw', ''))
    else:
        sugg_qs = result.get("quality_statements", [])
        st.markdown("**Model’s suggested Quality Statements:**")
        for q in sugg_qs:
            qid = q.get("id"); dom = q.get("domain") or qs_map.get(qid, {}).get("domain", "?"); title = q.get("title") or qs_map.get(qid, {}).get("title", "")
            st.write(f"- **{qid}** ({dom}) – {title} | confidence: {q.get('confidence','?')}

  rationale: {q.get('rationale','')}")
        default_ids = [q.get("id") for q in sugg_qs if q.get("id") in qs_map]
        selected_qs = st.multiselect("Confirm Quality Statements", options=qs_id_list, default=default_ids, format_func=lambda qid: f"[{qs_map[qid]['domain']}] {qid} – {qs_map[qid]['title']}" if qid in qs_map else qid)
        sugg_cats = [c for c in result.get("evidence_categories", []) if c in cat_options]
        selected_cats = st.multiselect("Confirm Evidence Categories (multi-select)", options=cat_options, default=sugg_cats or cat_options[:1])
        st.markdown("**Proposed storage paths:**")
        paths = propose_storage_paths(taxonomy, selected_qs, selected_cats)
        for p in paths:
            st.write(f"- {p}")
        notes = st.text_area("Reviewer notes (optional)", value=result.get("notes", ""))
        signed_off = st.checkbox("I confirm the above classification and approve filing.")
        reviewer = st.text_input("Your name for the audit log", value=os.getenv("USER", "Reviewer"))
        col1, col2 = st.columns([1,1])
        with col1: approve = st.button("Approve & File")
        with col2: reject = st.button("Reject (do not file)")
        def write_decision_log(row: Dict[str, Any]):
            file_exists = Path(DEFAULT_DECISIONS_LOG).exists()
            with open(DEFAULT_DECISIONS_LOG, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp","file","provider","model","quality_statements","evidence_categories","paths","reviewer","notes","action","mode"])
                if not file_exists: writer.writeheader()
                writer.writerow(row)
        if approve:
            if not signed_off:
                st.warning("Please tick the sign‑off checkbox before approving.")
            else:
                original_path = file_selected; original_name = file_selected.name
                first_dest_file = None
                for idx, rel in enumerate(paths):
                    dest_dir = output_dir / rel; dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = dest_dir / original_name
                    if idx == 0:
                        if move_or_copy == "Move":
                            shutil.move(str(original_path), str(dest_file)); first_dest_file = dest_file
                        else:
                            shutil.copy2(str(original_path), str(dest_file)); first_dest_file = dest_file
                    else:
                        src = first_dest_file if move_or_copy == "Move" else original_path
                        if src and Path(src).exists(): shutil.copy2(str(src), str(dest_file))
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
                    "mode": mode,
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
                "mode": mode,
            })
            st.info("Not filed. Decision recorded.")

# Convenience: allow `python app.py` to launch Streamlit
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
        if get_script_run_ctx() is None:
            import sys, subprocess
            print("Launching Streamlit server...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", sys.argv[0]])
    except Exception:
        pass