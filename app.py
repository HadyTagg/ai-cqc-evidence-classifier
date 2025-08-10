import os
import io
import csv
import json
import time
import shutil
import base64
import hashlib
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml
import pandas as pd
import streamlit as st

# Parsing libraries (used to turn non-visual docs into preview images)
import chardet
from bs4 import BeautifulSoup
from email import policy
from email.parser import BytesParser

# Documents
import docx  # used for DOCX text fallback to image

# Imaging / Preview
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
try:
    import pillow_heif as heif
    if heif:
        heif.register_heif_opener()
except Exception:
    # No explicit heif variable needed; registration is best-effort
    pass

APP_TITLE = "CQC Evidence Classifier"
DEFAULT_DECISIONS_LOG = "decisions.csv"
SUPPORTED_EXTS = {
    ".txt", ".pdf", ".docx", ".csv", ".xlsx", ".eml",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic"
}

# ---------------------
# Cache (LLM only)
# ---------------------
LLM_CACHE_DIR = Path('.llm_cache'); LLM_CACHE_DIR.mkdir(exist_ok=True)

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
# Simple extractors (only used to render texty files into images)
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
    """Return text content from an .eml. Attachments are ignored and not saved."""
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
        if msg.is_multipart():
            for part in msg.walk():
                cd = part.get_content_disposition()
                ct = part.get_content_type()
                if cd == "attachment":
                    continue
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
        full_text = header_text + "\n\n" + "\n".join(body_parts).strip()
        return full_text, []
    except Exception as e:
        return f"[EML parse error: {e}]", []

# ---------------------
# Rasterization helpers (image preview)
# ---------------------

def text_to_image(text: str, width: int = 1200, padding: int = 20) -> Image.Image:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow not available for text rasterization")
    text = (text or "").replace("\r", "")
    lines = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        wrap = 110
        while len(para) > wrap:
            cut = para.rfind(" ", 0, wrap)
            if cut == -1:
                cut = wrap
            lines.append(para[:cut])
            para = para[cut:].lstrip()
        lines.append(para)
    font = ImageFont.load_default()
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

# ---- Downscaling + JPEG for LLM to reduce size ----
def downscale_for_llm(img: Image.Image, max_w: int = 1024) -> Image.Image:
    if Image is None:
        raise RuntimeError("Pillow not available for image processing")
    if img.width <= max_w:
        return img
    h = int(img.height * (max_w / img.width))
    return img.resize((max_w, h))

def pil_to_data_url_jpeg(img: Image.Image, quality: int = 80) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality), optimize=True)
    b = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b}"

def rasterize_to_images(path: Path, dpi: int = 200, max_pages: int = 2) -> List[Image.Image]:
    ext = path.suffix.lower()
    # Raw images
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic"}:
        if Image is None:
            raise RuntimeError("Pillow not available to load images")
        return [Image.open(path)]
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
        # fallback to rendering extracted text
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
    return [text_to_image(f"[Unsupported for rasterization: {ext}] {path}")]

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
    s = (s or "").strip().replace("/", "-").replace("\\", "-")
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

# ---------- bullets parser ----------
def parse_bullets(block: str) -> List[str]:
    """Parse 'what_this_quality_statement_means' into a flat list of bullets."""
    if not block:
        return []
    lines = [l.rstrip() for l in block.splitlines()]
    bullets: List[str] = []
    cur: List[str] = []
    def flush():
        if cur:
            bullets.append(" ".join(" ".join(cur).split()))
            cur.clear()
    for l in lines:
        stripped = l.strip()
        if stripped.startswith("- "):
            flush()
            cur.append(stripped[2:])
        elif stripped == "":
            if cur:
                cur.append("")
        else:
            cur.append(stripped)
    flush()
    return [b for b in bullets if b]

# ---------------------
# Model normalization for Chat Completions
# ---------------------
def _normalize_model_for_chat(m: str) -> str:
    # Map generic GPT-5 ids to the chat-friendly alias
    m = (m or "").strip()
    if m in {"gpt-5", "gpt-5-latest", "gpt-5-preview"}:
        return "gpt-5-chat-latest"
    return m

# ---------------------
# LLM Provider (VISION ONLY)
# ---------------------
class LLMProvider:
    def __init__(self, provider: str, model: str, api_key: str = "", timeout: int = 60, max_retries: int = 3):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        # Optional runtime knobs (set from UI later)
        self.llm_image_max_width = 1024
        self.auto_fallback = True
        self.fallback_model = "gpt-5-mini"

    def classify_images(self, images: List[Image.Image], taxonomy: Dict[str, Any], max_images: int = 4) -> Dict[str, Any]:
        qs_brief = build_qs_brief(taxonomy)
        cats = taxonomy.get("evidence_categories", [])
        system_prompt = (
            "You are a compliance assistant for a CQC-regulated care service.\n"
            "You will be given image(s) of an evidence item (scan/photo). Map it to one or more CQC Quality Statements "
            "and to the main Evidence Category.\n\n"
            "GROUNDING MATERIAL provided for each Quality Statement includes:\n"
            "- 'we_statement' (verbatim)\n"
            "- 'we_explanation' (verbatim)\n"
            "- 'what_this_quality_statement_means' (verbatim block) and parsed 'means_bullets'\n"
            "- 'i_statements'\n"
            "- 'subtopics'\n"
            "- 'source_url'\n"
            "Use these verbatim texts to make precise mappings. Prefer precision over breadth. "
            "Justify each mapping with a short rationale referencing visible content, and select matching I-statements, "
            "subtopics, or 'means_bullets'. Return ONLY a JSON object per the schema."
        )
        schema_and_options = {
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
                                "matched_i_statements": {"type": "array", "items": {"type": "string"}},
                                "matched_subtopics": {"type": "array", "items": {"type": "string"}},
                                "matched_means_bullets": {"type": "array", "items": {"type": "string"}},
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
        }
        content = [{"type": "text", "text": json.dumps(schema_and_options)}]
        max_w = getattr(self, "llm_image_max_width", 1024)
        for img in images[:max_images]:
            ds = downscale_for_llm(img, max_w=max_w)
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_url_jpeg(ds)}})
        return self._chat_json(system_prompt, None, content_override=content)

    # --- transport ---
    def _chat_json(self, system_prompt: str, user_payload: Dict[str, Any] | None, content_override: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        if self.provider != "openai":
            return {"error": "Vision classification currently implemented for OpenAI Chat Completions only."}
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key or os.getenv('OPENAI_API_KEY','')}", "Content-Type": "application/json"}

        if content_override is None:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(user_payload)}]
        else:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content_override}]

        import requests, random  # type: ignore

        # Build model attempt list
        models_to_try = [self.model or "gpt-5-chat-latest"]
        if getattr(self, "auto_fallback", True):
            fb = getattr(self, "fallback_model", None)
            if fb and fb not in models_to_try:
                models_to_try.append(fb)

        last_err, last_retry_after, last_err_body = None, None, None

        for mdl in models_to_try:
            mdl = _normalize_model_for_chat(mdl)
            data = {
                "model": mdl,
                "messages": messages,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }

            attempt = 0
            while attempt <= int(self.max_retries):
                try:
                    resp = requests.post(url, headers=headers, json=data, timeout=float(self.timeout))
                    # Handle rate limits and server errors with backoff
                    if resp.status_code == 429 or 500 <= resp.status_code < 600:
                        ra = resp.headers.get("Retry-After")
                        wait = float(ra) if ra else min(20.0, 2 ** attempt + random.random())
                        last_retry_after = wait
                        time.sleep(wait)
                        attempt += 1
                        last_err = requests.HTTPError(f"HTTP {resp.status_code}")  # type: ignore
                        continue

                    # If it's a 4xx (other than 429), capture body and break to try fallback
                    if 400 <= resp.status_code < 500:
                        try:
                            last_err_body = resp.json()
                        except Exception:
                            last_err_body = resp.text
                        last_err = requests.HTTPError(f"HTTP {resp.status_code}")  # type: ignore
                        # Do not retry the same model on 4xx (likely bad params or access)
                        break

                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return {"error": "Model did not return valid JSON", "raw": content}
                except Exception as e:
                    last_err = e
                    attempt += 1
            # this model exhausted or returned 4xx—try the next model (fallback) if any

        # If we reach here, all attempts failed
        out = {"error": f"Request failed after retries: {last_err}"}
        if last_err_body is not None:
            out["raw"] = last_err_body
        if last_retry_after is not None:
            out["rate_limited"] = True
            out["retry_after_seconds"] = last_retry_after
        return out

# ---------------------
# Helper to include ALL QS context fields in options
# ---------------------
def build_qs_brief(taxonomy: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for q in taxonomy.get("quality_statements", []):
        means_block = q.get("what_this_quality_statement_means", q.get("what this quality statement means", "")) or ""
        out.append({
            "id": q.get("id"),
            "domain": q.get("domain"),
            "title": q.get("title"),
            "we_statement": q.get("we_statement", ""),
            "we_explanation": q.get("we_explanation", q.get("we explanation", "")),
            "what_this_quality_statement_means": means_block,
            "means_bullets": parse_bullets(means_block),
            "i_statements": q.get("i_statements", []),
            "subtopics": q.get("subtopics", []),
            "source_url": q.get("source_url", ""),
        })
    return out

# ---------------------
# UI
# ---------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Vision-only mode: documents are rasterized to images and classified from the images.")

with st.sidebar:
    st.header("Settings")
    input_dir = Path(st.text_input("Input folder (files to classify)", value=str(Path.cwd() / "input")))
    output_dir = Path(st.text_input("Output base folder (root for filing)", value=str(Path.cwd() / "classified")))

    taxonomy_path = st.text_input("Taxonomy file (YAML)", value=str(Path.cwd() / "cqc_taxonomy.yaml"))
    if st.button("Reload taxonomy"):
        st.cache_data.clear()

    provider = st.selectbox("LLM provider", ["openai"], index=0)
    # Default to chat-safe GPT-5 alias
    model = st.text_input("Model", value="gpt-5-chat-latest")
    api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")

    request_timeout = st.number_input("API timeout (sec)", min_value=10, max_value=120, value=60, step=5)
    max_retries = st.number_input("Max retries on 429/5xx", min_value=0, max_value=10, value=3, step=1)
    use_cache = st.checkbox("Use cached results (by content)", value=True)
    cooldown_secs = st.number_input("Cooldown between calls (sec)", min_value=0, max_value=120, value=10, step=5)

    st.markdown("**Preview settings**")
    preview_dpi = st.number_input("Preview DPI", min_value=100, max_value=600, value=100, step=50)
    preview_max_pages = st.number_input("Max pages/images to preview/classify", min_value=1, max_value=10, value=1, step=1)

    st.markdown("**LLM image controls**")
    llm_image_max_width = st.number_input("LLM image max width (px)", min_value=512, max_value=2048, value=1024, step=128)
    auto_fallback = st.checkbox("Auto-fallback on 429/5xx", value=True)
    fallback_model = st.text_input("Fallback model", value="gpt-5-mini")

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

# Provider
prov = LLMProvider(
    provider=provider,
    model=model,
    api_key=api_key,
    timeout=int(request_timeout),
    max_retries=int(max_retries),
)
# Attach runtime knobs from UI
prov.llm_image_max_width = int(llm_image_max_width)
prov.auto_fallback = bool(auto_fallback)
prov.fallback_model = (fallback_model or "").strip() or None

colA, colB = st.columns([2, 3])
with colA:
    file_selected = st.selectbox("Pick a file", files, format_func=lambda p: str(p.relative_to(input_dir))) if files else None
    if not files:
        st.info("Drop some files into the input folder to begin (supported: .pdf .docx .xlsx .csv .txt .eml .png .jpg .jpeg .tif .tiff .bmp .webp .heic).")

with colB:
    images = []
    if file_selected:
        try:
            images = rasterize_to_images(file_selected, dpi=int(preview_dpi), max_pages=int(preview_max_pages))
            st.markdown(f"**Preview (images): {file_selected.name}**")
            st.image(images, caption=[f"image {i+1}" for i in range(len(images))], use_container_width=True)
        except Exception as e:
            st.error(f"Preview error: {e}")

# Run classification (VISION ONLY)
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
            st.session_state['next_ok_at'] = now + float(cooldown_secs)
            with st.spinner("Asking the model…"):
                key = build_llm_cache_key("[image-mode]" + str(file_selected), taxonomy, model, "Vision-only")
                cached = llm_cache_read(key) if use_cache else None
                if cached:
                    result = cached
                else:
                    imgs = images or rasterize_to_images(file_selected, dpi=int(preview_dpi), max_pages=int(preview_max_pages))
                    result = prov.classify_images(imgs, taxonomy)
                    if use_cache and result and not result.get('error'):
                        llm_cache_write(key, result)
            st.session_state['llm_result'] = result

result = st.session_state.get('llm_result')
if result:
    if result.get('error'):
        if result.get("rate_limited"):
            ra = result.get("retry_after_seconds")
            if ra:
                st.warning(f"Rate limited. Server asked to retry after ~{int(ra)}s. Consider using the fallback model or increasing the cooldown.")
        st.error(f"LLM error: {result['error']}")
        with st.expander("Raw output"):
            st.code(result.get('raw', ''))
    else:
        sugg_qs = result.get("quality_statements", [])
        st.markdown("**Model’s suggested Quality Statements:**")
        for q in sugg_qs:
            qid = q.get("id")
            dom = q.get("domain") or qs_map.get(qid, {}).get("domain", "?")
            title = q.get("title") or qs_map.get(qid, {}).get("title", "")
            conf = q.get("confidence", "?")
            rationale = q.get("rationale", "")
            st.write(f"- **{qid}** ({dom}) – {title} | confidence: {conf}")
            if rationale:
                st.caption(f"Rationale: {rationale}")

            if qid in qs_map:
                qs = qs_map[qid]
                tab_labels = ["We statement", "We explanation", "What it means", "What it means (bullets)", "I statements", "Subtopics", "Source", "Matched (if any)"]
                tabs = st.tabs(tab_labels)

                with tabs[0]:
                    st.write(qs.get("we_statement", "_(none)_") or "_(none)_")
                with tabs[1]:
                    st.write(qs.get("we_explanation", qs.get("we explanation", "_(none)_")) or "_(none)_")
                with tabs[2]:
                    st.write(qs.get("what_this_quality_statement_means", qs.get("what this quality statement means", "_(none)_")) or "_(none)_")
                with tabs[3]:
                    bullets = parse_bullets(qs.get("what_this_quality_statement_means", ""))
                    if bullets:
                        for b in bullets:
                            st.write(f"- {b}")
                    else:
                        st.write("_(none)_")
                with tabs[4]:
                    i_list = qs.get("i_statements") or []
                    if i_list:
                        for s in i_list:
                            st.write(f"- {s}")
                    else:
                        st.write("_(none)_")
                with tabs[5]:
                    subs = qs.get("subtopics") or []
                    if subs:
                        for s in subs:
                            st.write(f"- {s}")
                    else:
                        st.write("_(none)_")
                with tabs[6]:
                    src = qs.get("source_url")
                    if src:
                        st.markdown(f"[Open the official CQC page]({src})")
                    else:
                        st.write("_(none)_")
                with tabs[7]:
                    mi = q.get("matched_i_statements") or []
                    ms = q.get("matched_subtopics") or []
                    mb = q.get("matched_means_bullets") or []
                    if mi:
                        st.write("**Matched I statements:**")
                        for s in mi:
                            st.write(f"- {s}")
                    if ms:
                        st.write("**Matched subtopics:**")
                        for s in ms:
                            st.write(f"- {s}")
                    if mb:
                        st.write("**Matched 'What it means' bullets:**")
                        for s in mb:
                            st.write(f"- {s}")
                    if not mi and not ms and not mb:
                        st.write("_(none returned by model)_")

        default_ids = [q.get("id") for q in sugg_qs if q.get("id") in qs_map]
        selected_qs = st.multiselect(
            "Confirm Quality Statements",
            options=qs_id_list,
            default=default_ids,
            format_func=lambda qid: f"[{qs_map[qid]['domain']}] {qid} – {qs_map[qid]['title']}" if qid in qs_map else qid
        )

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
        with col1:
            approve = st.button("Approve & File")
        with col2:
            reject = st.button("Reject (do not file)")

        def write_decision_log(row: Dict[str, Any]):
            file_exists = Path(DEFAULT_DECISIONS_LOG).exists()
            with open(DEFAULT_DECISIONS_LOG, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp","file","provider","model","quality_statements","evidence_categories","paths","reviewer","notes","action"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        if approve:
            if not signed_off:
                st.warning("Please tick the sign-off checkbox before approving.")
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
