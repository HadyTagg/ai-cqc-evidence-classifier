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

import warnings

import yaml
import pandas as pd
import streamlit as st

# Silence noisy openpyxl "Data Validation extension is not supported" warning (from openpyxl)
warnings.filterwarnings(
    "ignore",
    r"Data Validation extension is not supported",
    UserWarning,
)

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
    pass
try:
    import extract_msg  # for .msg (Outlook) files
except Exception:
    extract_msg = None

APP_TITLE = "CQC Evidence Classifier"
DEFAULT_DECISIONS_LOG = "decisions.csv"
SUPPORTED_EXTS = {
    ".txt", ".pdf", ".docx", ".csv", ".xlsx", ".xlsm", ".xls", ".eml", ".msg",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic"
}


def write_decision_log(row: Dict[str, Any]) -> None:
    file_exists = Path(DEFAULT_DECISIONS_LOG).exists()
    with open(DEFAULT_DECISIONS_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "file",
                "quality_statements",
                "evidence_categories",
                "paths",
                "reviewer",
                "notes",
                "action",
            ],
        )
        if not file_exists:
            writer.writeheader()
        if "timestamp" in row:
            row["timestamp"] = pd.to_datetime(row["timestamp"]).strftime("%d/%m/%y")
        file_val = row.get("file")
        row["file"] = Path(file_val).name if file_val else ""
        writer.writerow(row)

# ---------------------
# Simple extractors (only used as fallbacks)
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

# -------- Excel helpers & conversions ----------
def _read_excel_all_sheets(path: Path, engine: str | None = None) -> dict:
    return pd.read_excel(path, sheet_name=None, engine=engine)

def _xls_to_xlsx_with_libreoffice(xls_path: Path) -> Path | None:
    outdir = Path(".xls_convert"); outdir.mkdir(exist_ok=True)
    xlsx_out = outdir / (xls_path.stem + ".xlsx")
    try:
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "xlsx", "--outdir", str(outdir), str(xls_path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return xlsx_out if xlsx_out.exists() else None
    except Exception:
        return None

def extract_text_from_excel(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext in {".xlsx", ".xlsm"}:
            dfs = _read_excel_all_sheets(path)
        elif ext == ".xls":
            try:
                dfs = _read_excel_all_sheets(path, engine="xlrd")
            except Exception:
                conv = _xls_to_xlsx_with_libreoffice(path)
                if conv and conv.exists():
                    dfs = _read_excel_all_sheets(conv)
                else:
                    raise
        else:
            return f"[Excel extraction error: unsupported extension {ext}]"
        parts = []
        for name, df in dfs.items():
            parts.append(f"\n--- Sheet: {name} ---\n")
            parts.append(df.to_csv(index=False))
        return "\n".join(parts)
    except Exception as e:
        return f"[Excel read error: {e}]"

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

def extract_text_from_msg(path: Path) -> str:
    """Return text content from an Outlook .msg file (headers + body). Attachments ignored."""
    try:
        if extract_msg is None:
            return "[MSG parse error: 'extract-msg' is not installed. Run: pip install extract-msg]"
        m = extract_msg.Message(str(path))
        headers = []
        for key, label in (
            ("sender", "From"), ("to", "To"), ("cc", "Cc"), ("date", "Date"), ("subject", "Subject")
        ):
            val = getattr(m, key, None)
            if val:
                headers.append(f"{label}: {val}")
        header_text = "\n".join(headers)
        body_text = getattr(m, "body", None) or ""
        if not body_text:
            html_body = getattr(m, "htmlBody", None)
            if html_body:
                body_text = html_to_text(html_body)
        return (header_text + "\n\n" + (body_text or "")).strip()
    except Exception as e:
        return f"[MSG parse error: {e}]"

# ---------------------
# Font handling for rasterized text
# ---------------------
def _find_default_ttf() -> str | None:
    """Try to locate a reasonable TrueType font on Linux (Crostini)."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return str(p)
    # user fonts
    home = Path.home()
    for p in home.glob(".local/share/fonts/**/*.ttf"):
        return str(p)
    return None

def _find_monospace_ttf() -> str | None:
    """Prefer a monospaced font for emails to keep alignment (e.g., DejaVu Sans Mono)."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    # fallback to any default
    return _find_default_ttf()

def _load_ttf(font_path: str | None, size: int = 14):
    if ImageFont is None:
        return None
    try:
        if font_path and Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    except Exception:
        pass
    found = _find_default_ttf()
    if found:
        try:
            return ImageFont.truetype(found, size=size)
        except Exception:
            pass
    return ImageFont.load_default()

def _wrap_text_to_width(text: str, draw, font, max_width: int, padding: int) -> list[str]:
    """Wrap text so that each line fits within max_width using the provided font."""
    lines_out: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines_out.append("")
            continue
        words = paragraph.split(" ")
        cur = ""
        for w in words:
            tentative = w if not cur else (cur + " " + w)
            bbox = draw.textbbox((0, 0), tentative, font=font)
            if bbox[2] - bbox[0] + padding*2 <= max_width:
                cur = tentative
            else:
                if cur:
                    lines_out.append(cur)
                cur = w
        if cur:
            lines_out.append(cur)
    return lines_out

def _normalise_whitespace(text: str) -> str:
    # Tabs and non-breaking spaces often break alignment when rendering
    return (text or "").replace("\r", "").replace("\t", "    ").replace("\u00a0", " ")

def text_to_image(
    text: str,
    width: int = 1400,
    padding: int = 24,
    font_path: str | None = None,
    font_size: int = 15,
    supersample: int = 2,
    line_spacing: int = 4,
) -> Image.Image:
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow not available for text rasterization")
    text = _normalise_whitespace(text)
    # Supersample at higher resolution then downscale with LANCZOS to reduce "smeared" appearance
    s = max(1, int(supersample))
    width_ss = int(width) * s
    padding_ss = int(padding) * s
    font = _load_ttf(font_path, size=int(font_size) * s)
    tmp = Image.new("RGB", (width_ss, 10), "white")
    draw = ImageDraw.Draw(tmp)
    lines = _wrap_text_to_width(text, draw, font, width_ss, padding_ss)
    if hasattr(font, "getmetrics"):
        ascent, descent = font.getmetrics()
    else:
        ascent, descent = (14 * s, 4 * s)
    line_h = ascent + descent + int(line_spacing) * s
    height_ss = max(200 * s, padding_ss * 2 + line_h * (len(lines) + 1))
    img = Image.new("RGB", (width_ss, height_ss), "white")
    draw = ImageDraw.Draw(img)
    y = padding_ss
    for ln in lines:
        draw.text((padding_ss, y), ln, fill="black", font=font)
        y += line_h
        if y > height_ss - padding_ss:
            break
    if s > 1:
        # Pillow 9/10 compatibility for resampling enum
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        img = img.resize((int(width), int(height_ss / s)), resample=resample)
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

# ---------------------
# Raster cache for image previews
# ---------------------

RASTER_CACHE_DIR = Path(".raster_cache")
RASTER_CACHE_DIR.mkdir(exist_ok=True)

def _raster_params_dict(
    dpi: int,
    max_pages: int,
    font_path: str | None,
    font_size: int,
    text_image_width: int,
    email_font_path: str | None,
    email_font_size: int,
    email_supersample: int,
) -> Dict[str, Any]:
    return {
        "dpi": int(dpi),
        "max_pages": int(max_pages),
        "font_path": str(font_path or ""),
        "font_size": int(font_size),
        "text_image_width": int(text_image_width),
        "email_font_path": str(email_font_path or ""),
        "email_font_size": int(email_font_size),
        "email_supersample": int(email_supersample),
    }

def build_raster_cache_key(path: Path, params: Dict[str, Any]) -> str:
    try:
        stat = path.stat()
        payload = {
            "src": str(path.resolve()),
            "mtime": int(stat.st_mtime),
            "size": int(stat.st_size),
            **params,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.sha256(str(path).encode("utf-8")).hexdigest()

def _raster_cache_dir_for_key(key: str) -> Path:
    return RASTER_CACHE_DIR / key

def raster_cache_read(key: str) -> List[Image.Image] | None:
    d = _raster_cache_dir_for_key(key)
    meta = d / "meta.json"
    if not d.exists() or not meta.exists():
        return None
    try:
        _ = json.loads(meta.read_text(encoding="utf-8"))  # Reserved for future use
        images: List[Image.Image] = []
        i = 0
        while True:
            p = d / f"page_{i:03d}.png"
            if not p.exists():
                break
            if Image is None:
                return None
            images.append(Image.open(p))
            i += 1
        return images if images else None
    except Exception:
        return None

def raster_cache_write(key: str, params: Dict[str, Any], images: List[Image.Image]) -> None:
    try:
        d = _raster_cache_dir_for_key(key)
        if d.exists():
            for f in d.glob("*"):
                try:
                    f.unlink()
                except Exception:
                    pass
        d.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(images):
            out = d / f"page_{i:03d}.png"
            img.save(out, format="PNG")
        (d / "meta.json").write_text(json.dumps({"params": params, "count": len(images)}, indent=2), encoding="utf-8")
    except Exception:
        pass

def clear_raster_cache() -> int:
    """Delete all files in the raster cache. Returns number of files removed."""
    removed = 0
    try:
        if RASTER_CACHE_DIR.exists():
            for p in RASTER_CACHE_DIR.rglob("*"):
                try:
                    if p.is_file():
                        p.unlink()
                        removed += 1
                except Exception:
                    pass
    except Exception:
        pass
    return removed

# ---------------------
# Spreadsheet -> PDF (for visual classification)
# ---------------------
def _spreadsheet_to_pdf_with_libreoffice(path: Path, outdir: Path) -> Path | None:
    """Convert a spreadsheet (xlsx/xlsm/xls/csv) to a PDF using LibreOffice. Returns path or None."""
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_out = outdir / (path.stem + ".pdf")
    try:
        subprocess.run(
            ["soffice", "--headless", "--convert-to", "pdf:calc_pdf_Export", "--outdir", str(outdir), str(path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return pdf_out if pdf_out.exists() else None
    except Exception:
        return None

# ---------------------
# Rasterization helpers (image preview)
# ---------------------
def rasterize_to_images(
    path: Path,
    dpi: int = 200,
    max_pages: int = 2,
    font_path: str | None = None,
    font_size: int = 14,
    text_image_width: int = 1400,
    # Email-specific tuning
    email_font_path: str | None = None,
    email_font_size: int = 15,
    email_supersample: int = 2,
    # Cache controls
    use_cache: bool = True,
) -> List[Image.Image]:
    ext = path.suffix.lower()
    params = _raster_params_dict(
        dpi, max_pages, font_path, font_size, text_image_width, email_font_path, email_font_size, email_supersample
    )
    cache_key = build_raster_cache_key(path, params)

    # Raw images (no caching of original files)
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".heic"}:
        if Image is None:
            raise RuntimeError("Pillow not available to load images")
        return [Image.open(path)]

    if use_cache:
        cached = raster_cache_read(cache_key)
        if cached:
            return cached

    images: List[Image.Image] = []

    # PDFs
    if ext == ".pdf" and convert_from_path is not None:
        images = convert_from_path(str(path), dpi=int(dpi))
        if max_pages:
            images = images[:max_pages]
        if use_cache:
            raster_cache_write(cache_key, params, images)
        return images

    # DOCX -> try LibreOffice to PDF, then pdf2image; else fallback to text rasterization
    if ext == ".docx":
        outdir = Path(".docx_pdf"); outdir.mkdir(exist_ok=True)
        pdf_out = outdir / (path.stem + ".pdf")
        try:
            subprocess.run(
                ["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(outdir), str(path)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if pdf_out.exists() and convert_from_path is not None:
                images = convert_from_path(str(pdf_out), dpi=int(dpi))
                if max_pages:
                    images = images[:max_pages]
                if use_cache:
                    raster_cache_write(cache_key, params, images)
                return images
        except Exception:
            pass
        txt = extract_text_from_docx(path)
        images = [text_to_image(txt, width=int(text_image_width), font_path=font_path, font_size=int(font_size))]
        if use_cache:
            raster_cache_write(cache_key, params, images)
        return images

    # EML/MSG/TXT -> rasterize extracted text
    if ext == ".eml":
        txt, _ = extract_text_from_eml(path)
        images = [text_to_image(
            txt,
            width=int(text_image_width),
            font_path=email_font_path or font_path,
            font_size=int(email_font_size),
            supersample=int(email_supersample),
        )]
        if use_cache:
            raster_cache_write(cache_key, params, images)
        return images

    if ext == ".msg":
        images = [text_to_image(
            extract_text_from_msg(path),
            width=int(text_image_width),
            font_path=email_font_path or font_path,
            font_size=int(email_font_size),
            supersample=int(email_supersample),
        )]
        if use_cache:
            raster_cache_write(cache_key, params, images)
        return images

    if ext == ".txt":
        images = [text_to_image(
            extract_text_from_txt(path),
            width=int(text_image_width),
            font_path=font_path,
            font_size=int(font_size),
        )]
        if use_cache:
            raster_cache_write(cache_key, params, images)
        return images

    # SPREADSHEETS (XLSX/XLSM/XLS/CSV) -> PDF via LibreOffice, then to images
    if ext in {".xlsx", ".xlsm", ".xls", ".csv"}:
        pdf = _spreadsheet_to_pdf_with_libreoffice(path, Path(".sheet_pdf"))
        if pdf and convert_from_path is not None:
            images = convert_from_path(str(pdf), dpi=int(dpi))
            if max_pages:
                images = images[:max_pages]
            if use_cache:
                raster_cache_write(cache_key, params, images)
            return images
        # Fallback to text-based rendering if PDF conversion or pdf2image is unavailable
        if ext == ".csv":
            txt = extract_text_from_csv(path)
        else:
            txt = extract_text_from_excel(path)
        images = [text_to_image(txt, width=int(text_image_width), font_path=font_path, font_size=int(font_size))]
        if use_cache:
            raster_cache_write(cache_key, params, images)
        return images

    # Unknown -> best effort
    images = [text_to_image(f"[Unsupported for rasterization: {ext}] {path}", width=int(text_image_width), font_path=font_path, font_size=int(font_size))]
    if use_cache:
        raster_cache_write(cache_key, params, images)
    return images

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

# ---------------------
# Simple JSON file cache for LLM results
# ---------------------
LLM_CACHE_DIR = Path(".llm_cache")
LLM_CACHE_DIR.mkdir(exist_ok=True)

def _taxonomy_fingerprint(taxonomy: Dict[str, Any]) -> str:
    try:
        return hashlib.sha256(json.dumps(taxonomy, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "unknown"

def build_llm_cache_key(source_id: str, taxonomy: Dict[str, Any], model: str, mode: str, text: str | None = None) -> str:
    payload = {
        "source": source_id,
        "taxonomy_fp": _taxonomy_fingerprint(taxonomy),
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
# Model normalization for Chat Completions
# ---------------------
def _normalize_model_for_chat(m: str) -> str:
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
        self.llm_image_max_width = 1024
        self.auto_fallback = True
        self.fallback_model = "gpt-5-mini"

    def classify_images(self, images: List[Image.Image], taxonomy: Dict[str, Any], max_images: int = 4) -> Dict[str, Any]:
        qs_brief = build_qs_brief(taxonomy)
        cats_brief = build_evidence_category_brief(taxonomy)
        system_prompt = (
            """You are a compliance assistant for a CQC-regulated care service.
You will be given image(s) of an evidence item (scan/photo). Your task is to map it to:
1. One or more CQC Quality Statements
2. The main Evidence Category

GROUNDING MATERIAL is provided for each Quality Statement and contains:
- we_statement (verbatim)
- what_this_quality_statement_means (verbatim)
- i_statements
- subtopics
- source_url

Selection Rules:
- Primary filter: Only select a Quality Statement if its we_statement clearly and directly matches the evidence content.
- Precision over breadth: Do not select "close enough" or loosely related statements.
- If no clear match exists, return no Quality Statement for that evidence.

Additional Matching:
- For each selected Quality Statement, also check what_this_quality_statement_means, i_statements, and subtopics for exact or near-verbatim matches visible in the evidence.

Justification Requirements:
For every selected Quality Statement, provide:
- A short rationale explaining the match, referencing visible evidence content.
- The exact matching text for we_statement.
- The exact matching text for what_this_quality_statement_means.
- Any matching i_statements or subtopics (verbatim).

Output:
Return only a JSON object that follows the provided schema. Do not include any extra text outside the JSON."""
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
                                "matched_we_statement": {"type": "string"},
                                "matched_what_it_means": {"type": "string"},
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
            "evidence_categories_options": cats_brief,
        }
        content = [{"type": "text", "text": json.dumps(schema_and_options)}]
        max_w = getattr(self, "llm_image_max_width", 1024)
        for img in images[:max_images]:
            ds = downscale_for_llm(img, max_w=max_w)
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_url_jpeg(ds)}})
        return self._chat_json(system_prompt, None, content_override=content)

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
                    if resp.status_code == 429 or 500 <= resp.status_code < 600:
                        ra = resp.headers.get("Retry-After")
                        wait = float(ra) if ra else min(20.0, 2 ** attempt + random.random())
                        last_retry_after = wait
                        time.sleep(wait)
                        attempt += 1
                        last_err = requests.HTTPError(f"HTTP {resp.status_code}")  # type: ignore
                        continue

                    if 400 <= resp.status_code < 500:
                        try:
                            last_err_body = resp.json()
                        except Exception:
                            last_err_body = resp.text
                        last_err = requests.HTTPError(f"HTTP {resp.status_code}")  # type: ignore
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
            "what_this_quality_statement_means": means_block,
            "i_statements": q.get("i_statements", []),
            "subtopics": q.get("subtopics", []),
            "source_url": q.get("source_url", ""),
        })
    return out

def build_evidence_category_brief(taxonomy: Dict[str, Any]) -> List[Dict[str, str]]:
    cats = taxonomy.get("evidence_categories", [])
    desc = taxonomy.get("evidence_category_descriptions", {})
    return [{"name": c, "description": desc.get(c, "")} for c in cats]

# ---------------------
# UI
# ---------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("AI Powered CQC Evidence Classifier")

with st.sidebar:
    st.header("Settings")
    input_dir = Path(st.text_input("Input folder (files to classify)", value=str(Path.cwd() / "input")))
    output_dir = Path(st.text_input("Output base folder (root for filing)", value=str(Path.cwd() / "classified")))

    taxonomy_path = st.text_input("Taxonomy file (YAML)", value=str(Path.cwd() / "cqc_taxonomy.yaml"))
    if st.button("Reload taxonomy"):
        st.cache_data.clear()

    provider = st.selectbox("LLM provider", ["openai"], index=0)
    model = st.text_input("Model", value="gpt-5-chat-latest")
    api_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")

    request_timeout = st.number_input("API timeout (sec)", min_value=10, max_value=120, value=60, step=5)
    max_retries = st.number_input("Max retries on 429/5xx", min_value=0, max_value=10, value=3, step=1)
    use_cache = st.checkbox("Use cached results (by content)", value=True)
    cooldown_secs = st.number_input("Cooldown between calls (sec)", min_value=0, max_value=120, value=10, step=5)

    st.markdown("**Preview settings**")
    preview_dpi = st.number_input("Preview DPI (PDF/image sources)", min_value=100, max_value=600, value=150, step=50)
    preview_max_pages = st.number_input("Max pages/images to preview/classify", min_value=1, max_value=10, value=2, step=1)

    st.markdown("**Text preview font (fallback modes)**")
    font_path = st.text_input("TTF font path (optional)", value=_find_default_ttf() or "")
    font_size = st.number_input("Font size", min_value=10, max_value=28, value=14, step=1)
    text_image_width = st.number_input("Text image width (px)", min_value=800, max_value=2400, value=1400, step=50)

    st.markdown("**Email (.eml/.msg) preview fixes**")
    email_use_mono = st.checkbox("Use monospaced font for emails", value=True)
    email_font_path = st.text_input("Email TTF path (optional)", value=_find_monospace_ttf() or (font_path or ""))
    email_font_size = st.number_input("Email font size", min_value=10, max_value=28, value=15, step=1)
    email_supersample = st.number_input("Email supersample factor (anti‑smear)", min_value=1, max_value=4, value=2, step=1)

    st.markdown("**LLM image controls**")
    llm_image_max_width = st.number_input("LLM image max width (px)", min_value=512, max_value=2048, value=1024, step=128)
    auto_fallback = st.checkbox("Auto-fallback on 429/5xx", value=True)
    fallback_model = st.text_input("Fallback model", value="gpt-5-mini")

    st.markdown("**Classification filter**")
    we_conf_threshold = st.number_input(
        "Minimum 'we' statement confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
    )

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
prov.llm_image_max_width = int(llm_image_max_width)
prov.auto_fallback = bool(auto_fallback)
prov.fallback_model = (fallback_model or "").strip() or None

if files and st.button("Auto-classify all files (no manual review)"):
    progress = st.progress(0)
    for idx, f in enumerate(files):
        try:
            imgs = rasterize_to_images(
                f,
                dpi=int(preview_dpi),
                max_pages=int(preview_max_pages),
                font_path=font_path or None,
                font_size=int(font_size),
                text_image_width=int(text_image_width),
                email_font_path=(email_font_path if email_use_mono else (font_path or None)),
                email_font_size=int(email_font_size),
                email_supersample=int(email_supersample),
            )
            key = build_llm_cache_key("[image-mode]" + str(f), taxonomy, model, "Vision-only")
            cached = llm_cache_read(key) if use_cache else None
            if cached:
                result = cached
            else:
                result = prov.classify_images(imgs, taxonomy)
                if use_cache and result and not result.get("error"):
                    llm_cache_write(key, result)
            if result.get("error"):
                write_decision_log({
                    "timestamp": pd.Timestamp.utcnow(),
                    "file": str(f),
                    "quality_statements": json.dumps([]),
                    "evidence_categories": json.dumps([]),
                    "paths": json.dumps([]),
                    "reviewer": "automatic",
                    "notes": result.get("error"),
                    "action": "error",
                })
            else:
                filtered_qs = [
                    q
                    for q in result.get("quality_statements", [])
                    if q.get("confidence", 0) >= we_conf_threshold and q.get("matched_we_statement")
                ]
                selected_qs = [q.get("id") for q in filtered_qs if q.get("id") in qs_map]
                selected_q_titles = [qs_map[qid]["title"] for qid in selected_qs if qid in qs_map]
                selected_cats = [c for c in result.get("evidence_categories", []) if c in cat_options]
                paths = propose_storage_paths(taxonomy, selected_qs, selected_cats)
                original_path = f; original_name = f.name
                first_dest_file = None
                for j, rel in enumerate(paths):
                    dest_dir = output_dir / rel; dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = dest_dir / original_name
                    if j == 0:
                        if move_or_copy == "Move":
                            shutil.move(str(original_path), str(dest_file)); first_dest_file = dest_file
                        else:
                            shutil.copy2(str(original_path), str(dest_file)); first_dest_file = dest_file
                    else:
                        src = first_dest_file if move_or_copy == "Move" else original_path
                        if src and Path(src).exists():
                            shutil.copy2(str(src), str(dest_file))
                notes_val = result.get("notes")
                if notes_val:
                    notes_val = f"{notes_val} (not verified)"
                else:
                    notes_val = "not verified"
                write_decision_log({
                    "timestamp": pd.Timestamp.utcnow(),
                    "file": str(original_path),
                    "quality_statements": json.dumps(selected_q_titles),
                    "evidence_categories": json.dumps(selected_cats),
                    "paths": json.dumps(paths),
                    "reviewer": "automatic",
                    "notes": notes_val,
                    "action": "auto-filed (unverified)",
                })
        except Exception as e:
            write_decision_log({
                "timestamp": pd.Timestamp.utcnow(),
                "file": str(f),
                "quality_statements": json.dumps([]),
                "evidence_categories": json.dumps([]),
                "paths": json.dumps([]),
                "reviewer": "automatic",
                "notes": f"error: {e}",
                "action": "error",
            })
        progress.progress((idx + 1) / len(files))
    st.success("Auto classification completed.")

colA, colB = st.columns([2, 3])
with colA:
    file_selected = st.selectbox("Pick a file", files, format_func=lambda p: str(p.relative_to(input_dir))) if files else None
    if not files:
        st.info("Drop some files into the input folder to begin (supported: " + " ".join(sorted(SUPPORTED_EXTS)) + ").")

with colB:
    images = []
    if file_selected:
        try:
            images = rasterize_to_images(
                file_selected,
                dpi=int(preview_dpi),
                max_pages=int(preview_max_pages),
                font_path=font_path or None,
                font_size=int(font_size),
                text_image_width=int(text_image_width),
                email_font_path=(email_font_path if email_use_mono else (font_path or None)),
                email_font_size=int(email_font_size),
                email_supersample=int(email_supersample),
            )
            st.markdown(f"**Preview: {file_selected.name}**")
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
                    imgs = images or rasterize_to_images(
                        file_selected,
                        dpi=int(preview_dpi),
                        max_pages=int(preview_max_pages),
                        font_path=font_path or None,
                        font_size=int(font_size),
                        text_image_width=int(text_image_width),
                        email_font_path=(email_font_path if email_use_mono else (font_path or None)),
                        email_font_size=int(email_font_size),
                        email_supersample=int(email_supersample),
                    )
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
        all_qs = result.get("quality_statements", [])
        sugg_qs = [
            q
            for q in all_qs
            if q.get("confidence", 0) >= we_conf_threshold and q.get("matched_we_statement")
        ]
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
                tab_labels = [
                    "We statement",
                    "What it means",
                    "I statements",
                    "Subtopics",
                    "Source",
                    "Matched (if any)",
                ]
                tabs = st.tabs(tab_labels)

                with tabs[0]:
                    st.write(qs.get("we_statement", "_(none)_") or "_(none)_")
                with tabs[1]:
                    st.write(
                        qs.get(
                            "what_this_quality_statement_means",
                            qs.get("what this quality statement means", "_(none)_"),
                        )
                        or "_(none)_"
                    )
                with tabs[2]:
                    i_list = qs.get("i_statements") or []
                    if i_list:
                        for s in i_list:
                            st.write(f"- {s}")
                    else:
                        st.write("_(none)_")
                with tabs[3]:
                    subs = qs.get("subtopics") or []
                    if subs:
                        for s in subs:
                            st.write(f"- {s}")
                    else:
                        st.write("_(none)_")
                with tabs[4]:
                    src = qs.get("source_url")
                    if src:
                        st.markdown(f"[Open the official CQC page]({src})")
                    else:
                        st.write("_(none)_")
                with tabs[5]:
                    mi = q.get("matched_i_statements") or []
                    ms = q.get("matched_subtopics") or []
                    mw = q.get("matched_we_statement")
                    mm = q.get("matched_what_it_means") or q.get(
                        "matched_what_this_quality_statement_means"
                    )
                    if mi:
                        st.write("**Matched I statements:**")
                        for s in mi:
                            st.write(f"- {s}")
                    if ms:
                        st.write("**Matched subtopics:**")
                        for s in ms:
                            st.write(f"- {s}")
                    if mw:
                        st.write("**Matched we statement:**")
                        if isinstance(mw, str):
                            st.write(mw)
                        else:
                            st.write(qs.get("we_statement", ""))
                    if mm:
                        st.write("**Matched 'what it means':**")
                        if isinstance(mm, str):
                            st.write(mm)
                        else:
                            st.write(
                                qs.get(
                                    "what_this_quality_statement_means",
                                    qs.get("what this quality statement means", ""),
                                )
                            )
                    if not mi and not ms and not mw and not mm:
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

        selected_titles = [qs_map[qid]["title"] for qid in selected_qs if qid in qs_map]

        notes = st.text_area("Reviewer notes (optional)", value=result.get("notes", ""))
        signed_off = st.checkbox("I confirm the above classification and approve filing.")
        reviewer = st.text_input("Your name for the audit log", value=os.getenv("USER", "Reviewer"))
        col1, col2 = st.columns([1,1])
        with col1:
            approve = st.button("Approve & File")
        with col2:
            reject = st.button("Reject (do not file)")

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
                    "timestamp": pd.Timestamp.utcnow(),
                    "file": str(original_path),
                    "quality_statements": json.dumps(selected_titles),
                    "evidence_categories": json.dumps(selected_cats),
                    "paths": json.dumps(paths),
                    "reviewer": reviewer,
                    "notes": notes,
                    "action": "approved",
                })
                st.success("Filed to all selected locations.")
        if reject:
            write_decision_log({
                "timestamp": pd.Timestamp.utcnow(),
                "file": str(file_selected) if file_selected else "",
                "quality_statements": json.dumps(selected_titles if result else []),
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