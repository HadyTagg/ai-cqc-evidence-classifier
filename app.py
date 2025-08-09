import os
import io
import re
import csv
import json
import time
import glob
import uuid
import shutil
import chardet
import base64
import requests
import datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
import hashlib

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

APP_TITLE = "CQC Evidence Classifier"
DEFAULT_DECISIONS_LOG = "decisions.csv"
SUPPORTED_EXTS = {".txt", ".pdf", ".docx", ".csv", ".xlsx", ".eml"}

# Simple on-disk cache for LLM responses (keyed by content/model/taxonomy)
CACHE_DIR = Path('.llm_cache')
CACHE_DIR.mkdir(exist_ok=True)

def build_cache_key(text: str, taxonomy: Dict[str, Any], model: str) -> str:
    meta = taxonomy.get('metadata', {})
    payload = {
        'taxonomy_version': meta.get('version'),
        'model': model,
        'text_head': (text or '')[:10000],  # cap for key stability
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()

def cache_read(key: str) -> Dict[str, Any] | None:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    return None

def cache_write(key: str, data: Dict[str, Any]) -> None:
    try:
        with open(CACHE_DIR / f"{key}.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ---------------------
# Helpers – Text Extraction
# ---------------------

def read_file_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def detect_encoding(b: bytes) -> str:
    guess = chardet.detect(b)
    return guess.get("encoding") or "utf-8"


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
    # Remove script/style
    for tag in soup(["script", "style"]):
        tag.extract()
    return soup.get_text("\n", strip=True)


def extract_text_from_eml(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (email_text, attachments_meta).
    attachments_meta: list of {filename, temp_path}
    """
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
                content_disposition = part.get_content_disposition()
                content_type = part.get_content_type()
                if content_disposition == "attachment":
                    filename = part.get_filename() or f"attachment_{uuid.uuid4().hex}"
                    payload = part.get_payload(decode=True) or b""
                    tmpdir = Path(".eml_attachments")
                    tmpdir.mkdir(exist_ok=True)
                    temp_path = tmpdir / filename
                    with open(temp_path, "wb") as f:
                        f.write(payload)
                    attachments.append({"filename": filename, "temp_path": str(temp_path)})
                elif content_type == "text/plain":
                    body_parts.append(part.get_content())
                elif content_type == "text/html":
                    body_parts.append(html_to_text(part.get_content()))
        else:
            content_type = msg.get_content_type()
            if content_type == "text/plain":
                body_parts.append(msg.get_content())
            elif content_type == "text/html":
                body_parts.append(html_to_text(msg.get_content()))

        full_text = header_text + "\n\n" + "\n\n".join(body_parts).strip()
        return full_text, attachments
    except Exception as e:
        return f"[EML parse error: {e}]", []


def extract_text(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    ext = path.suffix.lower()
    if ext == ".txt":
        return extract_text_from_txt(path), []
    if ext == ".pdf":
        return extract_text_from_pdf(path), []
    if ext == ".docx":
        return extract_text_from_docx(path), []
    if ext == ".csv":
        return extract_text_from_csv(path), []
    if ext == ".xlsx":
        return extract_text_from_xlsx(path), []
    if ext == ".eml":
        return extract_text_from_eml(path)
    return f"[Unsupported file type: {ext}]", []

# ---------------------
# Helpers – Taxonomy & Paths
# ---------------------
@st.cache_data(show_spinner=False)
def load_taxonomy(taxonomy_path: str) -> Dict[str, Any]:
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_quality_statement_options(taxonomy: Dict[str, Any]) -> List[Dict[str, str]]:
    return taxonomy.get("quality_statements", [])


def list_evidence_categories(taxonomy: Dict[str, Any]) -> List[str]:
    return taxonomy.get("evidence_categories", [])


def propose_storage_paths(taxonomy: Dict[str, Any], qs_ids: List[str], categories: List[str]) -> List[str]:
    # Support templates with {domain}, {qs_id}, {qs_title}, {qs_id_and_title}, {category}
    def sanitize(s: str) -> str:
        s = (s or "").strip()
        # replace slashes with hyphen without using backslash literals
        s = s.replace('/', '-').replace(chr(92), '-')
        # remove other illegal filesystem characters
        illegal = '<>:"|?*'
        for ch in illegal:
            s = s.replace(ch, '')
        # collapse whitespace
        s = ' '.join(s.split())
        return s[:100]

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
                tmpl.replace("{domain}", sanitize(domain))
                    .replace("{qs_id}", sanitize(qid))
                    .replace("{qs_title}", sanitize(title))
                    .replace("{qs_id_and_title}", sanitize(f"{qid} – {title}"))
                    .replace("{category}", sanitize(cat))
            )
            paths.append(path)
    return sorted(set(paths))

# ---------------------
# LLM Provider Abstraction
# ---------------------
class LLMProvider:
    def __init__(self, provider: str, model: str, api_key: str = "", timeout: int = 60, max_retries: int = 3):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def classify(self, text: str, taxonomy: Dict[str, Any], max_chars: int = 6000) -> Dict[str, Any]:
        # Build prompt
        quality_statements = taxonomy.get("quality_statements", [])
        evidence_categories = taxonomy.get("evidence_categories", [])
        # Compact the QS for the prompt
        qs_brief = [
            {
                "id": q.get("id"),
                "domain": q.get("domain"),
                "title": q.get("title"),
            }
            for q in quality_statements
        ]
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
                    "evidence_categories": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
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
            raise ValueError("Unsupported provider")

    def _classify_openai(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        # Uses Chat Completions API compatible endpoint with retry/backoff on 429/5xx
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key or os.getenv('OPENAI_API_KEY','')}",
            "Content-Type": "application/json",
        }
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
                    wait = float(retry_after) if retry_after else min(10.0, (2 ** attempt) + random.random())
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
                time.sleep(min(10.0, (2 ** attempt) + random.random()))
                attempt += 1
            except requests.HTTPError as e:
                return {"error": f"HTTP error: {e}", "status": getattr(e.response, 'status_code', None), "raw": getattr(e.response, 'text', '')}
        return {"error": f"Request failed after retries: {last_err}", "status": getattr(getattr(last_err, 'response', None), 'status_code', None)}

    def _classify_ollama(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        # Local model via Ollama – simple single-shot prompt; instruct the model to output JSON
        prompt = (
            f"SYSTEM:\n{system_prompt}\n\n"
            f"USER (JSON payload follows):\n{json.dumps(user_payload)}\n\n"
            "Reply with ONLY valid JSON."
        )
        url = "http://localhost:11434/api/generate"
        data = {
            "model": self.model or "llama3:8b-instruct",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        resp = requests.post(url, json=data, timeout=120)
        resp.raise_for_status()
        content = resp.json().get("response", "{}")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Model did not return valid JSON", "raw": content}

# ---------------------
# UI Logic
# ---------------------
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
qs_labels = [f"[{q['domain']}] {q['id']} – {q['title']}" for q in qs_options]
qs_id_list = [q["id"] for q in qs_options]
cat_options = list_evidence_categories(taxonomy)

# Discover files
input_dir.mkdir(exist_ok=True, parents=True)
files = sorted([p for p in input_dir.glob("**/*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])
st.subheader("Files to review")
st.write(f"Found **{len(files)}** file(s) in `{input_dir}`.")

colA, colB = st.columns([2, 3])
with colA:
    if files:
        file_selected = st.selectbox(
            "Pick a file",
            files,
            format_func=lambda p: str(p.relative_to(input_dir)),
        )
    else:
        file_selected = None
        st.info("Drop some files into the input folder to begin (supported: .pdf .docx .xlsx .csv .txt .eml).")

with colB:
    text, attachments = ("", [])
    if file_selected:
        text, attachments = extract_text(file_selected)
        st.markdown(f"**Preview: {file_selected.name}**")
        st.code(text[:8000] if text else "[No text extracted]", language="markdown")
        if attachments:
            st.markdown("**Attachments detected:**")
            for att in attachments:
                st.write(f"- {att['filename']} → {att['temp_path']}")

# Classification controls
# ---------------------
# Batch queue (LLM propose only, no filing)
# ---------------------
st.subheader("Batch queue (propose only)")
if 'batch_queue' not in st.session_state:
    st.session_state['batch_queue'] = []
if 'batch_results' not in st.session_state:
    st.session_state['batch_results'] = {}

batch_select = st.multiselect(
    "Select files to add to queue",
    files,
    format_func=lambda p: str(p.relative_to(input_dir)),
)
if st.button("Add to queue"):
    for p in batch_select:
        if p not in st.session_state['batch_queue']:
            st.session_state['batch_queue'].append(p)

if st.session_state['batch_queue']:
    st.write("**Current queue:**")
    for p in st.session_state['batch_queue']:
        st.write(f"- {p.relative_to(input_dir)}")
    run_batch = st.button("Run queue (propose only)")
    if run_batch:
        progress = st.progress(0)
        total = len(st.session_state['batch_queue'])
        for i, fpath in enumerate(list(st.session_state['batch_queue'])):
            try:
                text_i, _ = extract_text(fpath)
                key_i = build_cache_key(text_i or "", taxonomy, model)
                cached_i = cache_read(key_i) if use_cache else None
                if cached_i:
                    res = cached_i
                else:
                    res = prov.classify(text_i or "", taxonomy, max_chars=int(max_chars))
                    if use_cache and res and not res.get('error'):
                        cache_write(key_i, res)
                st.session_state['batch_results'][str(fpath)] = res
            except Exception as e:
                st.session_state['batch_results'][str(fpath)] = {"error": str(e)}
            progress.progress(int(((i + 1) / total) * 100))
            if i < total - 1 and cooldown_secs:
                time.sleep(float(cooldown_secs))
        st.success("Batch complete. Open each file above to review and approve (nothing has been filed).")
        # Optional summary table
        summary_rows = []
        for f, res in st.session_state['batch_results'].items():
            if res and not res.get('error'):
                qs_ids = ",".join([q.get('id','') for q in res.get('quality_statements', [])])
                cats = ",".join(res.get('evidence_categories', []))
                summary_rows.append({"file": str(Path(f).name), "qs": qs_ids, "categories": cats})
            else:
                err = res.get('error','') if isinstance(res, dict) else '(error)'
                summary_rows.append({"file": str(Path(f).name), "qs": "(error)", "categories": err})
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows))

st.subheader("Proposed classification (by LLM)")
prov = LLMProvider(provider=provider, model=model, api_key=api_key, timeout=int(request_timeout), max_retries=int(max_retries))

# Cooldown + cache aware run
if 'next_ok_at' not in st.session_state:
    st.session_state['next_ok_at'] = 0.0

if st.button("Run LLM on this file"):
    if not file_selected:
        st.warning("Pick a file first.")
    else:
        # Try cache first
        key = build_cache_key(text or "", taxonomy, model)
        cached = cache_read(key) if use_cache else None
        if cached:
            st.info("Loaded cached classification for this content.")
            st.session_state["llm_result"] = cached
        else:
            now = time.time()
            if now < st.session_state['next_ok_at']:
                wait = int(st.session_state['next_ok_at'] - now)
                st.warning(f"Cooling down. Try again in {wait}s to avoid rate limits.")
            else:
                st.session_state['next_ok_at'] = now + float(cooldown_secs)
                with st.spinner("Asking the model…"):
                    result = prov.classify(text or "", taxonomy, max_chars=int(max_chars))
                st.session_state["llm_result"] = result
                if use_cache and result and not result.get('error'):
                    cache_write(key, result)

result = st.session_state.get("llm_result")

if result:
    if result.get("error"):
        st.error(f"LLM error: {result['error']}")
        with st.expander("Raw output"):
            st.code(result.get("raw", ""))
    else:
        # Render suggested QS
        sugg_qs = result.get("quality_statements", [])
        st.markdown("**Model’s suggested Quality Statements:**")
        for q in sugg_qs:
            qid = q.get("id")
            domain = q.get("domain") or qs_map.get(qid, {}).get("domain", "?")
            title = q.get("title") or qs_map.get(qid, {}).get("title", "")
            st.write(
                f"- **{qid}** ({domain}) – {title}"
                f"  | confidence: {q.get('confidence','?')}\n\n  rationale: {q.get('rationale','')}"
            )
        # Editable selection
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
                        "timestamp",
                        "file",
                        "provider",
                        "model",
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
                writer.writerow(row)

        col1, col2 = st.columns([1, 1])
        with col1:
            approve = st.button("Approve & File")
        with col2:
            reject = st.button("Reject (do not file)")

        if approve:
            if not signed_off:
                st.warning("Please tick the sign‑off checkbox before approving.")
            else:
                # File into all proposed paths
                first_dest_file = None
                original_path = file_selected
                for idx, rel in enumerate(paths):
                    dest_dir = output_dir / rel
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = dest_dir / file_selected.name
                    if move_or_copy == "Copy":
                        shutil.copy2(original_path, dest_file)
                    else:  # Move
                        if idx == 0:
                            shutil.move(str(original_path), str(dest_file))
                            first_dest_file = dest_file
                        else:
                            # subsequent paths: copy from the first destination
                            shutil.copy2(first_dest_file, dest_file)
                write_decision_log({
                    "timestamp": dt.datetime.utcnow().isoformat(),
                    "file": str(file_selected),
                    "provider": provider,
                    "model": model,
                    "quality_statements": ",".join(selected_qs),
                    "evidence_categories": ",".join(selected_cats),
                    "paths": ";".join(paths),
                    "reviewer": reviewer,
                    "notes": notes,
                    "action": f"Filed ({move_or_copy})",
                })
                st.success("Filed successfully and logged to decisions.csv.")
        if reject:
            write_decision_log({
                "timestamp": dt.datetime.utcnow().isoformat(),
                "file": str(file_selected),
                "provider": provider,
                "model": model,
                "quality_statements": ",".join(selected_qs),
                "evidence_categories": ",".join(selected_cats),
                "paths": ";".join(paths),
                "reviewer": reviewer,
                "notes": notes,
                "action": "Rejected (no filing)",
            })
            st.info("Decision recorded as Rejected. No files were moved/copied.")

st.divider()
st.caption(
    "If using a cloud LLM, input text is sent to the provider’s API. For sensitive content, use the Ollama local option.\n"
    "Keep `decisions.csv` as your audit trail showing human sign‑off for each item."
)

# --------------
# Allow `python app.py` to behave like `streamlit run app.py`
# --------------
if __name__ == "__main__":
    # If we are NOT already inside a Streamlit runtime, re-invoke via the CLI.
    try:
        from streamlit.web import cli as stcli
        import sys
        # Prevent infinite recursion if already invoked by streamlit
        if not any(x.endswith("streamlit") for x in sys.argv[:1]):
            sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
            raise SystemExit(stcli.main())
    except Exception:
        # Fallback: print helpful message
        print("\nRun this app with:  streamlit run app.py\n")