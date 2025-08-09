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
    qs_map = {q["id"]: q for q in taxonomy.get("quality_statements", [])}
    templates = taxonomy.get("path_templates", {})
    paths = []
    for qid in qs_ids:
        q = qs_map.get(qid)
        if not q:
            continue
        domain = q.get("domain", "Misc")
        tmpl = templates.get(domain, "{domain}/{qs_id}/{category}")
        for cat in categories:
            path = tmpl.replace("{domain}", domain).replace("{qs_id}", qid).replace("{category}", cat)
            paths.append(path)
    return sorted(set(paths))

# ---------------------
# LLM Provider Abstraction
# ---------------------
class LLMProvider:
    def __init__(self, provider: str, model: str, api_key: str = ""):
        self.provider = provider
        self.model = model
        self.api_key = api_key

    def classify(self, text: str, taxonomy: Dict[str, Any]) -> Dict[str, Any]:
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
            "evidence_text_first_6000_chars": text[:6000],
        }
        if self.provider == "openai":
            return self._classify_openai(system_prompt, user_prompt)
        elif self.provider == "ollama":
            return self._classify_ollama(system_prompt, user_prompt)
        else:
            raise ValueError("Unsupported provider")

    def _classify_openai(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        # Uses Chat Completions API compatible endpoint
        # Assumes OPENAI_API_KEY is set or provided
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
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"error": "Model did not return valid JSON", "raw": content}

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
    file_selected = st.selectbox("Pick a file", files, format_func=lambda p: p.relative_to(input_dir))

with colB:
    if file_selected:
        text, attachments = extract_text(file_selected)
        st.markdown(f"**Preview: {file_selected.name}**")
        st.code(text[:8000] if text else "[No text extracted]", language="markdown")
        if attachments:
            st.markdown("**Attachments detected:**")
            for att in attachments:
                st.write(f"- {att['filename']} → {att['temp_path']}")

# Classification controls
st.subheader("Proposed classification (by LLM)")
prov = LLMProvider(provider=provider, model=model, api_key=api_key)

if st.button("Run LLM on this file"):
    with st.spinner("Asking the model…"):
        result = prov.classify(text or "", taxonomy)
    st.session_state["llm_result"] = result

result = st.session_state.get("llm_result")

if result:
    if result.get("error"):
        st.error(f"LLM error: {result['error']}")
        with st.expander("Raw output"):
            st.code(result.get("raw", ""))
    else:
        # Render suggested QS
        sugg_qs = result.get("quality_statements", [])
        sugg_ids = [q.get("id") for q in sugg_qs if q.get("id")]
        st.markdown("**Model’s suggested Quality Statements:**")
        for q in sugg_qs:
            qid = q.get("id")
            meta = qs_map.get(qid, {})
            st.write(
                f"- **{qid}** ({meta.get('domain', q.get('domain','?'))}) – {meta.get('title', q.get('title',''))}"
                f"  | confidence: {q.get('confidence','?')}\n\n  rationale: {q.get('rationale','')}"
            )
        # Editable selection
        selected_qs = st.multiselect(
            "Confirm Quality Statements",
            options=qs_id_list,
            default=[qid for qid in sugg_qs if qid in qs_map],
            format_func=lambda qid: f"[{qs_map[qid]['domain']}] {qid} – {qs_map[qid]['title']}" if qid in qs_map else qid,
        )

        sugg_cats = result.get("evidence_categories", [])
        selected_cats = st.multiselect(
            "Confirm Evidence Categories (multi-select)",
            options=cat_options,
            default=[c for c in sugg_cats if c in cat_options] or cat_options[:1],
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
                for rel in paths:
                    dest = output_dir / rel
                    dest.mkdir(parents=True, exist_ok=True)
                    if move_or_copy == "Copy":
                        shutil.copy2(file_selected, dest / file_selected.name)
                    else:
                        # move once to the first path, copy to the rest
                        if rel == paths[0]:
                            shutil.move(str(file_selected), str(dest / file_selected.name))
                        else:
                            shutil.copy2(dest / file_selected.name, dest / file_selected.name)
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
    "Security note: If using a cloud LLM, input text is sent to the provider’s API. "
    "For sensitive content, consider the Ollama local model option or a private deployment. \n"
    "Compliance tip: Keep `decisions.csv` as your audit trail showing human sign‑off for each item."
)

