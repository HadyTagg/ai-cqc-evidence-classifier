
import ast
import io
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Canonical list of 34 Quality Statements (CQC SAF)
ALL_QS = ['Learning culture', 'Safe systems, pathways and transitions', 'Safeguarding', 'Involving people to manage risks', 'Safe environments', 'Safe and effective staffing', 'Infection prevention and control', 'Medicines optimisation', 'Assessing needs', 'Delivering evidence-based care and treatment', 'How staff and teams work together', 'Supporting people to live healthier lives', 'Monitoring and improving outcomes', 'Consent to care and treatment', 'Kindness, compassion and dignity', 'Treating people as individuals', 'Independence, choice and control', 'Responding to people’s immediate needs', 'Workforce wellbeing and enablement', 'Person centred care', 'Care provision, integration, and continuity', 'Providing information', 'Listening to and involving people', 'Equity in access', 'Equity in experiences and outcomes', 'Planning for the future', 'Shared direction and culture', 'Capable, compassionate and inclusive leaders', 'Freedom to speak up', 'Workforce equality, diversity and inclusion', 'Governance, management and sustainability', 'Partnerships and communities', 'Learning, improvement and innovation', 'Environmental sustainability, sustainable development']


# ----------------------------
# Page config & helpers
# ----------------------------
st.set_page_config(
    page_title="CQC Evidence Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-box {padding: 0.75rem 1rem; border-radius: 1rem; border: 1px solid #e5e7eb; background: #fafafa;}
    .small-muted {color:#6b7280; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

## Accessibility helpers
# ----------------------------
# Visual style & colours
# ----------------------------
# Okabe–Ito colour-blind safe palette
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00", "#000000"]

def apply_matplotlib_theme(high_contrast: bool = False):
    import matplotlib as mpl
    base_grid = "#D0D7DE" if not high_contrast else "#9AA0A6"
    face = "#FFFFFF"
    mpl.rcParams.update({
        "figure.facecolor": face,
        "axes.facecolor": face,
        "axes.edgecolor": "#111111",
        "axes.titleweight": "semibold",
        "axes.grid": True,
        "grid.color": base_grid,
        "grid.alpha": 0.5,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "legend.frameon": False,
    })

def category_color_map(categories):
    # Stable mapping of category -> palette colour
    cats = list(categories)
    cats.sort()
    mapping = {}
    for i, c in enumerate(cats):
        mapping[c] = OKABE_ITO[i % len(OKABE_ITO)]
    return mapping


# ----------------------------
# Accessibility helpers
# ----------------------------
def wrap_text(label: str, width: int = 26) -> str:
    if not isinstance(label, str):
        return label
    words = label.split()
    lines, line = [], ""
    for w in words:
        if len((line + " " + w).strip()) <= width:
            line = (line + " " + w).strip()
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return "\n".join(lines)

def apply_font_scale(scale: float = 1.0):
    import matplotlib as mpl
    base = 12 * scale
    mpl.rcParams.update({
        "font.size": base,
        "axes.titlesize": base * 1.1,
        "axes.labelsize": base,
        "xtick.labelsize": base * 0.9,
        "ytick.labelsize": base * 0.9,
        "legend.fontsize": base * 0.9,
    })


# Normalise common QS label variants to canonical forms
QS_NORMALISATION = {
    "Person-centred care": "Person centred care",
    "Care provision, integration and continuity": "Care provision, integration, and continuity",
}
def _normalise_qs_list(lst):
    out = []
    for x in lst or []:
        out.append(QS_NORMALISATION.get(x, x))
    return out


@st.cache_data(show_spinner=False)
def load_decisions(csv_bytes: bytes | None, default_path: str = "/home/supermunkey2k/PycharmProjects/ai-cqc-evidence-classifier/decisions.csv") -> pd.DataFrame:
    """
    Load the decisions CSV either from an uploaded file (csv_bytes) or a default path.
    Parse list-like columns to Python lists.
    """
    if csv_bytes is not None:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    else:
        df = pd.read_csv(default_path)

    # Parse list-like columns safely
    list_cols = ["quality_statements", "evidence_categories", "paths"]
    for col in list_cols:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else ([] if pd.isna(x) else [x]))

    # Normalise QS labels
    df["quality_statements"] = df["quality_statements"].apply(_normalise_qs_list)

    # Timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Normalised long form for counts
    long_df = df.explode("quality_statements").explode("evidence_categories")
    # Drop rows where after explode we don't have values
    long_df = long_df.dropna(subset=["quality_statements", "evidence_categories"])

    return df, long_df


def download_button_csv(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


# ----------------------------
# Sidebar: Data & Filters
# ----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload decisions.csv (optional)", type=["csv"])

df, long_df = load_decisions(uploaded.getvalue() if uploaded else None)

st.sidebar.header("Filters")

st.sidebar.header("Accessibility")
large_text = st.sidebar.toggle("Large text mode", value=False, help="Increase font sizes in all charts")
show_values = st.sidebar.toggle("Show values on bars", value=True)
wrap_labels = st.sidebar.toggle("Wrap long labels", value=False)

st.sidebar.header("Visual style")
high_contrast = st.sidebar.toggle("High-contrast theme", value=True)
apply_font_scale(1.25 if large_text else 1.0)
apply_matplotlib_theme(high_contrast=high_contrast)


# Date range
min_date = pd.to_datetime(df["timestamp"]).min()
max_date = pd.to_datetime(df["timestamp"]).max()
date_from, date_to = st.sidebar.date_input(
    "Date range",
    value=(min_date.date() if pd.notna(min_date) else None, max_date.date() if pd.notna(max_date) else None),
    min_value=min_date.date() if pd.notna(min_date) else None,
    max_value=max_date.date() if pd.notna(max_date) else None,
)
if isinstance(date_from, tuple):  # Streamlit API quirk safety
    date_from, date_to = date_from[0], date_from[1]
mask_date = (df["timestamp"].dt.date >= date_from) & (df["timestamp"].dt.date <= date_to)

# Reviewer & action
reviewers = sorted([x for x in df["reviewer"].dropna().unique()])
actions = sorted([x for x in df["action"].dropna().unique()])
sel_reviewers = st.sidebar.multiselect("Reviewer", reviewers, default=reviewers)
sel_actions = st.sidebar.multiselect("Action", actions, default=actions)

mask_rev = df["reviewer"].isin(sel_reviewers)
mask_act = df["action"].isin(sel_actions)

# Keyword search
kw = st.sidebar.text_input("Search in file / notes (optional)").strip()
if kw:
    kw_lower = kw.lower()
    mask_kw = df["file"].str.lower().str.contains(kw_lower, na=False) | df["notes"].str.lower().str.contains(kw_lower, na=False)
else:
    mask_kw = pd.Series([True] * len(df), index=df.index)

# Apply filters to the wide df, then regenerate a filtered long_df
df_filt = df[mask_date & mask_rev & mask_act & mask_kw].copy()
long_df_filt = df_filt.explode("quality_statements").explode("evidence_categories").dropna(subset=["quality_statements", "evidence_categories"])

st.sidebar.caption(f"{len(df_filt):,} items after filters")

# ----------------------------
# KPIs
# ----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Evidence items (rows)", f"{len(df_filt):,}")
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Unique files", f"{df_filt['file'].nunique():,}")
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    avg_qs_per_file = df_filt["quality_statements"].apply(len).mean() if len(df_filt) else 0
    st.metric("Avg QS per file", f"{avg_qs_per_file:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    covered_qs = long_df_filt["quality_statements"].nunique()
    st.metric("Quality statements covered", f"{covered_qs}")
    st.markdown('</div>', unsafe_allow_html=True)
with c5:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    st.metric("Last update", (df_filt["timestamp"].max().strftime("%Y-%m-%d") if not df_filt["timestamp"].isna().all() else "—"))
    st.markdown('</div>', unsafe_allow_html=True)


st.title("📊 CQC Evidence Insights (Interactive)")
st.caption("Explore coverage across quality statements and evidence categories. Use the filters in the sidebar.")

st.markdown(
    """
    <style>
    /* Visual polish */
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    h1, h2, h3 {letter-spacing: 0.2px;}
    .stMetric label, .stMetric [data-testid="stMetricValue"] {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# Section: Evidence per Quality Statement
# ----------------------------
st.subheader("Evidence per Quality Statement")

qs_counts = (
    long_df_filt.groupby("quality_statements")["file"].count()
    .reindex(ALL_QS, fill_value=0)
    .reset_index()
    .rename(columns={"index": "quality_statements", "file": "count"})
)


# Bar chart with matplotlib

fig1, ax1 = plt.subplots(figsize=(12, 10))
labels_qs = qs_counts["quality_statements"].apply(lambda s: wrap_text(s) if wrap_labels else s)
ax1.barh(labels_qs, qs_counts["count"], color=OKABE_ITO[0])
ax1.set_title("Evidence Count per Quality Statement")
ax1.set_xlabel("Count")
ax1.set_ylabel("Quality Statements")
ax1.grid(axis="x", which="major", alpha=0.3)
ax1.invert_yaxis()  # largest at top after sort
if show_values:
    for i, v in enumerate(qs_counts["count"]):
        ax1.text(v, i, f" {v}", va="center")
st.pyplot(fig1, use_container_width=True)


with st.expander("View table / download"):
    st.dataframe(qs_counts, use_container_width=True, hide_index=True)
    download_button_csv(qs_counts, "Download QS counts (CSV)", "qs_counts.csv")


# ----------------------------
# Section: Evidence Categories within each Quality Statement
# ----------------------------
st.subheader("Evidence categories within each Quality Statement")

qs_ec_counts = (long_df_filt.groupby(["quality_statements", "evidence_categories"])["file"]
                .count()
                .reset_index(name="count"))

pivot_qs_ec = (
    qs_ec_counts.pivot(index="quality_statements", columns="evidence_categories", values="count")
    .reindex(ALL_QS, fill_value=0)
    .fillna(0)
)


# Stacked bar via matplotlib


fig2, ax2 = plt.subplots(figsize=(12, 12))
labels_qs2 = [wrap_text(str(x)) if wrap_labels else str(x) for x in pivot_qs_ec.index]
left = [0] * len(pivot_qs_ec)
cat_palette = category_color_map(pivot_qs_ec.columns)
for col in pivot_qs_ec.columns:
    vals = pivot_qs_ec[col].values
    ax2.barh(labels_qs2, vals, left=left, label=col, color=cat_palette.get(col))
    left = (pd.Series(left) + pd.Series(vals)).tolist()
ax2.set_title("Evidence Categories within Each Quality Statement")

ax2.set_xlabel("Count")
ax2.set_ylabel("Quality Statements")
ax2.grid(axis="x", which="major", alpha=0.3)
ax2.invert_yaxis()
ax2.legend(title="Evidence Categories", bbox_to_anchor=(1.02, 1), loc="upper left")
st.pyplot(fig2, use_container_width=True)


with st.expander("View table / download"):
    st.dataframe(pivot_qs_ec.reset_index(), use_container_width=True, hide_index=True)
    download_button_csv(pivot_qs_ec.reset_index(), "Download QS x Category pivot (CSV)", "qs_by_category.csv")


# ----------------------------
# Section: Drill-down
# ----------------------------
st.subheader("Drill-down by Quality Statement")

qs_list = ALL_QS
sel_qs = st.selectbox("Choose a quality statement to drill into", ["—"] + qs_list, index=0)
if sel_qs != "—":
    subset = long_df_filt[long_df_filt["quality_statements"] == sel_qs]
    cat_counts = (subset.groupby("evidence_categories")["file"]
                  .count()
                  .reset_index(name="count")
                  .sort_values("count", ascending=False))

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(cat_counts["evidence_categories"], cat_counts["count"])
    ax3.set_title(f"Evidence categories for: {sel_qs}")
    ax3.set_xlabel("Evidence Categories")
    ax3.set_ylabel("Count")
    ax3.tick_params(axis="x", rotation=45)
    for lbl in ax3.get_xticklabels():
        lbl.set_ha("right")
    st.pyplot(fig3, use_container_width=True)

    st.markdown("**Example items (up to 50):**")
    examples = subset[["timestamp", "file", "evidence_categories", "reviewer", "action", "notes"]].head(50)
    st.dataframe(examples, use_container_width=True)

# ----------------------------
# Section: Category Mix & Balance
# ----------------------------
st.subheader("Overall category mix")

ec_counts = (long_df_filt.groupby("evidence_categories")["file"]
             .count()
             .reset_index(name="count")
             .sort_values("count", ascending=False))

fig4, ax4 = plt.subplots(figsize=(8, 4))
ax4.bar(ec_counts["evidence_categories"], ec_counts["count"])
ax4.set_title("Evidence by Category (overall)")
ax4.set_xlabel("Evidence Categories")
ax4.set_ylabel("Count")
ax4.tick_params(axis="x", rotation=45)
ax4.set_xticklabels(ax4.get_xticklabels(), ha="right")
st.pyplot(fig4, use_container_width=True)

with st.expander("View table / download"):
    st.dataframe(ec_counts, use_container_width=True, hide_index=True)
    download_button_csv(ec_counts, "Download category counts (CSV)", "category_counts.csv")

# ----------------------------
# Section: Tagging Density
# ----------------------------
st.subheader("Tagging density per file")

qs_per_file = df_filt["quality_statements"].apply(len)
ec_per_file = df_filt["evidence_categories"].apply(len)

col_a, col_b = st.columns(2)
with col_a:
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    qs_per_file.value_counts().sort_index().plot(kind="bar", ax=ax5)
    ax5.set_title("How many QS tags per file?")
    ax5.set_xlabel("Number of QS tags")
    ax5.set_ylabel("Files")
    ax5.grid(axis="y", which="major", alpha=0.3)
    if show_values:
        for p in ax5.patches:
            height = p.get_height()
            ax5.text(p.get_x() + p.get_width()/2, height, f"{int(height)}", ha="center", va="bottom")
    st.pyplot(fig5, use_container_width=True)

with col_b:
    fig6, ax6 = plt.subplots(figsize=(6, 4))
    ec_per_file.value_counts().sort_index().plot(kind="bar", ax=ax6)
    ax6.set_title("How many category tags per file?")
    ax6.set_xlabel("Number of category tags")
    ax6.set_ylabel("Files")
    ax6.grid(axis="y", which="major", alpha=0.3)
    if show_values:
        for p in ax6.patches:
            height = p.get_height()
            ax6.text(p.get_x() + p.get_width()/2, height, f"{int(height)}", ha="center", va="bottom")
    st.pyplot(fig6, use_container_width=True)

st.caption("Tip: heavy reliance on 'Processes' may indicate a need to strengthen Outcomes, People’s experience, and Observations.")

# ----------------------------
# Section: Data table & export
# ----------------------------
st.subheader("Filtered dataset")
st.dataframe(df_filt, use_container_width=True)
download_button_csv(df_filt, "Download filtered rows (CSV)", "filtered_decisions.csv")
