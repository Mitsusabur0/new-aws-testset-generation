from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st
import altair as alt


APP_DIR = Path(__file__).parent
PARQUET_PATH = APP_DIR / "testset_results.parquet"
CSV_PATH = APP_DIR / "evaluation_set_full.csv"


st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _to_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return [value]
    return [value]


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
    elif CSV_PATH.exists():
        # CSV fallback with encoding safety
        try:
            df = pd.read_csv(CSV_PATH, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(CSV_PATH, encoding="latin-1")
    else:
        return pd.DataFrame()

    list_cols = ["reference_contexts", "retrieved_contexts", "retrieved_file"]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(_to_list)

    metric_cols = [
        "custom_hit_rate",
        "custom_mrr",
        "custom_precision_at_k",
        "custom_recall_at_k",
        "deepeval_contextual_precision",
        "deepeval_contextual_recall",
        "deepeval_contextual_relevancy",
        "ragas_context_precision",
        "ragas_context_recall",
        "ragas_context_entity_recall",
    ]
    df = _coerce_numeric(df, metric_cols)

    # Convenience columns
    if "custom_hit_rate" in df.columns:
        df["is_hit"] = df["custom_hit_rate"].fillna(0).astype(int)
    if "custom_mrr" in df.columns:
        df["mrr_bucket"] = pd.cut(
            df["custom_mrr"].fillna(0),
            bins=[-0.01, 0, 0.33, 0.66, 1.0],
            labels=["0", "0–0.33", "0.33–0.66", "0.66–1.0"],
        )

    return df


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif;
}

.reportview-container {
    background: radial-gradient(1200px 600px at 20% -10%, #e7f3ff 0%, #f7f9fc 55%, #ffffff 100%);
}

section.main > div {
    padding-top: 1.5rem;
}

.hero {
    padding: 1.25rem 1.5rem;
    border-radius: 16px;
    background: #0f172a;
    color: #ffffff;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 20px 60px rgba(15, 23, 42, 0.2);
}

.hero h1 {
    font-size: 2.1rem;
    margin-bottom: 0.25rem;
}

.hero p {
    margin-top: 0.25rem;
    color: rgba(255, 255, 255, 0.78);
}

.kpi-card-custom {
    background: linear-gradient(180deg, rgba(37, 99, 235, 0.10), rgba(37, 99, 235, 0.02));
    border-color: rgba(37, 99, 235, 0.25);
}

.kpi-card-deepeval {
    background: linear-gradient(180deg, rgba(15, 118, 110, 0.10), rgba(15, 118, 110, 0.02));
    border-color: rgba(15, 118, 110, 0.25);
}

.kpi-card-ragas {
    background: linear-gradient(180deg, rgba(249, 115, 22, 0.12), rgba(249, 115, 22, 0.02));
    border-color: rgba(249, 115, 22, 0.25);
}

.kpi-label {
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748b;
}

.kpi-value {
    font-size: 1.6rem;
    font-weight: 600;
    color: #0f172a;
}

.code-block {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    background: #0f172a;
    color: #e2e8f0;
    padding: 0.75rem;
    border-radius: 10px;
}

.metric-pill {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    font-size: 0.75rem;
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #dbeafe;
}

.section-card {
    padding: 1.2rem;
    border-radius: 16px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    box-shadow: 0 14px 30px rgba(15, 23, 42, 0.05);
}

.section-title {
    font-size: 0.85rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 600;
    color: #475569;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.4rem;
}

.section-title::before {
    content: "";
    width: 12px;
    height: 12px;
    border-radius: 999px;
    background: currentColor;
}

.section-divider {
    height: 1px;
    margin: 0.75rem 0 1.1rem 0;
    background: linear-gradient(90deg, currentColor, rgba(148, 163, 184, 0));
    border: none;
}

.section-custom {
    color: #1d4ed8;
}

.section-deepeval {
    color: #0f766e;
}

.section-ragas {
    color: #ea580c;
}

@media (prefers-color-scheme: dark) {
    .section-divider {
        background: linear-gradient(90deg, rgba(148, 163, 184, 0.5), rgba(148, 163, 184, 0));
    }

    .section-custom {
        color: #93c5fd;
    }

    .section-deepeval {
        color: #5eead4;
    }

    .section-ragas {
        color: #fdba74;
    }
}

@media (prefers-color-scheme: dark) {
    .kpi-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.25);
        box-shadow: 0 12px 24px rgba(2, 6, 23, 0.35);
    }

    .kpi-card-custom {
        border-color: rgba(59, 130, 246, 0.35);
    }

    .kpi-card-deepeval {
        border-color: rgba(20, 184, 166, 0.35);
    }

    .kpi-card-ragas {
        border-color: rgba(249, 115, 22, 0.35);
    }

    .kpi-label {
        color: #94a3b8;
    }

    .kpi-value {
        color: #e2e8f0;
    }
}
</style>
"""


st.markdown(CSS, unsafe_allow_html=True)


def render_hero(df: pd.DataFrame) -> None:
    total = len(df)
    styles = df["query_style"].nunique() if "query_style" in df.columns else 0
    files = df["source_file"].nunique() if "source_file" in df.columns else 0
    st.markdown(
        f"""
        <div class="hero">
            <h1>RAG Evaluation Dashboard</h1>
            <p>End-to-end view of retrieval quality, LLM grounding, and testcase-level diagnostics.</p>
            <p><span class="metric-pill">{total} testcases</span> &nbsp; <span class="metric-pill">{styles} query styles</span> &nbsp; <span class="metric-pill">{files} source files</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_group(title: str, kpis: list[tuple[str, float, str]], tone: str) -> None:
    st.markdown(
        f"<div class='section-title section-{tone}'>{title}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='section-divider section-{tone}'></div>",
        unsafe_allow_html=True,
    )
    for label, value, fmt in kpis:
        if pd.isna(value):
            display = "N/A"
        elif fmt == "percent":
            display = f"{value:.2%}"
        else:
            display = f"{value:.4f}"
        st.markdown(
            f"""
            <div class="kpi-card kpi-card-{tone}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_query_style_chart(
    df: pd.DataFrame,
    metrics: list[str],
    label_map: dict[str, str],
    colors: list[str],
    title: str,
    stacked: bool = False,
    y_domain: tuple[float, float] | None = None,
) -> None:
    st.markdown(f"#### {title}")
    if "query_style" not in df.columns:
        st.info("No query_style column found.")
        return

    available_metrics = [metric for metric in metrics if metric in df.columns]
    if not available_metrics:
        st.info("No metrics available for this chart.")
        return

    grouped = (
        df.groupby("query_style")[available_metrics]
        .mean()
        .reset_index()
        .melt("query_style", var_name="metric", value_name="value")
    )
    grouped["metric_label"] = grouped["metric"].map(label_map).fillna(grouped["metric"])

    chart = (
        alt.Chart(grouped)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("query_style:N", title="Query Style"),
            y=alt.Y(
                "value:Q",
                title="Average",
                stack="zero" if stacked else None,
                scale=alt.Scale(domain=y_domain) if y_domain else alt.Undefined,
            ),
            color=alt.Color(
                "metric_label:N",
                scale=alt.Scale(range=colors),
                sort=[label_map.get(metric, metric) for metric in available_metrics],
                legend=alt.Legend(title="Metric") if len(available_metrics) > 1 else None,
            ),
            tooltip=["query_style", "metric_label", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)


def render_global_sections(df: pd.DataFrame) -> None:
    custom_kpis = [
        ("Hit Rate", df.get("custom_hit_rate", pd.Series(dtype=float)).mean(), "percent"),
        ("MRR", df.get("custom_mrr", pd.Series(dtype=float)).mean(), "float"),
        ("Precision@K", df.get("custom_precision_at_k", pd.Series(dtype=float)).mean(), "float"),
        ("Recall@K", df.get("custom_recall_at_k", pd.Series(dtype=float)).mean(), "float"),
    ]

    deepeval_kpis = [
        (
            "Precision",
            df.get("deepeval_contextual_precision", pd.Series(dtype=float)).mean(),
            "float",
            "deepeval_contextual_precision",
        ),
        (
            "Recall",
            df.get("deepeval_contextual_recall", pd.Series(dtype=float)).mean(),
            "float",
            "deepeval_contextual_recall",
        ),
        (
            "Relevancy",
            df.get("deepeval_contextual_relevancy", pd.Series(dtype=float)).mean(),
            "float",
            "deepeval_contextual_relevancy",
        ),
    ]

    ragas_kpis = [
        ("Precision", df.get("ragas_context_precision", pd.Series(dtype=float)).mean(), "float", "ragas_context_precision"),
        ("Recall", df.get("ragas_context_recall", pd.Series(dtype=float)).mean(), "float", "ragas_context_recall"),
        ("Entity Recall", df.get("ragas_context_entity_recall", pd.Series(dtype=float)).mean(), "float", "ragas_context_entity_recall"),
    ]

    st.markdown("### Custom Metrics")
    left, spacer, right = st.columns([1.05, 0.08, 1.25])
    with left:
        render_kpi_group("Custom Metrics", custom_kpis, "custom")
    with right:
        render_query_style_chart(
            df,
            metrics=["custom_mrr"],
            label_map={"custom_mrr": "MRR"},
            colors=["#2563eb"],
            title="MRR by Query Style",
            stacked=False,
            y_domain=(0, 1),
        )
        render_query_style_chart(
            df,
            metrics=["custom_hit_rate"],
            label_map={"custom_hit_rate": "Hit Rate"},
            colors=["#0f766e"],
            title="Hit Rate by Query Style",
            stacked=False,
            y_domain=(0, 1),
        )

    st.markdown("---")

    st.markdown("### DeepEval Metrics")
    left, spacer, right = st.columns([0.85, 0.08, 1.45])
    with left:
        render_kpi_group("DeepEval Metrics", deepeval_kpis, "deepeval")
        metric_options = [""] + [label for label, _, _, _ in deepeval_kpis]
        selected_label = st.radio(
            "Select DeepEval metric",
            metric_options,
            index=0,
            key="deepeval_metric_choice",
            label_visibility="collapsed",
        )
        label_to_metric = {label: metric for label, _, _, metric in deepeval_kpis}
        selected_metric = label_to_metric.get(selected_label, "")
    with right:
        if not selected_metric:
            st.info("Click a DeepEval metric to show its chart.")
        else:
            deepeval_colors = {
                "deepeval_contextual_precision": "#0f766e",
                "deepeval_contextual_recall": "#14b8a6",
                "deepeval_contextual_relevancy": "#5eead4",
            }
            deepeval_labels = {metric: label for label, _, _, metric in deepeval_kpis}
            label = deepeval_labels.get(selected_metric, selected_metric)
            color = deepeval_colors.get(selected_metric, "#0f766e")
            render_query_style_chart(
                df,
                metrics=[selected_metric],
                label_map={selected_metric: label},
                colors=[color],
                title=f"{label} by Query Style",
                stacked=False,
                y_domain=(0, 1),
            )

    st.markdown("---")

    st.markdown("### RAGAS Metrics")
    st.markdown("Click a KPI to update the chart.")
    selected_metric = st.session_state.get("ragas_selected_metric", ragas_kpis[0][3])
    ragas_colors = {
        "ragas_context_precision": "#f97316",
        "ragas_context_recall": "#fb923c",
        "ragas_context_entity_recall": "#fdba74",
    }
    ragas_labels = {metric: label for label, _, _, metric in ragas_kpis}

    kpi_cols = st.columns(len(ragas_kpis))
    for col, (label, value, fmt, metric) in zip(kpi_cols, ragas_kpis):
        if pd.isna(value):
            display = "N/A"
        elif fmt == "percent":
            display = f"{value:.2%}"
        else:
            display = f"{value:.4f}"
        button_label = f"{label}: {display}"
        if col.button(
            button_label,
            key=f"ragas_kpi_{metric}",
            type="primary" if metric == selected_metric else "secondary",
            use_container_width=True,
        ):
            st.session_state["ragas_selected_metric"] = metric
            selected_metric = metric

    label = ragas_labels.get(selected_metric, selected_metric)
    color = ragas_colors.get(selected_metric, "#f97316")
    render_query_style_chart(
        df,
        metrics=[selected_metric],
        label_map={selected_metric: label},
        colors=[color],
        title=f"{label} by Query Style",
        stacked=False,
        y_domain=(0, 1),
    )


def render_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    text_query = st.sidebar.text_input("Search in user input")

    if "query_style" in df.columns:
        styles = sorted(df["query_style"].dropna().unique().tolist())
        style_filter = st.sidebar.multiselect("Query Style", styles, default=styles)
    else:
        style_filter = None

    if "source_file" in df.columns:
        files = sorted(df["source_file"].dropna().unique().tolist())
        file_filter = st.sidebar.multiselect("Source File", files, default=files)
    else:
        file_filter = None

    min_mrr = None
    if "custom_mrr" in df.columns:
        min_mrr = st.sidebar.slider("Minimum MRR", 0.0, 1.0, 0.0, 0.05)

    show_failures_only = st.sidebar.toggle("Show failures only", value=False)

    filtered = df.copy()
    if text_query:
        filtered = filtered[filtered["user_input"].str.contains(text_query, case=False, na=False)]
    if style_filter is not None:
        filtered = filtered[filtered["query_style"].isin(style_filter)]
    if file_filter is not None:
        filtered = filtered[filtered["source_file"].isin(file_filter)]
    if min_mrr is not None:
        filtered = filtered[filtered["custom_mrr"].fillna(0) >= min_mrr]
    if show_failures_only and "custom_hit_rate" in filtered.columns:
        filtered = filtered[filtered["custom_hit_rate"].fillna(0) == 0]

    return filtered


def _format_context_list(contexts: list) -> str:
    if not contexts:
        return ""
    return "\n\n".join(str(c) for c in contexts)


def render_case_explorer(df: pd.DataFrame) -> None:
    st.markdown("### Testcase Explorer")

    display_cols = [
        col
        for col in ["user_input", "query_style", "source_file", "custom_mrr", "custom_hit_rate"]
        if col in df.columns
    ]

    selection = st.dataframe(
        df[display_cols],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=320,
    )

    selected_idx = None
    if selection.selection and selection.selection.rows:
        selected_row_idx = selection.selection.rows[0]
        selected_idx = df.index[selected_row_idx]

    if selected_idx is None:
        st.info("Select a row to view the full testcase details.")
        return

    row = df.loc[selected_idx]

    st.markdown("---")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### Prompt & Outputs")
        st.markdown(f"**User Input**\n\n{row.get('user_input', '')}")
        st.markdown(f"**Expected Output**\n\n{row.get('expected_output', '')}")
        st.markdown(f"**Actual Output**\n\n{row.get('actual_output', '')}")

    with col2:
        st.markdown("#### Metrics")
        metric_rows = [
            ("Custom Hit Rate", row.get("custom_hit_rate")),
            ("Custom MRR", row.get("custom_mrr")),
            ("Precision@K", row.get("custom_precision_at_k")),
            ("Recall@K", row.get("custom_recall_at_k")),
            ("DeepEval Precision", row.get("deepeval_contextual_precision")),
            ("DeepEval Recall", row.get("deepeval_contextual_recall")),
            ("DeepEval Relevancy", row.get("deepeval_contextual_relevancy")),
            ("RAGAS Precision", row.get("ragas_context_precision")),
            ("RAGAS Recall", row.get("ragas_context_recall")),
            ("Entity Recall", row.get("ragas_context_entity_recall")),
        ]
        for label, value in metric_rows:
            if pd.isna(value):
                display = "—"
            else:
                display = f"{value:.4f}" if isinstance(value, float) else str(value)
            st.markdown(f"**{label}:** {display}")

        st.markdown("#### Metadata")
        st.markdown(f"**Query Style:** {row.get('query_style', '—')}")
        st.markdown(f"**Source File:** {row.get('source_file', '—')}")

    st.markdown("---")

    context_col1, context_col2 = st.columns(2)

    with context_col1:
        st.markdown("#### Ground Truth Context")
        gt_contexts = row.get("reference_contexts", [])
        gt_text = _format_context_list(gt_contexts)
        if gt_text:
            st.markdown(f"<div class='code-block'>{gt_text}</div>", unsafe_allow_html=True)
        else:
            st.info("No reference context available.")

    with context_col2:
        st.markdown("#### Retrieved Contexts")
        retrieved = row.get("retrieved_contexts", [])
        retrieved_files = row.get("retrieved_file", [])
        source_file = str(row.get("source_file", "")).strip()

        if not retrieved:
            st.info("No retrieved contexts available.")
        else:
            for i, context in enumerate(retrieved, start=1):
                file_hint = ""
                if i - 1 < len(retrieved_files):
                    file_hint = str(retrieved_files[i - 1])
                is_match = bool(source_file and file_hint and source_file in file_hint)
                status = "Match" if is_match else ""
                st.markdown(
                    f"**Rank {i}** {status}  \n"
                    f"File: `{file_hint or '—'}`",
                )
                st.markdown(f"<div class='code-block'>{context}</div>", unsafe_allow_html=True)

    with st.expander("Reasoning Notes"):
        st.markdown("#### DeepEval Notes")
        st.markdown(f"**Precision:** {row.get('deepeval_contextual_precision_reason', '—')}")
        st.markdown(f"**Recall:** {row.get('deepeval_contextual_recall_reason', '—')}")
        st.markdown(f"**Relevancy:** {row.get('deepeval_contextual_relevancy_reason', '—')}")


def main() -> None:
    df = load_data()
    if df.empty:
        st.error("No evaluation results found. Place evaluation_set_full.csv or testset_results.parquet in streamlit/.")
        return

    render_hero(df)
    st.markdown("\n")

    filtered_df = render_filters(df)
    if filtered_df.empty:
        st.warning("No testcases match the selected filters.")
        return

    metrics_tab, explorer_tab = st.tabs(["Global Metrics", "Testcase Explorer"])

    with metrics_tab:
        st.markdown("### Global Metrics")
        render_global_sections(filtered_df)

        st.markdown("### Export")
        st.download_button(
            "Download filtered results as CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_eval_results.csv",
            mime="text/csv",
        )

    with explorer_tab:
        render_case_explorer(filtered_df)


if __name__ == "__main__":
    main()
