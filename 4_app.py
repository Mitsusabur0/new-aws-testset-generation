# app.py
import streamlit as st
import pandas as pd

# --- CONFIGURATION ---
INPUT_FILE = "testset_results.parquet"

st.set_page_config(page_title="RAG Offline Eval", layout="wide")

@st.cache_data
def load_data():
    try:
        return pd.read_parquet(INPUT_FILE)
    except FileNotFoundError:
        st.error("Results file not found. Please run the evaluation pipeline first.")
        return pd.DataFrame()

def main():
    st.title("🔎 RAG Pipeline Evaluation Dashboard")
    df = load_data()
    
    if df.empty:
        return

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")
    
    personas = st.sidebar.multiselect(
        "Select Persona", 
        options=df['persona_name'].unique(),
        default=df['persona_name'].unique()
    )
    
    styles = st.sidebar.multiselect(
        "Select Query Style", 
        options=df['query_style'].unique(),
        default=df['query_style'].unique()
    )

    # Apply filters
    filtered_df = df[
        (df['persona_name'].isin(personas)) & 
        (df['query_style'].isin(styles))
    ]

    # --- TABS ---
    tab1, tab2 = st.tabs(["📊 Dashboard", "🧐 Detailed Analysis"])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        st.subheader("Global Metrics")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Mean MRR", f"{filtered_df['mrr'].mean():.4f}")
        m_col2.metric("Hit Rate", f"{filtered_df['hit_rate'].mean():.2%}")
        m_col3.metric("Avg Precision@3", f"{filtered_df['precision_at_k'].mean():.4f}")
        m_col4.metric("Avg Recall@3", f"{filtered_df['recall_at_k'].mean():.4f}")

        st.divider()
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Hit Rate by Persona")
            persona_metrics = filtered_df.groupby('persona_name')['hit_rate'].mean()
            st.bar_chart(persona_metrics)
            
        with col_chart2:
            st.subheader("Hit Rate by Query Style")
            style_metrics = filtered_df.groupby('query_style')['hit_rate'].mean()
            st.bar_chart(style_metrics)

    # --- TAB 2: DETAILED ANALYSIS ---
    with tab2:
        st.subheader("Test Case Explorer")
        
        # Master View: Table with selection
        # We create a display dataframe with subset of columns
        display_cols = ['user_input', 'mrr', 'hit_rate', 'persona_name', 'query_style']
        
        # Using Streamlit's dataframe selection (requires newer Streamlit, fallback logic included)
        selection = st.dataframe(
            filtered_df[display_cols],
            use_container_width=True,
            hide_index=True,
            on_select="rerun", # Enables row selection
            selection_mode="single-row"
        )
        
        selected_index = None
        if selection.selection and selection.selection.rows:
            # Get the index of the selected row relative to the filtered dataframe
            selected_row_idx = selection.selection.rows[0]
            selected_index = filtered_df.index[selected_row_idx]
        
        st.divider()

        # Detailed View
        if selected_index is not None:
            row = df.loc[selected_index]
            
            st.markdown(f"### Question: *{row['user_input']}*")
            
            # Metrics Row for this case
            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            d_col1.info(f"MRR: {row['mrr']}")
            d_col2.info(f"Hit: {'✅' if row['hit_rate'] == 1 else '❌'}")
            d_col3.info(f"Persona: {row['persona_name']}")
            d_col4.info(f"Style: {row['query_style']}")
            
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.write("#### 📝 Ground Truth Context")
                # Handle list structure
                gt_list = row['reference_contexts']
                if isinstance(gt_list, list) and len(gt_list) > 0:
                    st.success(gt_list[0])
                else:
                    st.write(gt_list)

            with comp_col2:
                st.write("#### 🤖 Retrieved Contexts (Top 3)")
                retrieved = row['retrieved_contexts']
                
                # Check formatting
                if isinstance(retrieved, list):
                    for i, txt in enumerate(retrieved):
                        # Simple visual check for match
                        gt_txt_clean = " ".join(gt_list[0].lower().split()) if isinstance(gt_list, list) and gt_list else ""
                        is_match = gt_txt_clean in " ".join(txt.lower().split())
                        
                        emoji = "✅" if is_match else f"{i+1}."
                        color = "green" if is_match else "grey"
                        
                        with st.container(border=True):
                            st.markdown(f"**{emoji} Rank {i+1}**")
                            st.text(txt)
                else:
                    st.write("No retrieval data.")
        else:
            st.info("Select a row in the table above to view details.")

if __name__ == "__main__":
    main()