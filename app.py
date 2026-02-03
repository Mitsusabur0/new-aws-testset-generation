# app.py
import streamlit as st
import pandas as pd
import config

st.set_page_config(page_title="RAG Offline Eval", layout="wide")

# In app.py - load_data function
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet(config.OUTPUT_RESULTS_PARQUET)
        # Ensure list columns are actual Python lists, not numpy arrays
        list_cols = ['reference_contexts', 'retrieved_contexts', 'retrieved_file']
        for col in list_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x.tolist() if hasattr(x, 'tolist') else x)
        return df
    except FileNotFoundError:
        st.error("Results file not found...")
        return pd.DataFrame()

def main():
    st.title("RAG Pipeline Evaluation Dashboard")
    df = load_data()
    
    if df.empty:
        return

    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")
    
    styles = st.sidebar.multiselect(
        "Select Query Style", 
        options=df['query_style'].unique(),
        default=df['query_style'].unique()
    )

    # Apply filters
    filtered_df = df[
        (df['query_style'].isin(styles))
    ]

    # --- TABS ---
    tab1, tab2 = st.tabs(["📊 Dashboard", "Detailed Analysis"])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        st.subheader("Global Metrics")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Mean MRR", f"{filtered_df['mrr'].mean():.4f}")
        m_col2.metric("Hit Rate", f"{filtered_df['hit_rate'].mean():.2%}")
        m_col3.metric("Avg Precision@3", f"{filtered_df['precision_at_k'].mean():.4f}")
        m_col4.metric("Avg Recall@3", f"{filtered_df['recall_at_k'].mean():.4f}")

        st.divider()
        
        st.subheader("Hit Rate by Query Style")
        style_metrics = filtered_df.groupby('query_style')['hit_rate'].mean()
        st.bar_chart(style_metrics)

    # --- TAB 2: DETAILED ANALYSIS ---
    with tab2:
        st.subheader("Test Case Explorer")
        
        # Master View: Table with selection
        display_cols = ['source_file', 'user_input', 'mrr', 'query_style']
        
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
            
            # Metrics Row for this case (Reduced to 3 columns)
            d_col1, d_col2, d_col3 = st.columns(3)
            d_col1.info(f"MRR: {row['mrr']}")
            d_col2.info(f"Hit: {'✅' if row['hit_rate'] == 1 else '❌'}")
            d_col3.info(f"Style: {row['query_style']}")
            
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.write("#### Ground Truth Context")
                # Handle list structure
                gt_list = row['reference_contexts']
                if isinstance(gt_list, list) and len(gt_list) > 0:
                    st.success(gt_list[0])
                else:
                    st.write(gt_list)

            with comp_col2:
                st.write("#### Retrieved Contexts (Top 3)")
                retrieved = row['retrieved_contexts']
                retrieved_files = row.get('retrieved_file', [])
                
                # Check formatting
                if isinstance(retrieved, list):
                    for i, txt in enumerate(retrieved):
                        # File-name based match
                        source_file = str(row.get('source_file', "")).strip()
                        uri = ""
                        if isinstance(retrieved_files, list) and i < len(retrieved_files):
                            uri = str(retrieved_files[i])
                        is_match = bool(source_file and uri and source_file in uri)
                        
                        color = "green" if is_match else "grey"
                        
                        with st.container(border=True):
                            if is_match:
                                st.markdown(f"**Rank {i+1} ✅**")
                            else:
                                st.markdown(f"**Rank {i+1}**")
                            st.text(txt)
                else:
                    st.write("No retrieval data.")
        else:
            st.info("Select a row in the table above to view details.")

if __name__ == "__main__":
    main()
