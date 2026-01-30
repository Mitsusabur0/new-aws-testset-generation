# evaluator.py
import pandas as pd
import numpy as np
import ast
import config

def calculate_metrics(row):
    # Load lists (handling string conversion from CSV)
    gt_list = row['reference_contexts']
    retrieved_list = row['retrieved_contexts']
    
    if isinstance(gt_list, str): gt_list = ast.literal_eval(gt_list)
    if isinstance(retrieved_list, str): retrieved_list = ast.literal_eval(retrieved_list)
    
    # We assume 1 ground truth chunk for this pipeline
    gt_text = gt_list[0] if gt_list else ""
    
    hit = False
    rank = 0
    
    # Check for containment
    # Logic: Is the ground truth substring roughly contained in the retrieved chunk?
    # Or is the retrieved chunk contained in the ground truth (if chunks are small)?
    for i, ret_text in enumerate(retrieved_list):
        # Normalize for comparison
        clean_gt = " ".join(gt_text.lower().split())
        clean_ret = " ".join(ret_text.lower().split())
        
        if clean_gt in clean_ret or clean_ret in clean_gt:
            hit = True
            rank = i + 1
            break
    
    # Metrics
    hit_rate = 1 if hit else 0
    mrr = 1.0 / rank if hit else 0.0
    
    # Precision@K: (Relevant Items in Top K) / K
    precision = (1 / config.EVAL_K) if hit else 0
    
    # Recall@K: (Relevant Items in Top K) / Total Relevant Items
    # Total Relevant is 1 in this synthetic setup
    recall = 1 if hit else 0
    
    return pd.Series([hit_rate, mrr, precision, recall])

def main():
    print(f"Loading {config.OUTPUT_EVALSET_CSV}...")
    try:
        df = pd.read_csv(config.OUTPUT_EVALSET_CSV)
    except FileNotFoundError:
        print("Input file not found. Run File 2 first.")
        return

    print("Calculating metrics...")
    
    metrics_df = df.apply(calculate_metrics, axis=1)
    metrics_df.columns = ['hit_rate', 'mrr', 'precision_at_k', 'recall_at_k']
    
    final_df = pd.concat([df, metrics_df], axis=1)
    
    # Parse lists back to actual python objects for Parquet saving
    # (Parquet handles lists natively, unlike CSV)
    final_df['reference_contexts'] = final_df['reference_contexts'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    final_df['retrieved_contexts'] = final_df['retrieved_contexts'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    final_df.to_parquet(config.OUTPUT_RESULTS_PARQUET, index=False)
    print(f"Evaluation complete. Results saved to {config.OUTPUT_RESULTS_PARQUET}")

if __name__ == "__main__":
    main()
