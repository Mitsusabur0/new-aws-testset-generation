# retriever.py
import pandas as pd
import boto3
import ast
from botocore.exceptions import ClientError

# --- CONFIGURATION ---
INPUT_FILE = "testset.csv"
OUTPUT_FILE = "evaluation_set.csv"

# AWS Config
AWS_PROFILE = "sandbox"
AWS_REGION = "us-east-1"
SERVICE = "bedrock-agent-runtime"
KB_ID = "3TPM53DPBN"
TOP_K = 3

def get_runtime_client():
    session = boto3.Session(profile_name=AWS_PROFILE)
    return session.client(service_name=SERVICE, region_name=AWS_REGION)

def clean_text(text):
    """Helper to clean retrieved text for better comparison."""
    if not text:
        return ""
    # Remove excessive whitespace, newlines, etc.
    return " ".join(text.split())

def retrieve_contexts(query, client):
    try:
        response = client.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={
                'text': query
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': TOP_K
                }
            }
        )
        
        results = response.get('retrievalResults', [])
        # Extract text content
        retrieved_texts = [clean_text(res['content']['text']) for res in results]
        return retrieved_texts

    except ClientError as e:
        print(f"Retrieval Error for query '{query}': {e}")
        return []

def main():
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Input file not found. Run File 1 first.")
        return

    # Ensure reference_contexts is read as a list, not a string representation of a list
    df['reference_contexts'] = df['reference_contexts'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    client = get_runtime_client()
    
    print("Starting retrieval process...")
    retrieved_data = []
    
    for index, row in df.iterrows():
        query = row['user_input']
        print(f"[{index+1}/{len(df)}] Retrieving: {query[:30]}...")
        
        contexts = retrieve_contexts(query, client)
        retrieved_data.append(contexts)

    df['retrieved_contexts'] = retrieved_data
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Retrieval complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()