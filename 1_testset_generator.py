# testset_generator.py
import os
import json
import random
import glob
import re  # Added for Regex
import pandas as pd
import boto3
from botocore.exceptions import ClientError

# --- CONFIGURATION ---
KB_FOLDER = "./kb_nuevo_pipeline"
OUTPUT_FILE = "testset.csv"

# AWS Config
AWS_PROFILE = "default"
AWS_REGION = "us-east-2"
SERVICE = "bedrock-runtime"
MODEL_ID = "openai.gpt-oss-120b-1:0"
TEMPERATURE = 0.7 

# --- CHILEAN BANKING CONTEXT CONFIGURATION ---

# Personas adapted for Banco Estado / Chilean Financial context
PERSONAS = [
    {
        "name": "Cliente CuentaRUT",
        "desc": "Un usuario promedio de Chile. Usa lenguaje coloquial pero respetuoso. Le preocupan las comisiones, si le cobran por girar en Caja Vecina y la clave de internet."
    },
    {
        "name": "Micro-Emprendedor (Pyme)",
        "desc": "Dueño de un almacén o negocio local. Habla de capital de trabajo, créditos, tasas de interés y cómo pagar a proveedores."
    },
    {
        "name": "Pensionado / Adulto Mayor",
        "desc": "Usuario mayor. No entiende mucho de tecnología. Pregunta con desconfianza o buscando seguridad sobre su pensión y atención presencial."
    },
    {
        "name": "Estudiante Universitario",
        "desc": "Joven. Pregunta directo y al grano. Le interesa la App, pagar Spotify/Netflix, beneficios de la tarjeta Joven y compras online."
    }
]

# Query styles in Spanish
QUERY_STYLES = [
    "Directo (Pregunta corta y precisa)", 
    "Basado en un escenario ('Si me pasa esto, ¿qué hago?')", 
    "Comparativo ('¿Es mejor esto o aquello?')", 
    "Confundido ('No entiendo cómo funciona X')"
]

def get_bedrock_client():
    session = boto3.Session(profile_name=AWS_PROFILE)
    return session.client(service_name=SERVICE, region_name=AWS_REGION)

def clean_llm_output(text):
    """Removes <reasoning> tags and their content from the LLM output."""
    # The flag re.DOTALL makes '.' match newlines as well
    cleaned_text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def generate_question_only(chunk_text, persona, style, client):
    """
    Sends Context + Persona + Style to LLM.
    Returns ONLY the generated question string, stripped of reasoning.
    """
    
    prompt = f"""
    Instrucciones:
    Actúa como el siguiente usuario de Banco Estado (Chile): "{persona['name']}".
    Descripción del usuario: {persona['desc']}
    
    Tu tarea es leer el siguiente texto oficial del banco y formular UNA sola pregunta que este usuario haría, usando el estilo: "{style}".
    
    Texto Oficial (Contexto):
    ---
    {chunk_text}
    ---
    
    Requisitos:
    1. La pregunta debe estar en Español (Chile).
    2. La pregunta debe estar basada SOLAMENTE en la información del Texto Oficial.
    3. NO respondas la pregunta.
    4. NO incluyas encabezados, ni comillas, ni texto extra. Solo devuelve la pregunta.
    """

    # Payload structure
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": 500 # Increased slightly to allow for the reasoning step
    })

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=body
        )
        response_body = json.loads(response.get('body').read())
        
        # Parsing logic
        if 'choices' in response_body:
            content = response_body['choices'][0]['message']['content']
        elif 'output' in response_body:
            content = response_body['output']['message']['content']
        else:
            content = str(response_body)

        # --- CLEAN THE OUTPUT ---
        # 1. Remove reasoning tags
        content_no_reasoning = clean_llm_output(content)
        # 2. Remove any remaining quotes or newlines
        question_text = content_no_reasoning.replace('"', '').replace('\n', ' ').strip()
        
        return question_text

    except ClientError as e:
        print(f"AWS Error: {e}")
        return None
    except Exception as e:
        print(f"General Error: {e}")
        return None

def main():
    print(f"Scanning for files in {KB_FOLDER} (Recursive)...")
    
    if not os.path.exists(KB_FOLDER):
        print(f"Error: Directory {KB_FOLDER} does not exist.")
        return

    # Recursive glob search for .md files
    search_pattern = os.path.join(KB_FOLDER, "**", "*.md")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"No .md files found in {KB_FOLDER} or its subdirectories.")
        return

    print(f"Found {len(files)} Markdown files.")

    client = get_bedrock_client()
    dataset = []

    print("Generating synthetic questions...")

    for i, file_path in enumerate(files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                chunk_text = f.read()
                
                # Skip empty files
                if len(chunk_text) < 50: 
                    continue
                
                # --- PROGRAMMATIC SELECTION ---
                selected_persona = random.choice(PERSONAS)
                selected_style = random.choice(QUERY_STYLES)
                
                print(f"[{i+1}/{len(files)}] Processing {os.path.basename(file_path)} | Persona: {selected_persona['name']}")
                
                # --- CALL LLM FOR USER INPUT ONLY ---
                generated_question = generate_question_only(chunk_text, selected_persona, selected_style, client)
                
                if generated_question:
                    # --- CONSTRUCT ROW PROGRAMMATICALLY ---
                    row = {
                        "user_input": generated_question,
                        "reference_contexts": [chunk_text], 
                        "persona_name": selected_persona['name'],
                        "query_style": selected_style
                    }
                    dataset.append(row)
                    
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    if dataset:
        df = pd.DataFrame(dataset)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Successfully generated {len(df)} test cases. Saved to {OUTPUT_FILE}")
    else:
        print("No data generated.")

if __name__ == "__main__":
    main()