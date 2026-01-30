import os
import json
import random
import glob
import re  
import pandas as pd
import boto3
from botocore.exceptions import ClientError

# --- CONFIGURATION ---
# KB_FOLDER = "./testfolder"
KB_FOLDER = "./kb_nuevo_pipeline"
OUTPUT_FILE = "testset.csv"

# AWS Config
AWS_PROFILE = "default"
AWS_REGION = "us-east-2"

# LLM Config
MODEL_ID = "openai.gpt-oss-120b-1:0"
TEMPERATURE = 0.7 
# LLM PRICING PER 1K TOKENS
INPUT_PRICE = 0.00015
OUTPUT_PRICE = 0.0006

# --- CHILEAN BANKING CONTEXT CONFIGURATION ---

QUERY_STYLES = [
    {
        "style_name": "Buscador de Palabras Clave",
        "description": "El usuario no redacta una oración completa. Escribe fragmentos sueltos, como si estuviera buscando en Google. Ejemplo: 'requisitos pie', 'seguro desgravamen edad', 'renta minima postulacion'."
    },
    {
        "style_name": "Caso Hipotético en Primera Persona",
        "description": "El usuario plantea una situación personal (real o inventada) que incluye cifras o condiciones específicas para ver si el texto se aplica a él. Usa estructuras como 'Si yo tengo...', 'En caso de que gane...', '¿Qué pasa si...?'."
    },
    {
        "style_name": "Duda Directa sobre Restricciones",
        "description": "El usuario busca la 'letra chica', los límites o los impedimentos. Pregunta específicamente por lo que NO se puede hacer, los castigos, o los máximos/mínimos. Tono serio y pragmático."
    },
    {
        "style_name": "Colloquial Chileno Natural",
        "description": "Redacción relajada, usando modismos locales suaves y un tono de conversación por chat (WhatsApp). Usa términos como 'depa', 'lucas', 'chao', 'consulta', 'al tiro'. Trata al asistente con cercanía."
    },
    {
        "style_name": "Principiante / Educativo",
        "description": "El usuario admite no saber del tema y pide definiciones o explicaciones de conceptos básicos mencionados en el texto. Pregunta '¿Qué significa...?', '¿Cómo funciona...?', 'Explícame eso de...'."
    },
    {
        "style_name": "Orientado a la Acción",
        "description": "El usuario quiere saber el 'cómo' operativo. Pregunta por pasos a seguir, documentos a llevar, lugares dónde ir o botones que apretar. Ejemplo: '¿Dónde mando los papeles?', '¿Cómo activo esto?', '¿Con quién hablo?'."
    }
]



def get_bedrock_client():
    session = boto3.Session(profile_name=AWS_PROFILE)
    return session.client(service_name="bedrock-runtime", region_name=AWS_REGION)

def clean_llm_output(text):
    """Removes <reasoning> tags and their content from the LLM output."""
    # The flag re.DOTALL makes '.' match newlines as well
    cleaned_text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def generate_question_only(chunk_text, query_styles, client):    
    
    system_prompt = f"""
### ROL DEL SISTEMA
Eres un Generador de Datos Sintéticos especializado en Banca y Bienes Raíces de Chile.
Tu trabajo es crear el "Test Set" para evaluar un asistente de IA (RAG) del Banco Estado (Casaverso).

### TAREA PRINCIPAL
Se te entregará un fragmento de texto (La "Respuesta").
Tu objetivo es redactar la **Consulta del Usuario** (El "Input") que provocaría que el sistema recupere este texto como respuesta.

### REGLAS DE ORO (CRÍTICO: LEER CON ATENCIÓN)
1. **ASIMETRÍA DE INFORMACIÓN:** El usuario NO ha leído el texto. No sabe los términos técnicos exactos, ni los porcentajes, ni los artículos de la ley que aparecen en el texto.
2. **INTENCIÓN vs CONTENIDO:**
- MAL (Contaminado): "¿Cuáles son los requisitos del artículo 5 del subsidio DS19?" (El usuario no sabe que existe el artículo 5).
- BIEN (Realista): "Oye, ¿qué papeles me piden para postular al subsidio?"
3. **ABSTRACCIÓN:** Si el texto habla de "Tasa fija del 4.5%", el usuario NO pregunta "¿Es la tasa del 4.5%?". El usuario pregunta "¿Cómo están las tasas hoy?".
4. **SI EL TEXTO ES CORTO/PARCIAL:** Si el fragmento es muy específico o técnico, el usuario debe hacer una pregunta más amplia o vaga que este fragmento respondería parcialmente.
5. **CONTEXTO CHILENO:** Usa vocabulario local, modismos y el tono correspondiente al estilo solicitado.


### DOCUMENTO DE REFERENCIA:
Se te etregará un fragmento de texto que el asistente debería recuperar como respuesta a la consulta del usuario.

### ESTILOS DE CONSULTA DISPONIBLES:
Se te entregará una lista de 3 estilos. Debes seleccionar el estilo que permita más fácilmente crear una consulta realista en rol de usuario al documento específico entregado. 
Luego, debes redactar la consulta adoptando el estilo seleccionado.

### FORMATO DE SALIDA
Tu respuesta serán dos tags xml: <style_name> y <user_input>.
El texto dentro de style_name es el nombre del estilo seleccionado. Debes mantener el MISMO style_name entregado.
El texto dentro del <user_input> debe ser la consulta generada, sin comillas, sin saltos de línea, sin explicaciones adicionales. Es el texto plano de la consulta del usuario.
Responde ÚNICAMENTE con este formato XML (sin markdown, sin explicaciones):

<style_name>NOMBRE_DEL_ESTILO</style_name>
<user_input>TU_CONSULTA_GENERADA_AQUI</user_input>
"""

    prompt = f"""
### DOCUMENTO DE REFERENCIA:
{chunk_text}    
### ESTILOS DE CONSULTA DISPONIBLES:
{query_styles}
"""

    # Payload structure
    body = json.dumps({
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": 2000 
    })

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=body
        )
        response_body = json.loads(response.get('body').read().decode('utf-8'))
        
        # Parsing logic
        # CHECK RESPONSE AND ADJUST ACCORDINGLY
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
    print(f"Scanning for files in {KB_FOLDER}...")
    
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
                if len(chunk_text) < 30: 
                    continue
                
                # --- PROGRAMMATIC SELECTION ---
                #  We must change this so that it selects 3 random styles and passes them to the LLM
                # THIS IS WHERE WE MUST PUT THE LOGIC TO SELECT 3 STYLES RANDOMLY
                # selected_styles = ???
                
                print(f"[{i+1}/{len(files)}] Processing {os.path.basename(file_path)}")
                
                # --- CALL LLM FOR USER INPUT ONLY ---
                generated_question = generate_question_only(chunk_text, selected_styles, client)
                
                if generated_question:
                    # --- CONSTRUCT ROW PROGRAMMATICALLY ---
                    row = {
                        "user_input": generated_question,
                        "reference_contexts": [chunk_text], 
                        "query_style": selected_styles[0]["style_name"]
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