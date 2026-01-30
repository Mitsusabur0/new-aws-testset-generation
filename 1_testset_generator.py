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

# PERSONAS = [
#     {
#         "name": "El Primerizo Ahorrativo",
#         "desc": "Eres un joven chileno (25-30 años) que quiere comprar su primera vivienda pero tiene pocos ahorros. Tu principal preocupación es el 'Pie' y si puedes usar subsidios (DS1, DS19, FOGAES). No entiendes mucho de términos financieros complejos. Usas un lenguaje cercano, a veces coloquial ('depa', 'luca', 'bacán'), y sueles preguntar '¿qué necesito?' o '¿cómo lo hago?'."
#     },
#     {
#         "name": "El Trabajador Independiente",
#         "desc": "Eres un profesional que trabaja emitiendo boletas de honorarios (no tienes contrato indefinido). Tu mayor ansiedad es saber si el banco te prestará dinero a pesar de no ser empleado dependiente. Preguntas sobre 'castigos' al sueldo, antigüedad laboral requerida, requisitos para independientes y cómo demostrar ingresos siendo freelance."
#     },
#     {
#         "name": "El Planificador Familiar",
#         "desc": "Estás buscando comprar en conjunto. Tu foco principal es el 'Complemento de Renta'. Necesitas saber con quién puedes sumar sueldo (pareja sin casarse, padres, hermanos, amigos), qué requisitos piden al codeudor y a nombre de quién queda la propiedad. Eres pragmático y buscas maximizar el monto del crédito sumando ingresos."
#     },
#     {
#         "name": "El Inversionista Informado",
#         "desc": "Ya tienes una propiedad y entiendes cómo funciona el sistema. Buscas una segunda vivienda para inversión o quieres mejorar tus condiciones actuales. Preguntas por 'Refinanciamiento', 'Portabilidad', 'Tasa de interés' actual, 'Mutuarias' vs Bancos, y rentabilidad. Tu tono es directo, serio y financiero. No necesitas explicaciones básicas."
#     },
#     {
#         "name": "El Adulto Mayor Preocupado",
#         "desc": "Eres una persona mayor (55+ años) o jubilado. Tu preocupación es la edad límite para solicitar un hipotecario. Preguntas hasta qué edad te prestan, si los seguros son más caros, y por plazos de pago más cortos. Eres muy educado, formal y usas frases como 'Estimados', 'Quisiera consultar', 'Agradecería orientación'."
#     }
# ]

# Query styles in Spanish
# QUERY_STYLES = [
#     "Estilo 'Buscador (Keyword)': Frases cortas, telegráficas y sin conectores gramaticales. Como si el usuario estuviera escribiendo en la barra de Google. Ejemplo: 'requisitos pie', 'tasa interes hoy', 'simular dividendo'.",
    
#     "Estilo 'Narrativo con Cifras': El usuario plantea un escenario detallado incluyendo datos numéricos hipotéticos (sueldo líquido, monto ahorrado, valor de la propiedad) para validar si su caso específico califica. Ejemplo: 'Si gano 1.5 millones y tengo 10 millones de pie, ¿puedo comprar?'",
    
#     "Estilo 'Conversacional Informal': Redacción completa, natural y relajada. Incluye saludos ('Hola Vivi', 'Buenas'), tuteo, y modismos chilenos suaves ('al tiro', 'cachai', 'brígido'). Trata al asistente como a una persona.",
    
#     "Estilo 'Principiante Confundido': Preguntas abiertas, vagas o de solicitud de orientación general. El usuario admite explícitamente no saber por dónde empezar o no entender la terminología. Ejemplo: 'No entiendo nada, explícame cómo funciona lo del subsidio'.",
    
#     "Estilo 'Imperativo/Directo': El usuario va directo al grano, sin saludos ni rodeos. Busca un dato concreto, una confirmación de sí/no, o una acción inmediata. Ejemplo: 'Dime la tasa actual', '¿Se puede complementar renta con amigos? Sí o no'."
# ]

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

FEW_SHOT_EXAMPLES = """
EJEMPLOS DE ENTRENAMIENTO (FEW-SHOT):

---
Ejemplo 1:
[Contexto]: "Para trabajadores independientes, se requiere un mínimo de 1 año de antigüedad en el giro y las últimas 6 boletas de honorarios continuas. El castigo al ingreso es del 30%."
[Estilo]: Narrativo con Cifras
[Generación]: "Hola, mira yo boleteo hace 2 años y gano 1.2 millones bruto. Si me castigan el 30%, ¿cuánto sueldo me consideran para el banco? ¿Me sirve?"

---
Ejemplo 2:
[Contexto]: "El Subsidio DS19 es un beneficio automático que se aplica directamente al valor de la propiedad. No requiere postulación previa en el MINVU, solo cumplir los requisitos del crédito."
[Estilo]: Conversacional Informal
[Generación]: "Oye Vivi, una consulta. Eso del subsidio ds19 qué es? Porque no cacho mucho."

---
Ejemplo 3:
[Contexto]: "La Tasa Mixta ofrece un interés fijo por los primeros 5 años y luego variable por el resto del periodo. Tasa actual referencial: 4,5% anual."
[Estilo]: Imperativo/Directo
[Generación]: "Dime la tasa fija actual para un crédito a 20 años. ¿Tienen opciones mixtas?"

---
Ejemplo 4:
[Contexto]: "Podrán complementar renta los cónyuges, parejas con unión civil o parejas de hecho con hijos en común. También se permite padres con hijos."
[Estilo]: Narrativo con Cifras
[Generación]: "Somos una pareja, no estamos casados ni tenemos hijos, pero entre los dos sumamos 2 millones de pesos. ¿Podemos complementar renta para comprar la casa o nos van a rechazar?"

---
Ejemplo 5:
[Contexto]: "Requisitos: Ser chileno o extranjero con residencia definitiva. Tener buenos antecedentes comerciales (Sin DICOM)."
[Estilo]: Conversacional Informal
[Generación]: "Pucha, si estuve en Dicom el año pasado pero ya pagué todo y estoy limpio, ¿me dan el crédito igual o sigo castigado?"

---
Ejemplo 6:
[Contexto]: "Financiamiento máximo del 90% para primera vivienda. El cliente debe contar con el 10% de pie al momento de la escritura."
[Estilo]: Buscador (Keyword)
[Generación]: "credito hipotecario sin pie 10%"

---
Ejemplo 7:
[Contexto]: "Seguro de Desgravamen: Cubre el saldo insoluto de la deuda en caso de fallecimiento del titular. La prima aumenta con la edad del asegurado. Edad máxima de ingreso: 70 años."
[Estilo]: Conversacional Informal (Formal)
[Generación]: "Estimados, muy buenas tardes. Tengo 68 años y quisiera saber si todavía estoy a tiempo de pedir un préstamo o si el seguro me va a salir muy caro por la edad."

---
Ejemplo 8:
[Contexto]: "Glosario de Términos. UF: Unidad de Fomento. CAE: Carga Anual Equivalente. CTC: Costo Total del Crédito."
[Estilo]: Conversacional Informal
[Generación]: "No entiendo nada de las siglas. ¿Qué es eso del CAE que sale en la simulación? ¿Es lo que pago al final?"

---
Ejemplo 9:
[Contexto]: "Simulación para propiedad de 2.500 UF. Pie del 20% (500 UF). Monto del crédito: 2.000 UF. Dividendo estimado a 30 años: UF 9,5."
[Estilo]: Narrativo con Cifras
[Generación]: "Si la casa cuesta 2500 UF y yo solo tengo para el 10% del pie, ¿en cuánto me quedaría la cuota mensual más o menos?"

"""



def get_bedrock_client():
    session = boto3.Session(profile_name=AWS_PROFILE)
    return session.client(service_name="bedrock-runtime", region_name=AWS_REGION)

def clean_llm_output(text):
    """Removes <reasoning> tags and their content from the LLM output."""
    # The flag re.DOTALL makes '.' match newlines as well
    cleaned_text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def generate_question_only(chunk_text, style, client):
    """
    Sends Context + Style to LLM.
    MUST create a realistic user question based on the context.
    The following is just a placeholder
    """
    
    
    prompt = f"""
    ### ROL Y OBJETIVO
    Actúa como un simulador de usuarios chilenos reales de un asistente IA de un sitio web inmobiliario bancario.
    Estamos haciendo un testset para evaluar el retrieval del sistema RAG del asistente IA.
    Tu tarea es generar una Pregunta de Usuario basado ESTRICTAMENTE en el "Texto de Referencia" proporcionado abajo.


    ### INPUTS DE SIMULACIÓN
    **Estilo de Consulta:** {style}

    ### GUÍA DE EJEMPLOS (FEW-SHOT)
    Usa los siguientes ejemplos SOLAMENTE como guía de tono, vocabulario y estructura.
    ADVERTENCIA: NO uses la información/contenido de estos ejemplos. Usa SOLO LA INFORMACIÓN DEL TEXTO DE REFERENCIA para generar la respuesta.

    {FEW_SHOT_EXAMPLES}

    ---

    ### INSTRUCCIONES DE GENERACIÓN
    1. **Análisis de Contenido:** Lee el "Texto de Referencia" actual. 
    2. **Formulación de la Pregunta:**
    - La pregunta debe ser respondible **ÚNICAMENTE** con el texto provisto.
    - Usa modismos chilenos si el perfil lo dicta (UF, pie, depa, lucas, carnet).
    - Aplica el "Estilo de Consulta" (si es "Buscador", sé breve; si es "Narrativo", inventa una situación con cifras).

    ### FORMATO DE SALIDA (OBLIGATORIO)
    Debes responder ÚNICAMENTE con la Pregunta de Usuario generada, sin explicaciones ni comentarios adicionales. 
    Tu output es SÓLO el input real que el usuario le daría al asistente virtual, NADA MÁS.
    Las preguntas NO DEBEN ser muy extensas. Los ususarios reales tienden a ser concisos y no muy detallados. 

    ---

    ### TEXTO DE REFERENCIA (CONTEXTO REAL que responde la pregunta):
    {chunk_text}
    """


    # Payload structure
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": 1000 
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
                # selected_persona = random.choice(PERSONAS)
                selected_style = random.choice(QUERY_STYLES)
                
                print(f"[{i+1}/{len(files)}] Processing {os.path.basename(file_path)} | Style: {selected_style['style_name']}")
                
                # --- CALL LLM FOR USER INPUT ONLY ---
                generated_question = generate_question_only(chunk_text, selected_style, client)
                
                if generated_question:
                    # --- CONSTRUCT ROW PROGRAMMATICALLY ---
                    row = {
                        "user_input": generated_question,
                        "reference_contexts": [chunk_text], 
                        "query_style": selected_style["style_name"]
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