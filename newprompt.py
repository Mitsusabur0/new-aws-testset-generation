prompt = f"""
### ROL Y OBJETIVO
Actúa como un simulador de usuarios chilenos reales para el sitio inmobiliario de Banco Estado (Casaverso).
Tu tarea es generar un par de prueba sintético (Pregunta de Usuario + Respuesta Esperada) basado ESTRUCTAMENTE en el "Texto de Referencia" proporcionado.

### INPUTS DE SIMULACIÓN
1. **Perfil del Usuario:** {persona['desc']}
2. **Estilo de Consulta:** {style}

### GUÍA DE EJEMPLOS (FEW-SHOT)
Usa los siguientes ejemplos SOLAMENTE como guía de tono, vocabulario y estructura.
ADVERTENCIA: NO uses la información/contenido de estos ejemplos. Usa solo la información del texto nuevo.

{FEW_SHOT_EXAMPLES}

---

### INSTRUCCIONES DE GENERACIÓN
1. **Análisis de Contenido:** Lee el "Texto de Referencia" actual. Si es irrelevante (índices, pies de página, código, legal sin valor), responde con un JSON indicando "SKIP".
2. **Adopción de Persona:** Asume la personalidad descrita.
   - Si el tema del texto no encaja perfectamente con la persona (ej: un joven leyendo sobre pensiones), formula la pregunta como si el usuario tuviera curiosidad general o preguntara por un familiar, pero MANTENIENDO su vocabulario y estilo.
3. **Formulación de la Pregunta:**
   - La pregunta debe ser respondible **únicamente** con el texto provisto.
   - Usa modismos chilenos si el perfil lo dicta (UF, pie, depa, lucas, carnet).
   - Aplica rigurosamente el "Estilo de Consulta" (si es "Buscador", sé breve; si es "Narrativo", inventa una situación con cifras).
4. **Formulación de la Respuesta (Ground Truth):**
   - Redacta la respuesta correcta que un agente ideal daría, basada SOLO en el texto.
   - Si el texto no tiene la respuesta completa, la respuesta debe decir explícitamente qué falta.

### FORMATO DE SALIDA (OBLIGATORIO)
Debes responder ÚNICAMENTE con un bloque JSON válido. Sin texto antes ni después.

Estructura JSON requerida:
{{
    "user_question": "La pregunta generada según el perfil y estilo",
    "ground_truth_answer": "La respuesta precisa extraída del texto",
    "reasoning": "Breve explicación de por qué esta pregunta encaja con el texto y el perfil"
}}

Si el texto no sirve, responde:
{{
    "user_question": "SKIP",
    "ground_truth_answer": "N/A",
    "reasoning": "Texto irrelevante o sin contenido informativo"
}}

---

### TEXTO DE REFERENCIA (CONTEXTO REAL):
{chunk_text}
"""