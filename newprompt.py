prompt = f"""
### ROL Y OBJETIVO
Actúa como un simulador de usuarios chilenos reales de un asistente IA de un sitio web inmobiliario bancario.
Estamos haciendo un testset para evaluar el retrieval del sistema RAG del asistente IA.
Tu tarea es generar una Pregunta de Usuario basado ESTRICTAMENTE en el "Texto de Referencia" proporcionado.


### INPUTS DE SIMULACIÓN
1. **Perfil del Usuario:** {persona['desc']}
2. **Estilo de Consulta:** {style}

### GUÍA DE EJEMPLOS (FEW-SHOT)
Usa los siguientes ejemplos SOLAMENTE como guía de tono, vocabulario y estructura.
ADVERTENCIA: NO uses la información/contenido de estos ejemplos. Usa solo la información del texto nuevo.

{FEW_SHOT_EXAMPLES}

---

### INSTRUCCIONES DE GENERACIÓN
1. **Análisis de Contenido:** Lee el "Texto de Referencia" actual. 
2. **Adopción de Persona:** Asume la personalidad descrita.
   - Si el tema del texto no encaja perfectamente con la persona (ej: un joven leyendo sobre pensiones), formula la pregunta como si el usuario tuviera curiosidad general o preguntara por un familiar, pero MANTENIENDO su vocabulario y estilo.
3. **Formulación de la Pregunta:**
   - La pregunta debe ser respondible **únicamente** con el texto provisto.
   - Usa modismos chilenos si el perfil lo dicta (UF, pie, depa, lucas, carnet).
   - Aplica el "Estilo de Consulta" (si es "Buscador", sé breve; si es "Narrativo", inventa una situación con cifras).

### FORMATO DE SALIDA (OBLIGATORIO)
Debes responder ÚNICAMENTE con la Pregunta de Usuario generada, sin explicaciones ni comentarios adicionales. 
Tu output es SÓLO el input real que el usuario le daría al asistente virtual.

---

### TEXTO DE REFERENCIA (CONTEXTO REAL):
{chunk_text}
"""