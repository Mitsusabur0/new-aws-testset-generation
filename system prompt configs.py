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
Tu output es SÓLO el input real que el usuario le daría al asistente virtual, NADA MÁS.

---

### TEXTO DE REFERENCIA (CONTEXTO REAL):
{chunk_text}
"""




FEW_SHOT_EXAMPLES = """
EJEMPLOS DE ENTRENAMIENTO (FEW-SHOT):

---
Ejemplo 1:
[Contexto]: "Para trabajadores independientes, se requiere un mínimo de 1 año de antigüedad en el giro y las últimas 6 boletas de honorarios continuas. El castigo al ingreso es del 30%."
# [Persona]: El Trabajador Independiente
[Estilo]: Narrativo con Cifras
[Generación]: "Hola, mira yo boleteo hace 2 años y gano 1.2 millones bruto. Si me castigan el 30%, ¿cuánto sueldo me consideran para el banco? ¿Me sirve?"

---
Ejemplo 2:
[Contexto]: "El Subsidio DS19 es un beneficio automático que se aplica directamente al valor de la propiedad. No requiere postulación previa en el MINVU, solo cumplir los requisitos del crédito."
[Persona]: El Primerizo Ahorrativo
[Estilo]: Conversacional Informal
[Generación]: "Oye Vivi, una consulta. Eso del subsidio ds19 qué es? Porque no cacho mucho."

---
Ejemplo 3:
[Contexto]: "La Tasa Mixta ofrece un interés fijo por los primeros 5 años y luego variable por el resto del periodo. Tasa actual referencial: 4,5% anual."
[Persona]: El Inversionista Informado
[Estilo]: Imperativo/Directo
[Generación]: "Dime la tasa fija actual para un crédito a 20 años. ¿Tienen opciones mixtas?"

---
Ejemplo 4:
[Contexto]: "Podrán complementar renta los cónyuges, parejas con unión civil o parejas de hecho con hijos en común. También se permite padres con hijos."
[Persona]: El Planificador Familiar
[Estilo]: Narrativo con Cifras
[Generación]: "Somos una pareja, no estamos casados ni tenemos hijos, pero entre los dos sumamos 2 millones de pesos. ¿Podemos complementar renta para comprar la casa o nos van a rechazar?"

---
Ejemplo 5:
[Contexto]: "Requisitos: Ser chileno o extranjero con residencia definitiva. Tener buenos antecedentes comerciales (Sin DICOM)."
[Persona]: El Trabajador Independiente (Variante Ansioso)
[Estilo]: Conversacional Informal
[Generación]: "Pucha, si estuve en Dicom el año pasado pero ya pagué todo y estoy limpio, ¿me dan el crédito igual o sigo castigado?"

---
Ejemplo 6:
[Contexto]: "Financiamiento máximo del 90% para primera vivienda. El cliente debe contar con el 10% de pie al momento de la escritura."
[Persona]: El Primerizo Ahorrativo
[Estilo]: Buscador (Keyword)
[Generación]: "credito hipotecario sin pie 10%"

---
Ejemplo 7:
[Contexto]: "Seguro de Desgravamen: Cubre el saldo insoluto de la deuda en caso de fallecimiento del titular. La prima aumenta con la edad del asegurado. Edad máxima de ingreso: 70 años."
[Persona]: El Adulto Mayor Preocupado
[Estilo]: Conversacional Informal (Formal)
[Generación]: "Estimados, muy buenas tardes. Tengo 68 años y quisiera saber si todavía estoy a tiempo de pedir un préstamo o si el seguro me va a salir muy caro por la edad."

---
Ejemplo 8:
[Contexto]: "Glosario de Términos. UF: Unidad de Fomento. CAE: Carga Anual Equivalente. CTC: Costo Total del Crédito."
[Persona]: El Principiante Confundido
[Estilo]: Conversacional Informal
[Generación]: "No entiendo nada de las siglas. ¿Qué es eso del CAE que sale en la simulación? ¿Es lo que pago al final?"

---
Ejemplo 9:
[Contexto]: "Simulación para propiedad de 2.500 UF. Pie del 20% (500 UF). Monto del crédito: 2.000 UF. Dividendo estimado a 30 años: UF 9,5."
[Persona]: El Primerizo Ahorrativo
[Estilo]: Narrativo con Cifras
[Generación]: "Si la casa cuesta 2500 UF y yo solo tengo para el 10% del pie, ¿en cuánto me quedaría la cuota mensual más o menos?"

"""