# 🔍 AI-Powered CV Screener

Sistema RAG (Retrieval-Augmented Generation) para screening inteligente de CVs usando LlamaIndex y Streamlit.

## 📋 Características

- ✅ Generación automática de 30 CVs realistas en PDF
- ✅ Pipeline RAG con LlamaIndex para búsqueda semántica
- ✅ Interface de chat moderna con Streamlit
- ✅ Indicación de fuentes y scoring de relevancia
- ✅ Embeddings persistidos (no necesita regenerar)

## 🚀 Instalación Rápida

### 1. Clonar y Setup

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Configurar API Keys

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar .env y añadir tu API key de OpenAI
# Obtén una gratis en: https://platform.openai.com/api-keys
```

### 3. Generar CVs

```bash
python generate_cvs.py
```

Esto creará 30 CVs en la carpeta `cvs/`. Toma ~5-10 minutos porque usa LLM para generar descripciones realistas.

### 4. Ejecutar la App

```bash
streamlit run app.py
```

La app se abrirá en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
cv-screener/
├── app.py                  # Streamlit UI
├── rag_pipeline.py         # RAG con LlamaIndex
├── generate_cvs.py         # Generador de CVs
├── requirements.txt        # Dependencias
├── .env                    # API keys (no commitear)
├── .streamlit/
│   └── config.toml        # Theme config
├── cvs/                   # 30 CVs generados (PDF)
└── data/                  # Vector store (generado automáticamente)
    └── vector_store/
```

## 🎯 Uso

### Preguntas de Ejemplo

- "Who has Python experience?"
- "Find candidates with machine learning skills"
- "Which candidates worked at Google?"
- "Who graduated from MIT or Stanford?"
- "Show me Full Stack Developers with React experience"
- "Summarize the profile of [candidate name]"

### Funcionalidades

1. **Chat Interface**: Haz preguntas en lenguaje natural
2. **Sources**: Cada respuesta muestra qué CVs usó y su relevancia
3. **Persistent Index**: El vector store se guarda, solo se crea una vez
4. **Example Questions**: Sidebar con preguntas predefinidas

## 🛠️ Tecnologías Usadas

- **Frontend**: Streamlit + CSS custom
- **RAG Pipeline**: LlamaIndex
- **LLM**: OpenAI GPT-3.5-turbo
- **Embeddings**: OpenAI text-embedding-ada-002
- **Vector Store**: LlamaIndex (local, persistente)
- **PDF Generation**: ReportLab + Faker

## 🧪 Testing

```bash
# Test del RAG pipeline
python rag_pipeline.py
```

Esto ejecutará queries de prueba y mostrará los resultados en consola.

## 📊 Decisiones Técnicas

### ¿Por qué LlamaIndex?

- ✅ Abstracción high-level para RAG
- ✅ Manejo automático de chunking y embeddings
- ✅ Vector store persistente out-of-the-box
- ✅ Fácil obtener fuentes y scores de relevancia

### ¿Por qué Streamlit?

- ✅ Desarrollo rápido de UI
- ✅ Todo en Python (stack unificado)
- ✅ Fácil de personalizar con CSS
- ✅ Ideal para prototipos y demos

### ¿Por qué OpenAI?

- ✅ Embeddings de alta calidad
- ✅ GPT-3.5 es rápido y económico
- ✅ API bien documentada

**Alternativa**: Se puede usar Google Gemini (gratuito) modificando `rag_pipeline.py`

## 🎥 Demo para Video

### Flujo recomendado:

1. **Mostrar estructura del proyecto** (2 min)
   - Explicar cada archivo
   - Mostrar algunos CVs generados

2. **Demostrar la app** (3 min)
   - Hacer 4-5 preguntas variadas
   - Mostrar sources y relevance scores
   - Explicar cómo funciona el RAG

3. **Code walkthrough** (3 min)
   - `generate_cvs.py`: Cómo se generan
   - `rag_pipeline.py`: Pipeline RAG
   - `app.py`: UI y flujo

4. **Arquitectura** (2 min)
   - Diagrama: PDF → Chunks → Embeddings → Vector Store → Query → LLM → Response
   - Decisiones técnicas y trade-offs

## 🔄 Workflow Diagram

```
┌─────────────┐
│  30 CVs     │
│  (PDFs)     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ SimpleDirectory │
│ Reader          │ (LlamaIndex)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Text Chunking   │
│ (512 tokens)    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ OpenAI          │
│ Embeddings      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Vector Store    │
│ (Persistent)    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐     ┌──────────┐
│ User Query   ───┼────►│ Retrieval│
└─────────────────┘     └────┬─────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ Top-K Chunks │
                      │ (k=5)        │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ GPT-3.5      │
                      │ Synthesis    │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │ Answer +     │
                      │ Sources      │
                      └──────────────┘
```

## 📝 Mejoras Futuras

- [ ] Filtros avanzados (años de experiencia, skills específicos)
- [ ] Upload de CVs personalizados
- [ ] Comparación directa entre candidatos
- [ ] Export de resultados a Excel/CSV
- [ ] Multi-idioma (español/inglés)
- [ ] Análisis de fit para job description

## 🐛 Troubleshooting

### Error: "No API key found"
- Verifica que `.env` existe y tiene `OPENAI_API_KEY`
- Asegúrate de que el venv está activado

### Error: "No module named 'llama_index'"
```bash
pip install --upgrade llama-index
```

### Los CVs no se generan
- Verifica tu API key de OpenAI
- Si falla, el script usa fallback sin LLM

### Vector store toma mucho tiempo
- Es normal la primera vez (1-2 minutos)
- Las siguientes ejecuciones son instantáneas (carga del disco)

## 📧 Contacto

Proyecto creado para entrevista técnica - Full Stack AI Engineer

---

**Tiempo estimado de desarrollo**: 6-8 horas