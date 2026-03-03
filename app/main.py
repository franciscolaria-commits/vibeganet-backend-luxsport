import json
import os
import sys
import time
from collections import OrderedDict

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- VALIDAR VARIABLES CRÍTICAS AL INICIO ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ FATAL: La variable de entorno GROQ_API_KEY no está definida.")
    sys.exit(1)

# --- RATE LIMITER (Nivel IP) ---
limiter = Limiter(key_func=get_remote_address)

# --- FASTAPI CON DOCS DESACTIVADOS EN PRODUCCIÓN ---
IS_PRODUCTION = os.environ.get("ENVIRONMENT", "production").lower() == "production"

app = FastAPI(
    title="Vibe Agent MVP",
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CLIENTE GROQ ---
groq_client = Groq(api_key=GROQ_API_KEY)

# --- RAG: ChromaDB + SentenceTransformers ---
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "productos_ecommerce"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

try:
    print("🧠 Cargando modelo de embeddings...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"❌ FATAL: No se pudo cargar el modelo de embeddings '{EMBEDDING_MODEL_NAME}': {e}")
    sys.exit(1)

try:
    print(f"🗄️  Conectando a ChromaDB en '{CHROMA_DB_PATH}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"✅ Colección '{COLLECTION_NAME}' cargada ({collection.count()} chunks disponibles).")
except Exception as e:
    print(f"❌ FATAL: No se pudo conectar a ChromaDB o cargar la colección '{COLLECTION_NAME}': {e}")
    print("   ¿Ejecutaste generate_catalogo.py primero?")

# --- CONFIGURACIÓN DE CORS ---
def _parse_origins() -> list[str]:
    raw = os.environ.get("ALLOWED_ORIGINS", "*") 
    if raw and raw != "*":
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["*"]

origins = _parse_origins()
print(f"🔒 CORS orígenes permitidos: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["Content-Type", "ngrok-skip-browser-warning", "Authorization"],
)

# --- SECURITY HEADERS MIDDLEWARE ---
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# --- MODELO DE DATOS ---
class VibeEvent(BaseModel):
    event_type: str = Field(..., max_length=50)
    element_id: str = Field(..., max_length=200)
    meta: dict = Field(default_factory=dict)
    timestamp: str = Field(..., max_length=50)
    url: str = Field(..., max_length=2000)
    session_id: str = Field(..., max_length=64)

# --- MEMORIA DE SESIÓN Y ANTI-SPAM ---
MAX_SESSIONS = 1000
MAX_HISTORY = 3

class SessionMemory:
    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self._data: OrderedDict[str, dict] = OrderedDict()
        self._max = max_sessions

    def should_block(self, session_id: str, event_type: str, search_query: str) -> bool:
        """Retorna True si el evento debe ser bloqueado por ser ráfaga de spam."""
        now = time.time()
        if session_id in self._data:
            last = self._data[session_id].get("last_event", {})
            if not last: return False
            
            # Regla 1: Throttle global (cualquier evento) de 3 segundos para evitar colapso
            if now - last.get("time", 0) < 3.0:
                print(f"🛡️  Spam bloqueado (Regla 3s) para sesión {session_id}")
                return True
            
            # Regla 2: Deduplicación estricta. Mismo evento y query en < 15 segs = Bloqueado
            if now - last.get("time", 0) < 15.0 and last.get("type") == event_type and last.get("query") == search_query:
                print(f"🛡️  Spam bloqueado (Regla 15s Duplicado) para sesión {session_id}")
                return True
        return False

    def get_history(self, session_id: str, n: int = MAX_HISTORY) -> list:
        if session_id in self._data:
            return self._data[session_id].get("history", [])[-n:]
        return []

    def append(self, session_id: str, event_type: str, search_query: str, ai_response: dict = None):
        if session_id not in self._data:
            if len(self._data) >= self._max:
                self._data.popitem(last=False)
            self._data[session_id] = {"history": [], "last_event": {}}
        
        self._data.move_to_end(session_id)
        self._data[session_id]["last_event"] = {
            "time": time.time(),
            "type": event_type,
            "query": search_query
        }
        
        if ai_response:
            self._data[session_id]["history"].append({
                "user_action": event_type,
                "ai_button_offered": ai_response["button"],
            })
            if len(self._data[session_id]["history"]) > MAX_HISTORY * 2:
                self._data[session_id]["history"] = self._data[session_id]["history"][-MAX_HISTORY:]

session_memory = SessionMemory()

def search_product_context(search_query: str, n_results: int = 2) -> tuple[str, str]:
    """Busca contexto relevante en ChromaDB. Evita búsquedas inútiles."""
    
    # Filtro de basura RAG: Si el ID es genérico del frontend, no gastamos CPU en embeddings
    generic_ids = ["unknown", "size_selector", "size_button", "price_hover", "price_touch", "price_click"]
    if not search_query or search_query in generic_ids:
        return "Producto general", "El usuario está interactuando con talles o precios de un producto. Ofrecé asistencia por WhatsApp si tiene dudas."

    try:
        clean_query = search_query.replace("-", " ").replace("_", " ").strip()
        query_embedding = embedding_model.encode(clean_query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        documents = results.get("documents", [[]])[0]

        if not documents:
            return clean_query, "No se encontró información específica. Ofrecé atención personalizada."

        context = " ".join(documents)
        product_name = clean_query.title()
        print(f"🔍 Búsqueda RAG para '{clean_query}': {len(documents)} chunks encontrados.")
        return product_name, context
    except Exception as e:
        print(f"⚠️ Error en RAG search: {e}")
        return search_query, "Error consultando catálogo. Mantener un tono servicial."

# --- FUNCIÓN GENAI ---
def generate_vibe_response(event_type: str, product_name: str, context: str, history: str = "") -> dict:
    """Genera respuesta JSON con Groq optimizando el modelo y el system prompt."""
    fallback = {"message": "¡Tenemos ofertas increíbles! 🔥", "button": "whatsapp"}

    history_instruction = ""
    if history:
        history_instruction = (
            f"HISTORIAL RECIENTE: {history}. "
            "NO repitas el mismo mensaje exacto ni ofrezcas la misma acción si ya se ofreció recientemente."
        )

    try:
        # Se fuerza un modelo rápido y barato para evitar Error 429
        chat_completion = groq_client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama3-8b-8192"), 
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sos un asistente de ventas experto en e-commerce. "
                        "Responde SOLO con un JSON válido: {'message': '...', 'button': '...'}. "
                        "Reglas Button: 'whatsapp', 'checkout' o 'none'. "
                        "Reglas Message: Máx 15 palabras. Persuasivo y orientado a la conversión. "
                        "REGLA ESTRICTA: NUNCA uses la palabra 'irrelevante', 'desconocido' o frases negativas. Si falta información, invita a consultar. "
                        + history_instruction
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Evento del usuario: {event_type}. Producto: {product_name}. "
                        f"Contexto disponible: {context}. Generar respuesta JSON."
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
            max_tokens=150,
        )

        raw_response = chat_completion.choices[0].message.content
        parsed = json.loads(raw_response)

        message = parsed.get("message", fallback["message"])
        button = parsed.get("button", "none")
        if button not in ("whatsapp", "checkout", "none"):
            button = "whatsapp"

        return {"message": message, "button": button}

    except Exception as e:
        print(f"❌ Error llamando a Groq: {e}")
        return fallback

# --- ENDPOINTS ---
@app.head("/")
@app.get("/")
def root():
    return {"status": "Brain is running 🧠", "rag_chunks": collection.count() if 'collection' in globals() else 0}

# --- SISTEMA DE CACHÉ SEMÁNTICO (Añadir debajo de SessionMemory) ---
RESPONSE_CACHE = {}
CACHE_TTL = 1800  # 30 minutos de vida útil para las respuestas cacheadas

@app.post("/api/track")
@limiter.limit("60/minute") 
def track_event(request: Request, event: VibeEvent): 
    print(f"📥 Evento entrante: {event.event_type} -> {event.element_id} [S: {event.session_id}]")

    # 1. INTERCEPTOR DE MÉTRICAS (No consume IA)
    if event.event_type == "conversion_click":
        print(f"💰 [MÉTRICA MVP] ¡Conversión lograda a WhatsApp/Checkout! Sesión: {event.session_id}")
        return {"status": "ok", "action": "none"}

    # 2. PODA DE EVENTOS BASURA (No consume IA)
    # Si es solo scroll visual, se registra el interés pero no se interrumpe al usuario ni se gasta Groq
    if event.event_type == "interest":
        print(f"👁️ Registro de vista guardado silenciosamente. Sesión: {event.session_id}")
        return {"status": "ok", "action": "none"}

    # --- EXTRACCIÓN DE QUERY DE BÚSQUEDA ---
    search_query = event.element_id
    if event.event_type == "compare_price" and "text_selected" in event.meta:
        search_query = event.meta["text_selected"]
    elif event.event_type == "hesitation" and "options_compared" in event.meta:
        search_query = " ".join(event.meta["options_compared"])
    elif event.event_type == "size_select" and "selected_size" in event.meta:
        search_query = event.element_id 

    # --- FILTRO ANTI-SPAM (Backend Debounce) ---
    if session_memory.should_block(event.session_id, event.event_type, search_query):
        session_memory.append(event.session_id, event.event_type, search_query)
        return {"status": "blocked_by_throttle", "action": "none"}

    # --- CACHÉ SEMÁNTICO (Evita llamar a Groq por inputs idénticos) ---
    cache_key = f"{event.event_type}_{search_query}"
    now = time.time()
    
    # Limpieza pasiva del caché
    claves_viejas = [k for k, v in RESPONSE_CACHE.items() if now - v['time'] > CACHE_TTL]
    for k in claves_viejas: del RESPONSE_CACHE[k]

    # Mapeo de emoción
    emotion = "agent"
    if event.event_type == "rage_click": emotion = "rage"
    elif event.event_type in ["compare_price", "hesitation", "size_select"]: emotion = "doubt"

    # Retorno desde Caché
    if cache_key in RESPONSE_CACHE:
        print(f"⚡ Sirviendo desde CACHÉ (0 tokens gastados): {cache_key}")
        ai_response = RESPONSE_CACHE[cache_key]['data']
        session_memory.append(event.session_id, event.event_type, search_query, ai_response)
        return {
            "status": "ok",
            "action": "toast",
            "message": ai_response["message"],
            "emotion": emotion,
            "button": ai_response["button"],
        }

    # --- MEMORIA HISTÓRICA Y RAG ---
    user_history = session_memory.get_history(event.session_id)
    history_str = "; ".join(f"Acción: {h['user_action']}, Botón: {h['ai_button_offered']}" for h in user_history)
    product_name, context = search_product_context(search_query)

    # --- LLAMADA A IA (Solo llega acá si hay alta intención y no hay caché) ---
    ai_response = generate_vibe_response(event.event_type, product_name, context, history=history_str)
    print(f"🤖 Respuesta IA Generada (Sesión {event.session_id}): {ai_response}")

    # Guardar en memoria y caché
    session_memory.append(event.session_id, event.event_type, search_query, ai_response)
    RESPONSE_CACHE[cache_key] = {'time': now, 'data': ai_response}

    return {
        "status": "ok",
        "action": "toast",
        "message": ai_response["message"],
        "emotion": emotion,
        "button": ai_response["button"],
    }