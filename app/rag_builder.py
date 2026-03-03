"""
RAG Builder - Procesador de catálogo PDF para base de datos vectorial.

Lee un PDF, lo divide en chunks inteligentes, genera embeddings
y los almacena en ChromaDB para búsqueda semántica.

Uso:
    python rag_builder.py
    python rag_builder.py --pdf mi_catalogo.pdf --chunk-size 500
"""

import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

# --- CONFIGURACIÓN ---
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "productos_ecommerce"
JSON_PATH = "catalogo_extraido.json"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

def build_vector_db_from_json():
    if not os.path.exists(JSON_PATH):
        print(f"❌ Error: No se encontró el archivo {JSON_PATH}.")
        return

    print(f"📄 Cargando datos desde {JSON_PATH}...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        productos = json.load(f)

    if not productos:
        print("❌ Error: El catálogo JSON está vacío.")
        return

    print(f"🗄️ Conectando a ChromaDB en {CHROMA_DB_PATH}...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # 1. Purgar base de datos anterior (Evita mezclar taladros con zapatillas reales)
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print("🧹 Colección anterior eliminada para evitar contaminación de datos.")
    except Exception:
        pass # La colección no existía

    # 2. Crear colección limpia
    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    
    print(f"🧠 Cargando modelo de embeddings ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    documents = []
    metadatas = []
    ids = []

    # 3. Construir los chunks semánticos 1-a-1
    print("⚙️ Preparando chunks semánticos...")
    for i, prod in enumerate(productos):
        # Estructura optimizada para que el LLM entienda el contexto
        chunk = f"Producto: {prod['name']}. Precio: {prod['price']}. Descripción: {prod['description']}"
        documents.append(chunk)
        
        # Metadatos útiles para filtrar a futuro si el cliente escala
        metadatas.append({
            "name": prod['name'], 
            "price": prod['price']
        })
        ids.append(f"prod_{i}")

    # 4. Inserción por lotes (Batch Processing) -> Máximo rendimiento
    print("⚡ Generando embeddings e insertando en la base de datos (Batch)...")
    embeddings = model.encode(documents).tolist()
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Éxito: {collection.count()} productos reales indexados correctamente en ChromaDB.")

if __name__ == "__main__":
    build_vector_db_from_json()