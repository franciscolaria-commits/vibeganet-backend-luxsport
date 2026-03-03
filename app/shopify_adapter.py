import requests
import json
import re

# --- CONFIGURACIÓN ---
# Reemplazá por el dominio real si usa uno personalizado
SHOPIFY_URL = "https://luxsportt.myshopify.com/products.json?limit=250"

def clean_html(raw_html):
    """Elimina etiquetas HTML para no ensuciar los embeddings de ChromaDB"""
    if not raw_html:
        return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return ' '.join(cleantext.split())

def adaptar_catalogo():
    print(f"📡 Descargando catálogo desde: {SHOPIFY_URL}")
    try:
        response = requests.get(SHOPIFY_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ Error al conectar con Shopify o parsear JSON: {e}")
        return

    productos_shopify = data.get("products", [])
    if not productos_shopify:
        print("⚠️ No se encontraron productos. Verificá si el endpoint está habilitado.")
        return

    catalogo_limpio = []
    
    for prod in productos_shopify:
        name = prod.get("title", "Producto sin nombre")
        
        # Limpieza de descripción truncada a 500 caracteres
        body_html = prod.get("body_html", "")
        description = clean_html(body_html)[:500] 
        if len(body_html) > 500:
            description += "..."
            
        # Extracción de precio (Shopify guarda el precio dentro de un array 'variants')
        price = "Consultar"
        variants = prod.get("variants", [])
        if variants and isinstance(variants, list):
            price_raw = variants[0].get("price")
            if price_raw:
                price = f"${price_raw}"

        catalogo_limpio.append({
            "name": name,
            "price": price,
            "description": description
        })

    with open("catalogo_extraido.json", "w", encoding="utf-8") as f:
        json.dump(catalogo_limpio, f, ensure_ascii=False, indent=4)
        
    print(f"✅ Éxito: {len(catalogo_limpio)} productos adaptados y guardados en catalogo_extraido.json")

if __name__ == "__main__":
    adaptar_catalogo()