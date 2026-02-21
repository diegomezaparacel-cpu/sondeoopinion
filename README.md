# PARACEL · Monitor de Opinión (Web/Medios)

Repositorio listo para:
- Recolectar menciones diarias sobre PARACEL (GDELT + Google News RSS + RSS adicionales).
- Extraer texto de artículos (trafilatura).
- Clasificar tópicos y sentimiento (proxy lexicográfico).
- Publicar tablero interactivo vía GitHub Pages (carpeta /docs).

## Uso local
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_daily.py
python scripts/build_site.py
```

## GitHub Pages
Settings → Pages → Deploy from a branch → main → /docs

## Automatización diaria
El workflow está en `.github/workflows/daily.yml`.
