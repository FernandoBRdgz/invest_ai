# invest_ai
Herramienta conversacional para inversionistas que facilita consultar y entender información relevante de compañías listadas. Se enfoca en análisis fundamental (modelo de negocio, métricas financieras, ROIC, márgenes, FCF, moats, riesgos) para apoyar la toma de decisiones informadas y fomentar pensamiento independiente.

## Instalación
1. Clona este repositorio y sitúate en su carpeta raíz.
2. Crea un entorno virtual con Python 3.13 o superior (por ejemplo `python -m venv .venv`) y actívalo.
3. Instala `uv` si aún no lo tienes (`pip install uv`) y luego ejecuta `uv install` para resolver dependencias usando `pyproject.toml`.
4. Copia `.env.example` a `.env` y rellena `OPENAI_API_KEY` y `DEEPSEEK_API_KEY` con tus claves secretas.
5. Ejecuta la aplicación con `uv run streamlit run main_05.py` y abre el navegador cuando Streamlit indique la URL local.

## Árbol de dependencias
```
invest_ai
├─ openai (>=2.30.0)
├─ python-dotenv (>=1.2.2)
└─ streamlit (>=1.55.0)
```
