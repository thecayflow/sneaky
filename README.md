![sneaky icon](docs/icon.png)

# sneaky™ Semantic Report

App local (Streamlit) para analizar un feed de imágenes: extrae embeddings
CLIP, los agrupa en bloques semánticos y los visualiza como un radar
interactivo, con vistas complementarias de proyección 2D (t-SNE/UMAP) y
similitud visual a nivel de píxel (duplicados/ráfagas), además de un
informe PDF exportable ("Semantic Report by sneaky™").

## Qué hace

- Analiza cualquier carpeta local de imágenes (recursivo, sin copiar nada).
- Detecta automáticamente entre 3 y 25 ejes semánticos (clustering
  jerárquico sobre embeddings CLIP) y los etiqueta solo.
- Permite añadir ejes personalizados por texto libre, y excluir ejes
  automáticos si se quiere validar solo contra los personalizados.
- Compara dos feeds distintos superpuestos en el mismo radar.
- Detecta imágenes que no encajan en ningún eje ("Other") y duplicados/
  casi-duplicados visuales (ráfagas de cámara, etc.).
- Genera un informe PDF con el resumen completo, listo para compartir.

## Requisitos

- **Windows 11** con **GPU NVIDIA** (probado en RTX 3060) y drivers
  recientes — no hace falta instalar el CUDA Toolkit por separado, la
  rueda de PyTorch ya lo incluye.
- Python 3.11+ (usado durante el desarrollo).
- Sin GPU NVIDIA la app funciona pero muy lenta (no probado en CPU).

## Instalación

```powershell
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> El primer arranque de `streamlit run app.py` puede tardar 1-2 minutos
> (importa el stack de IA completo — torch, transformers, umap-learn...).
> Es normal, no significa que se haya colgado.

Para el diseño del PDF con la tipografía completa (Barlow / Barlow
Condensed), descarga esas familias desde Google Fonts y coloca los
`.ttf` en `src/report/fonts/` — si no están, el PDF sigue generándose
igual, solo que con una fuente de reserva (Helvetica).

## Uso

```powershell
venv\Scripts\activate
streamlit run app.py
```

Se abre en `http://localhost:8501`. Escribe la ruta de una carpeta local
con imágenes y pulsa **Analyze**.

## Estructura del proyecto

Ver el árbol de carpetas bajo `src/` — cada módulo tiene una
responsabilidad única (ingestion, embeddings, axes, scoring, viz,
similarity, report, persistence). Ver `BACKLOG.md` para el estado de
fases y mejoras pendientes.

## Decisiones de diseño relevantes

### Clustering jerárquico: `average` → `ward` (con PCA)

**Fecha**: durante la construcción de `src/axes/hierarchical.py`.

**Qué pasó**: la primera versión de `HierarchicalAxisEngine` usaba linkage
`average` sobre distancia coseno, directamente en el espacio de embeddings
de 512 dimensiones (salida de CLIP `ViT-B-32`). Al probarlo sobre el
dataset real (`E:\dataset_unificado`, 3.092 imágenes), el resultado fue muy
desequilibrado:

```
k=3: sizes=[3081, 6, 5]
k=5: sizes=[3069, 9, 6, 5, 3]
k=7: sizes=[3064, 9, 6, 5, 3, 3, 2]
```

Un cluster se comía casi todo el dataset, y el resto eran clusters
minúsculos — no bloques semánticos útiles para el radar (selfies,
paisajes...), sino imágenes atípicas del dataset (outliers).

**Por qué pasó**: `average` linkage sobre distancia coseno en espacios de
alta dimensión sufre un fenómeno conocido como *chaining*: en vez de formar
grupos compactos, va "encadenando" puntos uno a uno al cluster más grande.

**Solución aplicada**:
1. Cambiar el método de linkage por defecto a **`ward`** (minimiza varianza
   intra-cluster al fusionar — mucho más resistente al chaining).
2. Reducir los embeddings de 512 a **50 dimensiones con PCA** antes de
   clusterizar (`ward` necesita trabajar en espacio euclídeo real, y
   reducir dimensionalidad también ayuda a que el clustering salga mejor
   formado). Los centroides de cada cluster se siguen calculando en el
   espacio original — la PCA solo se usa para decidir la agrupación, no
   para representar los ejes.

**Resultado**: con `ward` + PCA, la misma prueba dio (k=5):
`sizes=[1063, 669, 658, 356, 346]` — mucho más equilibrado y
semánticamente razonable.

**Dónde se configura** (`src/axes/hierarchical.py`):

```python
HierarchicalAxisEngine(
    linkage_method="ward",   # antes: "average"
    pca_components=50,       # None para desactivar la reducción PCA
)
```

Ambos parámetros son configurables al instanciar `HierarchicalAxisEngine`,
y también desde un toggle en la propia interfaz.

### Modelos de Hugging Face y el bloqueo de `torch.load` (torch < 2.6)

**Fecha**: durante la construcción de `src/axes/labeling.py`.

**Qué pasó**: al cargar el modelo de captioning BLIP
(`Salesforce/blip-image-captioning-base`) con `transformers`, la carga
falló con:

```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with
`weights_only=True`, we now require users to upgrade torch to at least
v2.6 in order to use the function...
```

**Por qué pasa**: versiones recientes de `transformers` bloquean cargar
checkpoints en formato `.bin` (pickle, vía `torch.load`) si el PyTorch
instalado es menor a 2.6, por una vulnerabilidad de seguridad conocida
(CVE-2025-32434). El proyecto usa PyTorch 2.5.1 (fijado por la versión de
CUDA/GPU), así que este bloqueo aplica.

**Solución aplicada**: forzar el uso del formato **`.safetensors`** en vez
de `.bin` — es un formato de pesos que no usa pickle, así que no está
sujeto a esa restricción, y no requiere tocar la versión de PyTorch:

```python
BlipForConditionalGeneration.from_pretrained(model_name, use_safetensors=True)
```

**Por qué importa para el futuro**: cualquier modelo nuevo que se cargue
desde Hugging Face en este proyecto puede toparse con el mismo bloqueo si
el repositorio del modelo no publica pesos en `.safetensors` por defecto.
Añadir `use_safetensors=True` al `from_pretrained(...)` correspondiente es
la solución estándar.

### Actualización incremental de la caché

**Fecha**: al plantearse qué pasa si se añade una imagen nueva a un dataset
ya procesado.

**Qué pasaba antes**: `pipeline.py` comprobaba únicamente "¿existe caché
para esta ruta?" y, si existía, la usaba tal cual — sin comparar contra el
contenido real de la carpeta. Añadir o quitar imágenes no tenía ningún
efecto hasta que se borraba la caché a mano y se recalculaba todo desde
cero (~7 minutos).

**Solución aplicada**: `pipeline.py` ahora siempre escanea la carpeta antes
de decidir nada, y compara la lista de imágenes actual contra la que había
cacheada (`src/pipeline.py::_update_embeddings_incrementally`):
- Si no hay cambios, se usa la caché tal cual (rápido, como siempre).
- Si hay imágenes nuevas, se generan embeddings *solo* de esas — las
  demás se reutilizan sin recalcular.
- Si hay imágenes eliminadas, se descartan sus embeddings de la caché.
- En cualquiera de los dos últimos casos, se invalida el árbol jerárquico
  y las etiquetas por k (`cache.invalidate_tree_and_axes`), porque se
  calcularon a partir de la composición antigua del dataset — pero esto
  se recalcula rápido (decenas de segundos), ya que lo caro de verdad son
  los embeddings, no el clustering ni el etiquetado.

También se puede gestionar la caché manualmente desde la propia interfaz
("Cache management", al final de la página) sin tocar el disco a mano.
