# Latin Music Dataset Generator - Local Execution

Esta es una versión optimizada del `dataset-generator.ipynb` diseñada para ejecutarse localmente de la manera más eficiente y robusta posible.

## Ventajas de esta versión:
1. **Robustez**: Se ejecuta como un script de Python (`.py`), lo que consume menos RAM que un Jupyter Notebook y es menos propenso a colgarse en ejecuciones largas.
2. **Resumible**: Si la ejecución se interrumpe, continuará exactamente donde se quedó, leyendo el archivo de salida actual.
3. **Visibilidad**: Incluye barras de progreso (`tqdm`) y logs detallados en `dataset_generator.log`.
4. **Limpieza Automática**: Borra los archivos temporales de audio (.wav) inmediatamente después de procesar cada canción para ahorrar espacio en disco.

## Requisitos Previos

1. **Python 3.9+**
2. **FFmpeg**: Necesario para que `yt-dlp` extraiga el audio.
   - [Descargar FFmpeg](https://ffmpeg.org/download.html) y asegúrate de añadir el binario a tu PATH.
3. **Node.js**: `yt-dlp` lo usa para resolver desafíos de JavaScript en YouTube.
   - [Descargar Node.js](https://nodejs.org/)
4. **GPU (RTX 3060)**: He optimizado el script para usar **FP16 (Half Precision)**. Esto permitirá que tu RTX 3060 procese el modelo CLAP hasta un **50-70% más rápido** y consuma menos VRAM.

## Instalación con Anaconda (Recomendado para RTX 3060)

Esta es la forma más limpia y robusta de gestionar los drivers de CUDA y las dependencias en Windows:

1.  **Crea el entorno desde el archivo**:
    Abre **Anaconda Prompt** o **Terminal** y ejecuta:
    ```bash
    conda env create -f environment.yml
    ```

2.  **Activa el entorno**:
    ```bash
    conda activate latin_music
    ```

3.  **Configuración final**:
    Crea tu archivo `.env` (puedes copiar el `.env.example`) con tu token de Hugging Face.

---

## Instalación con Pip (Alternativa)
...
3. Crea un archivo `.env` en la raíz con tu token de Hugging Face (opcional, para evitar límites de descarga del modelo):
   ```text
   HF_TOKEN=tu_token_aqui
   ```

## Cómo ejecutar

Para iniciar el generador con los parámetros por defecto:
```bash
python generate_dataset.py --csv tu_playlist.csv --output mi_dataset.csv
```

### Argumentos disponibles:
- `--csv`: Ruta al archivo CSV de entrada (debe tener columnas `track` y `genre`).
- `--output`: Ruta al archivo CSV donde se guardarán los resultados.
- `--max_fragments`: Número de fragmentos a extraer por canción (por defecto: 3).
- `--cookies`: (Opcional) Ruta a un archivo de cookies de YouTube para evitar bloqueos.
- `--max_failures`: Cuántos fallos consecutivos permitir antes de abortar (por defecto: 40).

## Eficiencia Máxima
Si tienes mucha RAM y quieres acelerar el proceso, puedes ejecutar varias instancias con diferentes segmentos del CSV, aunque lo más recomendable es dejar que una sola instancia gestione el acceso a la GPU de forma secuencial para evitar errores de memoria de video (OOM).
