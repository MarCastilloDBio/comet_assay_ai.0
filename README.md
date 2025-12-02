# comet_assay_ai.0
Comet Assay AI es un proyecto que implementa un modelo de red neuronal basado en ResNet34 para el análisis automatizado de imágenes de ensayos cometa. Este proyecto permite procesar imágenes experimentales de ensayos cometa, segmentar, etiquetar y enmascarar las imágenes, y calcular métricas clave para evaluar el daño en el ADN

# 🧬 Comet Assay AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**Sistema automatizado de análisis de genotoxicidad mediante Deep Learning**

[Características](#características) • [Instalación](#instalación) • [Uso Rápido](#uso-rápido) • [Resultados](#resultados) • [Documentación](#documentación)

---

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Características](#características)
- [Arquitectura del Modelo](#arquitectura-del-modelo)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Preparación del Dataset](#preparación-del-dataset)
- [Entrenamiento](#entrenamiento)
- [Inferencia](#inferencia)
- [Generación de Reportes](#generación-de-reportes)
- [Resultados](#resultados)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Métricas Biológicas](#métricas-biológicas)
- [Validación Científica](#validación-científica)
- [Limitaciones](#limitaciones)
- [Contribuir](#contribuir)
- [Licencia](#licencia)
- [Citación](#citación)
- [Contacto](#contacto)

---

## 🔬 Descripción

**Comet Assay AI** es un sistema de análisis automatizado para ensayo cometa (comet assay) basado en redes neuronales convolucionales profundas (Deep Learning). El sistema permite la segmentación automática de cometas, separación de cabeza y cola, y cuantificación de métricas de genotoxicidad (Tail DNA%, Tail Moment, longitudes) con precisión comparable a análisis manual experto.

### ¿Qué es el Ensayo Cometa?

El ensayo cometa (Single Cell Gel Electrophoresis) es una técnica ampliamente utilizada en toxicología para detectar daño al ADN en células individuales. Las células dañadas forman una estructura característica de "cometa" bajo electroforesis, donde:
- **Cabeza**: ADN íntegro (núcleo compacto)
- **Cola**: Fragmentos de ADN migrados (indicativo de daño)

### Problema que Resuelve

El análisis manual de ensayo cometa es:
- ⏱️ **Lento**: 2-5 minutos por cometa
- 👤 **Subjetivo**: Variabilidad inter-observador
- 🔁 **No escalable**: Difícil analizar >100 cometas
- 📊 **Propenso a error**: Fatiga visual en análisis largos

**Comet Assay AI automatiza este proceso con:**
- ⚡ **Velocidad**: ~3 segundos por cometa
- 🎯 **Consistencia**: Criterios objetivos reproducibles
- 📈 **Escalabilidad**: Procesa cientos de imágenes
- 🤖 **Robustez**: Funciona con diferentes fluoróforos y condiciones

---

## ✨ Características

### Funcionalidades Principales

- 🖼️ **Segmentación Multi-clase**: Identificación automática de fondo, cabeza y cola
- 🎨 **Robustez a Colores**: Procesa imágenes con diferentes fluoróforos (DAPI, SYBR Green, Bromuro de Etidio)
- 📊 **Cuantificación Completa**: Calcula Tail DNA%, Tail Moment, longitudes, intensidades
- 📄 **Reportes Profesionales**: Genera PDFs con análisis estadístico y visualizaciones
- 🔍 **Visualización de Resultados**: Overlays con segmentación coloreada y centroides
- 🧪 **Validación Científica**: Métricas de desempeño (Dice Score, IoU) incluidas

### Tecnologías Utilizadas

- **Framework**: PyTorch 2.1.0
- **Arquitectura**: U-Net con encoder ResNet34 pre-entrenado (ImageNet)
- **Procesamiento**: OpenCV, scikit-image, scipy
- **Augmentations**: Albumentations
- **Reportes**: ReportLab, Matplotlib, Pandas
- **Transfer Learning**: Encoder pre-entrenado reduce datos necesarios

---

## 🏗️ Arquitectura del Modelo

### U-Net con Encoder Pre-entrenado
```
INPUT: Imagen en escala de grises (512×512, 3 canales)
  ↓
ENCODER (ResNet34 pre-trained)
  ├── Conv Block 1 → 64 features
  ├── Conv Block 2 → 128 features
  ├── Conv Block 3 → 256 features
  └── Conv Block 4 → 512 features
  ↓
BOTTLENECK → 512 features
  ↓
DECODER (U-Net)
  ├── UpConv Block 4 + Skip Connection → 256 features
  ├── UpConv Block 3 + Skip Connection → 128 features
  ├── UpConv Block 2 + Skip Connection → 64 features
  └── UpConv Block 1 + Skip Connection → 32 features
  ↓
OUTPUT: Máscara de segmentación (512×512, 3 clases)
  └── Clase 0: Fondo
  └── Clase 1: Cabeza
  └── Clase 2: Cola
```

### Loss Function

Combinación de **Cross-Entropy** y **Dice Loss** (50%/50%):
```python
Loss = 0.5 × CrossEntropy(logits, targets) + 0.5 × DiceLoss(logits, targets)
```

### Optimización

- **Optimizador**: AdamW (weight_decay=1e-5)
- **Learning Rate**: 1e-4 con Cosine Annealing
- **Batch Size**: 4-8 (ajustable según GPU/CPU)
- **Augmentations**: Rotación, flip, cambios de intensidad, ruido, desenfoque

---

## 💻 Requisitos

### Requisitos del Sistema

- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS
- **RAM**: 8GB mínimo (16GB recomendado)
- **GPU**: Opcional (NVIDIA con CUDA 11.x+) - acelera 10-20x
- **Almacenamiento**: ~2GB para código y modelo

### Requisitos de Software

- Python 3.8 - 3.10
- pip o conda

---

## 🚀 Instalación

### Opción 1: Instalación Estándar
```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/comet-assay-ai.git
cd comet-assay-ai

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Opción 2: Con Conda (Recomendado)
```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/comet-assay-ai.git
cd comet-assay-ai

# 2. Crear entorno conda
conda create -n comet python=3.9
conda activate comet

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Verificar Instalación
```bash
python -c "import torch; import cv2; import segmentation_models_pytorch as smp; print('✓ Instalación exitosa')"
```

---

## 📦 Preparación del Dataset

### 1. Obtener Imágenes

Necesitas **mínimo 30-50 imágenes** de cometas para entrenamiento inicial:

- Formatos soportados: PNG, JPG, TIFF
- Resolución mínima recomendada: 200×200 píxeles
- Fuentes válidas: Microscopía de fluorescencia, papers científicos, datasets públicos

### 2. Etiquetar con LabelMe
```bash
# Instalar LabelMe (si no está incluido)
pip install labelme

# Abrir herramienta de etiquetado
labelme
```

**Proceso de etiquetado:**

1. **File → Open Dir** → Selecciona `dataset/images/`
2. **File → Change Output Dir** → Selecciona `dataset/annotations/`
3. Para cada imagen:
   - Click en **"Create Polygons"**
   - Dibuja polígono alrededor de la **cabeza** → Etiqueta: `head`
   - Dibuja polígono alrededor de la **cola** → Etiqueta: `tail`
   - **Ctrl+S** para guardar
   - **D** para siguiente imagen

**Convenciones importantes:**
- ✅ Usa siempre los labels `head` y `tail` (minúsculas, inglés)
- ✅ Sé consistente en el criterio de delimitación
- ✅ Incluye todo el ADN visible en cada región
- ⏱️ Tiempo estimado: 2-3 minutos por imagen

### 3. Convertir Anotaciones a Máscaras
```bash
python src/convert_labelme_to_masks.py \
    --annotations dataset/annotations \
    --images dataset/images \
    --output dataset/masks \
    --visualize
```

**Salida:**
- `dataset/masks/`: Máscaras PNG (0=fondo, 1=cabeza, 2=cola)
- `dataset/visualizations/`: Imágenes con overlays para verificación
- `dataset/masks_info.csv`: Metadatos de las máscaras

---

## 🎓 Entrenamiento

### Entrenamiento Básico
```bash
python src/train.py \
    --data_dir dataset \
    --epochs 200 \
    --batch_size 4 \
    --lr 0.0001 \
    --val_split 0.15 \
    --early_stopping 30 \
    --output_dir checkpoints
```

### Parámetros Principales

| Parámetro | Descripción | Valor Recomendado |
|-----------|-------------|-------------------|
| `--epochs` | Número máximo de epochs | 150-200 |
| `--batch_size` | Imágenes por batch | 4 (CPU), 8-16 (GPU) |
| `--lr` | Learning rate | 1e-4 |
| `--val_split` | Proporción de validación | 0.15-0.20 |
| `--early_stopping` | Paciencia (epochs sin mejora) | 25-30 |
| `--augmentation_prob` | Probabilidad de augmentation | 0.6-0.7 |

### Monitoreo del Entrenamiento

Durante el entrenamiento verás:
```
Epoch 50/200
------------------------------------------------------------
Training: 100%|████████████| 11/11 [01:45<00:00]
Validation: 100%|██████████| 2/2 [00:20<00:00]

Train Loss: 0.3421
Val Loss:   0.3156
Dice Head:  0.7834
Dice Tail:  0.7612
Dice Mean:  0.7723 ← Métrica principal
IoU Mean:   0.6891
LR:         0.000087

✓ Mejor modelo guardado (Dice: 0.7723)
```

**Métricas a observar:**
- **Dice Mean**: Debe subir (objetivo: >0.70)
- **Train/Val Loss**: Deben bajar
- **Early Stopping**: Se activa automáticamente

### Tiempo de Entrenamiento

| Hardware | Tiempo/Epoch | Epochs Típicos | Tiempo Total |
|----------|--------------|----------------|--------------|
| CPU (Intel i5/i7) | 2-4 min | 80-120 | 3-8 horas |
| GPU (GTX 1060) | 20-40 seg | 80-120 | 30-80 min |
| GPU (RTX 3070+) | 10-20 seg | 80-120 | 15-40 min |

---

## 🔮 Inferencia

### Procesar Imágenes Individuales
```bash
python src/inference.py \
    --model checkpoints/[FECHA]/best_model.pth \
    --image path/to/image.png \
    --output results/
```

### Procesar Múltiples Imágenes
```bash
python src/inference.py \
    --model checkpoints/[FECHA]/best_model.pth \
    --image_dir path/to/images/ \
    --output results/ \
    --pixel_size_um 0.65  # Opcional: para conversión a µm
```

### Salidas Generadas
```
results/
├── overlays/          # Imágenes con segmentación superpuesta
│   ├── image_001.png  # Verde=cabeza, Rojo=cola, Línea=centroides
│   └── image_002.png
├── metrics.csv        # Tabla con todas las métricas
└── [timestamp]/       # Metadata del procesamiento
```

### Ejemplo de `metrics.csv`

| image_name | tail_dna_percent | tail_moment | comet_length_px | damage_level |
|------------|------------------|-------------|-----------------|--------------|
| comet_001.png | 15.3% | 234.5 | 187.2 | Bajo |
| comet_002.png | 67.8% | 1245.8 | 312.5 | Severo |
| comet_003.png | 23.1% | 421.3 | 205.8 | Moderado |

---

## 📊 Generación de Reportes

### Reporte Completo con Imágenes
```bash
python reports/generate_report.py \
    --csv results/metrics.csv \
    --output reporte_final.pdf \
    --overlays results/overlays \
    --dice_score 0.8265 \
    --include_individual
```

### Contenido del Reporte PDF

1. **Página 1**: Resumen ejecutivo con estadísticas clave
2. **Páginas 2-N**: Tabla completa con:
   - Nombre de imagen
   - Tail DNA %
   - Tail Moment
   - Nivel de daño (coloreado)
3. **Análisis Gráfico**:
   - Distribución de Tail DNA %
   - Correlaciones
   - Box plots por nivel de daño
4. **Interpretación y Metodología**:
   - Explicación de cálculo de métricas
   - Umbrales de daño
   - Recomendaciones científicas

---

## 📈 Resultados

### Desempeño del Modelo

En nuestro dataset de **50 imágenes** (43 train, 7 val):

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Dice Score** | 0.8265 | Excelente (>80%) |
| **IoU** | 0.7021 | Muy bueno |
| **Precisión (Cabeza)** | 0.8534 | Alta |
| **Precisión (Cola)** | 0.7996 | Alta |
| **Tiempo de inferencia** | ~3 seg/imagen | CPU |

### Comparación con Análisis Manual

| Aspecto | Manual | Comet Assay AI |
|---------|--------|----------------|
| Tiempo por imagen | 2-5 minutos | ~3 segundos |
| Consistencia | Variable (inter-observador) | 100% reproducible |
| Throughput | 10-30 imágenes/hora | 1200 imágenes/hora |
| Fatiga | Sí (después de 1-2 horas) | No |
| Costo | Alto (tiempo experto) | Bajo (automatizado) |

### Ejemplo de Segmentación

<div align="center">

| Original | Segmentación | Overlay |
|----------|--------------|---------|
| ![Original](docs/images/example_original.png) | ![Mask](docs/images/example_mask.png) | ![Overlay](docs/images/example_overlay.png) |

*Verde: Cabeza | Rojo: Cola | Línea: Vector cabeza-cola*

</div>

---

## 📁 Estructura del Proyecto
```
comet-assay-ai/
├── README.md                      # Este archivo
├── requirements.txt               # Dependencias
├── LICENSE                        # Licencia MIT
├── .gitignore                     # Archivos a ignorar
│
├── dataset/                       # Dataset de entrenamiento
│   ├── images/                    # Imágenes originales
│   ├── annotations/               # Anotaciones LabelMe (JSON)
│   ├── masks/                     # Máscaras generadas (PNG)
│   └── visualizations/            # Verificación de máscaras
│
├── src/                           # Código fuente
│   ├── __init__.py
│   ├── dataset_grayscale.py      # DataLoader (versión escala de grises)
│   ├── model.py                   # Arquitectura U-Net
│   ├── train.py                   # Script de entrenamiento
│   ├── inference.py               # Script de inferencia
│   ├── postprocessing.py          # Cálculo de métricas
│   ├── metrics.py                 # Métricas de evaluación
│   ├── utils.py                   # Utilidades
│   └── convert_labelme_to_masks.py # Conversión de anotaciones
│
├── reports/                       # Generación de reportes
│   └── generate_report.py         # Script para PDFs
│
├── checkpoints/                   # Modelos entrenados
│   └── [timestamp]/
│       ├── best_model.pth         # Mejor modelo
│       ├── last_checkpoint.pth    # Último checkpoint
│       ├── config.json            # Configuración
│       └── history.json           # Historial de entrenamiento
│
├── results/                       # Resultados de inferencia
│   ├── overlays/                  # Visualizaciones
│   ├── metrics.csv                # Métricas tabuladas
│   └── temp_plots/                # Gráficas temporales
│
├── docs/                          # Documentación adicional
│   ├── images/                    # Imágenes para README
│   ├── TRAINING_GUIDE.md          # Guía detallada de entrenamiento
│   └── API_REFERENCE.md           # Referencia de API
│
└── tests/                         # Tests unitarios (opcional)
    ├── test_postprocessing.py
    └── test_metrics.py
```

---

## 🧪 Métricas Biológicas

### Tail DNA %

**Definición**: Porcentaje de ADN fragmentado que migró a la cola.

**Cálculo**:
```
Tail DNA % = (Intensidad de Fluorescencia en Cola / Intensidad Total) × 100
```

**Interpretación**:
- **< 5%**: Sin daño o daño mínimo
- **5-20%**: Daño genotóxico bajo
- **20-40%**: Daño moderado
- **> 40%**: Daño severo

### Tail Moment

**Definición**: Métrica que incorpora cantidad y distribución del daño.

**Cálculo**:
```
Tail Moment = Tail DNA % × Distancia entre Centroides (píxeles)
```

**Ventaja**: Más sensible que Tail DNA% para detectar daños moderados.

### Otras Métricas

- **Comet Length**: Longitud total del cometa (head + tail)
- **Head/Tail Intensity**: Intensidad de fluorescencia total por región
- **Centroids Distance**: Distancia euclidiana entre centroides
- **Areas**: Área en píxeles (convertible a µm² con calibración)

---

## ✅ Validación Científica

### Recomendaciones para Publicación

1. **Validación Manual**:
   - Comparar 10-20% de segmentaciones con análisis experto
   - Calcular concordancia (Correlación de Pearson, ICC)
   - Generar Bland-Altman plots

2. **Controles Experimentales**:
   - Incluir controles positivos (H₂O₂, MMS)
   - Incluir controles negativos (PBS, medio de cultivo)
   - Documentar condiciones experimentales

3. **Análisis Estadístico**:
   - Usar pruebas no paramétricas (Mann-Whitney U, Kruskal-Wallis)
   - Reportar mediana, IQR además de media/SD
   - Analizar mínimo 50-100 cometas por grupo

4. **Documentación**:
   - Voltaje y tiempo de electroforesis
   - Tiempo de lisis
   - Tipo de fluoróforo
   - Magnificación del microscopio
   - Tamaño de píxel (µm)

### Script de Validación
```python
# compare_manual_vs_auto.py
import pandas as pd
from scipy import stats

auto = pd.read_csv('results/metrics.csv')
manual = pd.read_csv('manual_analysis.csv')

df = auto.merge(manual, on='image_name')

# Correlación
r, p = stats.pearsonr(df['tail_dna_percent_auto'], 
                       df['tail_dna_percent_manual'])

print(f"Correlación de Pearson: R = {r:.3f}, p = {p:.4f}")
```

---

## ⚠️ Limitaciones

### Limitaciones Técnicas

1. **Datos Requeridos**: Mínimo 30-50 imágenes etiquetadas manualmente
2. **Calidad de Imagen**: Funciona mejor con imágenes nítidas, bien expuestas
3. **Cometas Superpuestos**: Dificultad con múltiples cometas muy cercanos
4. **Variabilidad Extrema**: Puede fallar con condiciones muy fuera de distribución

### Limitaciones de Uso

- ⚠️ **NO para diagnóstico clínico** sin validación por expertos certificados
- ⚠️ **NO reemplaza** criterio científico ni controles experimentales
- ⚠️ **Requiere validación** con análisis manual antes de publicación
- ⚠️ **Sujeto a aprobación ética** para uso con muestras humanas/animales

### Mejoras Futuras

- [ ] Detección y separación automática de cometas superpuestos
- [ ] Soporte para múltiples cometas por imagen
- [ ] Calibración automática de píxel a µm desde metadatos TIFF
- [ ] Modelo ensemble para mayor robustez
- [ ] API REST para integración con LIMS
- [ ] Interfaz web completa (Streamlit/Flask)

---

### Áreas de Contribución

- 🐛 Reportar bugs
- 💡 Proponer nuevas features
- 📝 Mejorar documentación
- 🧪 Añadir tests unitarios
- 🎨 Mejorar visualizaciones
- 🌍 Traducir README a otros idiomas

---

## 📄 Licencia

Este proyecto está licenciado bajo la **MIT License** - ver el archivo [LICENSE](LICENSE) para detalles.
```
MIT License

Copyright (c) 2025 [Mario Esteban Castillo Díaz/ Universidad Nacional de Colombia]

Se concede permiso, de forma gratuita, a cualquier persona que obtenga una copia
de este software y archivos de documentación asociados (el "Software"), para usar
el Software sin restricciones, incluyendo sin limitación los derechos de usar,
copiar, modificar, fusionar, publicar, distribuir, sublicenciar y/o vender copias
del Software...
```

---

## 📚 Citación

Si usas este código en tu investigación, por favor cita:
```bibtex
@software{comet_assay_ai_2025,
  author = {[Tu Nombre]},
  title = {Comet Assay AI: Automated Genotoxicity Analysis using Deep Learning},
  year = {2025},
  url = {https://github.com/tu-usuario/comet-assay-ai},
  version = {1.0.0}
}
```

### Referencias Científicas

- **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.
- **Comet Assay**: Collins, A. R. (2004). The comet assay for DNA damage and repair. *Molecular Biotechnology*, 26(3), 249-261.
- **Tail Moment**: Olive, P. L., Banáth, J. P., & Durand, R. E. (1990). Heterogeneity in radiation-induced DNA damage and repair. *Radiation Research*, 122(1), 86-94.

---

## 👥 Autores

- **Mario Esteban Castillo Díaz** - *Desarrollo inicial* - [GitHub](https://github.com/tu-usuario) | [Email](mailto:macastillod@unal.edu.co)

### Agradecimientos

- Comunidad de PyTorch por herramientas excelentes
- Desarrolladores de segmentation-models-pytorch
- Revisores y testers del proyecto

---

## 📞 Contacto

- **Proyecto**: [https://github.com/tu-usuario/comet-assay-ai](https://github.com/tu-usuario/comet-assay-ai)
- **Issues**: [https://github.com/tu-usuario/comet-assay-ai/issues](https://github.com/tu-usuario/comet-assay-ai/issues)
- **Email**: tu-email@ejemplo.com
- **LinkedIn**: [Tu Perfil](https://linkedin.com/in/tu-perfil)

---

## ⭐ Soporte

Si este proyecto te fue útil, por favor considera:

- ⭐ Dar una estrella al repositorio
- 🐛 Reportar bugs o solicitar features
- 📢 Compartir con colegas que puedan beneficiarse
- 💬 Dejar feedback sobre tu experiencia

---

<div align="center">

**Desarrollado con 🧬 para la comunidad científica**

[⬆ Volver arriba](#-comet-assay-ai)

</div>
