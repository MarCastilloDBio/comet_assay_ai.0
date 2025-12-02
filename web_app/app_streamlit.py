"""
Aplicación web Streamlit para análisis de cometa
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import io
import zipfile
from pathlib import Path
import tempfile
import shutil

# Imports del proyecto
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))

from inference import CometInference
from postprocessing import visualize_segmentation, classify_damage_level

# Configuración de página
st.set_page_config(
    page_title="Comet Assay AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .damage-minimal {
        color: #27AE60;
        font-weight: bold;
    }
    .damage-low {
        color: #F39C12;
        font-weight: bold;
    }
    .damage-moderate {
        color: #E67E22;
        font-weight: bold;
    }
    .damage-severe {
        color: #E74C3C;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_cached(model_path: str, device: str = 'cpu'):
    """Carga modelo con caché"""
    return CometInference(
        model_path=model_path,
        device=device,
        image_size=512
    )


def process_uploaded_image(image_file, inference_model, pixel_size_um=None):
    """Procesa imagen subida"""
    # Leer imagen
    image = Image.open(image_file)
    image_np = np.array(image)

    # Convertir a escala de grises si es necesario
    if len(image_np.shape) == 3:
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_np

    # Predecir
    with st.spinner('Analizando imagen...'):
        mask = inference_model.predict(image_np)

        # Separar cabeza y cola
        from postprocessing import separate_head_tail, calculate_metrics
        head_mask, tail_mask = separate_head_tail(mask)

        # Calcular métricas
        metrics = calculate_metrics(
            image_gray,
            head_mask,
            tail_mask,
            pixel_size_um=pixel_size_um
        )

        # Visualización
        overlay = visualize_segmentation(image_gray, mask, head_mask, tail_mask)

    return metrics, overlay, image_np


def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 Comet Assay AI</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; color: #7F8C8D;'>
    Sistema automatizado de análisis de ensayo cometa para evaluación de genotoxicidad
    </p>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("⚙️ Configuración")

    # Selección de modelo
    model_path = st.sidebar.text_input(
        "Path del modelo entrenado",
        value="checkpoints/best_model.pth",
        help="Path al archivo .pth con el modelo entrenado"
    )

    # Device
    device = st.sidebar.selectbox(
        "Device",
        options=['cpu', 'cuda'],
        index=0 if not torch.cuda.is_available() else 1
    )

    # Calibración
    st.sidebar.subheader("📏 Calibración")
    use_calibration = st.sidebar.checkbox("Usar calibración física")
    pixel_size_um = None

    if use_calibration:
        pixel_size_um = st.sidebar.number_input(
            "Tamaño de píxel (µm)",
            min_value=0.01,
            max_value=10.0,
            value=0.65,
            step=0.01,
            help="Tamaño físico de un píxel en micrómetros"
        )

    # Cargar modelo
    try:
        if Path(model_path).exists():
            inference_model = load_model_cached(model_path, device)
            st.sidebar.success("✓ Modelo cargado")
        else:
            st.sidebar.error(f"❌ Modelo no encontrado: {model_path}")
            st.stop()
    except Exception as e:
        st.sidebar.error(f"Error cargando modelo: {e}")
        st.stop()

    # Tabs principales
    tab1, tab2, tab3 = st.tabs(["📤 Subir Imágenes", "📊 Resultados", "ℹ️ Información"])

    with tab1:
        st.header("Subir Imágenes")

        upload_option = st.radio(
            "Tipo de carga",
            options=["Imagen individual", "Múltiples imágenes", "ZIP"],
            horizontal=True
        )

        if upload_option == "Imagen individual":
            uploaded_file = st.file_uploader(
                "Seleccionar imagen",
                type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
                help="Formatos soportados: PNG, JPG, TIFF"
            )

            if uploaded_file:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Imagen Original")
                    image_display = Image.open(uploaded_file)
                    st.image(image_display, use_container_width=True)

                # Procesar
                metrics, overlay, original = process_uploaded_image(
                    uploaded_file,
                    inference_model,
                    pixel_size_um
                )

                with col2:
                    st.subheader("Segmentación")
                    st.image(overlay, use_container_width=True)

                # Métricas
                st.subheader("📈 Métricas Calculadas")

                # Clasificar daño
                damage_level = classify_damage_level(metrics['tail_dna_percent'])
                damage_class = f"damage-{damage_level.lower()}"

                # Mostrar métricas en columnas
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Tail DNA %", f"{metrics['tail_dna_percent']:.2f}%")

                with col2:
                    st.metric("Tail Moment", f"{metrics['tail_moment']:.2f}")

                with col3:
                    st.metric("Longitud Total (px)", f"{metrics['comet_length_px']:.0f}")

                with col4:
                    st.markdown(f"""
                    <div class='{damage_class}' style='font-size: 1.2rem; text-align: center; padding: 1rem;'>
                    Daño: {damage_level}
                    </div>
                    """, unsafe_allow_html=True)

                # Tabla detallada
                with st.expander("Ver métricas detalladas"):
                    metrics_df = pd.DataFrame([metrics]).T
                    metrics_df.columns = ['Valor']
                    st.dataframe(metrics_df, use_container_width=True)

                # Guardar en session state
                if 'results' not in st.session_state:
                    st.session_state.results = []

                if st.button("➕ Añadir a resultados"):
                    metrics['image_name'] = uploaded_file.name
                    st.session_state.results.append(metrics)
                    st.success(f"✓ Añadido: {uploaded_file.name}")

        elif upload_option == "Múltiples imágenes":
            uploaded_files = st.file_uploader(
                "Seleccionar imágenes",
                type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
                accept_multiple_files=True
            )

            if uploaded_files:
                if st.button("🚀 Procesar todas"):
                    results = []

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, file in enumerate(uploaded_files):
                        status_text.text(f"Procesando {file.name}...")

                        try:
                            metrics, _, _ = process_uploaded_image(
                                file,
                                inference_model,
                                pixel_size_um
                            )
                            metrics['image_name'] = file.name
                            results.append(metrics)
                        except Exception as e:
                            st.warning(f"Error en {file.name}: {e}")

                        progress_bar.progress((idx + 1) / len(uploaded_files))

                    status_text.text("✓ Procesamiento completado")

                    # Guardar resultados
                    st.session_state.results = results
                    st.success(f"✓ {len(results)} imágenes procesadas")

        elif upload_option == "ZIP":
            uploaded_zip = st.file_uploader("Seleccionar archivo ZIP", type=['zip'])

            if uploaded_zip:
                if st.button("🚀 Extraer y procesar"):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        # Extraer ZIP
                        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                            zip_ref.extractall(tmp_dir)

                        # Buscar imágenes
                        image_files = []
                        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
                            image_files.extend(Path(tmp_dir).rglob(ext))

                        st.info(f"Encontradas {len(image_files)} imágenes")

                        if image_files:
                            results = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for idx, img_path in enumerate(image_files):
                                status_text.text(f"Procesando {img_path.name}...")

                                try:
                                    with open(img_path, 'rb') as f:
                                        metrics, _, _ = process_uploaded_image(
                                            f,
                                            inference_model,
                                            pixel_size_um
                                        )
                                    metrics['image_name'] = img_path.name
                                    results.append(metrics)
                                except Exception as e:
                                    st.warning(f"Error en {img_path.name}: {e}")

                                progress_bar.progress((idx + 1) / len(image_files))

                            status_text.text("✓ Procesamiento completado")
                            st.session_state.results = results
                            st.success(f"✓ {len(results)} imágenes procesadas")

    with tab2:
        st.header("Resultados Acumulados")

        if 'results' in st.session_state and len(st.session_state.results) > 0:
            df = pd.DataFrame(st.session_state.results)

            # Estadísticas generales
            st.subheader("📊 Estadísticas Generales")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total de cometas", len(df))

            with col2:
                st.metric("Tail DNA % promedio", f"{df['tail_dna_percent'].mean():.2f}%")

            with col3:
                st.metric("Tail Moment promedio", f"{df['tail_moment'].mean():.2f}")

            with col4:
                damage_counts = df['tail_dna_percent'].apply(classify_damage_level).value_counts()
                st.metric("Daño predominante", damage_counts.index[0] if len(damage_counts) > 0 else "N/A")

            # Gráficas
            st.subheader("📈 Visualizaciones")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Distribución de Tail DNA %**")
                st.bar_chart(df['tail_dna_percent'])

            with col2:
                st.write("**Tail Moment por imagen**")
                st.line_chart(df['tail_moment'])

            # Tabla de resultados
            st.subheader("📋 Tabla de Resultados")

            display_cols = [
                'image_name', 'tail_dna_percent', 'tail_moment',
                'comet_length_px', 'head_intensity_total', 'tail_intensity_total'
            ]

            available_cols = [col for col in display_cols if col in df.columns]
            st.dataframe(df[available_cols], use_container_width=True)

            # Descargas
            st.subheader("💾 Descargar Resultados")

            col1, col2 = st.columns(2)

            with col1:
                # CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar CSV",
                    data=csv,
                    file_name="comet_results.csv",
                    mime="text/csv"
                )

            with col2:
                # Generar y descargar PDF
                if st.button("📄 Generar PDF"):
                    with st.spinner("Generando reporte PDF..."):
                        try:
                            from reports.generate_report import generate_pdf_report

                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                                generate_pdf_report(
                                    csv_path=st.session_state.results,
                                    output_path=tmp_pdf.name,
                                    title="Reporte Comet Assay AI"
                                )

                                with open(tmp_pdf.name, 'rb') as f:
                                    pdf_data = f.read()

                                st.download_button(
                                    label="📥 Descargar PDF",
                                    data=pdf_data,
                                    file_name="comet_report.pdf",
                                    mime="application/pdf"
                                )

                            Path(tmp_pdf.name).unlink()
                        except Exception as e:
                            st.error(f"Error generando PDF: {e}")

            # Limpiar resultados
            if st.button("🗑️ Limpiar resultados"):
                st.session_state.results = []
                st.rerun()

        else:
            st.info("No hay resultados aún. Sube y procesa imágenes en la pestaña anterior.")

    with tab3:
        st.header("Información del Sistema")

        st.markdown("""
        ### 🧬 Comet Assay AI

        Sistema automatizado de análisis de ensayo cometa basado en Deep Learning para
        evaluación de genotoxicidad.

        #### Características:
        - **Segmentación automática** de cabeza y cola mediante U-Net
        - **Cuantificación precisa** de métricas biológicas
        - **Procesamiento por lotes** para alta productividad
        - **Reportes profesionales** en PDF y CSV

        #### Métricas calculadas:
        - **Tail DNA %**: Porcentaje de ADN en la cola
        - **Tail Moment**: Producto de Tail DNA % y distancia entre centroides
        - **Longitudes**: Mediciones físicas del cometa
        - **Intensidades**: Fluorescencia total y media por región

        #### Clasificación de daño:
        - 🟢 **Mínimo**: Tail DNA % < 5%
        - 🟡 **Bajo**: Tail DNA % 5-20%
        - 🟠 **Moderado**: Tail DNA % 20-40%
        - 🔴 **Severo**: Tail DNA % > 40%

        #### ⚠️ Importante:
        Este sistema es una herramienta de apoyo para investigación. Los resultados deben
        ser validados por expertos y NO deben usarse con fines diagnósticos clínicos sin
        la supervisión apropiada.

        #### 📚 Referencias:
        - Collins et al. (1997) - Comet assay metrics
        - Ronneberger et al. (2015) - U-Net architecture
        - CASPLab - Software de referencia

        ---

        **Versión:** 1.0.0  
        **Licencia:** MIT  
        **Desarrollado con:** PyTorch, OpenCV, Streamlit
        """)


if __name__ == "__main__":
    main()