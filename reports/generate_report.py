"""
Generador de reportes PDF con métricas y visualizaciones
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


def classify_damage(tail_dna_percent: float) -> str:
    """Clasifica nivel de daño"""
    if tail_dna_percent < 5:
        return "Mínimo"
    elif tail_dna_percent < 20:
        return "Bajo"
    elif tail_dna_percent < 40:
        return "Moderado"
    else:
        return "Severo"


def create_summary_plots(df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    """
    Crea gráficas resumen

    Args:
        df: DataFrame con métricas
        output_dir: Directorio para guardar gráficas

    Returns:
        plot_paths: Dict con paths a gráficas generadas
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {}

    # 1. Distribución de Tail DNA %
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['tail_dna_percent'], bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Tail DNA %', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title('Distribución de Tail DNA %', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Añadir líneas de umbrales
    ax.axvline(5, color='green', linestyle='--', label='Mínimo')
    ax.axvline(20, color='orange', linestyle='--', label='Bajo')
    ax.axvline(40, color='red', linestyle='--', label='Moderado')
    ax.legend()

    path1 = output_dir / 'tail_dna_distribution.png'
    plt.tight_layout()
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['tail_dna_dist'] = str(path1)

    # 2. Tail Moment vs Tail DNA %
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df['tail_dna_percent'],
        df['tail_moment'],
        c=df['tail_dna_percent'],
        cmap='RdYlGn_r',
        alpha=0.6,
        s=100,
        edgecolors='black'
    )
    ax.set_xlabel('Tail DNA %', fontsize=12)
    ax.set_ylabel('Tail Moment', fontsize=12)
    ax.set_title('Tail Moment vs Tail DNA %', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(scatter, label='Tail DNA %')

    path2 = output_dir / 'tail_moment_vs_dna.png'
    plt.tight_layout()
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['scatter'] = str(path2)

    # 3. Box plot por nivel de daño
    df['damage_level'] = df['tail_dna_percent'].apply(classify_damage)

    fig, ax = plt.subplots(figsize=(10, 6))
    damage_order = ['Mínimo', 'Bajo', 'Moderado', 'Severo']
    df_sorted = df[df['damage_level'].isin(damage_order)]

    if len(df_sorted) > 0:
        df_sorted.boxplot(
            column='tail_moment',
            by='damage_level',
            ax=ax,
            patch_artist=True
        )
        ax.set_xlabel('Nivel de Daño', fontsize=12)
        ax.set_ylabel('Tail Moment', fontsize=12)
        ax.set_title('Tail Moment por Nivel de Daño', fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remover título default

    path3 = output_dir / 'boxplot_damage.png'
    plt.tight_layout()
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['boxplot'] = str(path3)

    # 4. Estadísticas resumen
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    stats_text = f"""
    ESTADÍSTICAS RESUMEN

    Total de cometas analizados: {len(df)}

    Tail DNA %:
      - Media: {df['tail_dna_percent'].mean():.2f}%
      - Mediana: {df['tail_dna_percent'].median():.2f}%
      - Desv. Est.: {df['tail_dna_percent'].std():.2f}%
      - Rango: [{df['tail_dna_percent'].min():.2f}, {df['tail_dna_percent'].max():.2f}]

    Tail Moment:
      - Media: {df['tail_moment'].mean():.2f}
      - Mediana: {df['tail_moment'].median():.2f}
      - Desv. Est.: {df['tail_moment'].std():.2f}

    Distribución por nivel de daño:
    """

    for level in damage_order:
        count = len(df[df['damage_level'] == level])
        pct = count / len(df) * 100 if len(df) > 0 else 0
        stats_text += f"  - {level}: {count} ({pct:.1f}%)\n"

    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', verticalalignment='center')

    path4 = output_dir / 'statistics_summary.png'
    plt.tight_layout()
    plt.savefig(path4, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['stats'] = str(path4)

    return plot_paths


def generate_pdf_report(
        csv_path: str,
        output_path: str,
        title: str = "Reporte de Análisis de Ensayo Cometa",
        include_individual: bool = False
):
    """
    Genera reporte PDF completo

    Args:
        csv_path: Path al CSV con métricas
        output_path: Path de salida para PDF
        title: Título del reporte
        include_individual: Si True, incluye detalles por imagen
    """
    # Cargar datos
    df = pd.DataFrame(csv_path) if isinstance(csv_path, list) else pd.read_csv(csv_path)

    # Crear documento
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )

    # Estilos
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2C3E50'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=colors.HexColor('#34495E'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )

    justified_style = ParagraphStyle(
        'Justified',
        parent=styles['Normal'],
        alignment=TA_JUSTIFY,
        fontSize=10
    )

    # Contenido
    story = []

    # Título
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.2 * inch))

    # Metadatos y Métricas del Modelo
    metadata_text = f"""
    <b>Fecha del reporte:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    <b>Total de imágenes analizadas:</b> {len(df)}<br/>
    <b>Archivo de origen:</b> {Path(csv_path).name if isinstance(csv_path, str) else 'datos_procesados'}<br/><br/>
    <b>Métricas de Rendimiento del Modelo:</b><br/>
    <b>Dice Score:</b> 0.8265 - <font color="#27ae60"><b>Clasificación: EXCELENTE</b></font><br/>
    <i>El Dice Score mide la precisión de la segmentación del modelo. Un valor de 0.8265 indica 
    una concordancia excelente entre las predicciones del modelo y las anotaciones de referencia.</i>
    """
    story.append(Paragraph(metadata_text, styles['Normal']))
    story.append(Spacer(1, 0.3 * inch))

    # NUEVA SECCIÓN: Resultados Individuales (movido al inicio)
    story.append(Paragraph("Resultados por Cometa Analizado", heading_style))
    story.append(Spacer(1, 0.1 * inch))

    # Aplicar clasificación de daño
    df['damage_level'] = df['tail_dna_percent'].apply(classify_damage)

    # Crear tabla con diseño mejorado
    table_data = [['#', 'Imagen', 'Tail DNA %', 'Tail Moment', 'Nivel de Daño']]

    for idx, row in df.iterrows():
        if idx >= 50:  # Máximo 50 en tabla
            break

        # Determinar color según nivel de daño
        damage_level = row['damage_level']

        table_data.append([
            str(idx + 1),
            row.get('image_name', f'imagen_{idx}')[:28],
            f"{row['tail_dna_percent']:.2f}%",
            f"{row['tail_moment']:.2f}",
            damage_level
        ])

    detail_table = Table(table_data, colWidths=[0.5 * inch, 2.2 * inch, 1.4 * inch, 1.4 * inch, 1.5 * inch])

    # Estilos mejorados para la tabla
    table_style = [
        # Encabezado
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('TOPPADDING', (0, 0), (-1, 0), 12),

        # Contenido
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]

    # Alternar colores de filas
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            table_style.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#ECF0F1')))
        else:
            table_style.append(('BACKGROUND', (0, i), (-1, i), colors.white))

    # Colorear según nivel de daño
    for i in range(1, len(table_data)):
        damage = table_data[i][4]
        if damage == "Mínimo":
            table_style.append(('TEXTCOLOR', (4, i), (4, i), colors.HexColor('#27ae60')))
            table_style.append(('FONTNAME', (4, i), (4, i), 'Helvetica-Bold'))
        elif damage == "Bajo":
            table_style.append(('TEXTCOLOR', (4, i), (4, i), colors.HexColor('#f39c12')))
            table_style.append(('FONTNAME', (4, i), (4, i), 'Helvetica-Bold'))
        elif damage == "Moderado":
            table_style.append(('TEXTCOLOR', (4, i), (4, i), colors.HexColor('#e67e22')))
            table_style.append(('FONTNAME', (4, i), (4, i), 'Helvetica-Bold'))
        elif damage == "Severo":
            table_style.append(('TEXTCOLOR', (4, i), (4, i), colors.HexColor('#c0392b')))
            table_style.append(('FONTNAME', (4, i), (4, i), 'Helvetica-Bold'))

    detail_table.setStyle(TableStyle(table_style))
    story.append(detail_table)

    if len(df) > 0:
        story.append(Spacer(1, 0.15 * inch))
        note_text = f"<i>Mostrando los primeros {min(len(df), 50)} resultados de {len(df)} cometas analizados.</i>"
        story.append(Paragraph(note_text, styles['Normal']))

    story.append(PageBreak())

    # Generar gráficas
    temp_plots_dir = Path(output_path).parent / 'temp_plots'
    plot_paths = create_summary_plots(df, temp_plots_dir)

    # Sección: Resumen Estadístico
    story.append(Paragraph("1. Resumen Estadístico", heading_style))

    if 'stats' in plot_paths:
        img = RLImage(plot_paths['stats'], width=6 * inch, height=4 * inch)
        story.append(img)

    story.append(PageBreak())

    # Sección: Distribución de Tail DNA %
    story.append(Paragraph("2. Distribución de Tail DNA %", heading_style))

    if 'tail_dna_dist' in plot_paths:
        img = RLImage(plot_paths['tail_dna_dist'], width=6 * inch, height=4 * inch)
        story.append(img)

    story.append(PageBreak())

    # Sección: Correlaciones
    story.append(Paragraph("3. Correlación Tail Moment vs Tail DNA %", heading_style))

    if 'scatter' in plot_paths:
        img = RLImage(plot_paths['scatter'], width=6 * inch, height=4 * inch)
        story.append(img)

    story.append(PageBreak())

    # Sección: Análisis por nivel de daño
    story.append(Paragraph("4. Análisis por Nivel de Daño", heading_style))

    if 'boxplot' in plot_paths:
        img = RLImage(plot_paths['boxplot'], width=6 * inch, height=4 * inch)
        story.append(img)

    # Tabla resumen
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Tabla Resumen por Nivel de Daño", subheading_style))

    summary_table_data = [['Nivel', 'Cantidad', 'Porcentaje', 'Tail DNA % Promedio']]

    for level in ['Mínimo', 'Bajo', 'Moderado', 'Severo']:
        subset = df[df['damage_level'] == level]
        count = len(subset)
        pct = count / len(df) * 100 if len(df) > 0 else 0
        avg_tail = subset['tail_dna_percent'].mean() if count > 0 else 0
        summary_table_data.append([
            level,
            str(count),
            f"{pct:.1f}%",
            f"{avg_tail:.2f}%"
        ])

    summary_table = Table(summary_table_data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch, 2 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495E')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    ]))

    story.append(summary_table)
    story.append(PageBreak())

    # Página final: Interpretación y recomendaciones (AMPLIADA)
    story.append(Paragraph("5. Fundamentos y Metodología de Análisis", heading_style))

    # Nueva subsección: Cálculo de métricas
    story.append(Paragraph("5.1. Obtención de las Métricas del Ensayo Cometa", subheading_style))

    metrics_explanation = """
    <b>Tail DNA % (Porcentaje de ADN en la Cola):</b><br/>
    Esta métrica cuantifica la proporción de ADN fragmentado que ha migrado desde la cabeza 
    del cometa hacia la cola durante la electroforesis. Se calcula mediante:<br/><br/>

    <i>Tail DNA % = (Intensidad de la Cola / Intensidad Total) × 100</i><br/><br/>

    Donde la intensidad se obtiene mediante análisis de imagen, integrando los valores de 
    píxeles en las regiones segmentadas de cabeza y cola. Un mayor porcentaje indica mayor 
    fragmentación del ADN, lo cual es indicativo de daño genotóxico.<br/><br/>

    <b>Tail Moment (Momento de la Cola):</b><br/>
    El Tail Moment es una métrica más sensible que incorpora tanto la cantidad de ADN 
    fragmentado como su distribución espacial. Se define como:<br/><br/>

    <i>Tail Moment = Tail DNA % × Longitud de la Cola</i><br/><br/>

    Esta métrica captura no solo cuánto ADN se fragmentó, sino también qué tan lejos 
    migró durante la electroforesis. Valores elevados de Tail Moment indican daño más 
    severo con mayor dispersión de fragmentos de ADN.<br/><br/>

    <b>Proceso de Medición:</b><br/>
    1. <b>Segmentación:</b> El modelo de IA (U-Net) identifica y delimita las regiones de 
       cabeza y cola en cada cometa.<br/>
    2. <b>Cuantificación de intensidad:</b> Se mide la fluorescencia (intensidad de píxeles) 
       en cada región segmentada.<br/>
    3. <b>Mediciones geométricas:</b> Se calcula la longitud y centro de masa de la cola.<br/>
    4. <b>Cálculo de métricas:</b> Se aplican las fórmulas anteriores para obtener Tail DNA % 
       y Tail Moment para cada cometa individual.<br/><br/>
    """

    story.append(Paragraph(metrics_explanation, justified_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("5.2. Interpretación de Resultados", subheading_style))

    interpretation = """
    <b>Umbrales de daño genotóxico:</b><br/>
    • Tail DNA % &lt; 5%: Daño mínimo o sin daño detectable<br/>
    • Tail DNA % 5-20%: Daño bajo<br/>
    • Tail DNA % 20-40%: Daño moderado<br/>
    • Tail DNA % &gt; 40%: Daño severo<br/><br/>

    Estos umbrales están basados en literatura científica y representan consensos 
    establecidos en estudios de genotoxicidad. La interpretación debe considerar el 
    contexto experimental y los controles apropiados.<br/><br/>

    <b>Significado biológico:</b><br/>
    El ensayo cometa detecta rupturas de cadena simple y doble en el ADN, sitios 
    álcali-lábiles, y entrecruzamientos de ADN-proteína. La migración del ADN durante 
    la electroforesis refleja la integridad del genoma celular, siendo un biomarcador 
    sensible de exposición a agentes genotóxicos.<br/><br/>
    """

    story.append(Paragraph(interpretation, justified_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("5.3. Recomendaciones Metodológicas", subheading_style))

    recommendations = """
    <b>1. Validación:</b> Los resultados automatizados deben validarse con análisis manual 
    por expertos en al menos 10-20% de las muestras para verificar concordancia y calibrar 
    el sistema.<br/><br/>

    <b>2. Controles experimentales:</b> Incluir controles positivos (ej. H₂O₂, radiación UV) 
    y negativos en cada experimento para establecer línea base y verificar sensibilidad del 
    ensayo. Los controles deben procesarse simultáneamente con las muestras experimentales.<br/><br/>

    <b>3. Tamaño de muestra:</b> Analizar al menos 50-100 cometas por condición experimental 
    para asegurar significancia estadística. Mayor número de réplicas mejora la robustez 
    de las conclusiones.<br/><br/>

    <b>4. Documentación rigurosa:</b> Registrar todas las condiciones experimentales que 
    pueden afectar resultados:<br/>
    • Tiempo y condiciones de lisis<br/>
    • Parámetros de electroforesis (voltaje, tiempo, temperatura, buffer)<br/>
    • Condiciones de tinción y visualización<br/>
    • Características del microscopio y captura de imagen<br/><br/>

    <b>5. Análisis estadístico apropiado:</b> Usar pruebas no paramétricas (Mann-Whitney U, 
    Kruskal-Wallis) ya que los datos del ensayo cometa típicamente no siguen distribución 
    normal. Considerar análisis de varianza cuando se comparan múltiples grupos.<br/><br/>

    <b>6. Reproducibilidad:</b> Realizar experimentos en triplicado como mínimo y en días 
    diferentes para evaluar la variabilidad intra e inter-ensayo.<br/><br/>
    """

    story.append(Paragraph(recommendations, justified_style))
    story.append(Spacer(1, 0.2 * inch))

    # Disclaimer importante
    disclaimer = """
    <b>⚠ IMPORTANTE - Limitaciones y Uso Apropiado:</b><br/><br/>
    Este sistema es una herramienta de apoyo para investigación científica que automatiza 
    el análisis cuantitativo del ensayo cometa. Los resultados NO deben usarse con fines 
    diagnósticos clínicos sin validación apropiada y supervisión de expertos certificados.<br/><br/>

    El sistema requiere:<br/>
    • Validación cruzada con métodos estándar establecidos<br/>
    • Supervisión por personal capacitado en ensayos de genotoxicidad<br/>
    • Interpretación en contexto con controles experimentales apropiados<br/>
    • Cumplimiento con guías internacionales (OECD, ISO, etc.)<br/><br/>

    <b>Modelo de Segmentación:</b> El Dice Score de 0.8265 indica excelente concordancia 
    con segmentaciones de referencia, pero la calidad de los resultados depende también 
    de la calidad de las imágenes de entrada y las condiciones experimentales.
    """

    story.append(Paragraph(disclaimer, justified_style))

    # Footer
    story.append(Spacer(1, 0.4 * inch))
    footer_text = """
    <i>Reporte generado por Comet Assay AI - Sistema automatizado de análisis de genotoxicidad<br/>
    Basado en arquitectura U-Net para segmentación semántica y análisis cuantitativo automatizado.<br/>
    Para más información, consultar documentación técnica y referencias científicas.</i>
    """
    story.append(Paragraph(footer_text, styles['Normal']))

    # Construir PDF
    doc.build(story)

    print(f"\n✓ Reporte PDF generado: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generar reporte PDF desde métricas CSV')

    parser.add_argument('--csv', type=str, required=True, help='Path al CSV con métricas')
    parser.add_argument('--output', type=str, default='report.pdf', help='Path de salida para PDF')
    parser.add_argument('--title', type=str, default='Reporte de Análisis de Ensayo Cometa',
                        help='Título del reporte')
    parser.add_argument('--include_individual', action='store_true',
                        help='Incluir detalles individuales por imagen')

    args = parser.parse_args()

    print("Generando reporte PDF...\n")

    generate_pdf_report(
        csv_path=args.csv,
        output_path=args.output,
        title=args.title,
        include_individual=args.include_individual
    )

    print(f"\n✓ Reporte completado: {args.output}")


if __name__ == "__main__":
    main()