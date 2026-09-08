"""
zigzag_pdf_generator.py
-----------------------
Dedicated ReportLab PDF compiler for ZigZag swing analysis & sensitivity sweeps.
Outputs generated documents to specified results directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_zigzag_pdf_report(symbol: str, timeframe: str, sweep_df: pd.DataFrame, valid_swings: dict,
                               adr_5d: float, output_filename: str, output_dir: str = os.path.join("results", "zigzag")):
    """
    Generates a PDF analysis document containing visual distribution charts and parameter sweep tables.
    Automatically creates the target directory structure (defaults to results/zigzag).
    """
    # Create directory tree automatically (e.g., results/zigzag)
    os.makedirs(output_dir, exist_ok=True)

    # Resolve full output paths inside target directory
    pdf_path = os.path.join(output_dir, output_filename)
    chart_image_path = os.path.join(output_dir, f"temp_zigzag_{symbol}_{timeframe}.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(f"{symbol} ({timeframe}) - Swing Distribution & Sensitivity Analysis", fontsize=14, fontweight='bold')

    valid_mask = sweep_df['Status'] == 'VALID'
    axes[0].bar(sweep_df['Pct_ADR'], sweep_df['Swings_Per_Day'],
                color=np.where(valid_mask, '#1f77b4', '#d62728'), alpha=0.85)
    axes[0].set_title("Swings Per Day (Blue = Valid, Red = Discarded)")
    axes[0].set_xlabel("% of 5-Day ADR")
    axes[0].set_ylabel("Frequency (Swings / Day)")
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    baseline_key = "10.0%" if "10.0%" in valid_swings else (list(valid_swings.keys())[0] if valid_swings else None)

    if baseline_key:
        baseline_swings = valid_swings[baseline_key]['size']
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        values = np.percentile(baseline_swings, percentiles)

        axes[1].plot(percentiles, values, marker='o', color='#2ca02c', linewidth=2)
        axes[1].set_title(f"Swing Size Quantiles ({baseline_key} ADR Baseline)")
        axes[1].set_xlabel("Percentile Distribution")
        axes[1].set_ylabel("Swing Size (Points)")
        axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(chart_image_path, dpi=200)
    plt.close()

    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    custom_title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1A2B4C'),
        spaceAfter=10
    )

    story = [
        Paragraph(f"<b>Statistical Swing Analysis Report: {symbol} ({timeframe})</b>", custom_title_style),
        Paragraph(f"<b>5-Day Average Daily Range (ADR_5d):</b> {adr_5d:.2f} pts", styles['Normal']),
        Spacer(1, 15),
        RLImage(chart_image_path, width=540, height=243),
        Spacer(1, 15),
        Paragraph("<b>Parameter Sweep & Noise Filtering Matrix</b>", styles['Heading2']),
        Spacer(1, 5)
    ]

    table_data = [[col.replace('_', ' ') for col in sweep_df.columns]]
    for _, row in sweep_df.iterrows():
        table_data.append([str(val) for val in row.values])

    col_widths = [45, 55, 55, 55, 60, 50, 50, 50, 50, 50, 55]
    table = Table(table_data, colWidths=col_widths)

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A2B4C')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CCCCCC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')]),
    ]))

    story.append(table)
    doc.build(story)

    # Clean up temporary chart file
    if os.path.exists(chart_image_path):
        os.remove(chart_image_path)