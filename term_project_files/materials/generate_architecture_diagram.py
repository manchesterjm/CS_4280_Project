"""
Generate Architecture Diagram for BiLSTM Cluster Model
Creates a visual representation of the model architecture for research paper

Requirements: matplotlib, graphviz (optional)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    """Create detailed architecture diagram using matplotlib"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')

    # Define colors
    color_input = '#E8F4F8'
    color_embed = '#FFE5CC'
    color_lstm = '#D4E6F1'
    color_fc = '#E8DAEF'
    color_output = '#D5F4E6'

    # Layer positions (x_center, y_center, width, height)
    layers = [
        # (name, y_pos, color, text)
        ("Input\nLight Curve Window\n(batch, 2048, 1)", 19, color_input, "2048 flux measurements"),
        ("Cluster ID\n(0-4)", 18, color_input, "K-means assignment"),
        ("Cluster Embedding\n5 → 32 dim", 16.5, color_embed, "Learnable embeddings"),
        ("BiLSTM Layer 1\n256 hidden × 2 directions", 14.5, color_lstm, "512 outputs"),
        ("Layer Norm + Dropout(0.4)", 13.5, color_lstm, "Regularization"),
        ("BiLSTM Layer 2\n256 hidden × 2 directions", 12, color_lstm, "512 outputs"),
        ("Layer Norm + Dropout(0.4)", 11, color_lstm, "Regularization"),
        ("BiLSTM Layer 3\n256 hidden × 2 directions", 9.5, color_lstm, "512 outputs"),
        ("Concatenate\n[LSTM_out, Cluster_emb]", 8, color_fc, "512 + 32 = 544 dim"),
        ("FC1: 544 → 256\n+ BatchNorm + ReLU + Dropout", 6.5, color_fc, ""),
        ("FC2: 256 → 128\n+ BatchNorm + ReLU + Dropout", 5, color_fc, ""),
        ("FC3: 128 → 1", 3.5, color_fc, "Logits"),
        ("Sigmoid Activation", 2, color_output, ""),
        ("Output Probability\n[0, 1]", 0.5, color_output, "Planet probability"),
    ]

    # Draw boxes
    box_width = 7
    box_height = 0.8

    for i, (name, y_pos, color, subtitle) in enumerate(layers):
        # Main box
        box = FancyBboxPatch(
            (5 - box_width/2, y_pos - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            linewidth=2
        )
        ax.add_patch(box)

        # Main text
        ax.text(5, y_pos, name,
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                multialignment='center')

        # Subtitle
        if subtitle:
            ax.text(5, y_pos - 0.5, subtitle,
                    ha='center', va='top',
                    fontsize=8, style='italic',
                    color='gray')

    # Draw arrows between layers
    arrow_props = dict(
        arrowstyle='->,head_width=0.4,head_length=0.4',
        lw=2,
        color='black'
    )

    # Connect layers
    connections = [
        (19, 18), (18, 16.5),  # Input to cluster embedding
        (19, 14.5), (16.5, 8),  # Light curve to LSTM, embedding to concat
        (14.5, 13.5), (13.5, 12),  # LSTM chain
        (12, 11), (11, 9.5),
        (9.5, 8), (8, 6.5),  # Concat to FC
        (6.5, 5), (5, 3.5), (3.5, 2), (2, 0.5)  # FC chain
    ]

    for start_y, end_y in connections:
        if (start_y == 19 and end_y == 14.5):
            # Curve for main input
            arrow = FancyArrowPatch(
                (5, start_y - 0.4), (5, end_y + 0.4),
                **arrow_props
            )
        elif (start_y == 16.5 and end_y == 8):
            # Curve for embedding to concat
            arrow = FancyArrowPatch(
                (8.5, start_y - 0.4), (8.5, end_y + 0.4),
                connectionstyle="arc3,rad=.3",
                **arrow_props
            )
        else:
            arrow = FancyArrowPatch(
                (5, start_y - 0.4), (5, end_y + 0.4),
                **arrow_props
            )
        ax.add_patch(arrow)

    # Add side annotations
    ax.text(0.5, 14.5, "Sequence\nEncoder",
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
            ha='center', va='center')

    ax.text(0.5, 6.5, "Classifier\nHead",
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.5),
            ha='center', va='center')

    # Title
    ax.text(5, 20.5, 'Cluster-Enhanced BiLSTM Architecture',
            fontsize=16, fontweight='bold', ha='center')

    # Add parameter count
    param_text = "Total Parameters: ~2.1M\nTrainable: 2.1M"
    ax.text(9.5, 0.5, param_text,
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
            ha='right', va='bottom')

    plt.tight_layout()
    return fig

def create_data_pipeline_flowchart():
    """Create data processing pipeline flowchart"""

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Define colors for different stages
    color_data = '#E8F4F8'
    color_process = '#FFF4E6'
    color_feature = '#E8DAEF'
    color_model = '#D5F4E6'

    # Pipeline stages
    stages = [
        ("Raw TESS Light Curves\n(FITS files)", 13, color_data, 4),
        ("Preprocessing\n• Detrending\n• Normalization\n• Quality filtering", 11, color_process, 4),
        ("Window Extraction\n• Phase folding (BLS period)\n• 2048-point windows\n• Positive: phase=0±0.05\n• Negative: |phase-0.5|>0.18", 8.5, color_process, 4),
        ("Feature Engineering\n• Period (P)\n• Depth (δ)\n• Duration (T)\n• BLS power (S/N)", 6, color_feature, 4),
        ("K-means Clustering\nk=5 clusters on\n[P, δ, T, S/N]", 4, color_feature, 4),
        ("BiLSTM Training\nInput: Windows + Cluster IDs\nOutput: Planet probability", 1.5, color_model, 4),
    ]

    # Draw stages
    for name, y_pos, color, width in stages:
        box = FancyBboxPatch(
            (6 - width/2, y_pos - 0.6),
            width, 1.2,
            boxstyle="round,pad=0.15",
            edgecolor='black',
            facecolor=color,
            linewidth=2.5
        )
        ax.add_patch(box)

        ax.text(6, y_pos, name,
                ha='center', va='center',
                fontsize=10, fontweight='bold',
                multialignment='center')

    # Draw arrows
    arrow_props = dict(
        arrowstyle='->,head_width=0.5,head_length=0.5',
        lw=3,
        color='black'
    )

    arrow_positions = [
        (13 - 0.6, 11 + 0.6),
        (11 - 0.6, 8.5 + 0.6),
        (8.5 - 0.6, 6 + 0.6),
        (6 - 0.6, 4 + 0.6),
        (4 - 0.6, 1.5 + 0.6),
    ]

    for start_y, end_y in arrow_positions:
        arrow = FancyArrowPatch(
            (6, start_y), (6, end_y),
            **arrow_props
        )
        ax.add_patch(arrow)

    # Add side annotations for data statistics
    annotations = [
        (10, 13, "~100 confirmed\nexoplanet hosts", "left"),
        (10, 8.5, "655 windows total\n150 positive\n505 negative", "left"),
        (10, 4, "5 clusters capture\nstellar diversity", "left"),
        (10, 1.5, "AUC: 0.6947\nF1: 0.34", "left"),
    ]

    for x, y, text, align in annotations:
        ax.text(x, y, text,
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
                ha=align, va='center',
                multialignment='left')

    # Title
    ax.text(6, 14, 'End-to-End Data Pipeline',
            fontsize=16, fontweight='bold', ha='center')

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Generate architecture diagram
    print("Generating architecture diagram...")
    fig1 = create_architecture_diagram()
    fig1.savefig('C:\\CS_4280_Project\\research_paper\\architecture_diagram.png',
                 dpi=300, bbox_inches='tight')
    print("Saved: architecture_diagram.png")

    # Generate data pipeline flowchart
    print("Generating data pipeline flowchart...")
    fig2 = create_data_pipeline_flowchart()
    fig2.savefig('C:\\CS_4280_Project\\research_paper\\data_pipeline.png',
                 dpi=300, bbox_inches='tight')
    print("Saved: data_pipeline.png")

    print("\nDiagrams generated successfully!")
    plt.show()
