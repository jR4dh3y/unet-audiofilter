#!/usr/bin/env python3
"""
Generate Data Flow Diagram for Speech Enhancement System
Creates a visual representation of the audio processing pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_dataflow_diagram():
    """Create and save the data flow diagram"""
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#e74c3c',      # Red
        'process': '#3498db',    # Blue  
        'model': '#9b59b6',      # Purple
        'output': '#27ae60',     # Green
        'eval': '#f39c12',       # Orange
        'arrow': '#34495e'       # Dark gray
    }
    
    # Define box positions and sizes
    boxes = [
        # (x, y, width, height, text, color_key)
        (0.5, 6.5, 1.8, 0.8, 'Noisy Audio\nInput', 'input'),
        (3, 6.5, 1.8, 0.8, 'STFT\nPreprocessing', 'process'),
        (5.5, 6.5, 1.8, 0.8, 'Magnitude/Phase\nSeparation', 'process'),
        (8, 6.5, 1.8, 0.8, 'U-Net Model\nEnhancement', 'model'),
        (8, 4.5, 1.8, 0.8, 'Enhanced\nMagnitude', 'model'),
        (5.5, 4.5, 1.8, 0.8, 'ISTFT\nReconstruction', 'process'),
        (3, 4.5, 1.8, 0.8, 'Clean Audio\nOutput', 'output'),
        (0.5, 4.5, 1.8, 0.8, 'Quality\nEvaluation', 'eval'),
        (1.5, 2.5, 2, 0.8, 'Metrics:\nPESQ, STOI, SNR', 'eval'),
        (5, 2.5, 2, 0.8, 'Visualization:\nSpectrograms', 'eval'),
        (8, 2.5, 1.8, 0.8, 'Audio Files:\nWAV, MP3, FLAC', 'output')
    ]
    
    # Draw boxes
    drawn_boxes = []
    for x, y, w, h, text, color_key in boxes:
        # Create fancy box
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.1",
            facecolor=colors[color_key],
            edgecolor='white',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(x + w/2, y + h/2, text, 
               ha='center', va='center', 
               fontsize=10, fontweight='bold',
               color='white',
               wrap=True)
        
        drawn_boxes.append((x + w/2, y + h/2))
    
    # Define arrows (from_index, to_index)
    arrow_connections = [
        (0, 1),   # Input → STFT
        (1, 2),   # STFT → Magnitude/Phase
        (2, 3),   # Magnitude/Phase → U-Net
        (3, 4),   # U-Net → Enhanced Magnitude
        (4, 5),   # Enhanced Magnitude → ISTFT
        (5, 6),   # ISTFT → Clean Output
        (6, 7),   # Clean Output → Evaluation
        (7, 8),   # Evaluation → Metrics
        (7, 9),   # Evaluation → Visualization
        (6, 10),  # Clean Output → Audio Files
    ]
    
    # Draw arrows
    for from_idx, to_idx in arrow_connections:
        from_x, from_y = drawn_boxes[from_idx]
        to_x, to_y = drawn_boxes[to_idx]
        
        # Calculate arrow positions
        if from_y == to_y:  # Horizontal arrow
            start_x = from_x + 0.9 if from_x < to_x else from_x - 0.9
            end_x = to_x - 0.9 if from_x < to_x else to_x + 0.9
            start_y = end_y = from_y
        else:  # Vertical arrow
            start_x = end_x = from_x
            start_y = from_y - 0.4 if from_y > to_y else from_y + 0.4
            end_y = to_y + 0.4 if from_y > to_y else to_y - 0.4
        
        # Create arrow
        arrow = patches.FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            arrowstyle='->', 
            mutation_scale=20,
            color=colors['arrow'],
            linewidth=2.5,
            alpha=0.8
        )
        ax.add_patch(arrow)
    
    # Add title
    ax.text(5, 7.7, 'AI Speech Enhancement - Data Flow Pipeline', 
           ha='center', va='center', 
           fontsize=16, fontweight='bold',
           color='#2c3e50')
    
    # Add phase labels
    ax.text(0.5, 0.5, 'Input Phase', ha='left', va='center', 
           fontsize=12, fontweight='bold', color=colors['input'])
    ax.text(4, 0.5, 'Processing Phase', ha='center', va='center', 
           fontsize=12, fontweight='bold', color=colors['process'])
    ax.text(8.5, 0.5, 'AI Enhancement', ha='center', va='center', 
           fontsize=12, fontweight='bold', color=colors['model'])
    ax.text(2, 1.5, 'Output & Evaluation', ha='center', va='center', 
           fontsize=12, fontweight='bold', color=colors['output'])
    
    # Add technical details
    tech_details = [
        "• STFT: n_fft=1024, hop_length=256",
        "• U-Net: 1.9M parameters, 3-layer depth",
        "• Processing: 4-second chunks with overlap",
        "• Output: 16kHz sample rate, multiple formats"
    ]
    
    for i, detail in enumerate(tech_details):
        ax.text(0.2, 1.2 - i*0.2, detail, 
               fontsize=9, color='#34495e',
               bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor='white', alpha=0.8))
    
    # Save the diagram
    plt.tight_layout()
    plt.savefig('dataflow_diagram.png', 
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('dataflow_diagram.pdf', 
                bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("✅ Data flow diagram saved as:")
    print("   - dataflow_diagram.png (for presentation)")
    print("   - dataflow_diagram.pdf (high quality)")
    
    # Show the plot
    plt.show()

def create_simplified_diagram():
    """Create a simplified version for slides"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Simplified flow
    steps = [
        (1, 1.5, "Audio\nInput", '#e74c3c'),
        (3, 1.5, "STFT", '#3498db'),
        (5, 1.5, "U-Net", '#9b59b6'),
        (7, 1.5, "ISTFT", '#3498db'),
        (9, 1.5, "Enhanced\nAudio", '#27ae60'),
        (11, 1.5, "Metrics", '#f39c12')
    ]
    
    # Draw simplified boxes and arrows
    for i, (x, y, text, color) in enumerate(steps):
        # Box
        box = FancyBboxPatch(
            (x-0.6, y-0.4), 1.2, 0.8,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='white',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(box)
        
        # Text
        ax.text(x, y, text, ha='center', va='center', 
               fontsize=11, fontweight='bold', color='white')
        
        # Arrow to next step
        if i < len(steps) - 1:
            arrow = patches.FancyArrowPatch(
                (x + 0.6, y), (steps[i+1][0] - 0.6, y),
                arrowstyle='->', mutation_scale=18,
                color='#34495e', linewidth=2.5
            )
            ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('dataflow_simple.png', 
                dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("✅ Simplified diagram saved as dataflow_simple.png")
    plt.show()

if __name__ == "__main__":
    print("🎨 Generating Speech Enhancement Data Flow Diagrams...")
    print("=" * 50)
    
    # Create main detailed diagram
    create_dataflow_diagram()
    
    print("\n" + "=" * 50)
    
    # Create simplified version
    create_simplified_diagram()
    
    print("\n🎉 All diagrams generated successfully!")
    print("Use 'dataflow_diagram.png' in your presentation.")
