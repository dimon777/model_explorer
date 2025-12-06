import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List
from pathlib import Path
from .loader import ModelLoader
from .utils import format_size

def visualize_model(files: List[Path]):
    """
    Visualizes the model structure as an interactive sunburst chart with metadata inset.
    """
    print(f"Loading {len(files)} file(s) for visualization...")
    loader = ModelLoader(files)
    loader.load()
    
    if not loader.tensors:
        print("No tensors found to visualize.")
        return

    print(f"Found {len(loader.tensors)} tensors. Preparing visualization...")
    
    # Prepare data for Sunburst DataFrame
    data = []
    
    # Add root node
    data.append({
        "id": "root",
        "parent": "",
        "name": "Model",
        "value": 0,
        "color_val": 0,
        "is_tensor": False
    })
    
    # Helper to track created groups
    created_groups = {"root"}
    
    # Process tensors
    for tensor in loader.tensors:
        parts = tensor.name.split(".")
        
        # Create hierarchy
        current_path = "root"
        for i, part in enumerate(parts):
            node_id = f"{current_path}.{part}"
            parent_id = current_path
            
            if node_id not in created_groups:
                is_leaf = (i == len(parts) - 1)
                
                # Leaf node (Tensor)
                if is_leaf:
                    size_fmt = format_size(tensor.size_bytes)
                    data.append({
                        "id": node_id,
                        "parent": parent_id,
                        "name": part,
                        "value": tensor.size_bytes,
                        "color_val": 1, # Count as 1 tensor
                        "is_tensor": True,
                        "hover_text": f"{tensor.name}<br>Shape: {tensor.shape}<br>Type: {tensor.dtype}<br>Size: {size_fmt}"
                    })
                else:
                    # Group node
                    data.append({
                        "id": node_id,
                        "parent": parent_id,
                        "name": part,
                        "value": 0, # Will be aggregated by plotly
                        "color_val": 0,
                        "is_tensor": False,
                        "hover_text": f"Group: {part}"
                    })
                created_groups.add(node_id)
            
            current_path = node_id

    df = pd.DataFrame(data)
    
    # Create Sunburst Figure
    sunburst_fig = px.sunburst(
        df,
        names='name',
        parents='parent',
        ids='id',
        values='value',
        color='color_val',
        color_continuous_scale='Viridis',
        hover_name='name',
        hover_data={'hover_text': True, 'value': False, 'color_val': False, 'id': False, 'parent': False, 'name': False},
    )
    
    # Create main figure with subplots
    # 1 row, 2 columns. Left: Sunburst (larger), Right: Metadata Table (smaller)
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        specs=[[{"type": "domain"}, {"type": "table"}]],
        subplot_titles=("Model Structure", "Metadata")
    )
    
    # Add Sunburst trace
    # We extract the trace from the px figure and add it to the subplot
    for trace in sunburst_fig.data:
        # Update trace to use custom hovertemplate
        trace.hovertemplate = "<b>%{label}</b><br>%{customdata[0]}<extra></extra>"
        trace.textinfo = "label+percent entry"
        fig.add_trace(trace, row=1, col=1)

    # Add Metadata Table trace
    if loader.metadata:
        meta_names = [m.name for m in loader.metadata]
        meta_values = [m.value for m in loader.metadata]
        meta_types = [m.value_type for m in loader.metadata]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Key", "Value"],
                    font=dict(size=12, color="white"),
                    align="left",
                    fill_color="#444"
                ),
                cells=dict(
                    values=[meta_names, meta_values],
                    align="left",
                    font=dict(size=11),
                    fill_color="#F5F5F5"
                )
            ),
            row=1, col=2
        )
    else:
        # Add empty table or text if no metadata
        fig.add_trace(
            go.Table(
                header=dict(values=["No Metadata"]),
                cells=dict(values=[[]])
            ),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        title_text=f"Model Structure Visualization ({len(loader.tensors)} tensors)",
        margin=dict(t=60, l=10, r=10, b=10),
        coloraxis=sunburst_fig.layout.coloraxis, # Copy coloraxis from px figure
        coloraxis_colorbar=dict(title="Tensor Count", x=0.65), # Position colorbar between plots
    )

    print("Opening visualization in browser...")
    fig.show()
