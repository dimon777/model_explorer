import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict
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

    total_tensors = len(loader.tensors)
    print(f"Found {total_tensors} tensors. Preparing visualization...")
    
    # 1. Calculate tensor counts for every node in the hierarchy
    # Map: node_id -> tensor_count
    node_counts: Dict[str, int] = {}
    
    # Initialize root
    node_counts["root"] = 0
    
    # Set of all unique node IDs to ensure we create them later
    all_node_ids = {"root"}
    
    for tensor in loader.tensors:
        parts = tensor.name.split(".")
        
        # Walk down the path, creating/updating counts for each node
        current_path = "root"
        
        # Increment root count for every tensor
        node_counts["root"] += 1
        
        for part in parts:
            node_id = f"{current_path}.{part}"
            all_node_ids.add(node_id)
            
            # Increment count for this node
            node_counts[node_id] = node_counts.get(node_id, 0) + 1
            
            current_path = node_id

    # 2. Prepare data for DataFrame
    data = []
    
    # Helper to track created groups for the DataFrame loop
    # (We iterate tensors again to build the structure, but we could also iterate all_node_ids if we reconstructed the tree)
    # To keep the structure clean and ordered, we'll iterate tensors and create nodes as we encounter them,
    # but use the pre-calculated counts.
    created_groups = set()
    
    # Add root node explicitly
    data.append({
        "id": "root",
        "parent": "",
        "name": "Model",
        "value": 0,
        "color_val": node_counts["root"] / total_tensors if total_tensors > 0 else 0,
        "tensor_count": node_counts["root"],
        "is_tensor": False
    })
    created_groups.add("root")
    
    for tensor in loader.tensors:
        parts = tensor.name.split(".")
        current_path = "root"
        
        for i, part in enumerate(parts):
            node_id = f"{current_path}.{part}"
            parent_id = current_path
            
            if node_id not in created_groups:
                is_leaf = (i == len(parts) - 1)
                count = node_counts[node_id]
                normalized_count = count / total_tensors if total_tensors > 0 else 0
                
                if is_leaf:
                    size_fmt = format_size(tensor.size_bytes)
                    data.append({
                        "id": node_id,
                        "parent": parent_id,
                        "name": part,
                        "value": tensor.size_bytes,
                        "color_val": normalized_count,
                        "tensor_count": count,
                        "is_tensor": True,
                        "hover_text": f"{tensor.name}<br>Shape: {tensor.shape}<br>Type: {tensor.dtype}<br>Size: {size_fmt}<br>Count: {count} ({(normalized_count*100):.1f}%)"
                    })
                else:
                    # Group node
                    data.append({
                        "id": node_id,
                        "parent": parent_id,
                        "name": part,
                        "value": 0, # Aggregated by plotly
                        "color_val": normalized_count,
                        "tensor_count": count,
                        "is_tensor": False,
                        "hover_text": f"Group: {part}<br>Tensors: {count} ({(normalized_count*100):.1f}%)"
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
        color_continuous_scale='Blues', # Light blue to Dark blue
        range_color=[0, 1], # Normalize 0 to 1
        hover_name='name',
        hover_data={'hover_text': True, 'value': False, 'color_val': False, 'id': False, 'parent': False, 'name': False, 'tensor_count': False},
    )
    
    # Create main figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        specs=[[{"type": "domain"}, {"type": "table"}]],
        subplot_titles=("Model Structure", "Metadata")
    )
    
    # Add Sunburst trace
    for trace in sunburst_fig.data:
        trace.hovertemplate = "<b>%{label}</b><br>%{customdata[0]}<extra></extra>"
        trace.textinfo = "label+percent entry"
        fig.add_trace(trace, row=1, col=1)

    # Add Metadata Table trace
    if loader.metadata:
        meta_names = [m.name for m in loader.metadata]
        meta_values = [m.value for m in loader.metadata]
        
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
        fig.add_trace(
            go.Table(
                header=dict(values=["No Metadata"]),
                cells=dict(values=[[]])
            ),
            row=1, col=2
        )

    # Update layout
    fig.update_layout(
        title_text=f"Model Structure Visualization ({total_tensors} tensors)",
        margin=dict(t=60, l=10, r=10, b=10),
        coloraxis=sunburst_fig.layout.coloraxis,
        coloraxis_colorbar=dict(
            title="Tensor Count Fraction", 
            x=0.65,
            tickformat=".0%"
        ),
    )

    print("Opening visualization in browser...")
    fig.show()
