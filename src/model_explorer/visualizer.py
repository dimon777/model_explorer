import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict
from pathlib import Path
from .loader import ModelLoader
from .utils import format_size

# Color definitions
COLORS = {
    "ENTRY": "#BBDEFB",      # Light Blue
    "BLOCK_START": "#E1BEE7", # Light Purple
    "ATTN_QKV": "#FFCDD2",    # Light Red
    "ATTN_OUT": "#C62828",    # Dark Red
    "MID_BLOCK": "#db92e8",   # Medium Purple
    "MLP_GATE_UP": "#C8E6C9", # Light Green
    "MLP_DOWN": "#2E7D32",    # Dark Green
    "EXIT_NORM": "#9C27B0",   # Dark Purple
    "EXIT_HEAD": "#1565C0",   # Dark Blue
    "DEFAULT": "#EEEEEE",     # Grey for groups/unknown
}

# Legend descriptions
LEGEND_INFO = [
    ("ENTRY", COLORS["ENTRY"], "Input / Embeddings"),
    ("BLOCK START", COLORS["BLOCK_START"], "Norm (Pre-Attn)"),
    ("MID-BLOCK", COLORS["MID_BLOCK"], "Norm (Pre-FFN)"),
    ("EXIT NORM", COLORS["EXIT_NORM"], "Final Norm"),
    ("ATTENTION", COLORS["ATTN_QKV"], "Query / Key / Value"),
    ("ATTENTION OUT", COLORS["ATTN_OUT"], "Attention Output"),
    ("MLP (FFN)", COLORS["MLP_GATE_UP"], "Gate / Up Projection"),
    ("MLP DOWN", COLORS["MLP_DOWN"], "Down Projection"),
    ("EXIT HEAD", COLORS["EXIT_HEAD"], "Output Head"),
    ("GROUP", COLORS["DEFAULT"], "Layer Group / Other"),
]

def get_layer_color(name: str) -> str:
    """
    Determines the color of a layer based on its name using GGUF and Safetensors patterns.
    """
    n = name.lower()
    
    # ENTRY
    if "token_embd.weight" in n or "embed_tokens.weight" in n:
        return COLORS["ENTRY"]
        
    # BLOCK START (Norm)
    if "attn_norm.weight" in n or "input_layernorm.weight" in n:
        return COLORS["BLOCK_START"]
        
    # ATTENTION (Q/K/V)
    if any(x in n for x in ["attn_q.weight", "attn_k.weight", "attn_v.weight", 
                           "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]):
        return COLORS["ATTN_QKV"]
        
    # ATTENTION (Out)
    if "attn_output.weight" in n or "self_attn.o_proj.weight" in n:
        return COLORS["ATTN_OUT"]
        
    # MID-BLOCK (Norm)
    if "ffn_norm.weight" in n or "post_attention_layernorm.weight" in n:
        return COLORS["MID_BLOCK"]
        
    # MLP (Gate/Up)
    if any(x in n for x in ["ffn_gate.weight", "ffn_up.weight", 
                           "mlp.gate_proj.weight", "mlp.up_proj.weight"]):
        return COLORS["MLP_GATE_UP"]
        
    # MLP (Down)
    if "ffn_down.weight" in n or "mlp.down_proj.weight" in n:
        return COLORS["MLP_DOWN"]
        
    # EXIT (Norm)
    if "output_norm.weight" in n or "model.norm.weight" in n:
        return COLORS["EXIT_NORM"]
        
    # EXIT (Head)
    if "output.weight" in n or "lm_head.weight" in n:
        return COLORS["EXIT_HEAD"]
        
    return COLORS["DEFAULT"]

def visualize_model(files: List[Path], input_paths: List[str] = None):
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

    # ... (rest of the function)

    # Construct title
    path_str = " ".join(input_paths) if input_paths else "Model"
    title = f"Model Structure<br>{path_str} ({total_tensors} tensors)"


    
    # 1. Calculate tensor counts for every node in the hierarchy
    node_counts: Dict[str, int] = {}
    node_counts["root"] = 0
    
    all_node_ids = {"root"}
    
    for tensor in loader.tensors:
        parts = tensor.name.split(".")
        current_path = "root"
        node_counts["root"] += 1
        
        for part in parts:
            node_id = f"{current_path}.{part}"
            all_node_ids.add(node_id)
            node_counts[node_id] = node_counts.get(node_id, 0) + 1
            current_path = node_id

    # 2. Prepare data for DataFrame
    data = []
    created_groups = set()
    
    # Add root node explicitly
    data.append({
        "id": "root",
        "parent": "",
        "name": f"Model ({node_counts['root']})",
        "value": 0,
        "color": COLORS["DEFAULT"],
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
                display_name = f"{part} ({count})"
                
                if is_leaf:
                    size_fmt = format_size(tensor.size_bytes)
                    # Determine color based on full tensor name
                    color = get_layer_color(tensor.name)
                    
                    data.append({
                        "id": node_id,
                        "parent": parent_id,
                        "name": display_name,
                        "value": tensor.size_bytes,
                        "color": color,
                        "is_tensor": True,
                        "hover_text": f"{tensor.name}<br>Shape: {tensor.shape}<br>Type: {tensor.dtype}<br>Size: {size_fmt}"
                    })
                else:
                    # Group node - use default grey
                    data.append({
                        "id": node_id,
                        "parent": parent_id,
                        "name": display_name,
                        "value": 0,
                        "color": COLORS["DEFAULT"],
                        "is_tensor": False,
                        "hover_text": f"Group: {part}<br>Tensors: {count}"
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
        hover_name='name',
        hover_data={'hover_text': True, 'value': False, 'id': False, 'parent': False, 'name': False, 'color': False},
    )
    
    # Apply colors manually since we have specific hex codes per node
    sunburst_fig.update_traces(marker=dict(colors=df['color']))
    
    # Create main figure with subplots
    # 2 rows, 2 columns. 
    # Left column (Sunburst) spans both rows.
    # Right column: Row 1 = Metadata, Row 2 = Legend
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.7, 0.3],
        row_heights=[0.5, 0.5],
        specs=[
            [{"type": "domain", "rowspan": 2}, {"type": "table"}],
            [None, {"type": "table"}]
        ],
        subplot_titles=(title, "Metadata", "Legend")
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

    # Add Legend Table trace
    legend_names = [x[0] for x in LEGEND_INFO]
    legend_colors = [x[1] for x in LEGEND_INFO]
    legend_desc = [x[2] for x in LEGEND_INFO]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Category", "Meaning"],
                font=dict(size=12, color="white"),
                align="left",
                fill_color="#444"
            ),
            cells=dict(
                values=[legend_names, legend_desc],
                align="left",
                font=dict(size=11),
                fill_color=[legend_colors, "#F5F5F5"] # Color the Category column
            )
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="", #title,
        margin=dict(t=60, l=10, r=10, b=10),
    )

    print("Opening visualization in browser...")
    fig.show()
