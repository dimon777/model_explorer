from typing import List

def format_shape(shape: List[int]) -> str:
    """Format a tensor shape as a string."""
    return f"({', '.join(map(str, shape))})"

def format_size(size_bytes: int) -> str:
    """Format a byte size into human-readable units."""
    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    unit_idx = 0

    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1

    if unit_idx == 0:
        return f"{int(size)} {units[unit_idx]}"
    else:
        return f"{size:.1f} {units[unit_idx]}"

def format_parameters(params: int) -> str:
    """Format a parameter count into human-readable units (K, M, B)."""
    if params < 1_000:
        return str(params)
    elif params < 1_000_000:
        return f"{params / 1_000.0:.1f}K"
    elif params < 1_000_000_000:
        return f"{params / 1_000_000.0:.1f}M"
    else:
        return f"{params / 1_000_000_000.0:.1f}B"
