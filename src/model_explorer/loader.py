import os
from typing import List, Optional, Tuple, Dict
from pathlib import Path
from safetensors.torch import load_file as load_safetensors
from safetensors import safe_open
import gguf

from .tree import TensorInfo, MetadataInfo

class ModelLoader:
    def __init__(self, files: List[Path]):
        self.files = files
        self.tensors: List[TensorInfo] = []
        self.metadata: List[MetadataInfo] = []
        self.total_parameters = 0

    def load(self) -> None:
        for file_path in self.files:
            ext = file_path.suffix.lower()
            try:
                if ext == ".safetensors":
                    self.load_safetensors_file(file_path)
                elif ext == ".gguf":
                    self.load_gguf_file(file_path)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        # Deduplicate tensors by name (if multiple files have same tensors)
        unique_tensors = {}
        for t in self.tensors:
            if t.name not in unique_tensors:
                unique_tensors[t.name] = t
        self.tensors = list(unique_tensors.values())
        
        self.total_parameters = sum(t.num_elements for t in self.tensors)

    def load_safetensors_file(self, path: Path) -> None:
        # Read metadata first
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata()
            if metadata:
                for k, v in metadata.items():
                    self.metadata.append(MetadataInfo(k, v, "string"))
            
            for key in f.keys():
                tensor = f.get_slice(key)
                shape = tensor.get_shape()
                dtype = str(tensor.get_dtype())
                # Calculate size roughly
                num_elements = 1
                for dim in shape:
                    num_elements *= dim
                
                # Approximate bytes based on dtype string
                bytes_per_elem = 4 # Default to float32
                if "16" in dtype: bytes_per_elem = 2
                elif "8" in dtype: bytes_per_elem = 1
                elif "64" in dtype: bytes_per_elem = 8
                
                size_bytes = num_elements * bytes_per_elem
                
                self.tensors.append(TensorInfo(
                    name=key,
                    dtype=dtype,
                    shape=list(shape),
                    size_bytes=size_bytes,
                    num_elements=num_elements
                ))

    def load_gguf_file(self, path: Path) -> None:
        reader = gguf.GGUFReader(str(path))
        
        # Metadata
        for field in reader.fields.values():
            # Skip arrays for metadata display simplicity or format them
            val_str = str(field.parts[-1])
            if len(val_str) > 100: val_str = val_str[:97] + "..."
            self.metadata.append(MetadataInfo(
                name=field.name,
                value=val_str,
                value_type=str(field.types[-1].name)
            ))

        # Tensors
        for tensor in reader.tensors:
            shape = tensor.shape
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            
            self.tensors.append(TensorInfo(
                name=tensor.name,
                dtype=tensor.tensor_type.name,
                shape=list(shape),
                size_bytes=tensor.n_bytes,
                num_elements=num_elements
            ))
