import re
from dataclasses import dataclass
from typing import List, Dict, Union, Optional

@dataclass
class TensorInfo:
    name: str
    dtype: str
    shape: List[int]
    size_bytes: int
    num_elements: int

@dataclass
class MetadataInfo:
    name: str
    value: str
    value_type: str

@dataclass
class TreeNode:
    name: str
    # If it's a group
    children: Optional[List['TreeNode']] = None
    expanded: bool = False
    tensor_count: int = 0
    total_size: int = 0
    # If it's a tensor
    tensor_info: Optional[TensorInfo] = None
    # If it's metadata
    metadata_info: Optional[MetadataInfo] = None

    @property
    def is_group(self) -> bool:
        return self.children is not None

    @property
    def is_tensor(self) -> bool:
        return self.tensor_info is not None

    @property
    def is_metadata(self) -> bool:
        return self.metadata_info is not None

def natural_sort_key(text: str) -> List[Union[str, int]]:
    """
    Splits the string into a list of text and numbers for natural sorting.
    e.g., "layer10" -> ["layer", 10]
    """
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    
    return [convert(c) for c in re.split('([0-9]+)', text)]

class TreeBuilder:
    @staticmethod
    def build_tree_mixed(tensors: List[TensorInfo], metadata: List[MetadataInfo]) -> List[TreeNode]:
        tree = []

        # Add metadata as a separate group
        if metadata:
            metadata_children = []
            for meta in metadata:
                metadata_children.append(TreeNode(
                    name=meta.name,
                    metadata_info=meta
                ))
            
            # Sort metadata by name
            metadata_children.sort(key=lambda x: natural_sort_key(x.name))

            tree.append(TreeNode(
                name="🔧 Metadata",
                children=metadata_children,
                expanded=False,
                tensor_count=0,
                total_size=0
            ))

        # Build tensor tree
        tensor_tree = TreeBuilder.build_tree(tensors)
        tree.extend(tensor_tree)

        return tree

    @staticmethod
    def build_tree(tensors: List[TensorInfo]) -> List[TreeNode]:
        root_map: Dict[str, List[TensorInfo]] = {}

        for tensor in tensors:
            parts = tensor.name.split('.')
            if len(parts) > 1:
                prefix = parts[0]
                if prefix not in root_map:
                    root_map[prefix] = []
                root_map[prefix].append(tensor)
            else:
                if "_root" not in root_map:
                    root_map["_root"] = []
                root_map["_root"].append(tensor)

        tree = []
        for prefix, group_tensors in root_map.items():
            if prefix == "_root":
                for tensor in group_tensors:
                    tree.append(TreeNode(
                        name=tensor.name,
                        tensor_info=tensor
                    ))
            else:
                # Sort tensors within the group
                group_tensors.sort(key=lambda x: natural_sort_key(x.name))
                
                tensor_count = len(group_tensors)
                total_size = sum(t.size_bytes for t in group_tensors)
                
                children = TreeBuilder.build_subtree(group_tensors, prefix)
                
                tree.append(TreeNode(
                    name=prefix,
                    children=children,
                    expanded=True,
                    tensor_count=tensor_count,
                    total_size=total_size
                ))

        # Sort root nodes
        tree.sort(key=lambda x: natural_sort_key(x.name))
        return tree

    @staticmethod
    def build_subtree(tensors: List[TensorInfo], prefix: str) -> List[TreeNode]:
        groups: Dict[str, List[TensorInfo]] = {}
        direct_tensors: List[TensorInfo] = []

        for tensor in tensors:
            # Remove prefix + dot
            if tensor.name.startswith(prefix + "."):
                remaining = tensor.name[len(prefix) + 1:]
            else:
                remaining = tensor.name

            parts = remaining.split('.')
            
            if len(parts) == 1:
                direct_tensors.append(tensor)
            else:
                next_prefix = parts[0]
                if next_prefix not in groups:
                    groups[next_prefix] = []
                groups[next_prefix].append(tensor)

        result = []

        for tensor in direct_tensors:
            result.append(TreeNode(
                name=tensor.name,
                tensor_info=tensor
            ))

        for group_name, group_tensors in groups.items():
            tensor_count = len(group_tensors)
            total_size = sum(t.size_bytes for t in group_tensors)
            full_prefix = f"{prefix}.{group_name}"
            
            children = TreeBuilder.build_subtree(group_tensors, full_prefix)
            
            result.append(TreeNode(
                name=group_name,
                children=children,
                expanded=False,
                tensor_count=tensor_count,
                total_size=total_size
            ))

        result.sort(key=lambda x: natural_sort_key(x.name))
        return result
