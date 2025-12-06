import os
from typing import List, Optional, Tuple
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Tree, Static, Input, Label
from textual.containers import Container, Vertical, Horizontal
from textual.screen import ModalScreen
from textual.binding import Binding
from textual.reactive import reactive
from textual import events

from safetensors.torch import load_file as load_safetensors
from safetensors import safe_open
import gguf

from .tree import TreeBuilder, TreeNode, TensorInfo, MetadataInfo
from .utils import format_shape, format_size, format_parameters
from .loader import ModelLoader

class DetailScreen(ModalScreen):
    """Screen for showing details of a tensor or metadata."""
    
    def __init__(self, title: str, content: str):
        super().__init__()
        self.title_text = title
        self.content_text = content

    def compose(self) -> ComposeResult:
        yield Container(
            Label(self.title_text, classes="detail-title"),
            Static(self.content_text, classes="detail-content"),
            Label("Press any key to return...", classes="detail-footer"),
            classes="detail-container"
        )

    def on_key(self, event: events.Key) -> None:
        self.dismiss()

class SafetensorsExplorerApp(App):
    """A Textual app to explore SafeTensors and GGUF files."""

    CSS = """
    Screen {
        align: center middle;
    }

    .detail-container {
        width: 80%;
        height: 80%;
        border: solid green;
        background: $surface;
        padding: 1 2;
        align: center middle;
    }

    .detail-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
        width: 100%;
        background: $primary;
        color: $text;
    }

    .detail-content {
        height: 1fr;
        overflow-y: auto;
    }

    .detail-footer {
        text-align: center;
        margin-top: 1;
        color: $text-muted;
    }

    Tree {
        height: 1fr;
    }
    
    #search-container {
        height: auto;
        dock: top;
        display: none;
    }
    
    #search-input {
        width: 100%;
    }

    .search-active #search-container {
        display: block;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("/", "toggle_search", "Search"),
        Binding("escape", "clear_search", "Clear Search"),
        Binding("space", "show_details", "Show Details"),
        Binding("enter", "select_node", "Expand/Details"),
    ]



    def __init__(self, files: List[Path]):
        super().__init__()
        self.files = files
        self.loader = ModelLoader(files)
        self.tensors: List[TensorInfo] = []
        self.metadata: List[MetadataInfo] = []
        self.root_nodes: List[TreeNode] = []
        self.total_parameters = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Input(placeholder="Search...", id="search-input"),
            id="search-container"
        )
        yield Tree("Root", id="file-tree")
        yield Footer()

    def on_mount(self) -> None:
        self.load_files()
        self.build_tree()
        self.title = f"SafeTensors Explorer - {len(self.files)} file(s)"

    def load_files(self) -> None:
        self.loader.load()
        self.tensors = self.loader.tensors
        self.metadata = self.loader.metadata
        self.total_parameters = self.loader.total_parameters

    def build_tree(self, filter_text: str = "") -> None:
        tree_widget = self.query_one("#file-tree", Tree)
        tree_widget.clear()
        tree_widget.root.expand()

        # Filter tensors if needed
        filtered_tensors = self.tensors
        filtered_metadata = self.metadata
        
        if filter_text:
            filtered_tensors = [t for t in self.tensors if filter_text.lower() in t.name.lower()]
            filtered_metadata = [m for m in self.metadata if filter_text.lower() in m.name.lower()]

        nodes = TreeBuilder.build_tree_mixed(filtered_tensors, filtered_metadata)
        
        for node in nodes:
            self.add_node_to_tree(tree_widget.root, node)

    def add_node_to_tree(self, parent_node, node_data: TreeNode) -> None:
        label = self.format_node_label(node_data)
        tree_node = parent_node.add(label, data=node_data, expand=node_data.expanded)
        
        if node_data.children:
            for child in node_data.children:
                self.add_node_to_tree(tree_node, child)
        else:
            tree_node.allow_expand = False

    def format_node_label(self, node: TreeNode) -> str:
        if node.is_group:
            return f"📁 {node.name} ({node.tensor_count} tensors, {format_size(node.total_size)})"
        elif node.is_tensor:
            info = node.tensor_info
            name = info.name.split('.')[-1]
            return f"📄 {name} [{info.dtype}, {format_shape(info.shape)}, {format_size(info.size_bytes)}]"
        elif node.is_metadata:
            info = node.metadata_info
            val = info.value
            if len(val) > 30: val = val[:27] + "..."
            return f"🏷️ {info.name}: {val}"
        return node.name

    def action_toggle_search(self) -> None:
        self.add_class("search-active")
        self.query_one("#search-input").focus()

    def action_clear_search(self) -> None:
        self.remove_class("search-active")
        self.query_one("#search-input").value = ""
        self.build_tree()
        self.query_one("#file-tree").focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input":
            self.build_tree(event.value)

    def action_select_node(self) -> None:
        tree = self.query_one("#file-tree", Tree)
        if tree.cursor_node:
            node_data = tree.cursor_node.data
            if node_data:
                if node_data.is_group:
                    tree.cursor_node.toggle()
                else:
                    self.show_details_for_node(node_data)

    def action_show_details(self) -> None:
        tree = self.query_one("#file-tree", Tree)
        if tree.cursor_node and tree.cursor_node.data:
            self.show_details_for_node(tree.cursor_node.data)

    def show_details_for_node(self, node: TreeNode) -> None:
        if node.is_tensor:
            info = node.tensor_info
            content = (
                f"Name: {info.name}\n"
                f"Data Type: {info.dtype}\n"
                f"Shape: {format_shape(info.shape)}\n"
                f"Size: {format_size(info.size_bytes)}\n"
                f"Elements: {format_parameters(info.num_elements)}\n"
            )
            self.push_screen(DetailScreen("Tensor Details", content))
        elif node.is_metadata:
            info = node.metadata_info
            content = (
                f"Key: {info.name}\n"
                f"Type: {info.value_type}\n"
                f"Value:\n{info.value}"
            )
            self.push_screen(DetailScreen("Metadata Details", content))
