import unittest
from pathlib import Path
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model_explorer.visualizer import visualize_model, get_layer_color, COLORS
from tests.create_test_safetensors import create_safetensors
from unittest.mock import patch, MagicMock

class TestVisualizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).parent.parent
        cls.safetensors_path = cls.root_dir / "test_viz.safetensors"
        create_safetensors(str(cls.safetensors_path))

    @classmethod
    def tearDownClass(cls):
        if cls.safetensors_path.exists():
            os.remove(cls.safetensors_path)

    def test_get_layer_color(self):
        """Test color mapping logic."""
        # Entry
        self.assertEqual(get_layer_color("model.embed_tokens.weight"), COLORS["ENTRY"])
        self.assertEqual(get_layer_color("token_embd.weight"), COLORS["ENTRY"])
        
        # Block Start
        self.assertEqual(get_layer_color("model.layers.0.input_layernorm.weight"), COLORS["BLOCK_START"])
        
        # Attention
        self.assertEqual(get_layer_color("model.layers.0.self_attn.q_proj.weight"), COLORS["ATTN_QKV"])
        self.assertEqual(get_layer_color("model.layers.0.self_attn.o_proj.weight"), COLORS["ATTN_OUT"])
        
        # MLP
        self.assertEqual(get_layer_color("model.layers.0.mlp.gate_proj.weight"), COLORS["MLP_GATE_UP"])
        self.assertEqual(get_layer_color("model.layers.0.mlp.down_proj.weight"), COLORS["MLP_DOWN"])
        
        # Exit
        self.assertEqual(get_layer_color("model.norm.weight"), COLORS["EXIT_NORM"])
        self.assertEqual(get_layer_color("lm_head.weight"), COLORS["EXIT_HEAD"])
        
        # Default
        self.assertEqual(get_layer_color("unknown.layer"), COLORS["DEFAULT"])

    @patch('model_explorer.visualizer.make_subplots')
    @patch('model_explorer.visualizer.px.sunburst')
    @patch('model_explorer.visualizer.pd.DataFrame')
    def test_visualize_model(self, mock_df, mock_sunburst, mock_subplots):
        """Test that visualize_model calls plotly with correct data."""
        
        # Mock DataFrame to capture data passed to it
        def side_effect(data):
            # Verify data structure
            self.assertIsInstance(data, list)
            self.assertTrue(len(data) > 0)
            
            # Check for color attribute
            tensors = [item for item in data if item.get("is_tensor")]
            self.assertTrue(len(tensors) > 0)
            self.assertIn("color", tensors[0])
            
            return MagicMock() # Return a mock DataFrame
            
        mock_df.side_effect = side_effect
        
        # Mock figures
        mock_sunburst_fig = MagicMock()
        mock_sunburst_fig.data = [MagicMock()] # One trace
        mock_sunburst.return_value = mock_sunburst_fig
        
        mock_main_fig = MagicMock()
        mock_subplots.return_value = mock_main_fig
        
        visualize_model([self.safetensors_path])
        
        # Verify calls
        self.assertTrue(mock_df.called)
        self.assertTrue(mock_sunburst.called)
        self.assertTrue(mock_subplots.called)
        
        # Verify make_subplots called with 2 rows
        _, kwargs = mock_subplots.call_args
        self.assertEqual(kwargs.get('rows'), 2)
        self.assertEqual(kwargs.get('cols'), 2)
        
        # Verify traces added: Sunburst, Metadata, Legend
        self.assertTrue(mock_main_fig.add_trace.called)
        self.assertEqual(mock_main_fig.add_trace.call_count, 3)

if __name__ == '__main__':
    unittest.main()
