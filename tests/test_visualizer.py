import unittest
from pathlib import Path
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model_explorer.visualizer import visualize_model
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
            
            # Check for root node
            root = next((item for item in data if item["id"] == "root"), None)
            self.assertIsNotNone(root)
            
            # Check for tensor nodes
            tensors = [item for item in data if item.get("is_tensor")]
            self.assertTrue(len(tensors) > 0)
            
            # Metadata should NOT be in the main dataframe anymore
            metadata = [item for item in data if item.get("name") == "Metadata"]
            self.assertEqual(len(metadata), 0)
            
            return MagicMock() # Return a mock DataFrame
            
        mock_df.side_effect = side_effect
        
        # Mock figures
        mock_sunburst_fig = MagicMock()
        mock_sunburst_fig.data = [MagicMock()] # One trace
        mock_sunburst_fig.layout.coloraxis = {}
        mock_sunburst.return_value = mock_sunburst_fig
        
        mock_main_fig = MagicMock()
        mock_subplots.return_value = mock_main_fig
        
        visualize_model([self.safetensors_path])
        
        # Verify calls
        self.assertTrue(mock_df.called)
        self.assertTrue(mock_sunburst.called)
        self.assertTrue(mock_subplots.called)
        
        # Verify traces added
        # Should add sunburst trace and table trace
        self.assertTrue(mock_main_fig.add_trace.called)
        self.assertEqual(mock_main_fig.add_trace.call_count, 2)

if __name__ == '__main__':
    unittest.main()
