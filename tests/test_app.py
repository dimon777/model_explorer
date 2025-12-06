import unittest
import os
import sys
from pathlib import Path
import shutil
import numpy as np

# Add src to path so we can import model_explorer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from model_explorer.app import SafetensorsExplorerApp
from tests.create_test_gguf import create_gguf
from tests.create_test_safetensors import create_safetensors

class TestModelExplorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Paths
        cls.root_dir = Path(__file__).parent.parent
        cls.safetensors_path = Path(__file__).parent / "test_model.safetensors"
        cls.gguf_path = Path(__file__).parent / "test_model.gguf"

        # Create sample model files
        create_safetensors(str(cls.safetensors_path))
        create_gguf(str(cls.gguf_path))

    @classmethod
    def tearDownClass(cls):
        # Clean up generated GGUF
        return
        if cls.gguf_path.exists():
            os.remove(cls.gguf_path)
        if cls.safetensors_path.exists():
            os.remove(cls.safetensors_path)

    def test_load_safetensors(self):
        """Test loading a safetensors file."""
        if not self.safetensors_path.exists():
            self.skipTest("test_model.safetensors not found")
            
        app = SafetensorsExplorerApp(files=[self.safetensors_path])
        app.load_files()
        
        self.assertTrue(len(app.tensors) > 0, "No tensors loaded from safetensors file")
        # Check if we can find a tensor (assuming the file has some)
        # We don't know the exact content of the user's file, but we can check structure
        first_tensor = app.tensors[0]
        self.assertIsNotNone(first_tensor.name)
        self.assertIsNotNone(first_tensor.dtype)
        self.assertIsNotNone(first_tensor.shape)

    def test_load_gguf(self):
        """Test loading a GGUF file."""
        if not self.gguf_path.exists():
            self.fail("Failed to create test GGUF file")
            
        app = SafetensorsExplorerApp(files=[self.gguf_path])
        app.load_files()
        
        self.assertTrue(len(app.tensors) > 0, "No tensors loaded from GGUF file")
        
        # We know what we put in the GGUF file
        tensor_names = [t.name for t in app.tensors]
        self.assertIn("tensor1", tensor_names)
        
        tensor1 = next(t for t in app.tensors if t.name == "tensor1")
        self.assertEqual(tensor1.shape, [10, 10])

    def test_mixed_loading(self):
        """Test loading both types together."""
        if not self.safetensors_path.exists():
            self.skipTest("test_model.safetensors not found")
            
        app = SafetensorsExplorerApp(files=[self.safetensors_path, self.gguf_path])
        app.load_files()
        
        self.assertTrue(len(app.tensors) > 0)
        # Should have tensors from both (unless names collide, but here they shouldn't)
        # Note: app dedupes by name, so if both have "tensor1", only one appears.
        
if __name__ == '__main__':
    unittest.main()
