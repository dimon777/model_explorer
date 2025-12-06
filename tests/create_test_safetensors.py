import torch
import sys
from safetensors.torch import save_file


def create_safetensors(path):
    tensors = {
        "model.layers.0.weight": torch.zeros((10, 10)),
        "model.layers.0.bias": torch.zeros((10)),
        "model.layers.1.weight": torch.zeros((10, 10)),
        "model.layers.1.bias": torch.zeros((10)),
        "model.embed_tokens.weight": torch.zeros((100, 10)),
    }

    metadata = {
        "format": "pt",
        "arch": "llama"
    }

    save_file(tensors, path, metadata)
    print(f"Created {path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        create_safetensors(sys.argv[1])
    else:
        create_safetensors("test_model.safetensors")
