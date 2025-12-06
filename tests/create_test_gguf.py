import gguf
import numpy as np
import sys
import os

def create_gguf(path):
    gguf_writer = gguf.GGUFWriter(path, "llama")
    
    # Add some metadata
    gguf_writer.add_string("general.architecture", "llama")
    gguf_writer.add_uint32("llama.context_length", 2048)
    
    # Add a tensor
    data = np.ones((10, 10), dtype=np.float32)
    gguf_writer.add_tensor("tensor1", data)
    
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"Created {path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        create_gguf(sys.argv[1])
    else:
        create_gguf("test_model.gguf")
