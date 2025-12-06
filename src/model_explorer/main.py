import argparse
import sys
import glob
from pathlib import Path
from typing import List

from .app import SafetensorsExplorerApp

def collect_files(paths: List[str], recursive: bool) -> List[Path]:
    collected_files = []
    
    for path_str in paths:
        # Expand glob patterns
        expanded_paths = glob.glob(path_str, recursive=recursive)
        if not expanded_paths:
            # If glob didn't match anything, try treating it as a literal path
            # (glob.glob might return empty if no wildcards and file doesn't exist, 
            # or if it's just a direct path)
            if Path(path_str).exists():
                expanded_paths = [path_str]
            else:
                # If it was a glob pattern that matched nothing, or a non-existent file
                # we can warn or skip. For now, let's just skip.
                pass

        for p in expanded_paths:
            path = Path(p)
            if not path.exists():
                continue
                
            if path.is_file():
                ext = path.suffix.lower()
                if ext in [".safetensors", ".gguf"]:
                    collected_files.append(path)
            elif path.is_dir():
                # Check for index file
                index_path = path / "model.safetensors.index.json"
                if index_path.exists():
                    # TODO: Parse index file to get list of files
                    # For now, just scan directory
                    pass
                
                # Scan directory
                patterns = ["*.safetensors", "*.gguf"]
                if recursive:
                    patterns = [f"**/{p}" for p in patterns]
                
                for pattern in patterns:
                    for f in path.glob(pattern):
                        collected_files.append(f)
    
    return sorted(list(set(collected_files)))

def main():
    parser = argparse.ArgumentParser(
        description="Interactive explorer for SafeTensors and GGUF files"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="SafeTensors and GGUF files, directories, or glob patterns to explore"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively search directories"
    )
    
    parser.add_argument(
        "-v", "--visualize",
        action="store_true",
        help="Visualize model structure as an interactive sunburst chart"
    )
    
    args = parser.parse_args()
    
    files = collect_files(args.paths, args.recursive)
    
    if not files:
        print("Error: No SafeTensors or GGUF files found in the specified paths.", file=sys.stderr)
        sys.exit(1)
        
    if args.visualize:
        from .visualizer import visualize_model
        visualize_model(files)
    else:
        app = SafetensorsExplorerApp(files)
        app.run()

if __name__ == "__main__":
    main()
