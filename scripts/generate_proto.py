#!/usr/bin/env python
"""
Generate Python code from protobuf definitions.
Run this script after modifying .proto files.
"""
import os
import subprocess
import sys
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
PROTO_DIR = PROJECT_ROOT / "protos"
OUTPUT_DIR = PROJECT_ROOT / "gen" / "python"


def main():
    """Generate Python protobuf code."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all .proto files
    proto_files = list(PROTO_DIR.glob("**/*.proto"))

    if not proto_files:
        print("No .proto files found")
        return 1

    print(f"Found {len(proto_files)} proto files")

    # Generate Python code using grpc_tools.protoc
    for proto_file in proto_files:
        print(f"Generating code for: {proto_file}")

        # Calculate relative paths
        proto_rel = proto_file.relative_to(PROTO_DIR)
        output_subdir = OUTPUT_DIR / proto_rel.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Create __init__.py files
        for parent in output_subdir.parents:
            if parent >= OUTPUT_DIR:
                init_file = parent / "__init__.py"
                if not init_file.exists():
                    init_file.touch()

        (output_subdir / "__init__.py").touch()

        # Run protoc
        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            f"--proto_path={PROTO_DIR}",
            f"--python_out={OUTPUT_DIR}",
            f"--pyi_out={OUTPUT_DIR}",
            f"--grpc_python_out={OUTPUT_DIR}",
            str(proto_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error generating {proto_file}:")
            print(result.stderr)
            return 1

    # Fix imports in generated files (grpc_tools generates absolute imports)
    fix_imports()

    print("Proto generation complete!")
    return 0


def fix_imports():
    """Fix import statements in generated files."""
    for py_file in OUTPUT_DIR.glob("**/*.py"):
        content = py_file.read_text()
        original = content

        # Fix imports like "from rag.v1 import" to "from gen.python.rag.v1 import"
        # This is needed because grpc_tools generates imports relative to the proto path
        content = content.replace(
            "from rag.v1 import",
            "from gen.python.rag.v1 import"
        )

        if content != original:
            py_file.write_text(content)
            print(f"Fixed imports in: {py_file}")


if __name__ == "__main__":
    sys.exit(main())
