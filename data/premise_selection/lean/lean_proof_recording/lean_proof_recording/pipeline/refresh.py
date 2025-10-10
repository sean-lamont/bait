import json
import os
from pathlib import Path
import shutil
import subprocess


# Constants
TARGET_DIR = Path("_target")
LEAN_LIBRARY_PATH = TARGET_DIR / "deps" / "lean" / "library"
LEANPKG_PATH_FILE = Path("leanpkg.path")


def print_section_header(title: str):
    """Print a formatted section header."""
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def remove_target_directory():
    """Remove the _target directory if it exists."""
    print_section_header("Remove `_target` directory")
    if TARGET_DIR.is_dir():
        print("_target directory found.")
        shutil.rmtree(TARGET_DIR)
        print("_target directory removed.")
    else:
        print("No _target directory.")


def build_lean_project():
    """Build the Lean project according to leanpkg.toml."""
    print_section_header("Configure and build lean project")
    os.system("leanproject build")


def get_lean_library_path():
    """Get the Lean library path from the lean command."""
    print("Getting lean paths.")
    with subprocess.Popen(
        ["lean", "--path"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as out:
        stdout, stderr = out.communicate()
    assert stderr is None, stderr
    s = stdout.decode("utf-8")
    path_data = json.loads(s)
    
    for p in path_data["path"]:
        if p.endswith("lean/library"):
            return Path(p)
    
    raise RuntimeError("Could not find lean/library path in lean --path output")


def copy_lean_library():
    """Copy base lean library to _target/deps/lean/library."""
    print_section_header("Copy base lean library to _target/deps/lean/library")
    
    lean_library = get_lean_library_path()
    print("Found lean library path: ", lean_library)
    print(f"Copying lean library to {LEAN_LIBRARY_PATH} ...")
    
    (TARGET_DIR / "deps" / "lean").mkdir(parents=True)
    shutil.copytree(lean_library, LEAN_LIBRARY_PATH)


def update_leanpkg_path():
    """Change leanpkg.path to point to the local lean files."""
    print_section_header("Change lean path to _target/deps/lean/library")
    
    with open(LEANPKG_PATH_FILE, "r") as f:
        lines = [line.replace("builtin_path", f"path {LEAN_LIBRARY_PATH}") for line in f]
    
    with open(LEANPKG_PATH_FILE, "w") as f:
        f.writelines(lines)
    
    print("Path changed")


def _main():
    """
    Refreshes the Lean setup:

    - Deletes the target directory.
    - Refresh Lean and Mathlib according to what's in leanpkg.toml.
    - Copy base lean files from elan to `_target/deps/lean/library/`
    - Change leanpkg.path to point to the local lean files.
    """
    remove_target_directory()
    build_lean_project()
    copy_lean_library()
    update_leanpkg_path()



if __name__ == "__main__":
    _main()
