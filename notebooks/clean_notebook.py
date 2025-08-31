# import nbformat
# from pathlib import Path

# # Folder containing notebooks
# notebook_folder = Path(".")

# # Scan all .ipynb files
# for notebook_path in notebook_folder.rglob("*.ipynb"):
#     try:
#         nb = nbformat.read(notebook_path, as_version=4)
#         if "widgets" in nb.metadata:
#             del nb.metadata["widgets"]
#             nbformat.write(nb, notebook_path)
#             print(f"Cleaned widgets metadata in: {notebook_path}")
#         else:
#             print(f"No widgets metadata to clean in: {notebook_path}")
#     except Exception as e:
#         print(f"Failed to process {notebook_path}: {e}")


import nbformat
from pathlib import Path

notebook_folder = Path(".")

for notebook_path in notebook_folder.rglob("*.ipynb"):
    try:
        nb = nbformat.read(notebook_path, as_version=4)
        cleaned = False

        # Remove top-level widgets
        if "widgets" in nb.metadata:
            del nb.metadata["widgets"]
            cleaned = True

        # Remove widgets from each cell
        for cell in nb.cells:
            if "widgets" in cell.metadata:
                del cell.metadata["widgets"]
                cleaned = True

        if cleaned:
            nbformat.write(nb, notebook_path)
            print(f"Cleaned widgets metadata in: {notebook_path}")
        else:
            print(f"No widgets metadata to clean in: {notebook_path}")

    except Exception as e:
        print(f"Failed to process {notebook_path}: {e}")
