import json
import shutil

notebook_path = "run_red_teaming.ipynb"

# Read the current (already-correct) file from disk
with open(notebook_path, "r", encoding="utf-8") as f:
    content = f.read()

# Write to a backup first
shutil.copy2(notebook_path, "run_red_teaming_backup.ipynb")

# Parse and verify
nb = json.loads(content)
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        src = "".join(cell.get("source", []))
        if "generate_summary" in src:
            if "r.get('metadata'" in src:
                print("✅ Notebook already has the CORRECT code (uses metadata)")
            elif "r.get('vars'" in src:
                print("❌ Notebook still has OLD code (uses vars)")
            print()
            print("Cell source code:")
            print(src)
            break

print("\nBackup saved to: run_red_teaming_backup.ipynb")
