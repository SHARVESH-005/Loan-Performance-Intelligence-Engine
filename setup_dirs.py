import os

directories = [
    "data/raw", "data/interim", "data/processed",
    "notebooks",
    "src/data", "src/features", "src/models", "src/survival", 
    "src/anomaly", "src/scenario", "src/explainability", "src/llm_copilot", "src/utils",
    "reports",
    "submission",
    "logs",
    "tests",
    "demo"
]

for d in directories:
    os.makedirs(d, exist_ok=True)
    if d.startswith("src"):
        with open(os.path.join(d, "__init__.py"), "w") as f:
            pass
    elif not d.startswith("src/data"):
        with open(os.path.join(d, ".gitkeep"), "w") as f:
            pass

# Root src __init__
with open(os.path.join("src", "__init__.py"), "w") as f:
    pass

print("Directories and init files created successfully.")
