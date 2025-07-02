def access_utilities():
    from pathlib import Path
    import sys
    cwd = Path.cwd()
    parent_dir = cwd.resolve().parents[0]
    sys.path.insert(0, str(Path(parent_dir) / "utilities-main"))