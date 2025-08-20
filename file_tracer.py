import builtins
import os
import sys
import sqlite3
import time
import runpy

# --- Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALL_FILES = []
for root, dirs, files in os.walk(BASE_DIR):
    for f in files:
        ALL_FILES.append(os.path.relpath(os.path.join(root, f), BASE_DIR))

LOGGED = set()

# --- Wrap open() ---
orig_open = builtins.open
def logged_open(file, *args, **kwargs):
    try:
        abs_path = os.path.abspath(file)
        if abs_path.startswith(BASE_DIR):
            LOGGED.add(os.path.relpath(abs_path, BASE_DIR))
    except Exception:
        pass
    return orig_open(file, *args, **kwargs)
builtins.open = logged_open

# --- Wrap sqlite3.connect ---
orig_connect = sqlite3.connect
def logged_connect(db, *args, **kwargs):
    abs_path = os.path.abspath(db)
    if abs_path.startswith(BASE_DIR):
        LOGGED.add(os.path.relpath(abs_path, BASE_DIR))
    return orig_connect(db, *args, **kwargs)
sqlite3.connect = logged_connect

# --- Run your app.py ---
print("🔍 Tracing file usage while running app.py ...")
start = time.time()
runpy.run_path(os.path.join(BASE_DIR, "app.py"), run_name="__main__")
end = time.time()

# --- Report ---
print("\n⏱ Runtime: %.2f seconds" % (end - start))
print("\n📂 Files accessed during execution:")
for f in sorted(LOGGED):
    print("   ✅", f)

unused = sorted(set(ALL_FILES) - LOGGED)
print("\n🗑 Unused files (never touched):")
for f in unused:
    print("   ❌", f)
