#!/usr/bin/env python3
"""
Shows a native OS "Open File" dialog and prints the selected path to stdout.
Launched as a short-lived subprocess by POST /matches/browse so the frontend
can let the user pick a match video via the real file explorer instead of
typing an absolute path by hand. Prints nothing if the user cancels.
"""

import sys


def main():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("ERROR: tkinter not available in this Python installation", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select match video file",
        filetypes=[
            ("Video files", "*.mp4 *.mkv *.mov *.avi *.m4v *.webm"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if path:
        print(path)


if __name__ == "__main__":
    main()
