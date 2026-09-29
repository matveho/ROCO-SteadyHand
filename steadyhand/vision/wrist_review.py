"""Pixel selection on an actual wrist frame; no robot commands in the UI."""
from __future__ import annotations

import math
import os


def select_pixel(rgb, path, title, *, use_viewer=True):
    """Use a zoomable desktop picker when available, or explicit terminal UV."""
    print(f"{title}\nIMAGE: {path}", flush=True)
    if use_viewer and (os.environ.get("DISPLAY") or os.name == "nt"):
        try:
            return _window(rgb, title)
        except InterruptedError:
            raise
        except Exception:
            print("Tk/Pillow unavailable; enter pixels from the saved image.")
    h, w = rgb.shape[:2]
    while True:
        raw = input(f"{title}: enter u v (image {w}x{h}), or abort: ").strip()
        if raw.lower() in ("abort", "q", "exit"):
            raise InterruptedError("Pixel selection cancelled")
        try:
            u, v = (float(s) for s in raw.replace(",", " ").split())
            if math.isfinite(u) and math.isfinite(v) and 0 <= u < w and 0 <= v < h:
                return u, v
        except ValueError:
            pass
        print("Enter two pixel coordinates inside the image.")


def _window(rgb, title):
    import tkinter as tk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(title)
    tk.Label(root, text=title + " | click: select | wheel: zoom | right drag: pan | Ctrl+Z: undo selection").pack()
    canvas = tk.Canvas(root, width=1000, height=700, background="#222")
    canvas.pack(fill="both", expand=True)
    image = Image.fromarray(rgb)
    state = {"scale": min(1000 / image.width, 700 / image.height), "x": 0., "y": 0., "marks": [], "result": None}

    def draw():
        scale = state["scale"]
        resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
        state["photo"] = ImageTk.PhotoImage(resized)
        canvas.delete("all")
        canvas.create_image(state["x"], state["y"], image=state["photo"], anchor="nw")
        if state["marks"]:
            u, v = state["marks"][-1]
            x, y = state["x"] + u * scale, state["y"] + v * scale
            canvas.create_line(x-15, y, x+15, y, fill="lime", width=2)
            canvas.create_line(x, y-15, x, y+15, fill="lime", width=2)
            canvas.create_text(x+20, y-15, text=f"{u:.1f}, {v:.1f}", fill="lime", anchor="w")

    def click(event):
        uv = ((event.x-state["x"])/state["scale"], (event.y-state["y"])/state["scale"])
        if 0 <= uv[0] < image.width and 0 <= uv[1] < image.height:
            state["marks"].append(uv)
            draw()

    def zoom(event):
        factor = 1.2 if getattr(event, "num", 0) == 4 or getattr(event, "delta", 0) > 0 else 1/1.2
        old = state["scale"]
        new = min(5., max(.1, old * factor))
        state["x"] = event.x - (event.x - state["x"]) * new / old
        state["y"] = event.y - (event.y - state["y"]) * new / old
        state["scale"] = new
        draw()

    def drag(event):
        state["x"] += event.x-state["drag"][0]
        state["y"] += event.y-state["drag"][1]
        state["drag"] = (event.x, event.y)
        draw()

    def undo(_event=None):
        if state["marks"]:
            state["marks"].pop()
            draw()

    def accept(_event=None):
        if state["marks"]:
            state["result"] = state["marks"][-1]
            root.destroy()

    canvas.bind("<Button-1>", click)
    canvas.bind("<Button-3>", lambda e: state.update(drag=(e.x, e.y)))
    canvas.bind("<B3-Motion>", drag)
    for binding in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        canvas.bind(binding, zoom)
    root.bind("<Control-z>", undo)
    root.bind("<Return>", accept)
    tk.Button(root, text="Use selected pixel", command=accept).pack()
    draw()
    root.mainloop()
    if state["result"] is None:
        raise InterruptedError("Pixel selection cancelled")
    return state["result"]
