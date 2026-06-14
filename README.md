# Voxels Viewer

A single-file desktop application for inspecting, windowing, measuring, and
aligning 3-D volumetric (CT) datasets on **macOS** and **Windows 11**.

Everything lives in [`ct_viewer.py`](ct_viewer.py) and runs on top of Qt
(PySide6 or PyQt5) + pyqtgraph.

## Versioning

Voxels Viewer uses a calendar version `YY.M` (2-digit year + month), shown in
the title bar — e.g. `26.6` for June 2026. The full build number (`YY.M.build`)
appears in **Help → About** and is written into saved project files for
forward/backward-compatibility checks.

## Features

### Viewports & layout
- Four viewports: **XY**, **YZ**, **XZ** slices and an interactive **3D** view.
- Two 2×2 layouts via **View → Layout**: **Classic** and **Engineering**
  (Engineering = XY / 3D on top, XZ / YZ on the bottom). The choice is remembered.
- Maximize/restore any single viewport.
- Per-axis slice sliders for the 2D views.
- A lock button to sync zoom/pan and crosshair coordinate lines across the 2D
  viewports; coordinate lines also appear briefly when you zoom/scroll. Lock
  state is remembered between sessions.
- Orientation tripod overlay in each viewport.

### Histogram & windowing
- Live histogram whose bars are shaded by their mapped gray value (black below
  the window minimum, white above the maximum).
- A single **Window** range slider, plus draggable red dots on the histogram
  (drag horizontally to set the window — synced with the slider).
- **Auto Min Max**: drag a rectangle in any 2D viewport to set the window from
  the min/max inside it. An on-screen hint shows while active; press **Esc** to
  exit. Windowing updates in real time.
- Linear / logarithmic histogram scale (Preferences).

### Measurements (2D viewports)
- Dropdown tool with **Distance**, **Angle**, and **Diameter**, with live values
  in millimetres (computed from the voxel size). Yellow, draggable, with grab
  handles; they pan/zoom with the image and clear on slice/alignment changes.

### Gray-value tools (2D viewports)
- **Gray Value Picker** — a draggable crosshair showing live X/Y/Z coordinates
  and the gray value (shown as a plain integer for integer datasets).
- **Gray Value Profile** — a draggable line with a live intensity plot drawn
  above it (with Y-axis tick labels). **Right-click** the profile to export the
  samples to **CSV**.

### 3D view
- **Isosurface** (marching cubes, with an isovalue slider) and **Phong Volume**
  rendering; the Phong render respects the histogram window.
- Free turntable rotation (no pole flipping / inversion).
- Right-click → **Quality**: **Low** / **Default** / **High** render resolution.

### Alignment (non-destructive, view-time)
- **Operations → Alignment → Simple Alignment…** — rotate/translate with a live
  preview.
- **Operations → Alignment → 3-2-1 Alignment…** — pick a plane, a line, and an
  origin to align to the standard frame.

### Import / projects
- **File → Import → Import Slice Files…** (TIFF or raw) and **Import Volume…**
  (TIFF or single raw volume), with a metadata dialog for raw data (dimensions,
  voxel size, data type, byte order / endianness, Z flip). Large imports show a
  progress bar.
- **File → Open / Save Voxels Project…** — saves the source reference, window,
  viewport state, alignment, lock state, render mode, and camera to a `.voxels`
  file.
- **Operations → Volume Information…** — a read-only histogram plus a table of
  voxel dimensions, physical dimensions, voxel size, data type, and volume size.

### Appearance & platform
- Dark / Light / Automatic theme (Preferences).
- High-DPI aware (correct rendering at fractional scaling, e.g. 150% on Windows).

## Requirements

- **Python 3.8+** (developed/tested on 3.11–3.13).
- A Qt binding: **PySide6** *or* **PyQt5**.

## Installation

Install the dependencies into your Python environment:

```bash
python3 -m pip install PySide6 pyqtgraph numpy imageio tifffile scikit-image PyOpenGL scipy
```

If you prefer PyQt5, swap the Qt binding:

```bash
python3 -m pip install PyQt5 pyqtgraph numpy imageio tifffile scikit-image PyOpenGL scipy
```

On **macOS**, to show the application name ("Voxels Viewer") in the menu bar when
running from source, also install PyObjC (optional):

```bash
python3 -m pip install pyobjc-framework-Cocoa
```

## Dependencies

| Package | Purpose |
|---|---|
| PySide6 **or** PyQt5 | Qt GUI toolkit |
| pyqtgraph | 2D/3D plotting, image views, ROIs |
| numpy | Volume data and math |
| PyOpenGL | Required by pyqtgraph for the 3D view |
| scipy | Alignment resampling and gray-value profile sampling |
| scikit-image | Isosurface (marching cubes) in the 3D view |
| imageio | Image/TIFF reading |
| tifffile | More reliable TIFF reading (recommended) |
| pyobjc-framework-Cocoa | macOS-only, optional; sets the app menu name from source |

If `scipy` or `scikit-image` is missing, the related features (alignment /
isosurface) degrade gracefully with a message instead of failing.

## Run

From the project folder:

```bash
python3 ct_viewer.py
```

## License

Copyright © 2026 **Stepan Rumyantsev**. All rights reserved.

Permission is granted, free of charge, to **use** the Software for any lawful
purpose — personal, academic, or commercial — subject to:

1. **Permitted use** — any individual or organisation may run the Software
   without restriction.
2. **Redistribution** — verbatim (unmodified) redistribution is permitted
   provided this copyright notice and license text are retained in full and
   **Stepan Rumyantsev** is clearly credited as the original author.
3. **No modifications** — modification, adaptation, translation,
   reverse-engineering, decompilation, or derivative works are **not permitted**
   without explicit prior written permission from the copyright holder.
4. **No warranty** — the Software is provided "as is", without warranty of any
   kind; the copyright holder is not liable for any claim, damages, or other
   liability arising from its use.

The full license and third-party library notices are available in **Help →
About** within the application.
