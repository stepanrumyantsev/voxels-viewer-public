#!/usr/bin/env python3
"""
CT Volume Viewer

Single-file Python application for macOS and Windows 11.
Dependencies:
    PySide6 or PyQt5, pyqtgraph, numpy, imageio, tifffile, scikit-image

Usage:
    python ct_viewer.py
"""

import os
import sys
import json
import math
import numpy as np

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QFileDialog,
        QDialog,
        QFormLayout,
        QLineEdit,
        QComboBox,
        QCheckBox,
        QGroupBox,
        QGridLayout,
        QMessageBox,
        QSplitter,
        QTextBrowser,
        QProgressDialog,
    )
    from PySide6.QtCore import Qt, Signal, Slot
except ImportError:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QFileDialog,
        QDialog,
        QFormLayout,
        QLineEdit,
        QComboBox,
        QCheckBox,
        QGroupBox,
        QGridLayout,
        QMessageBox,
        QSplitter,
        QTextBrowser,
        QProgressDialog,
    )
    from PyQt5.QtCore import Qt, pyqtSignal as Signal, pyqtSlot as Slot

try:
    import pyqtgraph as pg
except ImportError:
    print("pyqtgraph is required. Install with pip install pyqtgraph")
    raise

try:
    import pyqtgraph.opengl as gl
except ModuleNotFoundError as exc:
    if exc.name == 'OpenGL':
        print("PyOpenGL is required for the 3D view. Install with pip install PyOpenGL")
    raise

try:
    from OpenGL import GL as _GL
    # Always-on-top lines (frames): no depth test. Also restores depth/colour
    # state to defaults — frames are drawn last, so this cleans up after the
    # coordinate-plane depth pre-pass below.
    _ONTOP_GLOPTS = {
        _GL.GL_DEPTH_TEST: False,
        _GL.GL_BLEND: True,
        _GL.GL_CULL_FACE: False,
        'glDepthMask': (_GL.GL_TRUE,),
        'glDepthFunc': (_GL.GL_LESS,),
        'glColorMask': (_GL.GL_TRUE, _GL.GL_TRUE, _GL.GL_TRUE, _GL.GL_TRUE),
        'glBlendFunc': (_GL.GL_SRC_ALPHA, _GL.GL_ONE_MINUS_SRC_ALPHA),
    }
    # Coordinate-plane depth pre-pass: write only the NEAREST plane's depth per
    # pixel (no colour).
    _PLANE_DEPTH_GLOPTS = {
        _GL.GL_DEPTH_TEST: True,
        _GL.GL_BLEND: False,
        _GL.GL_CULL_FACE: False,
        'glDepthFunc': (_GL.GL_LESS,),
        'glDepthMask': (_GL.GL_TRUE,),
        'glColorMask': (_GL.GL_FALSE, _GL.GL_FALSE, _GL.GL_FALSE, _GL.GL_FALSE),
    }
    # Colour pass: only fragments matching the pre-pass depth draw, so exactly
    # one (the nearest) plane colours each pixel — never a blend of two planes.
    _PLANE_COLOR_GLOPTS = {
        _GL.GL_DEPTH_TEST: True,
        _GL.GL_BLEND: True,
        _GL.GL_CULL_FACE: False,
        'glDepthFunc': (_GL.GL_LEQUAL,),
        'glDepthMask': (_GL.GL_FALSE,),
        'glColorMask': (_GL.GL_TRUE, _GL.GL_TRUE, _GL.GL_TRUE, _GL.GL_TRUE),
        'glBlendFunc': (_GL.GL_SRC_ALPHA, _GL.GL_ONE_MINUS_SRC_ALPHA),
    }
except Exception:
    _GL = None
    _ONTOP_GLOPTS = 'additive'
    _PLANE_DEPTH_GLOPTS = 'opaque'
    _PLANE_COLOR_GLOPTS = 'translucent'


class _GLDepthClear(gl.GLGraphicsItem.GLGraphicsItem):
    """Invisible item that clears the GL depth buffer (and resets depth/colour
    state) when painted, so items drawn after it depth-test against a fresh
    buffer. Lets the coordinate planes be layered independently of the
    volume/surface depth."""

    def paint(self):
        self.setupGLState()
        if _GL is None:
            return
        try:
            _GL.glDepthMask(_GL.GL_TRUE)
            _GL.glDepthFunc(_GL.GL_LESS)
            _GL.glColorMask(_GL.GL_TRUE, _GL.GL_TRUE, _GL.GL_TRUE, _GL.GL_TRUE)
            _GL.glClear(_GL.GL_DEPTH_BUFFER_BIT)
        except Exception:
            pass

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    from skimage import measure
except ImportError:
    measure = None

try:
    from scipy.ndimage import affine_transform as _scipy_affine_transform
except ImportError:
    _scipy_affine_transform = None


APP_NAME = 'Voxels Viewer'
# Version scheme: YY.M (2-digit year + month) is the headline version shown in
# the title bar; the trailing build number increments with every code change
# and forms the full version (YY.M.build) used in the About dialog and saved
# into .voxels project files for compatibility checks.
APP_VERSION = '26.6'          # June 2026
APP_BUILD = 101
APP_VERSION_FULL = f'{APP_VERSION}.{APP_BUILD}'


def _parse_version(s):
    """Parse a 'YY.M.build' string into a comparable tuple of ints."""
    try:
        return tuple(int(p) for p in str(s).split('.'))
    except (ValueError, AttributeError):
        return (0,)


SUPPORTED_DTYPES = [
    ('8-bit unsigned', 'uint8'),
    ('8-bit signed', 'int8'),
    ('16-bit unsigned', 'uint16'),
    ('16-bit signed', 'int16'),
    ('32-bit float', 'float32'),
]

VIEWPORT_NAMES = ['XY', 'YZ', 'XZ', '3D']

try:
    _EV_RESIZE          = QtCore.QEvent.Type.Resize
    _EV_MOUSE_MOVE      = QtCore.QEvent.Type.MouseMove
    _EV_MOUSE_PRESS     = QtCore.QEvent.Type.MouseButtonPress
    _EV_WHEEL           = QtCore.QEvent.Type.Wheel
    _EV_NATIVE_GESTURE  = QtCore.QEvent.Type.NativeGesture
    _ANTIALIASING       = QtGui.QPainter.RenderHint.Antialiasing
    _ROUND_CAP          = Qt.PenCapStyle.RoundCap
except AttributeError:
    _EV_RESIZE          = QtCore.QEvent.Resize
    _EV_MOUSE_MOVE      = QtCore.QEvent.MouseMove
    _EV_MOUSE_PRESS     = QtCore.QEvent.MouseButtonPress
    _EV_WHEEL           = QtCore.QEvent.Wheel
    _EV_NATIVE_GESTURE  = QtCore.QEvent.NativeGesture
    _ANTIALIASING       = QtGui.QPainter.Antialiasing
    _ROUND_CAP          = Qt.RoundCap

try:
    _ZOOM_GESTURE_TYPE = Qt.NativeGestureType.ZoomNativeGesture
except AttributeError:
    try:
        _ZOOM_GESTURE_TYPE = Qt.ZoomNativeGesture
    except AttributeError:
        _ZOOM_GESTURE_TYPE = None


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def parse_dtype(selection):
    for name, dtype in SUPPORTED_DTYPES:
        if name == selection:
            return np.dtype(dtype)
    return np.uint8


def _detect_os_dark() -> bool:
    """Return True if the OS is currently using a dark colour scheme."""
    if sys.platform == 'darwin':
        try:
            import subprocess
            r = subprocess.run(
                ['defaults', 'read', '-g', 'AppleInterfaceStyle'],
                capture_output=True, text=True,
            )
            return r.returncode == 0 and r.stdout.strip() == 'Dark'
        except Exception:
            pass
    elif sys.platform == 'win32':
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize',
            )
            value, _ = winreg.QueryValueEx(key, 'AppsUseLightTheme')
            winreg.CloseKey(key)
            return value == 0
        except Exception:
            pass
    return False


def _make_dark_palette() -> QtGui.QPalette:
    p = QtGui.QPalette()
    c = QtGui.QColor
    p.setColor(QtGui.QPalette.Window,          c(45,  45,  45))
    p.setColor(QtGui.QPalette.WindowText,      c(220, 220, 220))
    p.setColor(QtGui.QPalette.Base,            c(30,  30,  30))
    p.setColor(QtGui.QPalette.AlternateBase,   c(45,  45,  45))
    p.setColor(QtGui.QPalette.ToolTipBase,     c(30,  30,  30))
    p.setColor(QtGui.QPalette.ToolTipText,     c(220, 220, 220))
    p.setColor(QtGui.QPalette.Text,            c(220, 220, 220))
    p.setColor(QtGui.QPalette.Button,          c(55,  55,  55))
    p.setColor(QtGui.QPalette.ButtonText,      c(220, 220, 220))
    p.setColor(QtGui.QPalette.BrightText,      c(255, 80,  80))
    p.setColor(QtGui.QPalette.Link,            c(42,  130, 218))
    p.setColor(QtGui.QPalette.Highlight,       c(42,  130, 218))
    p.setColor(QtGui.QPalette.HighlightedText, c(255, 255, 255))
    dim = c(128, 128, 128)
    for role in (QtGui.QPalette.Text, QtGui.QPalette.ButtonText, QtGui.QPalette.WindowText):
        p.setColor(QtGui.QPalette.Disabled, role, dim)
    return p


def _prefs_dir():
    """Directory for the user's preferences file.

    When running from source, keep it next to the script (legacy behaviour).
    When frozen (PyInstaller) the executable usually lives in a read-only,
    machine-wide location (e.g. Program Files), so store per-user under the
    platform's app-data directory instead — this lets every user on the machine
    keep their own settings and avoids write failures."""
    if getattr(sys, 'frozen', False):
        base = (os.environ.get('APPDATA')                       # Windows
                or os.environ.get('XDG_CONFIG_HOME')            # Linux
                or os.path.join(os.path.expanduser('~'), '.config'))
        d = os.path.join(base, 'Voxels Viewer')
        try:
            os.makedirs(d, exist_ok=True)
            return d
        except OSError:
            pass
    return os.path.dirname(os.path.abspath(__file__))


PREFS_FILE = os.path.join(_prefs_dir(), 'ct_viewer_prefs.json')

_PREFS_DEFAULTS: dict = {
    'theme': 'Dark',
    'histogram_scale': 'Logarithmic',
    'splitter_sizes': [220, 1200],
    'maximized_viewport': None,
    'last_import_dir': '',
    'viewport_layout': 'Engineering',
    'sync_locked': True,
    'measurement_tool': 'distance',
    'coord_lines_visible': True,
    'coord_planes_visible': False,
    'voxel_interpolation': False,
}


def _load_prefs() -> dict:
    prefs = dict(_PREFS_DEFAULTS)
    try:
        with open(PREFS_FILE, 'r') as f:
            prefs.update(json.load(f))
    except Exception:
        pass
    return prefs


def _save_prefs(prefs: dict) -> None:
    try:
        with open(PREFS_FILE, 'w') as f:
            json.dump(prefs, f, indent=2)
    except Exception:
        pass


_LICENSE_HTML = """
<h2 style="margin-bottom:4px;">Voxels Viewer</h2>
<p style="color:gray;">Copyright &copy; 2026 Stepan Rumyantsev. All rights reserved.<br>
Web Site: <a href="https://stepanrumyantsev.com">https://stepanrumyantsev.com</a><br>
Source Code: <a href="https://github.com/stepanrumyantsev/voxels-viewer-public">https://github.com/stepanrumyantsev/voxels-viewer-public</a></p>

<h3>License</h3>
<p>
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the &ldquo;Software&rdquo;), to
<b>use</b> the Software for any lawful purpose &mdash; personal, academic, or
commercial &mdash; free of charge, subject to the following conditions:
</p>
<ol>
  <li>
    <b>Permitted use</b> &ndash; Any individual or organisation may run the Software
    without restriction.
  </li>
  <li>
    <b>Redistribution</b> &ndash; Verbatim (unmodified) redistribution is permitted
    provided that:
    <ul>
      <li>this copyright notice and license text are retained in full; and</li>
      <li><b>Stepan Rumyantsev</b> is clearly credited as the original author in all
          copies and in any accompanying documentation or promotional materials.</li>
    </ul>
  </li>
  <li>
    <b>No modifications</b> &ndash; Modification, adaptation, translation,
    reverse-engineering, decompilation, or creation of derivative works based on
    this Software is <b>not permitted</b> without explicit prior written permission
    from the copyright holder.
  </li>
  <li>
    <b>No warranty</b> &ndash; The Software is provided &ldquo;as is&rdquo;, without
    warranty of any kind, express or implied, including but not limited to the
    warranties of merchantability, fitness for a particular purpose, and
    non-infringement. In no event shall the copyright holder be liable for any
    claim, damages, or other liability arising from the use of or inability to use
    the Software.
  </li>
</ol>
<p>
To request permission for uses not covered above, please contact the copyright
holder.
</p>
"""

_THIRD_PARTY_HTML = """
<p>
This application is built on the following open-source libraries. Full license
texts are available in each project&rsquo;s repository.
</p>

<h3>Qt &mdash; PySide6 / PyQt5</h3>
<p>
  Qt Framework &mdash; Copyright &copy; The Qt Company Ltd. and contributors.<br>
  PySide6 is distributed under the
  <b>GNU Lesser General Public License v3 (LGPL-3.0)</b>.<br>
  PyQt5 is distributed under the
  <b>GNU General Public License v3 (GPL-3.0)</b> or a commercial licence.<br>
  <a href="https://www.qt.io">https://www.qt.io</a>
</p>

<h3>pyqtgraph</h3>
<p>
  Copyright &copy; 2012 Luke Campagnola, Yale University.<br>
  Distributed under the <b>MIT License</b>.<br>
  <a href="https://pyqtgraph.readthedocs.io">https://pyqtgraph.readthedocs.io</a>
</p>

<h3>NumPy</h3>
<p>
  Copyright &copy; 2005&ndash;2024 NumPy Developers.<br>
  Distributed under the <b>BSD 3-Clause License</b>.<br>
  <a href="https://numpy.org">https://numpy.org</a>
</p>

<h3>imageio</h3>
<p>
  Copyright &copy; 2014&ndash;2024 imageio contributors.<br>
  Distributed under the <b>BSD 2-Clause License</b>.<br>
  <a href="https://imageio.readthedocs.io">https://imageio.readthedocs.io</a>
</p>

<h3>tifffile</h3>
<p>
  Copyright &copy; 2008&ndash;2024 Christoph Gohlke.<br>
  Distributed under the <b>BSD 3-Clause License</b>.<br>
  <a href="https://pypi.org/project/tifffile">https://pypi.org/project/tifffile</a>
</p>

<h3>scikit-image</h3>
<p>
  Copyright &copy; 2009&ndash;2024 the scikit-image team.<br>
  Distributed under the <b>BSD 3-Clause License</b>.<br>
  <a href="https://scikit-image.org">https://scikit-image.org</a>
</p>

<h3>PyOpenGL</h3>
<p>
  Copyright &copy; 2000&ndash;2011 Mike Fletcher.<br>
  Distributed under the <b>BSD-style License</b>.<br>
  <a href="https://pyopengl.sourceforge.net">https://pyopengl.sourceforge.net</a>
</p>
"""


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('About Voxels Viewer')
        self.setMinimumSize(580, 520)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel(APP_NAME)
        title_font = title.font()
        title_font.setPointSize(title_font.pointSize() + 4)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)

        version_label = QLabel(f'Version {APP_VERSION_FULL}')
        version_label.setStyleSheet('color: gray;')
        layout.addWidget(version_label)

        tabs = QtWidgets.QTabWidget()

        license_browser = QTextBrowser()
        license_browser.setHtml(_LICENSE_HTML)
        license_browser.setOpenLinks(False)
        license_browser.anchorClicked.connect(
            lambda url: QtGui.QDesktopServices.openUrl(url)
        )
        tabs.addTab(license_browser, 'License')

        third_party_browser = QTextBrowser()
        third_party_browser.setHtml(_THIRD_PARTY_HTML)
        third_party_browser.setOpenLinks(False)
        third_party_browser.anchorClicked.connect(
            lambda url: QtGui.QDesktopServices.openUrl(url)
        )
        tabs.addTab(third_party_browser, 'Third-Party Licenses')

        layout.addWidget(tabs)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class _ImportWorker(QtCore.QThread):
    progress = Signal(int, int)
    finished = Signal(object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            result = self._fn(self.progress.emit)
        except Exception:
            result = None
        self.finished.emit(result)


class VolumeData:
    def __init__(self):
        self.volume = None
        self.voxel_size = (1.0, 1.0, 1.0)
        self.dimensions = (0, 0, 0)
        self.dtype = np.uint8
        self.signed = False
        self.transform = np.eye(4, dtype=np.float32)
        self.histogram = None

    def is_loaded(self):
        return self.volume is not None

    def set_volume(self, volume, voxel_size=(1.0, 1.0, 1.0)):
        self.volume = volume
        self.dimensions = self.volume.shape[::-1]
        self.voxel_size = voxel_size
        self.dtype = self.volume.dtype
        self.signed = np.issubdtype(self.dtype, np.signedinteger)
        self.compute_histogram()

    def compute_histogram(self):
        if self.volume is None:
            self.histogram = None
            return
        flat = self.volume.ravel()   # view if C-contiguous, copy otherwise
        if flat.size == 0:
            self.histogram = None
            return
        try:
            # Sample large volumes so np.histogram never needs a full float64 copy
            # (963 M voxels × 8 bytes = 7.7 GB — we cap at ~32 MB of samples)
            if flat.size > 4_000_000:
                flat = flat[::max(1, flat.size // 4_000_000)]
            self.histogram, self.bin_edges = np.histogram(flat, bins=256)
        except Exception:
            self.histogram = None

    def apply_intensity_mapping(self, mapping_curve):
        if self.volume is None:
            return
        flat = self.volume.astype(np.float32).ravel()
        in_min, in_max = mapping_curve[0][0], mapping_curve[-1][0]
        scaled = (flat - in_min) / max(in_max - in_min, 1e-6)
        scaled = np.clip(scaled, 0.0, 1.0)
        mapped = np.interp(scaled, [0, 1], [mapping_curve[0][1], mapping_curve[-1][1]])
        self.volume = mapped.reshape(self.volume.shape)
        self.compute_histogram()

    def transform_voxel(self, matrix):
        self.transform = matrix @ self.transform


class MetadataDialog(QDialog):
    def __init__(self, parent=None, raw=False, initial_values=None):
        super().__init__(parent)
        self.setWindowTitle('Volume Metadata')
        self.setMinimumWidth(360)
        self.raw = raw
        self.initial_values = initial_values or {}
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)

        self.edit_width = QLineEdit(str(self.initial_values.get('width', 256)))
        self.edit_height = QLineEdit(str(self.initial_values.get('height', 256)))
        self.edit_depth = QLineEdit(str(self.initial_values.get('depth', 256)))
        self.edit_voxel_x = QLineEdit(str(self.initial_values.get('voxel_x', 1.0)))
        self.edit_voxel_y = QLineEdit(str(self.initial_values.get('voxel_y', 1.0)))
        self.edit_voxel_z = QLineEdit(str(self.initial_values.get('voxel_z', 1.0)))
        self.combo_dtype = QComboBox()
        self.combo_dtype.addItems([name for name, _ in SUPPORTED_DTYPES])
        self.combo_dtype.setCurrentText(self.initial_values.get('dtype', '8-bit unsigned'))
        self.combo_byteorder = QComboBox()
        self.combo_byteorder.addItems(['Little-endian', 'Big-endian'])
        self.combo_byteorder.setCurrentText(self.initial_values.get('byteorder', 'Little-endian'))
        self.check_flip = QCheckBox('Flip Z axis')
        self.check_flip.setChecked(False)

        layout.addRow('Width (X):', self.edit_width)
        layout.addRow('Height (Y):', self.edit_height)
        layout.addRow('Depth (Z):', self.edit_depth)
        layout.addRow('Voxel size X (mm):', self.edit_voxel_x)
        layout.addRow('Voxel size Y (mm):', self.edit_voxel_y)
        layout.addRow('Voxel size Z (mm):', self.edit_voxel_z)
        layout.addRow('Data type:', self.combo_dtype)
        if self.raw:
            layout.addRow('Byte order:', self.combo_byteorder)
            layout.addRow(self.check_flip)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            Qt.Horizontal,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_metadata(self):
        dims = (int(self.edit_width.text()), int(self.edit_height.text()), int(self.edit_depth.text()))
        voxels = (float(self.edit_voxel_x.text()), float(self.edit_voxel_y.text()), float(self.edit_voxel_z.text()))
        dtype = parse_dtype(self.combo_dtype.currentText())
        flip_z = self.check_flip.isChecked()
        big_endian = self.combo_byteorder.currentText() == 'Big-endian'
        return dims, voxels, dtype, flip_z, big_endian


class PreferencesDialog(QDialog):
    def __init__(self, parent=None, preferences=None):
        super().__init__(parent)
        self.setWindowTitle('Preferences')
        self.setMinimumWidth(320)
        self._prefs = dict(preferences or {})
        self.setup_ui()

    def setup_ui(self):
        layout = QFormLayout(self)

        self.combo_theme = QComboBox()
        self.combo_theme.addItems(['Automatic', 'Light', 'Dark'])
        self.combo_theme.setCurrentText(self._prefs.get('theme', 'Automatic'))
        layout.addRow('Theme:', self.combo_theme)

        self.combo_hist_scale = QComboBox()
        self.combo_hist_scale.addItems(['Linear', 'Logarithmic'])
        self.combo_hist_scale.setCurrentText(self._prefs.get('histogram_scale', 'Linear'))
        layout.addRow('Histogram vertical scale:', self.combo_hist_scale)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            Qt.Horizontal,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_preferences(self):
        return {
            'theme': self.combo_theme.currentText(),
            'histogram_scale': self.combo_hist_scale.currentText(),
        }


class RangeSlider(QWidget):
    """Horizontal slider with two handles defining a [low, high] range."""
    valueChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._minimum = 0
        self._maximum = 10000
        self._low = 0
        self._high = 10000
        self._handle_radius = 7
        self._drag = None          # 'low', 'high', or None
        self._last = 'high'        # which handle was touched last
        self._span_color = None    # optional override for the selected-span colour
        self.setMinimumHeight(24)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.NoFocus)

    # ── Public API ────────────────────────────────────────────────────────
    def setRange(self, minimum, maximum):
        self._minimum = int(minimum)
        self._maximum = int(maximum)
        self._low = max(self._minimum, min(self._low, self._maximum))
        self._high = max(self._minimum, min(self._high, self._maximum))
        self.update()

    def low(self):
        return self._low

    def high(self):
        return self._high

    def setSpanColor(self, color):
        """Override the colour of the selected span + handles (None = default)."""
        self._span_color = color
        self.update()

    def last_handle(self):
        """'low' or 'high' — whichever handle was moved most recently."""
        return self._last

    def last_value(self):
        return self._low if self._last == 'low' else self._high

    def setLow(self, value):
        value = int(max(self._minimum, min(value, self._high)))
        if value != self._low:
            self._low = value
            self.update()
            self.valueChanged.emit()

    def setHigh(self, value):
        value = int(max(self._low, min(value, self._maximum)))
        if value != self._high:
            self._high = value
            self.update()
            self.valueChanged.emit()

    def setValues(self, low, high):
        low = int(max(self._minimum, min(low, self._maximum)))
        high = int(max(self._minimum, min(high, self._maximum)))
        if low > high:
            low, high = high, low
        changed = (low != self._low or high != self._high)
        self._low, self._high = low, high
        self.update()
        if changed:
            self.valueChanged.emit()

    # ── Geometry helpers ──────────────────────────────────────────────────
    def _span(self):
        m = self._handle_radius + 1
        return m, max(m + 1, self.width() - m)

    def _value_to_x(self, value):
        lo, hi = self._span()
        if self._maximum == self._minimum:
            return lo
        frac = (value - self._minimum) / (self._maximum - self._minimum)
        return lo + frac * (hi - lo)

    def _x_to_value(self, x):
        lo, hi = self._span()
        if hi <= lo:
            return self._minimum
        frac = max(0.0, min(1.0, (x - lo) / (hi - lo)))
        return int(round(self._minimum + frac * (self._maximum - self._minimum)))

    # ── Mouse handling ────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        x = event.pos().x()
        xlow = self._value_to_x(self._low)
        xhigh = self._value_to_x(self._high)
        if self._low == self._high:
            self._drag = 'high' if x >= xhigh else 'low'
        elif abs(x - xlow) <= abs(x - xhigh):
            self._drag = 'low'
        else:
            self._drag = 'high'
        self._move_to(x)

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            self._move_to(event.pos().x())

    def mouseReleaseEvent(self, event):
        self._drag = None

    def _move_to(self, x):
        value = self._x_to_value(x)
        if self._drag == 'low':
            self._last = 'low'
            self.setLow(value)
        elif self._drag == 'high':
            self._last = 'high'
            self.setHigh(value)

    # ── Painting ──────────────────────────────────────────────────────────
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(_ANTIALIASING)
        pal = self.palette()
        enabled = self.isEnabled()

        gy = self.height() // 2
        lo_x, hi_x = self._span()
        xlow = self._value_to_x(self._low)
        xhigh = self._value_to_x(self._high)

        # groove background
        p.setPen(Qt.NoPen)
        p.setBrush(pal.color(QtGui.QPalette.Mid))
        p.drawRoundedRect(QtCore.QRectF(lo_x, gy - 2, hi_x - lo_x, 4), 2, 2)

        # selected span between the two handles
        if enabled and self._span_color is not None:
            span_col = QtGui.QColor(self._span_color)
        elif enabled:
            span_col = pal.color(QtGui.QPalette.Highlight)
        else:
            span_col = pal.color(QtGui.QPalette.Mid)
        p.setBrush(span_col)
        p.drawRoundedRect(QtCore.QRectF(xlow, gy - 2, xhigh - xlow, 4), 2, 2)

        # handles
        handle_col = span_col if enabled else pal.color(QtGui.QPalette.Button)
        r = self._handle_radius
        p.setBrush(handle_col)
        p.setPen(QtGui.QPen(pal.color(QtGui.QPalette.Window), 1))
        for hx in (xlow, xhigh):
            p.drawEllipse(QtCore.QPointF(hx, gy), r, r)
        p.end()


def _make_voxel_icon(color, size=32):
    """A square cut into a 3x3 grid — the 'voxels' glyph for the Volume node."""
    px = QtGui.QPixmap(size, size)
    px.fill(Qt.transparent)
    p = QtGui.QPainter(px)
    pen = QtGui.QPen(QtGui.QColor(color))
    pen.setWidth(2)
    pen.setJoinStyle(Qt.MiterJoin)
    p.setPen(pen)
    m = 4
    s = size - 2 * m
    p.drawRect(m, m, s, s)
    for i in (1, 2):
        off = round(i * s / 3)
        p.drawLine(m + off, m, m + off, m + s)
        p.drawLine(m, m + off, m + s, m + off)
    p.end()
    return QtGui.QIcon(px)


def _make_folder_icon(color, size=32):
    """A simple outlined folder glyph, matching the line aesthetic of the
    voxel icon (so the tree icons stay monochrome and theme-aware)."""
    px = QtGui.QPixmap(size, size)
    px.fill(Qt.transparent)
    p = QtGui.QPainter(px)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    pen = QtGui.QPen(QtGui.QColor(color))
    pen.setWidth(2)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    m = 4
    w = size - 2 * m
    h = w * 0.74
    top = (size - h) / 2.0
    tab_w = w * 0.42
    tab_h = h * 0.20
    path = QtGui.QPainterPath()
    path.moveTo(m, top + tab_h)
    path.lineTo(m, top)
    path.lineTo(m + tab_w, top)
    path.lineTo(m + tab_w + tab_h, top + tab_h)
    path.lineTo(m + w, top + tab_h)
    path.lineTo(m + w, top + h)
    path.lineTo(m, top + h)
    path.closeSubpath()
    p.drawPath(path)
    p.end()
    return QtGui.QIcon(px)


def _triangle_icon(color, size=32):
    """An outlined triangle — the surface-mesh glyph for the project tree."""
    px = QtGui.QPixmap(size, size)
    px.fill(Qt.transparent)
    p = QtGui.QPainter(px)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    pen = QtGui.QPen(QtGui.QColor(color))
    pen.setWidth(2)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    m = 4.5
    path = QtGui.QPainterPath()
    path.moveTo(size / 2.0, m)               # apex
    path.lineTo(size - m, size - m)          # bottom-right
    path.lineTo(m, size - m)                 # bottom-left
    path.closeSubpath()
    p.drawPath(path)
    p.end()
    return QtGui.QIcon(px)


class ProjectStructureWidget(QWidget):
    """Hierarchical project tree shown above the histogram in the sidebar.

        Volume                (renamable, voxel icon)
          ├─ Alignments
          ├─ Measurements
          └─ Gray Values

    The folders are placeholders for now (expandable but empty). Right-clicking
    the Volume offers Properties (→ Volume Information) and Rename.
    """
    properties_requested = Signal()
    volume_renamed = Signal(str)
    alignment_activate_requested = Signal(int)
    alignment_remove_requested = Signal(int)
    measurement_activate_requested = Signal(int)
    measurement_remove_requested = Signal(int)
    grayvalue_activate_requested = Signal(int)
    grayvalue_remove_requested = Signal(int)
    grayvalue_export_requested = Signal(int)
    create_surface_requested = Signal()
    surface_activate_requested = Signal(int)
    surface_remove_requested = Signal(int)
    surface_export_stl_requested = Signal(int)
    surface_promote_requested = Signal(int)
    projectmesh_activate_requested = Signal(int)
    projectmesh_export_stl_requested = Signal(int)
    projectmesh_remove_requested = Signal(int)

    FOLDERS = ('Alignments', 'Measurements', 'Gray Values')
    _MESH_FOLDERS = ('Alignments', 'Measurements')   # meshes have no Gray Values

    # Custom item-data roles (offset from UserRole).
    _ROLE_KIND = Qt.UserRole          # 'volume' | 'folder' | 'alignment'
    _ROLE_INDEX = Qt.UserRole + 1     # alignment index in MainWindow's list
    _ROLE_INITIAL = Qt.UserRole + 2   # True for the non-removable Initial Alignment
    _ROLE_SUBKIND = Qt.UserRole + 3   # tool sub-kind (e.g. 'profile' gray value)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._icon_color = QtGui.QColor('#6e6e6e')   # neutral gray (light theme)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        title = QLabel('Project')
        title.setStyleSheet('font-weight: bold; padding: 2px 4px;')
        layout.addWidget(title)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        # Single click never edits; F2 (on the editable Volume item) and the
        # context-menu Rename are the explicit rename paths.
        self.tree.setEditTriggers(QtWidgets.QAbstractItemView.EditKeyPressed)
        # Small, uniform icons so the tool glyphs sit neatly beside the entries.
        self.tree.setIconSize(QtCore.QSize(14, 14))
        layout.addWidget(self.tree)

        self._build_tree()
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)

    def _build_tree(self):
        self.volume_item = QtWidgets.QTreeWidgetItem(self.tree, ['Volume'])
        self.volume_item.setIcon(0, _make_voxel_icon(self._icon_color))
        self.volume_item.setData(0, self._ROLE_KIND, 'volume')
        self.volume_item.setFlags(self.volume_item.flags() | Qt.ItemIsEditable)

        folder_icon = _make_folder_icon(self._icon_color)
        self.folder_items = {}
        for name in self.FOLDERS:
            child = QtWidgets.QTreeWidgetItem(self.volume_item, [name])
            child.setIcon(0, folder_icon)
            child.setData(0, self._ROLE_KIND, 'folder')
            # Folders are not renamable (no ItemIsEditable flag).
            self.folder_items[name] = child
        self.alignment_items = []
        self.measurement_items = []
        self.grayvalue_items = []
        self.surface_items = []
        self.project_mesh_items = []
        self.volume_item.setExpanded(True)

    def set_project_meshes(self, entries):
        """Top-level mesh objects (siblings of Volume, shown below it), each a
        first-class node carrying the same folder children as the Volume."""
        for it in list(self.project_mesh_items):
            idx = self.tree.indexOfTopLevelItem(it)
            if idx >= 0:
                self.tree.takeTopLevelItem(idx)
        self.project_mesh_items = []
        tri = _triangle_icon(self._icon_color)
        folder_icon = _make_folder_icon(self._icon_color)
        for i, e in enumerate(entries):
            node = QtWidgets.QTreeWidgetItem(self.tree, [e['name']])
            node.setData(0, self._ROLE_KIND, 'projectmesh')
            node.setData(0, self._ROLE_INDEX, i)
            node.setIcon(0, tri)
            for fname in self._MESH_FOLDERS:
                child = QtWidgets.QTreeWidgetItem(node, [fname])
                child.setIcon(0, folder_icon)
                child.setData(0, self._ROLE_KIND, 'folder')
            node.setExpanded(True)
            self.project_mesh_items.append(node)

    def set_surfaces(self, entries):
        """Rebuild the surface entries — direct children of Volume, shown right
        under it (above the folders), each with a triangle icon."""
        for it in self.surface_items:
            self.volume_item.removeChild(it)
        self.surface_items = []
        icon = _triangle_icon(self._icon_color)
        for i, e in enumerate(entries):
            item = QtWidgets.QTreeWidgetItem([e['name']])
            item.setData(0, self._ROLE_KIND, 'surface')
            item.setData(0, self._ROLE_INDEX, i)
            item.setIcon(0, icon)
            self.volume_item.insertChild(i, item)   # before the folders
            self.surface_items.append(item)
        self.volume_item.setExpanded(True)

    def set_alignments(self, entries, active_index):
        """Rebuild the Alignments folder.

        ``entries`` is a list of {'name': str, 'initial': bool}; the entry at
        ``active_index`` is shown with a ' (active)' suffix."""
        folder = self.folder_items['Alignments']
        folder.takeChildren()
        self.alignment_items = []
        tripod = _axes_tripod_icon(self._icon_color)
        for i, e in enumerate(entries):
            label = e['name'] + (' (active)' if i == active_index else '')
            item = QtWidgets.QTreeWidgetItem(folder, [label])
            item.setData(0, self._ROLE_KIND, 'alignment')
            item.setData(0, self._ROLE_INDEX, i)
            item.setData(0, self._ROLE_INITIAL, bool(e.get('initial')))
            item.setIcon(0, tripod)
            self.alignment_items.append(item)
        if entries:
            folder.setExpanded(True)

    def set_measurements(self, entries):
        """Rebuild the Measurements folder. ``entries`` is a list of
        {'name': str}; index in the list is the measurement's id."""
        folder = self.folder_items['Measurements']
        folder.takeChildren()
        self.measurement_items = []
        for i, e in enumerate(entries):
            item = QtWidgets.QTreeWidgetItem(folder, [e['name']])
            item.setData(0, self._ROLE_KIND, 'measurement')
            item.setData(0, self._ROLE_INDEX, i)
            kind = e.get('kind')
            item.setData(0, self._ROLE_SUBKIND, kind)
            if kind:
                item.setIcon(0, _measure_icon(kind, self._icon_color))
            self.measurement_items.append(item)
        if entries:
            folder.setExpanded(True)

    def set_gray_values(self, entries):
        """Rebuild the Gray Values folder. ``entries`` is a list of
        {'name': str}; index in the list is the gray-value's id."""
        folder = self.folder_items['Gray Values']
        folder.takeChildren()
        self.grayvalue_items = []
        for i, e in enumerate(entries):
            item = QtWidgets.QTreeWidgetItem(folder, [e['name']])
            item.setData(0, self._ROLE_KIND, 'grayvalue')
            item.setData(0, self._ROLE_INDEX, i)
            kind = e.get('kind')
            item.setData(0, self._ROLE_SUBKIND, kind)
            if kind:
                item.setIcon(0, _grayvalue_icon(kind, self._icon_color))
            self.grayvalue_items.append(item)
        if entries:
            folder.setExpanded(True)

    def set_volume_name(self, name):
        """Set the Volume node's text without emitting volume_renamed."""
        self.tree.blockSignals(True)
        self.volume_item.setText(0, name or 'Volume')
        self.tree.blockSignals(False)

    def volume_name(self):
        return self.volume_item.text(0)

    def _on_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if item is None:
            return
        kind = item.data(0, self._ROLE_KIND)
        if kind == 'volume':
            menu = QtWidgets.QMenu(self)
            rename_action = menu.addAction('Rename')
            surface_action = menu.addAction('Create Surface...')
            props_action = menu.addAction('Properties...')
            chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if chosen == props_action:
                self.properties_requested.emit()
            elif chosen == surface_action:
                self.create_surface_requested.emit()
            elif chosen == rename_action:
                self.tree.editItem(item, 0)
        elif kind == 'surface':
            idx = int(item.data(0, self._ROLE_INDEX))
            menu = QtWidgets.QMenu(self)
            show_action = menu.addAction('Show in 3D')
            export_menu = menu.addMenu('Export')
            stl_action = export_menu.addAction('As STL...')
            # 'Add to Project Structure' is hidden for now — to be reactivated later.
            remove_action = menu.addAction('Remove')
            chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if chosen == show_action:
                self.surface_activate_requested.emit(idx)
            elif chosen == stl_action:
                self.surface_export_stl_requested.emit(idx)
            elif chosen == remove_action:
                self.surface_remove_requested.emit(idx)
        elif kind == 'projectmesh':
            idx = int(item.data(0, self._ROLE_INDEX))
            menu = QtWidgets.QMenu(self)
            show_action = menu.addAction('Show in 3D')
            stl_action = menu.addAction('Export As STL...')
            remove_action = menu.addAction('Remove')
            chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if chosen == show_action:
                self.projectmesh_activate_requested.emit(idx)
            elif chosen == stl_action:
                self.projectmesh_export_stl_requested.emit(idx)
            elif chosen == remove_action:
                self.projectmesh_remove_requested.emit(idx)
        elif kind == 'alignment':
            idx = int(item.data(0, self._ROLE_INDEX))
            is_initial = bool(item.data(0, self._ROLE_INITIAL))
            menu = QtWidgets.QMenu(self)
            activate_action = menu.addAction('Activate')
            # Initial Alignment is activatable but never removable.
            remove_action = None if is_initial else menu.addAction('Remove')
            chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if chosen == activate_action:
                self.alignment_activate_requested.emit(idx)
            elif remove_action is not None and chosen == remove_action:
                self.alignment_remove_requested.emit(idx)
        elif kind == 'measurement':
            idx = int(item.data(0, self._ROLE_INDEX))
            menu = QtWidgets.QMenu(self)
            goto_action = menu.addAction('Go To')
            remove_action = menu.addAction('Remove')
            chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if chosen == goto_action:
                self.measurement_activate_requested.emit(idx)
            elif chosen == remove_action:
                self.measurement_remove_requested.emit(idx)
        elif kind == 'grayvalue':
            idx = int(item.data(0, self._ROLE_INDEX))
            menu = QtWidgets.QMenu(self)
            goto_action = menu.addAction('Go To')
            # Only the profile tool produces a sampled curve to export.
            export_action = (menu.addAction('Export to CSV...')
                             if item.data(0, self._ROLE_SUBKIND) == 'profile'
                             else None)
            remove_action = menu.addAction('Remove')
            chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if chosen == goto_action:
                self.grayvalue_activate_requested.emit(idx)
            elif export_action is not None and chosen == export_action:
                self.grayvalue_export_requested.emit(idx)
            elif chosen == remove_action:
                self.grayvalue_remove_requested.emit(idx)

    def _on_item_double_clicked(self, item, _col):
        if item is None:
            return
        kind = item.data(0, self._ROLE_KIND)
        if kind == 'alignment':
            self.alignment_activate_requested.emit(int(item.data(0, self._ROLE_INDEX)))
        elif kind == 'measurement':
            self.measurement_activate_requested.emit(int(item.data(0, self._ROLE_INDEX)))
        elif kind == 'grayvalue':
            self.grayvalue_activate_requested.emit(int(item.data(0, self._ROLE_INDEX)))
        elif kind == 'surface':
            self.surface_activate_requested.emit(int(item.data(0, self._ROLE_INDEX)))
        elif kind == 'projectmesh':
            self.projectmesh_activate_requested.emit(int(item.data(0, self._ROLE_INDEX)))

    def _on_item_changed(self, item, _col):
        if item is self.volume_item:
            name = (item.text(0) or '').strip()
            if not name:
                name = 'Volume'
                self.set_volume_name(name)
            self.volume_renamed.emit(name)

    def apply_theme(self, dark):
        self._icon_color = QtGui.QColor('#b0b0b0' if dark else '#6e6e6e')
        self.tree.blockSignals(True)
        self.volume_item.setIcon(0, _make_voxel_icon(self._icon_color))
        folder_icon = _make_folder_icon(self._icon_color)
        for child in self.folder_items.values():
            child.setIcon(0, folder_icon)
        # Re-tint the per-entry tool icons to match the new theme.
        tripod = _axes_tripod_icon(self._icon_color)
        for item in self.alignment_items:
            item.setIcon(0, tripod)
        for item in self.measurement_items:
            kind = item.data(0, self._ROLE_SUBKIND)
            if kind:
                item.setIcon(0, _measure_icon(kind, self._icon_color))
        for item in self.grayvalue_items:
            kind = item.data(0, self._ROLE_SUBKIND)
            if kind:
                item.setIcon(0, _grayvalue_icon(kind, self._icon_color))
        triangle = _triangle_icon(self._icon_color)
        for item in self.surface_items:
            item.setIcon(0, triangle)
        for node in self.project_mesh_items:
            node.setIcon(0, triangle)
            for c in range(node.childCount()):
                node.child(c).setIcon(0, folder_icon)
        self.tree.blockSignals(False)


class BrightnessCurveWidget(QWidget):
    curve_changed = Signal(object)
    auto_minmax_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(200)
        self._raw_hist_bins = None
        self._raw_hist_counts = None
        self._bin_centers = None
        self._log_scale = False
        self._data_min = 0.0
        self._data_max = 255.0
        self._integer = False
        self._syncing_handles = False
        self._hist_max = 1.0          # max bar height; the mapping ramp is scaled to it
        self.init_ui()
        # Full-range window by default so both dots sit at their proper corners
        # (low at the left baseline, high at the far right) before any data loads.
        self.points = [(self._data_min, 0.0), (self._data_min, 0.0),
                       (self._data_max, 1.0), (self._data_max, 1.0)]
        self.update_plot()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # match the Project Structure panel
        layout.setSpacing(0)                    # no gap below the title
        title = QLabel('Histogram')
        title.setStyleSheet('font-weight: bold; padding: 0 4px 2px 4px;')  # tiny padding below only
        layout.addWidget(title)
        self.plot_widget = pg.PlotWidget(background='w')
        self.plot_widget.setLabel('bottom', 'Intensity')
        self.plot_widget.setLabel('left', 'Mapped')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)

        self.hist_bar = pg.BarGraphItem(x=[], height=[], width=0.005, brush='lightgray')
        self.plot_widget.addItem(self.hist_bar)
        self.curve_plot = self.plot_widget.plot([], [], pen=pg.mkPen('#0077cc', width=2))
        # Static endpoint dots (data min/max); the window points are draggable.
        self.point_plot = self.plot_widget.plot([], [], pen=None, symbol='o', symbolBrush='#cc3300', symbolSize=8)

        # Draggable red dots for the window min (y=0) and max (y=1). Dragging
        # them horizontally adjusts the window and stays in sync with the slider.
        self.low_handle = pg.TargetItem(
            pos=(0.0, 0.0), size=10, symbol='o', movable=True,
            pen=pg.mkPen('#cc3300'), brush=pg.mkBrush('#cc3300'),
            hoverPen=pg.mkPen('#ff5a33'), hoverBrush=pg.mkBrush('#ff5a33'))
        self.high_handle = pg.TargetItem(
            pos=(0.0, 1.0), size=10, symbol='o', movable=True,
            pen=pg.mkPen('#cc3300'), brush=pg.mkBrush('#cc3300'),
            hoverPen=pg.mkPen('#ff5a33'), hoverBrush=pg.mkBrush('#ff5a33'))
        for h in (self.low_handle, self.high_handle):
            h.setZValue(10)
            self.plot_widget.addItem(h, ignoreBounds=True)
        self.low_handle.sigPositionChanged.connect(lambda: self._on_handle_dragged('low'))
        self.high_handle.sigPositionChanged.connect(lambda: self._on_handle_dragged('high'))

        self.range_slider = RangeSlider()
        self.range_slider.setRange(0, 10000)
        self.range_slider.setValues(0, 10000)
        self.range_slider.valueChanged.connect(self.on_slider_changed)

        self.label_min_val = QLabel('0')
        self.label_max_val = QLabel('255')

        window_row = QHBoxLayout()
        window_row.addWidget(QLabel('Window'))
        window_row.addStretch()
        window_row.addWidget(self.label_min_val)
        window_row.addWidget(QLabel('–'))
        window_row.addWidget(self.label_max_val)

        layout.addLayout(window_row)
        layout.addWidget(self.range_slider)

        self.auto_minmax_btn = QCheckBox('Auto Min Max')
        self.auto_minmax_btn.toggled.connect(self._on_auto_minmax_toggled)
        layout.addWidget(self.auto_minmax_btn)

    def _on_auto_minmax_toggled(self, enabled):
        self.range_slider.setEnabled(not enabled)
        # In Auto Min Max mode the window is driven by region selection, so the
        # red dots become grey and non-draggable.
        self._set_handles_interactive(not enabled)
        self.auto_minmax_toggled.emit(enabled)

    def _set_handles_interactive(self, interactive):
        color = '#cc3300' if interactive else '#888888'
        for h in (self.low_handle, self.high_handle):
            h.movable = interactive
            h.setPen(pg.mkPen(color))
            h.setBrush(pg.mkBrush(color))
            h.setHoverPen(pg.mkPen(color if not interactive else '#ff5a33'))
            h.setHoverBrush(pg.mkBrush(color if not interactive else '#ff5a33'))
        self.point_plot.setSymbolBrush(color)

    def _value_to_pos(self, v):
        rng = self._data_max - self._data_min
        if rng <= 0:
            return 0
        return int(np.clip((v - self._data_min) / rng * 10000, 0, 10000))

    def _position_handles(self):
        """Snap the draggable dots to the current window points (horizontal only).

        The mapped Y values (0/1) are scaled to the histogram bar height so the
        dots span the full plot height and stay grabbable.
        """
        if not hasattr(self, 'low_handle') or len(self.points) < 4:
            return
        H = self._hist_max
        prev = self._syncing_handles
        self._syncing_handles = True
        try:
            self.low_handle.setPos(self.points[1][0], self.points[1][1] * H)
            self.high_handle.setPos(self.points[2][0], self.points[2][1] * H)
        finally:
            self._syncing_handles = prev

    def _on_handle_dragged(self, which):
        if self._syncing_handles:
            return
        self._syncing_handles = True
        try:
            handle = self.low_handle if which == 'low' else self.high_handle
            pos = self._value_to_pos(handle.pos().x())
            if which == 'low':
                self.range_slider.setLow(pos)
            else:
                self.range_slider.setHigh(pos)
        finally:
            self._syncing_handles = False
        # Keep the dot on its horizontal line even if the slider value didn't change.
        self._position_handles()

    def set_window_minmax(self, min_val, max_val):
        rng = self._data_max - self._data_min
        if rng <= 0:
            return

        def to_pos(v):
            return int(np.clip((v - self._data_min) / rng * 10000, 0, 10000))

        self.range_slider.blockSignals(True)
        self.range_slider.setValues(to_pos(min_val), to_pos(max_val))
        self.range_slider.blockSignals(False)
        self.on_slider_changed()

    def set_histogram(self, bins, counts):
        if bins is None or counts is None:
            return
        self._raw_hist_bins = bins
        self._raw_hist_counts = counts
        self._render_histogram()

    def _render_histogram(self):
        if self._raw_hist_bins is None or self._raw_hist_counts is None:
            return
        bins = self._raw_hist_bins
        counts = self._raw_hist_counts
        centers = (bins[:-1] + bins[1:]) / 2.0
        if self._log_scale:
            heights = np.log1p(counts.astype(float))
        else:
            heights = counts.astype(float)
        self._bin_centers = centers
        h_max = float(np.max(heights)) if heights.size else 1.0
        self._hist_max = h_max if h_max > 0 else 1.0
        brushes, pens = self._bar_styles(centers)
        self.hist_bar.setOpts(x=centers, height=heights,
                              width=(bins[1] - bins[0]) * 0.9,
                              brushes=brushes, pens=pens)
        self.plot_widget.setXRange(float(bins[0]), float(bins[-1]))
        # Rescale the mapping ramp + draggable dots to the new bar height.
        self.update_plot()

    def _bar_styles(self, centers):
        """Per-bar grayscale brushes + matching pens for the window mapping:
        black below window-min, white above window-max, gray ramp between.

        The pen matches the brush so each thin bar is a solid block of its
        gray value (no contrasting outline diluting the colour).
        """
        lo, hi = self.window_min(), self.window_max()
        lo, hi = min(lo, hi), max(lo, hi)
        levels = np.clip((centers - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        levels = (levels * 255).astype(int)
        colors = [QtGui.QColor(int(v), int(v), int(v)) for v in levels]
        return [pg.mkBrush(c) for c in colors], [pg.mkPen(c) for c in colors]

    def _update_histogram_colors(self):
        if getattr(self, '_bin_centers', None) is None:
            return
        brushes, pens = self._bar_styles(self._bin_centers)
        self.hist_bar.setOpts(brushes=brushes, pens=pens)

    def set_histogram_scale(self, log: bool):
        self._log_scale = log
        self._render_histogram()

    def apply_theme(self, dark: bool):
        self.plot_widget.setBackground('#1e1e1e' if dark else 'w')
        # Bars are coloured by their grayscale value (see _bar_brushes), so the
        # theme only affects the plot background and axes — not the bar fill.
        text_color = '#cccccc' if dark else '#000000'
        pen = pg.mkPen(color=text_color)
        for name in ('bottom', 'left'):
            ax = self.plot_widget.getAxis(name)
            ax.setPen(pen)
            ax.setTextPen(pen)

    def set_data_range(self, min_val: float, max_val: float, integer: bool = False):
        self._data_min = float(min_val)
        self._data_max = float(max_val) if max_val > min_val else float(min_val) + 1.0
        self._integer = bool(integer)
        self.range_slider.blockSignals(True)
        self.range_slider.setValues(0, 10000)
        self.range_slider.blockSignals(False)
        self.on_slider_changed()

    def _slider_to_value(self, pos: int) -> float:
        return self._data_min + pos / 10000.0 * (self._data_max - self._data_min)

    def _fmt(self, value: float) -> str:
        if self._integer:
            return f'{int(round(value))}'
        return f'{value:.4g}'

    def window_min(self) -> float:
        return self._slider_to_value(self.range_slider.low())

    def window_max(self) -> float:
        return self._slider_to_value(self.range_slider.high())

    def _update_range_labels(self):
        self.label_min_val.setText(self._fmt(self.window_min()))
        self.label_max_val.setText(self._fmt(self.window_max()))

    def on_slider_changed(self):
        low = self.window_min()
        high = self.window_max()
        low, high = min(low, high), max(low, high)
        self.points = [
            (self._data_min, 0.0),
            (low, 0.0),
            (high, 1.0),
            (self._data_max, 1.0),
        ]
        self._update_range_labels()
        self._update_histogram_colors()
        self.update_plot()
        self.curve_changed.emit(self.points)

    def update_plot(self):
        # Scale the mapping ramp's 0/1 range to the histogram bar height so it
        # overlays the bars instead of being squished at the bottom.
        H = self._hist_max
        xs = [p[0] for p in self.points]
        ys = [p[1] * H for p in self.points]
        self.curve_plot.setData(xs, ys)
        # Only the endpoints are static dots; window points are the handles.
        self.point_plot.setData([xs[0], xs[-1]], [ys[0], ys[-1]])
        self._position_handles()


class TripodWidget(QWidget):
    """Axis orientation tripod overlaid in the bottom-left corner of a viewport.

    For 2D orientations the axes are fixed; for '3D' they rotate with the
    camera by reading azimuth/elevation from a GLViewWidget reference.
    """

    _SIZE     = 68
    _AXIS_LEN = 22
    _CX       = 34
    _CY       = 34
    _COLORS   = {
        'X': QtGui.QColor(220,  60,  60),
        'Y': QtGui.QColor( 60, 200,  60),
        'Z': QtGui.QColor( 60, 120, 255),
    }

    def __init__(self, orientation, parent=None, gl_view=None):
        super().__init__(parent)
        self.orientation = orientation
        self._gl_view = gl_view
        self.setFixedSize(self._SIZE, self._SIZE)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)

    # ------------------------------------------------------------------
    # Axis computation
    # ------------------------------------------------------------------

    def _axes(self):
        """Return [(label, QColor, sx, sy, alpha), ...] sorted back-to-front."""
        C = self._COLORS
        if self.orientation == 'XY':
            return [('X', C['X'], 1.0, 0.0, 1.0), ('Y', C['Y'], 0.0,  1.0, 1.0)]
        if self.orientation == 'YZ':
            return [('Y', C['Y'], 1.0, 0.0, 1.0), ('Z', C['Z'], 0.0,  1.0, 1.0)]
        if self.orientation == 'XZ':
            return [('X', C['X'], 1.0, 0.0, 1.0), ('Z', C['Z'], 0.0,  1.0, 1.0)]
        return self._axes_3d()

    def _axes_3d(self):
        if self._gl_view is None:
            return []
        R = getattr(self._gl_view, '_view_rot', None)
        rows = []
        for label, world in [('X', (1, 0, 0)), ('Y', (0, 1, 0)), ('Z', (0, 0, 1))]:
            if R is not None:
                d  = np.array(world, dtype=np.float64)
                sx, sy, sz = float(R[0] @ d), float(R[1] @ d), float(R[2] @ d)
            else:
                az = self._gl_view.opts.get('azimuth', 45)
                el = self._gl_view.opts.get('elevation', 30)
                sx, sy, sz = self._project(*world, az, el)
            alpha = 1.0 if sz >= 0 else 0.4
            rows.append((label, self._COLORS[label], sx, sy, sz, alpha))
        rows.sort(key=lambda r: r[4])           # draw back-facing axes first
        return [(lbl, col, sx, sy, a) for lbl, col, sx, sy, _sz, a in rows]

    @staticmethod
    def _project(dx, dy, dz, az_deg, el_deg):
        """Project a world-space direction to GLViewWidget screen space.

        pyqtgraph viewMatrix applies rotate(azimuth+90, 0,0,-1) then
        rotate(elevation-90, 1,0,0), so we must add 90 to the azimuth.
        """
        az = math.radians(az_deg + 90)   # pyqtgraph uses azimuth+90 internally
        el = math.radians(el_deg - 90)
        # rotate around -Z by azimuth
        x1 =  dx * math.cos(az) + dy * math.sin(az)
        y1 = -dx * math.sin(az) + dy * math.cos(az)
        # rotate around X by (elevation - 90)
        sx = x1
        sy = y1 * math.cos(el) - dz * math.sin(el)
        sz = y1 * math.sin(el) + dz * math.cos(el)
        return sx, sy, sz

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        axes = self._axes()
        p = QtGui.QPainter(self)
        p.setRenderHint(_ANTIALIASING)

        # translucent disc background
        p.setPen(Qt.NoPen)
        p.setBrush(QtGui.QColor(20, 20, 20, 110))
        p.drawEllipse(2, 2, self._SIZE - 4, self._SIZE - 4)

        font = QtGui.QFont()
        font.setPixelSize(10)
        font.setBold(True)
        p.setFont(font)

        for label, color, sx, sy, alpha in axes:
            c = QtGui.QColor(color)
            c.setAlphaF(alpha)
            ex = int(self._CX + sx * self._AXIS_LEN)
            ey = int(self._CY - sy * self._AXIS_LEN)   # flip y for screen coords

            p.setPen(QtGui.QPen(c, 2, Qt.SolidLine, _ROUND_CAP))
            p.drawLine(self._CX, self._CY, ex, ey)

            p.setPen(Qt.NoPen)
            p.setBrush(c)
            p.drawEllipse(ex - 3, ey - 3, 6, 6)

            p.setPen(QtGui.QPen(c))
            lx = int(self._CX + sx * (self._AXIS_LEN + 9)) - 4
            ly = int(self._CY - sy * (self._AXIS_LEN + 9)) + 4
            p.drawText(lx, ly, label)

        p.end()


class SelectionOverlay(QWidget):
    region_selected = Signal(QtCore.QRect)

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self._pressing = False
        self._start = QtCore.QPoint()
        self._rect = QtCore.QRect()
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def set_active(self, enabled):
        self.setAttribute(Qt.WA_TransparentForMouseEvents, not enabled)
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.unsetCursor()
            self._pressing = False
            self._rect = QtCore.QRect()
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._start = event.pos()
            self._rect = QtCore.QRect(self._start, self._start)
            self._pressing = True
            self.update()

    def mouseMoveEvent(self, event):
        if self._pressing:
            self._rect = QtCore.QRect(self._start, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._pressing and event.button() == Qt.LeftButton:
            self._pressing = False
            r = QtCore.QRect(self._start, event.pos()).normalized()
            self._rect = QtCore.QRect()
            self.update()
            if r.width() > 2 and r.height() > 2:
                self.region_selected.emit(r)

    def paintEvent(self, event):
        if not self._rect.isValid():
            return
        p = QtGui.QPainter(self)
        p.fillRect(self._rect, QtGui.QColor(255, 255, 0, 40))
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 2))
        p.drawRect(self._rect.adjusted(0, 0, -1, -1))
        p.end()


_AXIS_LINE_COLORS = {'X': '#d94a4a', 'Y': '#4ab54a', 'Z': '#4a90d9'}

# Green background applied to a header button while its mode is active (Lock,
# Clip, coordinate lines/planes), as a quick visual cue that the mode is on.
_ACTIVE_BTN_QSS = (
    'QPushButton { background-color: #557a55; border: 1px solid #486848; '
    'border-radius: 4px; }'
    'QPushButton:hover { background-color: #608a60; }'
)


def _set_button_active(button, active: bool):
    """Toggle the green 'active mode' background on a header button."""
    button.setStyleSheet(_ACTIVE_BTN_QSS if active else '')


# Red background for the record button while slice-video record mode is armed.
_RECORD_BTN_QSS = (
    'QPushButton { background-color: #8e1c1c; border: 1px solid #6e1414; '
    'border-radius: 4px; }'
    'QPushButton:hover { background-color: #a02323; }'
)


def _lock_icon(locked: bool) -> QtGui.QIcon:
    """Monochrome padlock icon — drawn at 1× and 2× into 20×20 logical px."""
    icon = QtGui.QIcon()
    for dpr in (1.0, 2.0):
        # Physical canvas size for this DPR variant
        S_phys = round(20 * dpr)
        px = QtGui.QPixmap(S_phys, S_phys)
        px.setDevicePixelRatio(dpr)
        px.fill(Qt.transparent)

        p = QtGui.QPainter(px)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        # Map the 0–20 design space to physical pixels so the same
        # coordinate values produce equally-sized strokes at both DPRs.
        p.scale(S_phys / 20.0, S_phys / 20.0)

        fg  = QtGui.QColor('#cccccc')
        pen = QtGui.QPen(fg, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

        # Body
        p.setPen(pen)
        p.setBrush(fg)
        p.drawRoundedRect(QtCore.QRectF(2, 11, 16, 8), 2.5, 2.5)

        # Keyhole dot
        p.setPen(Qt.NoPen)
        p.setBrush(QtGui.QColor('#3a3a3a'))
        p.drawEllipse(QtCore.QRectF(8.5, 13, 3, 3))

        # Shackle: semicircle + arms
        # Arc rect (5,3,10,8) → centre (10,7), ends at (5,7) and (15,7), top at (10,3)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawArc(QtCore.QRectF(5, 3, 10, 8), 0, 180 * 16)
        p.drawLine(QtCore.QPointF(5, 7), QtCore.QPointF(5, 11.5))   # left arm → body
        if locked:
            p.drawLine(QtCore.QPointF(15, 7), QtCore.QPointF(15, 11.5))  # right → body
        else:
            p.drawLine(QtCore.QPointF(15, 7), QtCore.QPointF(15, 3))     # right → up (open)

        p.end()
        icon.addPixmap(px)
    return icon


def _measure_icon(kind: str, color: str = '#cccccc') -> QtGui.QIcon:
    """Monochrome symbolic icon for a measurement tool, drawn at 1× and 2×
    into a 20×20 logical-pixel canvas (matches the lock-icon style). ``color``
    defaults to the viewport-toolbar gray; the project tree passes a themed gray."""
    icon = QtGui.QIcon()
    for dpr in (1.0, 2.0):
        S_phys = round(20 * dpr)
        px = QtGui.QPixmap(S_phys, S_phys)
        px.setDevicePixelRatio(dpr)
        px.fill(Qt.transparent)

        p = QtGui.QPainter(px)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.scale(S_phys / 20.0, S_phys / 20.0)

        col = QtGui.QColor(color)
        pen = QtGui.QPen(col, 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)

        if kind == 'distance':
            # Diagonal ruler: a long body with tick marks along one edge.
            p.save()
            p.translate(10, 10)
            p.rotate(-45)
            p.drawRoundedRect(QtCore.QRectF(-9, -3, 18, 6), 1.0, 1.0)
            for i, x in enumerate((-6, -3, 0, 3, 6)):
                length = 3.0 if i % 2 == 0 else 2.0
                p.drawLine(QtCore.QLineF(x, -3, x, -3 + length))
            p.restore()
        elif kind == 'angle':
            # Two arms meeting at a vertex with an arc between them.
            p.drawLine(QtCore.QLineF(4, 15, 17.5, 15))   # horizontal arm
            p.drawLine(QtCore.QLineF(4, 15, 15, 4.5))    # diagonal arm
            p.drawArc(QtCore.QRectF(4 - 7, 15 - 7, 14, 14), 0, 42 * 16)
        elif kind == 'diameter':
            # Diameter sign: a circle with a diagonal slash.
            p.drawEllipse(QtCore.QRectF(4, 4, 12, 12))
            p.drawLine(QtCore.QLineF(4.5, 15.5, 15.5, 4.5))

        p.end()
        icon.addPixmap(px)
    return icon


_MEAS_COLOR = '#FFD400'          # yellow measurement colour
_MEAS_HOVER = '#FFF24D'


def _style_handles(roi):
    """Colour an ROI's grab handles to match the measurement colour."""
    for h in roi.getHandles():
        try:
            h.pen = pg.mkPen(_MEAS_COLOR, width=2)
            h.currentPen = h.pen
            h.update()
        except Exception:
            pass


class _OutlinedLabel(QtWidgets.QGraphicsObject):
    """Constant-size text anchored in view coords with a subtle dark outline,
    so the yellow measurement values stay readable over light gray values."""

    def __init__(self, color=_MEAS_COLOR, anchor=(0.5, 0.5), px=12, offset=(0.0, 0.0)):
        super().__init__()
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)
        self.setZValue(20)
        self._anchor = anchor
        self._offset = offset           # fixed screen-pixel offset (dx, dy down)
        self._brush = QtGui.QBrush(QtGui.QColor(color))
        self._outline = QtGui.QPen(QtGui.QColor(0, 0, 0, 170), 2.0)
        self._outline.setJoinStyle(Qt.RoundJoin)
        self._font = QtGui.QFont()
        self._font.setPixelSize(px)
        self._path = QtGui.QPainterPath()
        self._brect = QtCore.QRectF()

    def setText(self, text):
        path = QtGui.QPainterPath()
        fm = QtGui.QFontMetricsF(self._font)
        line_h = fm.height()
        for i, line in enumerate(str(text).split('\n')):
            sub = QtGui.QPainterPath()
            sub.addText(0, 0, self._font, line)
            sbr = sub.boundingRect()
            # centre each line horizontally and stack vertically
            sub.translate(-sbr.center().x(), i * line_h)
            path.addPath(sub)
        br = path.boundingRect()
        ax, ay = self._anchor
        path.translate(-br.left() - ax * br.width() + self._offset[0],
                       -br.top()  - ay * br.height() + self._offset[1])
        self.prepareGeometryChange()
        self._path = path
        self._brect = path.boundingRect().adjusted(-3, -3, 3, 3)
        self.update()

    def boundingRect(self):
        return self._brect

    def paint(self, p, option, widget=None):
        p.setRenderHint(_ANTIALIASING, True)
        p.setBrush(Qt.NoBrush)
        p.setPen(self._outline)
        p.drawPath(self._path)
        p.setPen(Qt.NoPen)
        p.setBrush(self._brush)
        p.drawPath(self._path)


class _BaseMeasurement:
    """A yellow, live, draggable measurement anchored in image (data) coords.

    Subclasses build a pyqtgraph ROI (the grab handles/ticks) plus a TextItem
    showing the live value; both live in view coordinates so they pan and zoom
    with the image automatically.
    """

    def __init__(self, viewer):
        self.viewer = viewer
        self.view = viewer.image_view.getView()        # PlotItem
        self.vb = self.view.getViewBox()
        self.roi = None
        self.label = None

    def _finish_setup(self):
        _style_handles(self.roi)
        self.view.addItem(self.roi, ignoreBounds=True)
        self.view.addItem(self.label, ignoreBounds=True)
        self.roi.sigRegionChanged.connect(self.update)
        self.update()

    def _handle_view_points(self):
        # Map each handle from ROI-local coords straight to the viewbox (data)
        # coordinate system via the ROI's own transform. This deliberately
        # avoids the scene→view round-trip (getSceneHandlePositions +
        # mapSceneToView), which depends on the live view/scene transform and
        # returns wrong (devicePixelRatio-scaled) values on Retina displays
        # before layout settles — that corrupted captured geometry, making a
        # measurement's value jump ~2× when it was restored.
        pts = []
        for _name, lp in self.roi.getLocalHandlePositions():
            pp = self.roi.mapToParent(lp)
            pts.append(np.array([pp.x(), pp.y()], dtype=float))
        return pts

    def update(self):
        pass

    def remove(self):
        for item in (self.roi, self.label):
            if item is not None:
                try:
                    self.view.removeItem(item)
                except Exception:
                    pass


class _DistanceMeasurement(_BaseMeasurement):
    def __init__(self, viewer, p1, p2):
        super().__init__(viewer)
        self.roi = pg.LineSegmentROI(
            [p1, p2], pen=pg.mkPen(_MEAS_COLOR, width=2),
            hoverPen=pg.mkPen(_MEAS_HOVER, width=3))
        self.label = _OutlinedLabel(anchor=(0.5, 1.0))
        self._finish_setup()

    def update(self):
        pts = self._handle_view_points()
        if len(pts) < 2:
            return
        a, b = pts[0], pts[1]
        sx, sy = self.viewer.plane_scales()
        d = float(np.hypot((b[0] - a[0]) * sx, (b[1] - a[1]) * sy))
        mid = (a + b) / 2.0
        self.label.setText(f'{d:.2f} mm')
        self.label.setPos(float(mid[0]), float(mid[1]))

    def geometry(self):
        pts = self._handle_view_points()
        if len(pts) < 2:
            return [[0.0, 0.0], [0.0, 0.0]]
        return [[float(pts[0][0]), float(pts[0][1])],
                [float(pts[1][0]), float(pts[1][1])]]


class _AngleMeasurement(_BaseMeasurement):
    def __init__(self, viewer, a, b, c):
        super().__init__(viewer)
        self.roi = pg.PolyLineROI([a, b, c], closed=False,
                                  pen=pg.mkPen(_MEAS_COLOR, width=2))
        # Keep it a strict 3-point angle: stop segment clicks from adding points.
        self.roi.segmentClicked = lambda *a, **k: None
        for seg in self.roi.segments:
            seg.setAcceptedMouseButtons(Qt.NoButton)
        self.label = _OutlinedLabel(anchor=(0.5, 0.0))
        self._finish_setup()

    def update(self):
        pts = self._handle_view_points()
        if len(pts) < 3:
            return
        a, b, c = pts[0], pts[1], pts[2]
        sx, sy = self.viewer.plane_scales()
        # Measure the true physical angle by scaling deltas to mm first.
        v1 = np.array([(a[0] - b[0]) * sx, (a[1] - b[1]) * sy])
        v2 = np.array([(c[0] - b[0]) * sx, (c[1] - b[1]) * sy])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        ang = float(np.degrees(np.arccos(cosang)))
        self.label.setText(f'{ang:.1f}°')
        self.label.setPos(float(b[0]), float(b[1]))

    def geometry(self):
        pts = self._handle_view_points()
        if len(pts) < 3:
            return [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        return [[float(p[0]), float(p[1])] for p in pts[:3]]


class _DiameterMeasurement(_BaseMeasurement):
    def __init__(self, viewer, cx, cy, r):
        super().__init__(viewer)
        self.roi = pg.CircleROI([cx - r, cy - r], [2 * r, 2 * r],
                                pen=pg.mkPen(_MEAS_COLOR, width=2))
        self.roi.addTranslateHandle([0.5, 0.5])   # centre handle to move it
        # Anchor below the centre point (+ a small gap) so the value clears
        # the centre grab handle.
        self.label = _OutlinedLabel(anchor=(0.5, 0.0), offset=(0.0, 10.0))
        self._finish_setup()

    def update(self):
        size = self.roi.size()
        pos = self.roi.pos()
        sx, sy = self.viewer.plane_scales()
        # Circle is round in voxel space; use the mean in-plane scale for mm.
        d = float(size[0]) * (sx + sy) / 2.0
        cx = float(pos[0]) + float(size[0]) / 2.0
        cy = float(pos[1]) + float(size[1]) / 2.0
        self.label.setText(f'Ø {d:.2f} mm')
        self.label.setPos(cx, cy)

    def geometry(self):
        size = self.roi.size()
        pos = self.roi.pos()
        r = float(size[0]) / 2.0
        return [float(pos[0]) + r, float(pos[1]) + r, r]


def _grayvalue_icon(kind: str, color: str = '#cccccc') -> QtGui.QIcon:
    """Monochrome symbolic icon for a gray-value tool (picker / profile).
    ``color`` defaults to the viewport-toolbar gray; the tree passes a themed gray."""
    icon = QtGui.QIcon()
    for dpr in (1.0, 2.0):
        S_phys = round(20 * dpr)
        px = QtGui.QPixmap(S_phys, S_phys)
        px.setDevicePixelRatio(dpr)
        px.fill(Qt.transparent)
        p = QtGui.QPainter(px)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.scale(S_phys / 20.0, S_phys / 20.0)
        pen = QtGui.QPen(QtGui.QColor(color), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        if kind == 'picker':
            # Target crosshair: cross with a gap and a centre circle.
            p.drawLine(QtCore.QLineF(10, 2, 10, 7))
            p.drawLine(QtCore.QLineF(10, 13, 10, 18))
            p.drawLine(QtCore.QLineF(2, 10, 7, 10))
            p.drawLine(QtCore.QLineF(13, 10, 18, 10))
            p.drawEllipse(QtCore.QRectF(7, 7, 6, 6))
        elif kind == 'profile':
            # A baseline with an intensity waveform above it.
            p.drawLine(QtCore.QLineF(2.5, 16, 17.5, 16))
            poly = QtGui.QPolygonF([QtCore.QPointF(*pt) for pt in
                                    ((2.5, 13), (6, 6), (9, 10), (12, 4), (15, 9), (17.5, 7))])
            p.drawPolyline(poly)
        p.end()
        icon.addPixmap(px)
    return icon


def _axes_tripod_icon(color: str = '#cccccc', width: float = 1.6) -> QtGui.QIcon:
    """Coordinate tripod — three lines (X/Y/Z axes) from a common origin. Used
    as the alignment symbol in the project tree and the Operations menu.

    Drawn as a single high-resolution pixmap (Qt down-scales it cleanly to any
    icon size). A multi-resolution / devicePixelRatio icon was tried first but
    rendered as a single line at larger sizes (e.g. in the menu)."""
    S = 64
    px = QtGui.QPixmap(S, S)
    px.fill(Qt.transparent)
    p = QtGui.QPainter(px)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.scale(S / 20.0, S / 20.0)
    pen = QtGui.QPen(QtGui.QColor(color), width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    ox, oy = 9.0, 11.0                              # common origin
    p.drawLine(QtCore.QLineF(ox, oy, ox, 2.5))      # Z — straight up
    p.drawLine(QtCore.QLineF(ox, oy, 18.0, oy))     # X — to the right
    p.drawLine(QtCore.QLineF(ox, oy, 2.5, 18.0))    # Y — diagonal, down-left
    p.end()
    return QtGui.QIcon(px)


def _clip_icon(state: str) -> QtGui.QIcon:
    """Scissors icon for the clipping-plane toggle.

    'off' → full scissors; 'left' → only the right half is drawn (left half of
    the scissors removed); 'right' → only the left half is drawn. The removed
    half mirrors the part of the volume being clipped away.
    """
    icon = QtGui.QIcon()
    for dpr in (1.0, 2.0):
        S = round(20 * dpr)
        px = QtGui.QPixmap(S, S)
        px.setDevicePixelRatio(dpr)
        px.fill(Qt.transparent)
        p = QtGui.QPainter(px)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.scale(S / 20.0, S / 20.0)
        pen = QtGui.QPen(QtGui.QColor('#cccccc'), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

        def scissors():
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QtCore.QRectF(4.0, 14.0, 5.0, 5.0))    # left handle ring
            p.drawEllipse(QtCore.QRectF(11.0, 14.0, 5.0, 5.0))   # right handle ring
            p.drawLine(QtCore.QLineF(6.5, 16.5, 15.5, 4.0))      # blade
            p.drawLine(QtCore.QLineF(13.5, 16.5, 4.5, 4.0))      # blade

        if state == 'left':       # remove the left half → draw only the right half
            p.setClipRect(QtCore.QRectF(10, 0, 10, 20))
        elif state == 'right':    # remove the right half → draw only the left half
            p.setClipRect(QtCore.QRectF(0, 0, 10, 20))
        scissors()
        p.end()
        icon.addPixmap(px)
    return icon


def _coord_axes_icon() -> QtGui.QIcon:
    """Coordinate-system icon: a horizontal + vertical pair plus a diagonal,
    all through the same point (toggles the 3D coordinate planes)."""
    icon = QtGui.QIcon()
    for dpr in (1.0, 2.0):
        S = round(20 * dpr)
        px = QtGui.QPixmap(S, S)
        px.setDevicePixelRatio(dpr)
        px.fill(Qt.transparent)
        p = QtGui.QPainter(px)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.scale(S / 20.0, S / 20.0)
        p.setPen(QtGui.QPen(QtGui.QColor('#cccccc'), 1.6, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(QtCore.QLineF(2.0, 10.0, 18.0, 10.0))     # horizontal axis
        p.drawLine(QtCore.QLineF(10.0, 2.0, 10.0, 18.0))     # vertical axis
        p.drawLine(QtCore.QLineF(4.7, 15.3, 15.3, 4.7))      # diagonal axis
        p.end()
        icon.addPixmap(px)
    return icon


def _draw_camera(p, x, w):
    """Draw a camera (body + viewfinder bump + lens) into a horizontal band of
    width ``w`` starting at ``x`` (20-px-tall logical canvas)."""
    s = w / 15.0                      # scale relative to the full-width camera
    cx = x + w / 2.0
    p.drawPolyline(QtGui.QPolygonF([
        QtCore.QPointF(cx - 3.0 * s, 6.0), QtCore.QPointF(cx - 1.7 * s, 3.8),
        QtCore.QPointF(cx + 1.7 * s, 3.8), QtCore.QPointF(cx + 3.0 * s, 6.0)]))
    p.drawRoundedRect(QtCore.QRectF(x, 6.0, w, 11.0), 2.0 * s, 2.0 * s)
    p.drawEllipse(QtCore.QRectF(cx - 2.7 * s, 8.3, 5.4 * s, 5.4 * s))


def _camera_icon(arrow=False) -> QtGui.QIcon:
    """Monochrome camera icon. With ``arrow=True`` a down-arrow is drawn to the
    right of a slightly smaller camera (the Save Image variant)."""
    icon = QtGui.QIcon()
    for dpr in (1.0, 2.0):
        S = round(20 * dpr)
        px = QtGui.QPixmap(S, S)
        px.setDevicePixelRatio(dpr)
        px.fill(Qt.transparent)
        p = QtGui.QPainter(px)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.scale(S / 20.0, S / 20.0)
        pen = QtGui.QPen(QtGui.QColor('#cccccc'), 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        if arrow:
            _draw_camera(p, 1.5, 11.0)                  # compact camera on the left
            p.drawLine(QtCore.QLineF(16.3, 5.5, 16.3, 12.0))                 # arrow shaft
            p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(13.9, 9.6),
                                            QtCore.QPointF(16.3, 12.6),
                                            QtCore.QPointF(18.7, 9.6)]))     # arrowhead
        else:
            _draw_camera(p, 2.5, 15.0)                  # full-width camera
        p.end()
        icon.addPixmap(px)
    return icon


def _record_icon(active=False) -> QtGui.QIcon:
    """A video-camera (movie) glyph: a rounded body with a lens triangle. When
    ``active`` the lens fills red to signal that recording mode is armed."""
    # Active state pairs with a dark-red button background, so the glyph goes
    # white (not red) to stay clearly visible on it.
    accent = '#ffffff' if active else '#cccccc'
    icon = QtGui.QIcon()
    for dpr in (1.0, 2.0):
        S = round(20 * dpr)
        px = QtGui.QPixmap(S, S)
        px.setDevicePixelRatio(dpr)
        px.fill(Qt.transparent)
        p = QtGui.QPainter(px)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.scale(S / 20.0, S / 20.0)
        pen = QtGui.QPen(QtGui.QColor(accent), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        # Camcorder body (rounded rectangle) on the left.
        p.drawRoundedRect(QtCore.QRectF(2.5, 6.0, 10.0, 8.0), 1.5, 1.5)
        # Lens/viewfinder spout pointing right.
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(13.0, 8.0),
                                       QtCore.QPointF(17.5, 6.0),
                                       QtCore.QPointF(17.5, 14.0),
                                       QtCore.QPointF(13.0, 12.0)]))
        # A small record dot inside the body.
        p.setBrush(QtGui.QColor(accent))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QtCore.QPointF(6.0, 10.0), 1.6, 1.6)
        p.end()
        icon.addPixmap(px)
    return icon


def _make_camera_button(on_copy, on_save):
    """A drop-down capture button: the main click runs the current action
    (default Copy to Clipboard); the menu switches between Copy to Clipboard and
    Save Image (and runs the chosen one). The button icon reflects the current
    mode — plain camera for copy, camera+down-arrow for save."""
    copy_icon = _camera_icon(arrow=False)
    save_icon = _camera_icon(arrow=True)
    btn = QtWidgets.QToolButton()
    btn.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
    btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
    btn.setIconSize(QtCore.QSize(16, 16))
    btn.setFixedHeight(24)
    state = {'mode': 'copy'}
    menu = QtWidgets.QMenu(btn)

    def set_mode(mode):
        state['mode'] = mode
        btn.setToolTip('Copy to Clipboard' if mode == 'copy' else 'Save Image')
        btn.setIcon(copy_icon if mode == 'copy' else save_icon)

    def run():
        (on_copy if state['mode'] == 'copy' else on_save)()

    act_copy = menu.addAction(copy_icon, 'Copy to Clipboard')
    act_save = menu.addAction(save_icon, 'Save Image')
    act_copy.triggered.connect(lambda: (set_mode('copy'), run()))
    act_save.triggered.connect(lambda: (set_mode('save'), run()))
    btn.setMenu(menu)
    btn.clicked.connect(run)
    set_mode('copy')
    return btn


class _GrayValuePicker:
    """Draggable yellow crosshair showing the X/Y/Z coords and gray value."""

    def __init__(self, viewer, x, y):
        self.viewer = viewer
        self.view = viewer.image_view.getView()
        self.target = pg.TargetItem(pos=(x, y), size=14, movable=True,
                                    pen=pg.mkPen(_MEAS_COLOR, width=2),
                                    hoverPen=pg.mkPen(_MEAS_HOVER, width=2))
        self.target.setZValue(15)
        self.label = _OutlinedLabel(anchor=(0.5, 1.0), offset=(0.0, -12.0))
        self.view.addItem(self.target, ignoreBounds=True)
        self.view.addItem(self.label, ignoreBounds=True)
        self.target.sigPositionChanged.connect(self.update)
        self.update()

    def update(self):
        pos = self.target.pos()
        res = self.viewer.gray_value_at(float(pos.x()), float(pos.y()))
        if res is None:
            return
        x, y, z, val = res
        self.label.setText(f'X {x}  Y {y}  Z {z}\nGV {self.viewer.gray_format(val)}')
        self.label.setPos(float(pos.x()), float(pos.y()))

    def geometry(self):
        pos = self.target.pos()
        return [float(pos.x()), float(pos.y())]

    def changed_signal(self):
        return self.target.sigPositionChanged

    def remove(self):
        for item in (self.target, self.label):
            try:
                self.view.removeItem(item)
            except Exception:
                pass


class _GrayValueProfile:
    """Yellow line with a live intensity profile plotted above it; the profile
    can be exported to CSV via right-click."""

    _SAMPLES = 200

    def __init__(self, viewer, p1, p2):
        self.viewer = viewer
        self.view = viewer.image_view.getView()
        self.vb = self.view.getViewBox()
        self.roi = pg.LineSegmentROI([p1, p2], pen=pg.mkPen(_MEAS_COLOR, width=2),
                                     hoverPen=pg.mkPen(_MEAS_HOVER, width=3))
        _style_handles(self.roi)
        self.curve = pg.PlotCurveItem(pen=pg.mkPen(_MEAS_COLOR, width=1.5))
        self.curve.setZValue(15)
        self.axis = pg.PlotCurveItem(pen=pg.mkPen(_MEAS_COLOR, width=1))
        self.axis.setZValue(15)
        # Six Y-axis tick value labels (min, max, and four in between).
        self._n_ticks = 6
        self.tick_labels = [_OutlinedLabel(anchor=(1.0, 0.5), px=10, offset=(-4.0, 0.0))
                            for _ in range(self._n_ticks)]
        self.view.addItem(self.roi, ignoreBounds=True)
        self.view.addItem(self.axis, ignoreBounds=True)
        self.view.addItem(self.curve, ignoreBounds=True)
        for lbl in self.tick_labels:
            self.view.addItem(lbl, ignoreBounds=True)

        # CSV export is offered from the Project Structure tree (right-click the
        # profile entry), not from the 2D viewport.
        self.roi.sigRegionChanged.connect(self.update)
        self._profile = None    # (distance_mm, values, px, py) for export
        self.update()

    def _endpoints(self):
        # Use the ROI's own transform (not the scene→view round-trip) so the
        # endpoints are independent of zoom/pan/devicePixelRatio/layout timing —
        # see _BaseMeasurement._handle_view_points for the Retina rationale.
        pts = []
        for _n, lp in self.roi.getLocalHandlePositions():
            pp = self.roi.mapToParent(lp)
            pts.append(np.array([pp.x(), pp.y()], dtype=float))
        return pts

    def geometry(self):
        pts = self._endpoints()
        if len(pts) < 2:
            return [[0.0, 0.0], [0.0, 0.0]]
        return [[float(pts[0][0]), float(pts[0][1])],
                [float(pts[1][0]), float(pts[1][1])]]

    def changed_signal(self):
        return self.roi.sigRegionChanged

    def update(self):
        pts = self._endpoints()
        if len(pts) < 2:
            return
        a, b = pts[0], pts[1]
        n = self._SAMPLES
        ts = np.linspace(0.0, 1.0, n)
        xs = a[0] + ts * (b[0] - a[0])
        ys = a[1] + ts * (b[1] - a[1])
        sampled = self.viewer.sample_gray_line(a, b, n)
        if sampled is None:
            return
        vals, px, py = sampled

        sx, sy = self.viewer.plane_scales()
        length_mm = float(np.hypot((b[0] - a[0]) * sx, (b[1] - a[1]) * sy))
        self._profile = (ts * length_mm, vals, px, py)

        d = b - a
        L = float(np.hypot(d[0], d[1]))
        if L < 1e-6:
            return
        u = d / L
        # Perpendicular pointing toward screen-up (the profile sits above the
        # line). The viewport may have an inverted Y axis, so consult it.
        nrm = np.array([-u[1], u[0]])
        up_is_plus_y = not self.vb.yInverted()
        if (nrm[1] < 0) == up_is_plus_y:
            nrm = -nrm

        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        norm = (vals - vmin) / max(vmax - vmin, 1e-9)
        # Plot height is a fixed fraction of the viewport height (in view units,
        # aspect-locked), so it stays ~25% of the viewport and does NOT grow
        # with the profile length.
        (_yr0, _yr1) = self.vb.viewRange()[1]
        vph = abs(_yr1 - _yr0)
        height = 0.25 * vph
        gap = 0.02 * vph
        base = np.stack([xs, ys], axis=1)
        plot_pts = base + nrm * (gap + norm[:, None] * height)
        self.curve.setData(plot_pts[:, 0], plot_pts[:, 1])

        # Y-axis (intensity) at the start with short outward tick marks; the
        # line itself is the X axis.
        y0 = a + nrm * gap
        y1 = a + nrm * (gap + height)
        ax_x = [y0[0], y1[0]]
        ax_y = [y0[1], y1[1]]
        tick_len = 0.10 * height
        fracs = np.linspace(0.0, 1.0, self._n_ticks)
        for i, fr in enumerate(fracs):
            pt = a + nrm * (gap + fr * height)
            tip = pt - u * tick_len
            ax_x += [np.nan, pt[0], tip[0]]
            ax_y += [np.nan, pt[1], tip[1]]
            val = vmin + fr * (vmax - vmin)
            self.tick_labels[i].setText(self.viewer.gray_format(val))
            self.tick_labels[i].setPos(float(tip[0]), float(tip[1]))
        self.axis.setData(ax_x, ax_y, connect='finite')

    def export_csv(self):
        if self._profile is None:
            QMessageBox.warning(self.viewer, 'Export Gray Value Profile',
                                'No profile samples available to export.')
            return
        dist_mm, vals, px, py = self._profile
        path, _ = QFileDialog.getSaveFileName(
            self.viewer, 'Export Gray Value Profile', '', 'CSV (*.csv)')
        if not path:
            return
        if not path.lower().endswith('.csv'):
            path += '.csv'
        try:
            import csv
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['index', 'distance_mm', 'gray_value'])
                for i, (dmm, v) in enumerate(zip(dist_mm, vals)):
                    w.writerow([i, f'{dmm:.4f}', f'{float(v):.6g}'])
        except OSError as exc:
            QMessageBox.critical(self.viewer, 'Export Failed', str(exc))

    def remove(self):
        for item in [self.roi, self.curve, self.axis] + list(self.tick_labels):
            try:
                self.view.removeItem(item)
            except Exception:
                pass


class SliceViewer(QWidget):
    point_placed = Signal(tuple)
    region_selected = Signal(float, float)
    axis_position_changed = Signal(str, int, int)  # (axis, index, total)
    lock_clicked = Signal()
    measurement_tool_changed = Signal(str)         # user picked a tool from the menu
    measurement_created = Signal(object)           # a new measurement graphic was made
    gray_value_created = Signal(object)            # a new gray-value graphic was made
    clip_changed = Signal()                        # clipping-plane toggle changed
    planes_toggled = Signal(bool)                  # coordinate-planes visibility toggle
    record_mode_changed = Signal(bool)             # video-record mode entered/left
    record_range_changed = Signal(int, int, int)   # (low, high, total) for the record axis

    _CLIP_STATES = ('off', 'left', 'right')
    _CLIP_TOOLTIP = {'off': 'Clipping plane: off',
                     'left': 'Clipping plane: hide below this slice in 3D',
                     'right': 'Clipping plane: hide above this slice in 3D'}

    _MEAS_LABELS = {'distance': 'Distance', 'angle': 'Angle', 'diameter': 'Diameter'}
    _GRAY_LABELS = {'picker': 'Gray Value Picker', 'profile': 'Gray Value Profile'}

    # Longest in-plane dimension sampled for interactive (live preview) slices.
    # The full-resolution slice is restored once the transform is applied.
    _PREVIEW_MAX_DIM = 384

    def __init__(self, orientation='XY', parent=None):
        super().__init__(parent)
        self.orientation = orientation
        self.volume_data = None
        self.current_index = 0
        self._measurements = []
        self._measure_tool = 'distance'
        self._gray_items = []
        self._gray_tool = 'picker'
        self._surf_tri = None          # (M,3,3) surface triangle verts (display frame)
        self._surf_shape = None        # display-frame dims for the plot flip
        self._surf_tri_min = None      # (M,3) per-triangle min along each axis (cache)
        self._surf_tri_max = None      # (M,3) per-triangle max along each axis (cache)
        self._surf_outline_idx = None  # slice index the outline was last computed for
        self._surf_outline_item = None # pg.PlotDataItem of the slice ∩ surface
        self._clip_state = 'off'
        self._display_levels = None
        self._auto_range_pending = True
        self._preview_R      = None   # 3×3 rotation for live Simple Alignment preview
        self._preview_offset = None   # (3,) offset
        self._preview_vol    = None   # original volume (not resampled)
        self._perm_R      = None      # permanent non-destructive alignment transform
        self._perm_offset = None
        self._perm_shape  = None      # output bounding-box shape for the permanent transform
        self._aligned_vol = None      # full pre-resampled aligned volume (fast slicing)
        self._axis_lines  = {}   # axis -> pg.InfiniteLine
        self._axis_timers = {}   # axis -> QTimer
        self._lines_pinned = False
        self._coord_lines_enabled = True
        self._pip = None   # picture-in-picture 3D preview, shown when maximized
        self._interpolate = False   # smooth (trilinear) vs nearest-neighbour voxels
        self._record_mode = False   # slice-video recording range mode
        self._recording = False     # actively grabbing/encoding frames
        self._record_lines = []     # InfiniteLines marking another viewport's record range
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header_bar = QWidget()
        header_bar.setFixedHeight(28)
        header = QHBoxLayout(header_bar)
        header.setContentsMargins(2, 2, 2, 2)
        self.lock_button = QPushButton()
        self.lock_button.setIcon(_lock_icon(False))
        self.lock_button.setFixedSize(24, 24)
        self.lock_button.setIconSize(QtCore.QSize(16, 16))
        self.lock_button.setToolTip('Lock viewport zoom/pan in sync with others')
        self.lock_button.clicked.connect(self.lock_clicked)
        header.addWidget(self.lock_button)

        self.measure_button = QtWidgets.QToolButton()
        self.measure_button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.measure_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.measure_button.setIconSize(QtCore.QSize(16, 16))
        self.measure_button.setFixedHeight(24)
        self._meas_icons = {k: _measure_icon(k) for k in ('distance', 'angle', 'diameter')}
        meas_menu = QtWidgets.QMenu(self.measure_button)
        meas_menu.setToolTipsVisible(True)
        for kind in ('distance', 'angle', 'diameter'):
            # Icon + text in the dropdown list; the collapsed button stays
            # icon-only (ToolButtonIconOnly) to keep the header compact.
            act = meas_menu.addAction(self._meas_icons[kind], self._MEAS_LABELS[kind])
            act.setToolTip(self._MEAS_LABELS[kind])
            act.triggered.connect(lambda _checked=False, k=kind: self._on_measure_menu(k))
        self.measure_button.setMenu(meas_menu)
        self.measure_button.clicked.connect(lambda: self._create_measurement(self._measure_tool))
        self.set_measurement_tool(self._measure_tool)
        header.addWidget(self.measure_button)

        self.gray_button = QtWidgets.QToolButton()
        self.gray_button.setPopupMode(QtWidgets.QToolButton.MenuButtonPopup)
        self.gray_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.gray_button.setIconSize(QtCore.QSize(16, 16))
        self.gray_button.setFixedHeight(24)
        self._gray_icons = {k: _grayvalue_icon(k) for k in ('picker', 'profile')}
        gray_menu = QtWidgets.QMenu(self.gray_button)
        gray_menu.setToolTipsVisible(True)
        for kind in ('picker', 'profile'):
            act = gray_menu.addAction(self._gray_icons[kind], self._GRAY_LABELS[kind])
            act.setToolTip(self._GRAY_LABELS[kind])
            act.triggered.connect(lambda _checked=False, k=kind: self._on_gray_menu(k))
        self.gray_button.setMenu(gray_menu)
        self.gray_button.clicked.connect(lambda: self._create_gray_value(self._gray_tool))
        self.set_gray_tool(self._gray_tool)
        header.addWidget(self.gray_button)

        self._clip_icons = {s: _clip_icon(s) for s in self._CLIP_STATES}
        self.clip_button = QPushButton()
        self.clip_button.setFixedSize(24, 24)
        self.clip_button.setIconSize(QtCore.QSize(16, 16))
        self.clip_button.clicked.connect(self._cycle_clip)
        self._update_clip_button()
        header.addWidget(self.clip_button)

        self.planes_button = QPushButton()
        self.planes_button.setCheckable(True)
        self.planes_button.setChecked(True)
        self.planes_button.setFixedSize(24, 24)
        self.planes_button.setIconSize(QtCore.QSize(16, 16))
        self.planes_button.setIcon(_coord_axes_icon())
        self.planes_button.setToolTip('Show coordinate lines in the 2D viewports')
        self.planes_button.toggled.connect(self.planes_toggled)
        header.addWidget(self.planes_button)

        self.camera_button = _make_camera_button(self._camera_copy, self._camera_save)
        header.addWidget(self.camera_button)

        self.record_button = QPushButton()
        self.record_button.setIcon(_record_icon(False))
        self.record_button.setFixedSize(24, 24)
        self.record_button.setIconSize(QtCore.QSize(16, 16))
        self.record_button.setToolTip('Record a slice-range video of this viewport')
        self.record_button.clicked.connect(self.toggle_record_mode)
        header.addWidget(self.record_button)

        header.addStretch()
        self.maximize_button = QPushButton('▲')
        self.maximize_button.setFixedSize(24, 24)
        header.addWidget(self.maximize_button)
        layout.addWidget(header_bar)

        self.image_view = pg.ImageView(view=pg.PlotItem())
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        self.image_view.getImageItem().mouseClickEvent = self.on_image_click
        layout.addWidget(self.image_view, 1)

        self.tripod = TripodWidget(self.orientation, parent=self.image_view)

        self.auto_hint = QLabel('Auto Min Max is on. Press Esc to exit.', self.image_view)
        self.auto_hint.setStyleSheet(
            'color: #FFD400; background: rgba(0,0,0,120);'
            'padding: 2px 6px; border-radius: 3px; font-size: 11px;')
        self.auto_hint.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.auto_hint.adjustSize()
        self.auto_hint.hide()

        self.image_view.installEventFilter(self)

        gv = self.image_view.ui.graphicsView
        self.selection_overlay = SelectionOverlay(gv.viewport())
        self.selection_overlay.resize(gv.viewport().size())
        self.selection_overlay.region_selected.connect(self._on_region_selected)
        gv.viewport().installEventFilter(self)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        _axis_color = {'XY': '#4a90d9', 'YZ': '#d94a4a', 'XZ': '#4ab54a'}.get(self.orientation, '#888888')
        self._axis_color = _axis_color
        self.slice_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 4px;
                background: #555;
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {_axis_color};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {_axis_color};
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
        """)
        # Distance-unit readout of the slice position, fully synced with the
        # slider (XY → Z, YZ → X, XZ → Y, in mm). Range mirrors the slider's
        # range mapped to mm; step is 1/100th of that range.
        self.slice_spin = QtWidgets.QDoubleSpinBox()
        self.slice_spin.setDecimals(3)
        self.slice_spin.setRange(0.0, 0.0)
        self.slice_spin.setSuffix(' mm')
        self.slice_spin.setFixedWidth(96)
        self.slice_spin.setKeyboardTracking(False)
        self.slice_spin.setToolTip('Slice position in distance units')
        self.slice_spin.valueChanged.connect(self.on_slice_spin_changed)

        # ── Record-mode widgets (hidden until the record button is pressed) ──
        self.record_start_btn = QPushButton('Start')
        self.record_start_btn.setFixedHeight(22)
        self.record_start_btn.setToolTip('Render the slice range to a video file')
        self.record_start_btn.clicked.connect(self._start_recording)
        self.record_start_btn.hide()

        self.record_len_spin = QtWidgets.QDoubleSpinBox()
        self.record_len_spin.setDecimals(1)
        self.record_len_spin.setRange(0.5, 600.0)
        self.record_len_spin.setValue(10.0)
        self.record_len_spin.setSingleStep(1.0)
        self.record_len_spin.setSuffix(' s')
        self.record_len_spin.setFixedWidth(72)
        self.record_len_spin.setKeyboardTracking(False)
        self.record_len_spin.setToolTip('Video length in seconds (30 FPS)')
        self.record_len_spin.hide()

        self.record_slider = RangeSlider()
        self.record_slider.setSpanColor(self._axis_color)
        self.record_slider.valueChanged.connect(self._on_record_range_changed)
        self.record_slider.hide()

        self.record_progress = QtWidgets.QProgressBar()
        self.record_progress.setFixedHeight(18)
        self.record_progress.setTextVisible(True)
        self.record_progress.hide()

        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(28)
        bottom = QHBoxLayout(bottom_bar)
        bottom.setContentsMargins(2, 2, 2, 2)
        bottom.addWidget(self.record_start_btn)
        bottom.addWidget(self.record_len_spin)
        bottom.addWidget(self.slice_slider)
        bottom.addWidget(self.record_slider)
        bottom.addWidget(self.record_progress)
        bottom.addWidget(self.slice_spin)
        layout.addWidget(bottom_bar)

    def eventFilter(self, obj, event):
        gv = self.image_view.ui.graphicsView
        if obj is self.image_view and event.type() == _EV_RESIZE:
            self._reposition_tripod()
            self._reposition_auto_hint()
            self._reposition_pip()
            self.selection_overlay.resize(gv.viewport().size())
        elif obj is gv.viewport():
            et = event.type()
            if et == _EV_WHEEL:
                if not (event.modifiers() & Qt.ControlModifier):
                    return True  # eat non-Ctrl wheel; zoom requires Ctrl
                return False     # Ctrl held: let pyqtgraph ViewBox zoom normally
            if et == _EV_NATIVE_GESTURE and _ZOOM_GESTURE_TYPE is not None:
                try:
                    if event.gestureType() == _ZOOM_GESTURE_TYPE:
                        factor = 1.0 + event.value()
                        if factor > 0:
                            vb = self.image_view.getView().getViewBox()
                            vb.scaleBy((1.0 / factor, 1.0 / factor))
                        return True
                except Exception:
                    pass
        return super().eventFilter(obj, event)

    def _reposition_tripod(self):
        m = 6
        self.tripod.move(m, self.image_view.height() - self.tripod.height() - m)
        self.tripod.raise_()

    # ── Picture-in-picture 3D preview (shown only while maximized) ────────────
    def attach_pip(self, pip):
        """Overlay a 3D preview widget in the bottom-right corner."""
        self._pip = pip
        pip.setParent(self.image_view)
        pip.show()
        self._reposition_pip()

    def detach_pip(self):
        if self._pip is not None:
            self._pip.hide()
            self._pip.setParent(None)
            self._pip = None

    def _reposition_pip(self):
        if self._pip is None:
            return
        iv = self.image_view
        # 1/5 of the viewport, 30% larger; lifted off the bottom so it clears the
        # coordinate scale / tick labels.
        w = max(80, int(iv.width()  * 0.20 * 1.3))
        h = max(80, int(iv.height() * 0.20 * 1.3))
        right_m, bottom_m = 6, 44
        self._pip.resize(w, h)
        self._pip.move(iv.width() - w - right_m, iv.height() - h - bottom_m)
        self._pip.raise_()

    def _reposition_auto_hint(self):
        m = 6
        self.auto_hint.adjustSize()
        self.auto_hint.move(self.image_view.width() - self.auto_hint.width() - m, m)
        self.auto_hint.raise_()

    def set_levels(self, min_val, max_val):
        self._display_levels = (float(min_val), float(max_val))
        # Update the display LUT directly instead of re-slicing the image — the
        # pixels don't change, only the black/white mapping, so this is real-time.
        if self.volume_data is not None and self.volume_data.volume is not None:
            try:
                self.image_view.getImageItem().setLevels((float(min_val), float(max_val)))
            except Exception:
                pass

    def set_interpolation(self, enabled):
        """Smoothly interpolate the displayed slice (bilinear in-plane) instead
        of drawing blocky voxels. It's just a render hint on the graphics view —
        Qt does the scaling, so there's essentially no performance cost."""
        self._interpolate = bool(enabled)
        gv = self.image_view.ui.graphicsView
        gv.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, self._interpolate)
        gv.viewport().update()

    def set_locked(self, locked):
        self.lock_button.setIcon(_lock_icon(locked))
        _set_button_active(self.lock_button, locked)

    def set_auto_mode(self, enabled):
        self.selection_overlay.set_active(enabled)
        self.auto_hint.setVisible(enabled)
        if enabled:
            self._reposition_auto_hint()

    # ── Measurements ──────────────────────────────────────────────────────
    def plane_scales(self):
        """mm-per-voxel for this viewport's (horizontal, vertical) plot axes.

        Plot-x / plot-y map to volume axes per orientation:
          XY → (X, Y),  YZ → (Y, Z),  XZ → (X, Z)
        """
        vs = (1.0, 1.0, 1.0)
        if self.volume_data is not None and getattr(self.volume_data, 'voxel_size', None):
            vs = self.volume_data.voxel_size
        vx, vy, vz = (float(vs[0]), float(vs[1]), float(vs[2]))
        return {
            'XY': (vx, vy),
            'YZ': (vy, vz),
            'XZ': (vx, vz),
        }.get(self.orientation, (vx, vy))

    def set_measurement_tool(self, kind):
        """Set the current tool shown on the dropdown button (no creation)."""
        if kind not in self._MEAS_LABELS:
            return
        self._measure_tool = kind
        self.measure_button.setIcon(self._meas_icons[kind])
        self.measure_button.setToolTip(self._MEAS_LABELS[kind])

    def _on_measure_menu(self, kind):
        self.set_measurement_tool(kind)
        self.measurement_tool_changed.emit(kind)
        self._create_measurement(kind)

    def _create_measurement(self, kind):
        if self.volume_data is None or self.volume_data.volume is None:
            return
        (x0, x1), (y0, y1) = self.image_view.getView().getViewBox().viewRange()
        w, h = x1 - x0, y1 - y0
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0 + 0.20 * h        # 20% above centre, towards the top
        if kind == 'distance':
            half = 0.125 * w                   # line ≈ 25% of the viewport
            m = _DistanceMeasurement(self, [cx - half, cy], [cx + half, cy])
        elif kind == 'angle':
            ax, ay = 0.18 * w, 0.16 * h
            m = _AngleMeasurement(self, [cx - ax, cy + ay], [cx, cy], [cx + ax, cy + ay])
        elif kind == 'diameter':
            m = _DiameterMeasurement(self, cx, cy, 0.125 * w)
        else:
            return
        m.kind = kind
        m._record = None
        self._measurements.append(m)
        # The owning MainWindow registers a persistent record for this graphic.
        self.measurement_created.emit(m)

    def build_measurement(self, kind, points):
        """Recreate a measurement graphic from stored geometry (used when a
        persistent measurement scrolls back into its slice). Returns the graphic
        or None. Does NOT emit measurement_created."""
        if self.volume_data is None or self.volume_data.volume is None:
            return None
        if kind == 'distance':
            p1, p2 = points
            m = _DistanceMeasurement(self, list(p1), list(p2))
        elif kind == 'angle':
            a, b, c = points
            m = _AngleMeasurement(self, list(a), list(b), list(c))
        elif kind == 'diameter':
            cx, cy, r = points
            m = _DiameterMeasurement(self, float(cx), float(cy), float(r))
        else:
            return None
        m.kind = kind
        m._record = None
        self._measurements.append(m)
        return m

    def clear_measurements(self):
        """Remove the currently-displayed measurement graphics from this view.
        Persistent records (owned by MainWindow) are detached, not deleted."""
        for m in self._measurements:
            m.remove()
            rec = getattr(m, '_record', None)
            if rec is not None:
                rec['graphic'] = None
        self._measurements = []

    # ── Gray-value tools ──────────────────────────────────────────────────
    def set_gray_tool(self, kind):
        if kind not in self._GRAY_LABELS:
            return
        self._gray_tool = kind
        self.gray_button.setIcon(self._gray_icons[kind])
        self.gray_button.setToolTip(self._GRAY_LABELS[kind])

    def _on_gray_menu(self, kind):
        self.set_gray_tool(kind)
        self._create_gray_value(kind)

    def _create_gray_value(self, kind):
        if self.volume_data is None or self.volume_data.volume is None:
            return
        (x0, x1), (y0, y1) = self.image_view.getView().getViewBox().viewRange()
        w, h = x1 - x0, y1 - y0
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0 + 0.20 * h
        if kind == 'picker':
            obj = _GrayValuePicker(self, cx, cy)
        elif kind == 'profile':
            half = 0.125 * w
            obj = _GrayValueProfile(self, [cx - half, cy], [cx + half, cy])
        else:
            return
        obj.kind = kind
        obj._record = None
        self._gray_items.append(obj)
        self.gray_value_created.emit(obj)

    def build_gray_value(self, kind, points):
        """Recreate a gray-value graphic from stored geometry. Returns the
        object or None. Does NOT emit gray_value_created."""
        if self.volume_data is None or self.volume_data.volume is None:
            return None
        if kind == 'picker':
            x, y = points
            obj = _GrayValuePicker(self, float(x), float(y))
        elif kind == 'profile':
            p1, p2 = points
            obj = _GrayValueProfile(self, list(p1), list(p2))
        else:
            return None
        obj.kind = kind
        obj._record = None
        self._gray_items.append(obj)
        return obj

    def clear_gray_values(self):
        """Remove displayed gray-value graphics; detach persistent records."""
        for g in self._gray_items:
            g.remove()
            rec = getattr(g, '_record', None)
            if rec is not None:
                rec['graphic'] = None
        self._gray_items = []

    # ── Clipping plane ────────────────────────────────────────────────────
    def _update_clip_button(self):
        self.clip_button.setIcon(self._clip_icons[self._clip_state])
        self.clip_button.setToolTip(self._CLIP_TOOLTIP[self._clip_state])
        _set_button_active(self.clip_button, self._clip_state != 'off')

    def _cycle_clip(self):
        i = self._CLIP_STATES.index(self._clip_state)
        self._clip_state = self._CLIP_STATES[(i + 1) % len(self._CLIP_STATES)]
        self._update_clip_button()
        self.clip_changed.emit()

    def clip_info(self):
        """Return (axis_index, mode, fraction) for the active clip, or None.

        fraction is the slice position along this viewer's traversal axis in
        [0, 1], so the 3D view can map it onto its own (downsampled) volume.
        """
        if self._clip_state == 'off':
            return None
        shape = self._active_shape()
        if shape is None:
            return None
        ax = self._traversal_axis()
        depth = shape[ax]
        frac = self.current_index / max(depth - 1, 1)
        return ax, self._clip_state, float(frac)

    def slice_fraction(self):
        """(traversal axis, slice fraction in [0,1]) for the coordinate plane."""
        shape = self._active_shape()
        if shape is None:
            return None
        ax = self._traversal_axis()
        return ax, float(self.current_index / max(shape[ax] - 1, 1))

    def _displayed_pixel(self, vx, vy):
        """Map a view-coord point to (image_array, px, py) of the displayed slice."""
        item = self.image_view.getImageItem()
        image = item.image
        if image is None:
            return None
        ip = item.mapFromView(QtCore.QPointF(vx, vy))
        px = int(np.clip(int(np.floor(ip.x())), 0, image.shape[0] - 1))
        py = int(np.clip(int(np.floor(ip.y())), 0, image.shape[1] - 1))
        return image, px, py

    def gray_format(self, value):
        """Format a gray value as a plain integer for integer datasets, else %.4g."""
        dt = self.volume_data.dtype if self.volume_data is not None else None
        if dt is not None and np.issubdtype(dt, np.integer):
            return f'{int(round(float(value)))}'
        return f'{float(value):.4g}'

    def gray_value_at(self, vx, vy):
        """Return (X, Y, Z, value) for a view-coord point, sampling the displayed
        slice and mapping the pixel back to volume coordinates (display flips
        included)."""
        if self.volume_data is None or self.volume_data.volume is None:
            return None
        got = self._displayed_pixel(vx, vy)
        if got is None:
            return None
        image, px, py = got
        h = image.shape[1]
        idx = self.current_index
        if self.orientation == 'XY':
            x, y, z = px, h - 1 - py, idx
        elif self.orientation == 'YZ':
            x, y, z = idx, px, h - 1 - py
        elif self.orientation == 'XZ':
            x, y, z = px, idx, h - 1 - py
        else:
            x, y, z = px, py, idx
        return int(x), int(y), int(z), float(image[px, py])

    def sample_gray_line(self, a_view, b_view, n):
        """Sample n gray values along a view-coord line from the displayed slice.
        Returns (values, px, py) in image-pixel coordinates, or None."""
        item = self.image_view.getImageItem()
        image = item.image
        if image is None:
            return None
        pa = item.mapFromView(QtCore.QPointF(float(a_view[0]), float(a_view[1])))
        pb = item.mapFromView(QtCore.QPointF(float(b_view[0]), float(b_view[1])))
        px = np.linspace(pa.x(), pb.x(), n)
        py = np.linspace(pa.y(), pb.y(), n)
        try:
            from scipy.ndimage import map_coordinates
            vals = map_coordinates(image, np.vstack([px, py]), order=1, mode='nearest')
        except Exception:
            xi = np.clip(px.astype(int), 0, image.shape[0] - 1)
            yi = np.clip(py.astype(int), 0, image.shape[1] - 1)
            vals = image[xi, yi]
        return np.asarray(vals, dtype=float), px, py

    def _on_region_selected(self, rect):
        if self.volume_data is None or self.volume_data.volume is None:
            return
        # Sample the *displayed* slice so the rectangle always lands on the
        # pixels actually on screen — raw or alignment-transformed. Mapping the
        # screen rect through the image item's own transform (mapFromScene)
        # accounts for any setRect() stretching applied by update_image().
        item = self.image_view.getImageItem()
        image = item.image
        if image is None:
            return

        gv = self.image_view.ui.graphicsView

        def to_img(qp):
            ip = item.mapFromScene(gv.mapToScene(qp))
            return ip.x(), ip.y()

        x1f, y1f = to_img(rect.topLeft())
        x2f, y2f = to_img(rect.bottomRight())

        x1 = int(np.clip(min(x1f, x2f), 0, image.shape[0]))
        x2 = int(np.clip(max(x1f, x2f), 0, image.shape[0]))
        y1 = int(np.clip(min(y1f, y2f), 0, image.shape[1]))
        y2 = int(np.clip(max(y1f, y2f), 0, image.shape[1]))

        if x2 <= x1 or y2 <= y1:
            return

        region = image[x1:x2, y1:y2]
        if region.size == 0:
            return

        min_val = float(np.min(region))
        max_val = float(np.max(region))
        if min_val >= max_val:
            max_val = min_val + 1.0

        self.region_selected.emit(min_val, max_val)

    def _traversal_axis(self):
        """Index of the volume axis this viewer scrolls through."""
        return {'XY': 2, 'YZ': 0, 'XZ': 1}.get(self.orientation, 2)

    def _active_shape(self):
        """Shape of the frame currently displayed (aligned output if a permanent
        transform is active, otherwise the raw volume)."""
        if self._perm_shape is not None:
            return self._perm_shape
        if self.volume_data is not None and self.volume_data.volume is not None:
            return self.volume_data.volume.shape
        return None

    def _traversal_spacing(self):
        """mm-per-slice along this viewport's traversal axis (XY→Z, YZ→X, XZ→Y)."""
        vs = (1.0, 1.0, 1.0)
        if self.volume_data is not None and getattr(self.volume_data, 'voxel_size', None):
            vs = self.volume_data.voxel_size
        s = float(vs[self._traversal_axis()])
        return s if s > 0 else 1.0

    def _update_slice_spin_range(self):
        """Mirror the slider's range onto the distance spinbox (in mm)."""
        sp = self._traversal_spacing()
        maximum = max(0, self.slice_slider.maximum()) * sp
        self.slice_spin.blockSignals(True)
        self.slice_spin.setRange(0.0, maximum)
        self.slice_spin.setSingleStep(maximum / 100.0 if maximum > 0 else 1.0)
        self.slice_spin.blockSignals(False)

    def _sync_slice_spin(self):
        """Reflect the current slice index on the distance spinbox."""
        self.slice_spin.blockSignals(True)
        self.slice_spin.setValue(self.current_index * self._traversal_spacing())
        self.slice_spin.blockSignals(False)

    def _apply_slider_range(self, recenter=True):
        """Set the slider maximum from the active frame's traversal depth."""
        shape = self._active_shape()
        if shape is None:
            self.slice_slider.setMaximum(0)
            self._update_slice_spin_range()
            self._sync_slice_spin()
            return
        depth = shape[self._traversal_axis()]
        if recenter:
            self.current_index = depth // 2
        else:
            self.current_index = int(np.clip(self.current_index, 0, depth - 1))
        self.slice_slider.setMaximum(depth - 1)
        self.slice_slider.setValue(self.current_index)
        self._update_slice_spin_range()
        self._sync_slice_spin()

    def set_volume(self, volume_data):
        if self._record_mode:
            self.set_record_mode(False)
        self.clear_record_lines()
        self.volume_data = volume_data
        self._display_levels = None
        self._auto_range_pending = True
        self._perm_R = None
        self._perm_offset = None
        self._perm_shape = None
        self._aligned_vol = None
        self.clear_measurements()
        self.clear_gray_values()
        if self._clip_state != 'off':
            self._clip_state = 'off'
            self._update_clip_button()
        if volume_data is None or not volume_data.is_loaded():
            self.image_view.clear()
            self.slice_slider.setMaximum(0)
            self._update_slice_spin_range()
            self._sync_slice_spin()
            return
        self._apply_slider_range(recenter=True)
        self.update_image()

    def on_slice_changed(self, value):
        self.current_index = value
        self._sync_slice_spin()
        self.clear_measurements()
        self.clear_gray_values()
        self.update_image()
        shape = self._active_shape()
        if shape is not None:
            ax_idx = self._traversal_axis()
            axis   = {'XY': 'Z', 'YZ': 'X', 'XZ': 'Y'}[self.orientation]
            self.axis_position_changed.emit(axis, value, shape[ax_idx])

    def on_slice_spin_changed(self, value):
        """Map an edited distance value back to the nearest slice index."""
        sp = self._traversal_spacing()
        idx = int(round(value / sp)) if sp > 0 else 0
        if self._record_mode:
            # In record mode the spin drives the last-touched range handle.
            idx = int(np.clip(idx, 0, max(0, self.slice_slider.maximum())))
            if self.record_slider.last_handle() == 'low':
                self.record_slider.setLow(idx)
            else:
                self.record_slider.setHigh(idx)
            return
        idx = int(np.clip(idx, 0, max(0, self.slice_slider.maximum())))
        if idx != self.slice_slider.value():
            self.slice_slider.setValue(idx)   # → on_slice_changed re-syncs the spin
        else:
            self._sync_slice_spin()           # snap display back onto the slice grid

    def set_preview_transform(self, R, offset, orig_vol):
        """Activate live preview: update_image() will sample orig_vol via R/offset."""
        self._preview_R      = R
        self._preview_offset = offset
        self._preview_vol    = orig_vol
        self.clear_measurements()
        self.clear_gray_values()
        self.update_image()

    def clear_preview_transform(self):
        """Return to normal volume display."""
        self._preview_R      = None
        self._preview_offset = None
        self._preview_vol    = None
        self.clear_measurements()
        self.clear_gray_values()
        self.update_image()

    def set_permanent_transform(self, R, offset, out_shape, aligned_vol=None):
        """Set a persistent non-destructive alignment transform (survives preview cycles).

        ``out_shape`` is the bounding-box shape of the aligned output frame; the
        viewer samples / scrolls over this frame rather than the raw volume.

        ``aligned_vol`` (optional) is the full volume already resampled into the
        output frame. When provided, update_image() slices it directly — as fast
        as the raw path — instead of resampling each slice on demand. It is shared
        (read-only) across the three viewers, so panning/scrolling stays smooth
        after an alignment. When None (e.g. a very large frame), the viewer falls
        back to per-slice sampling.
        """
        self._perm_R      = np.array(R,      dtype=np.float64)
        self._perm_offset = np.array(offset, dtype=np.float64)
        self._perm_shape  = tuple(int(s) for s in out_shape)
        self._aligned_vol = aligned_vol
        self.clear_measurements()
        self.clear_gray_values()
        self._auto_range_pending = True
        self._apply_slider_range(recenter=True)
        self.update_image()

    def clear_permanent_transform(self):
        self._perm_R      = None
        self._perm_offset = None
        self._perm_shape  = None
        self._aligned_vol = None
        self.clear_measurements()
        self.clear_gray_values()
        self._auto_range_pending = True
        self._apply_slider_range(recenter=True)
        self.update_image()

    def _sample_transformed_slice(self, R, off, vol, out_shape, stride=1):
        """Return 2D float32 slice by mapping output coords through R/off into vol.

        The slice is taken from the aligned *output* frame of shape ``out_shape``
        (so the full rotated volume is visible), at ``self.current_index`` along
        this viewer's traversal axis. Only the one visible slice is resampled, so
        this stays cheap even for large volumes.

        ``stride`` (>1) samples a coarse grid for fast interactive previews; the
        caller stretches the result back to the full extent. Sampling cost scales
        with the number of output points, so a coarse grid is dramatically faster
        on large volumes (the full grid thrashes cache on multi-GB arrays)."""
        try:
            from scipy.ndimage import map_coordinates
        except ImportError:
            return None
        s   = out_shape
        idx = self.current_index
        st  = max(1, int(stride))
        ax  = lambda n: np.arange(0, n, st, dtype=np.float64)  # coarse grid when st > 1
        if self.orientation == 'XY':
            ii = ax(s[0])
            jj = ax(s[1])
            xi, yj = np.meshgrid(ii, jj, indexing='ij')
            zk = np.full_like(xi, idx, dtype=np.float64)
        elif self.orientation == 'YZ':
            jj = ax(s[1])
            kk = ax(s[2])
            yj, zk = np.meshgrid(jj, kk, indexing='ij')
            xi = np.full_like(yj, idx, dtype=np.float64)
        else:                                  # XZ
            ii = ax(s[0])
            kk = ax(s[2])
            xi, zk = np.meshgrid(ii, kk, indexing='ij')
            yj = np.full_like(xi, idx, dtype=np.float64)
        pts_out = np.stack([xi.ravel(), yj.ravel(), zk.ravel()], axis=0)
        pts_in  = R @ pts_out + off.reshape(3, 1)
        # Sample the original volume directly (no full-volume astype copy);
        # write float32 so integer volumes are interpolated, not rounded.
        vals = map_coordinates(vol, pts_in, order=1, mode='constant',
                               cval=0.0, output=np.float32)
        img = vals.reshape(xi.shape).astype(np.float32)
        img = img[:, ::-1]    # flip Y (XY) or Z (YZ/XZ) — same for all orientations
        return img

    def update_image(self):
        if self.volume_data is None or self.volume_data.volume is None:
            return
        # Fast path for an applied alignment: the whole volume was pre-resampled
        # into the aligned output frame once, so a slice is a cheap numpy view —
        # identical pixels to per-slice sampling, but pan/scroll stay smooth.
        # (Live preview still uses on-the-fly sampling; it has no cached volume.)
        if (self._preview_R is None and self._perm_R is not None
                and self._aligned_vol is not None):
            av = self._aligned_vol
            ax = {'XY': 2, 'YZ': 0, 'XZ': 1}[self.orientation]
            idx = int(np.clip(self.current_index, 0, av.shape[ax] - 1))
            if self.orientation == 'XY':
                image = av[:, ::-1, idx]
            elif self.orientation == 'YZ':
                image = av[idx, :, ::-1]
            else:                                  # XZ
                image = av[:, idx, ::-1]
            ar = self._auto_range_pending
            self._auto_range_pending = False
            if self._display_levels is not None:
                self.image_view.setImage(image, autoLevels=False,
                                         autoRange=ar, levels=self._display_levels)
            else:
                self.image_view.setImage(image, autoLevels=True, autoRange=ar)
            self.image_view.getImageItem().setRect(
                QtCore.QRectF(0, 0, image.shape[0], image.shape[1]))
            self.image_view.getView().setAspectLocked(True)
            self._update_surface_outline()
            return
        # Temporary preview overrides permanent transform; permanent overrides raw slice.
        R_use = off_use = vol_use = shape_use = None
        if self._preview_R is not None and self._preview_vol is not None:
            R_use, off_use, vol_use = self._preview_R, self._preview_offset, self._preview_vol
            shape_use = self._preview_vol.shape
        elif self._perm_R is not None:
            R_use, off_use, vol_use = self._perm_R, self._perm_offset, self.volume_data.volume
            shape_use = self._perm_shape
        if R_use is not None:
            # In-plane (width, height) of the displayed frame for this orientation.
            wh = {'XY': (shape_use[0], shape_use[1]),
                  'YZ': (shape_use[1], shape_use[2]),
                  'XZ': (shape_use[0], shape_use[2])}[self.orientation]
            # Interactive preview samples a coarse grid (then stretches to the full
            # extent); an applied/permanent transform is rendered full-resolution.
            stride = 1
            if self._preview_R is not None:
                stride = max(1, int(np.ceil(max(wh) / self._PREVIEW_MAX_DIM)))
            image = self._sample_transformed_slice(R_use, off_use, vol_use, shape_use, stride)
            if image is None and self._preview_R is not None:
                self._preview_R = None   # scipy missing, fall through to raw slice
            elif image is not None:
                ar = self._auto_range_pending
                self._auto_range_pending = False
                if self._display_levels is not None:
                    self.image_view.setImage(image, autoLevels=False,
                                             autoRange=ar, levels=self._display_levels)
                else:
                    self.image_view.setImage(image, autoLevels=True, autoRange=ar)
                # Map the (possibly coarse) image back onto the full frame extent
                # so plot coordinates and aspect stay correct.
                self.image_view.getImageItem().setRect(QtCore.QRectF(0, 0, wh[0], wh[1]))
                self.image_view.getView().setAspectLocked(True)
                self._update_surface_outline()
                return
        vol = self.volume_data.volume
        if self.orientation == 'XY':
            image = vol[:, ::-1, self.current_index]   # flip Y: Y=0 at bottom, Y=max at top
        elif self.orientation == 'YZ':
            image = vol[self.current_index, :, ::-1]   # flip Z: Z=0 at bottom, Z=max at top
        elif self.orientation == 'XZ':
            image = vol[:, self.current_index, ::-1]   # flip Z: Z=0 at bottom, Z=max at top
        else:
            image = vol[:, :, self.current_index]
        ar = self._auto_range_pending
        self._auto_range_pending = False
        if self._display_levels is not None:
            self.image_view.setImage(image.astype(np.float32), autoLevels=False,
                                     autoRange=ar, levels=self._display_levels)
        else:
            self.image_view.setImage(image.astype(np.float32), autoLevels=True,
                                     autoRange=ar)
        # Reset any image rect left over from a coarse preview so this full-res
        # slice maps 1:1 to plot coordinates.
        self.image_view.getImageItem().setRect(
            QtCore.QRectF(0, 0, image.shape[0], image.shape[1]))
        self.image_view.getView().setAspectLocked(True)
        self._update_surface_outline()

    # ── Surface outline (slice ∩ surface mesh) ────────────────────────────────
    def set_surface_outline(self, tri_verts, shape):
        """Show the contour where ``tri_verts`` (an (M,3,3) array of surface mesh
        triangles, in the display frame) crosses this viewport's current slice."""
        self._surf_tri = tri_verts
        self._surf_shape = shape
        # Precompute per-triangle min/max along each axis ONCE so the per-slice
        # spanning test is just two array comparisons (no reductions over the
        # whole mesh every scroll/redraw).
        if tri_verts is not None and len(tri_verts):
            self._surf_tri_min = tri_verts.min(axis=1)   # (M, 3)
            self._surf_tri_max = tri_verts.max(axis=1)   # (M, 3)
        else:
            self._surf_tri_min = self._surf_tri_max = None
        self._surf_outline_idx = None                    # force a recompute
        self._update_surface_outline()

    def clear_surface_outline(self):
        self._surf_tri = None
        self._surf_shape = None
        self._surf_tri_min = self._surf_tri_max = None
        self._surf_outline_idx = None
        if self._surf_outline_item is not None:
            self._surf_outline_item.setData([], [])

    def _update_surface_outline(self):
        tri = self._surf_tri
        item = self._surf_outline_item
        if tri is None or self._surf_shape is None or self._surf_tri_min is None:
            if item is not None:
                item.setData([], [])
            self._surf_outline_idx = None
            return
        idx = int(self.current_index)
        # The contour depends only on the slice index, not on zoom/pan — the item
        # lives in data coordinates and transforms with the view for free. So skip
        # the whole recompute when the slice hasn't changed (the hot path during
        # panning), and only rebuild it when actually scrolling to a new slice.
        if idx == self._surf_outline_idx and item is not None:
            return
        self._surf_outline_idx = idx
        axis = {'XY': 2, 'YZ': 0, 'XZ': 1}[self.orientation]
        idx = float(idx)
        span = (self._surf_tri_min[:, axis] <= idx) & (self._surf_tri_max[:, axis] >= idx)
        xs = ys = None
        if span.any():
            t = tri[span]
            s = t[:, :, axis] - idx                    # (K,3) signed distance to plane
            neg = s < 0
            c01 = neg[:, 0] != neg[:, 1]
            c12 = neg[:, 1] != neg[:, 2]
            c20 = neg[:, 2] != neg[:, 0]
            keep = (c01.astype(np.int8) + c12 + c20) == 2   # clean crossings
            if keep.any():
                t = t[keep]; s = s[keep]
                c01, c12, c20 = c01[keep], c12[keep], c20[keep]
                s0, s1, s2 = s[:, 0], s[:, 1], s[:, 2]
                v0, v1, v2 = t[:, 0], t[:, 1], t[:, 2]

                def interp(sa, sb, va, vb):
                    # Non-crossing edges divide by ~0 → NaN, but those entries are
                    # discarded by the np.where below; silence the warning.
                    with np.errstate(divide='ignore', invalid='ignore'):
                        tt = (sa / (sa - sb))[:, None]
                        return va + tt * (vb - va)

                p01, p12, p20 = interp(s0, s1, v0, v1), interp(s1, s2, v1, v2), interp(s2, s0, v2, v0)
                A = np.where(c01[:, None], p01, np.where(c12[:, None], p12, p20))
                B = np.where(c20[:, None], p20, np.where(c12[:, None], p12, p01))
                h, v, vdim = self._outline_plot_axes()
                n = A.shape[0]
                xs = np.empty(2 * n, dtype=np.float32)
                ys = np.empty(2 * n, dtype=np.float32)
                xs[0::2] = A[:, h]; xs[1::2] = B[:, h]
                ys[0::2] = (vdim - 1) - A[:, v]; ys[1::2] = (vdim - 1) - B[:, v]
        if item is None:
            # antialias=False keeps the (often many-segment) contour cheap to
            # repaint while panning/zooming. (clipToView is avoided: it can split
            # connect='pairs' segments at the view edge.)
            item = pg.PlotDataItem(pen=pg.mkPen(_MEAS_COLOR, width=2), antialias=False)
            item.setZValue(20)
            self.image_view.getView().addItem(item, ignoreBounds=True)
            self._surf_outline_item = item
        if xs is None:
            item.setData([], [])
        else:
            item.setData(xs, ys, connect='pairs', skipFiniteCheck=True)

    def _outline_plot_axes(self):
        """(horizontal volume axis, vertical volume axis, vertical dim) for the
        current orientation — vertical axis is shown flipped in the plot."""
        sh = self._surf_shape
        if self.orientation == 'XY':
            return 0, 1, int(sh[1])   # plot-x = X, plot-y = ny-1-Y
        if self.orientation == 'YZ':
            return 1, 2, int(sh[2])   # plot-x = Y, plot-y = nz-1-Z
        return 0, 2, int(sh[2])       # XZ: plot-x = X, plot-y = nz-1-Z

    def on_image_click(self, event):
        if self.volume_data is None or self.volume_data.volume is None:
            return
        if not _pick_modifier(event):
            return  # Ctrl/⌘+click required for point placement
        pos = event.pos()
        view_box = self.image_view.getImageItem().parentItem()
        scene_pos = view_box.mapSceneToView(pos)
        x = int(np.clip(scene_pos.x(), 0, self.image_view.width()))
        y = int(np.clip(scene_pos.y(), 0, self.image_view.height()))
        mapped = self.map_to_volume_coordinates(x, y)
        if mapped is not None:
            self.point_placed.emit(mapped)

    def _axis_line_geom(self, axis, index, total):
        """(angle, pos) for a coordinate line of ``axis`` in this viewport's plot
        space, or None if ``axis`` is this viewport's own traversal axis.
        Vertical (angle=90): axis is the horizontal dim, no flip. Horizontal
        (angle=0): axis is the vertical dim, always flipped ([::-1])."""
        pairs = {'XY': ('X', 'Y'), 'YZ': ('Y', 'Z'), 'XZ': ('X', 'Z')}.get(self.orientation)
        if pairs is None:
            return None
        h_axis, v_axis = pairs
        if axis == h_axis:
            return 90, index
        if axis == v_axis:
            return 0, total - 1 - index
        return None  # this viewport's own traversal axis

    def show_axis_line(self, axis, index, total, seg_range=None):
        if not self._coord_lines_enabled:
            return
        geom = self._axis_line_geom(axis, index, total)
        if geom is None:
            return
        angle, pos = geom

        color = _AXIS_LINE_COLORS[axis]
        view = self.image_view.getView()
        # Reuse persistent line items per axis and just reposition them, rather
        # than recreating graphics items on every call. Recreation (addItem /
        # removeItem / signal connect-disconnect) churns the scene graph and was
        # a major cost during pan and locked slice-sync, especially after an
        # alignment. The dashed InfiniteLine never affects ViewBox auto-range.
        entry = self._axis_lines.get(axis)
        if entry is None:
            line = pg.InfiniteLine(
                pos=pos, angle=angle,
                pen=pg.mkPen(color=color, width=1, style=Qt.DashLine))
            seg = pg.PlotDataItem(
                pen=pg.mkPen(color=color, width=2, style=Qt.SolidLine))
            view.addItem(line, ignoreBounds=True)
            view.addItem(seg, ignoreBounds=True)
            entry = {'line': line, 'seg': seg}
            self._axis_lines[axis] = entry
        line, seg = entry['line'], entry['seg']
        line.setAngle(angle)
        line.setValue(pos)
        line.setVisible(True)
        if seg_range is not None:
            a, b = min(seg_range), max(seg_range)
            if angle == 0:  # horizontal at y=pos
                seg.setData([a, b], [pos, pos])
            else:           # vertical at x=pos
                seg.setData([pos, pos], [a, b])
            seg.setVisible(True)
        else:
            seg.setVisible(False)

        if self._lines_pinned:
            # Pinned mode: stop any running hide timer; line stays until unpinned.
            if axis in self._axis_timers:
                self._axis_timers[axis].stop()
        else:
            if axis not in self._axis_timers:
                timer = QtCore.QTimer()
                timer.setSingleShot(True)
                timer.timeout.connect(lambda a=axis: self.hide_axis_line(a))
                self._axis_timers[axis] = timer
            self._axis_timers[axis].start(3000)

    def hide_axis_line(self, axis):
        if self._lines_pinned:
            return
        entry = self._axis_lines.get(axis)
        if entry is not None:   # keep the items for reuse; just hide them
            entry['line'].setVisible(False)
            entry['seg'].setVisible(False)
        if axis in self._axis_timers:
            self._axis_timers.pop(axis).stop()

    def set_lines_pinned(self, pinned):
        self._lines_pinned = pinned
        if pinned:
            for timer in self._axis_timers.values():
                timer.stop()
        else:
            self._clear_axis_lines()

    def _clear_axis_lines(self):
        for timer in list(self._axis_timers.values()):
            timer.stop()
        self._axis_timers.clear()
        view = self.image_view.getView()
        for entry in list(self._axis_lines.values()):
            view.removeItem(entry['line'])
            view.removeItem(entry['seg'])
        self._axis_lines.clear()

    def set_coord_lines_enabled(self, enabled):
        """Enable/disable the coordinate lines in this 2D viewport."""
        self._coord_lines_enabled = bool(enabled)
        _set_button_active(self.planes_button, bool(enabled))
        if not enabled:
            self._clear_axis_lines()

    # ── Slice-video recording ─────────────────────────────────────────────────
    def toggle_record_mode(self):
        self.set_record_mode(not self._record_mode)

    def set_record_mode(self, on):
        """Enter/leave the slice-video range-selection mode for this viewport."""
        if self._recording:
            return
        on = bool(on) and self.volume_data is not None and self.volume_data.is_loaded()
        self._record_mode = on
        self.record_button.setIcon(_record_icon(on))
        self.record_button.setStyleSheet(_RECORD_BTN_QSS if on else '')
        if on:
            mx = max(0, self.slice_slider.maximum())
            self.record_slider.setRange(0, mx)
            self.record_slider.blockSignals(True)
            self.record_slider.setValues(0, mx)
            self.record_slider._last = 'high'
            self.record_slider.blockSignals(False)
            self.slice_slider.hide()
            self.record_progress.hide()
            self.record_slider.show()
            self.record_start_btn.show()
            self.record_len_spin.show()
            self.current_index = self.record_slider.last_value()
            self.update_image()
            self._sync_slice_spin()
            self._emit_record_range()
        else:
            self.record_slider.hide()
            self.record_start_btn.hide()
            self.record_len_spin.hide()
            self.record_progress.hide()
            self.slice_slider.show()
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(int(np.clip(self.current_index, 0,
                                                   max(0, self.slice_slider.maximum()))))
            self.slice_slider.blockSignals(False)
            self._sync_slice_spin()
        self.record_mode_changed.emit(on)

    def _emit_record_range(self):
        shape = self._active_shape()
        if shape is None:
            return
        total = shape[self._traversal_axis()]
        self.record_range_changed.emit(self.record_slider.low(),
                                       self.record_slider.high(), total)

    def _on_record_range_changed(self):
        """A record handle moved: show the last-touched endpoint as the live slice."""
        self.current_index = self.record_slider.last_value()
        self.update_image()
        self._sync_slice_spin()
        self._emit_record_range()

    def show_record_lines(self, axis, low, high, total):
        """Draw two solid lines marking another viewport's record range (its two
        endpoints). Persistent until cleared; independent of the coord-line toggle."""
        self.clear_record_lines()
        view = self.image_view.getView()
        color = _AXIS_LINE_COLORS.get(axis)
        for idx in (low, high):
            geom = self._axis_line_geom(axis, idx, total)
            if geom is None:
                continue
            angle, pos = geom
            line = pg.InfiniteLine(pos=pos, angle=angle,
                                   pen=pg.mkPen(color=color, width=2, style=Qt.SolidLine))
            view.addItem(line, ignoreBounds=True)
            self._record_lines.append(line)

    def clear_record_lines(self):
        view = self.image_view.getView()
        for ln in self._record_lines:
            try:
                view.removeItem(ln)
            except Exception:
                pass
        self._record_lines = []

    def _grab_frame_array(self):
        """Grab the current 2D view as a contiguous (H, W, 3) uint8 RGB array."""
        gv = self.image_view.ui.graphicsView
        img = gv.viewport().grab().toImage().convertToFormat(QtGui.QImage.Format_RGB888)
        w, h, bpl = img.width(), img.height(), img.bytesPerLine()
        ptr = img.constBits()
        try:
            ptr.setsize(h * bpl)          # PyQt5: sip.voidptr needs an explicit size
        except AttributeError:
            pass                          # PySide6: already a sized memoryview
        arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape(h, bpl)
        return np.ascontiguousarray(arr[:, : w * 3].reshape(h, w, 3))

    def _start_recording(self):
        if self._recording or self.volume_data is None or not self.volume_data.is_loaded():
            return
        low, high = self.record_slider.low(), self.record_slider.high()
        if high < low:
            low, high = high, low
        fps = 30
        n_frames = max(2, int(round(float(self.record_len_spin.value()) * fps)))
        path, sel = QFileDialog.getSaveFileName(
            self, 'Save Slice Video', '',
            'MP4 Video (*.mp4);;Animated GIF (*.gif)')
        if not path:
            return
        is_gif = path.lower().endswith('.gif') or 'GIF' in (sel or '')
        if not is_gif and not path.lower().endswith('.mp4'):
            path += '.mp4'
        indices = [int(round(v)) for v in np.linspace(low, high, n_frames)]
        self._run_recording(path, indices, fps, is_gif)

    def _run_recording(self, path, indices, fps, is_gif):
        import tempfile, shutil
        try:
            import imageio
        except Exception:
            QMessageBox.warning(self, 'Record Video',
                                'The imageio package is required to export video.')
            return
        # Encode to a temp file first, then move it to the user-chosen path. Under
        # the App Store sandbox the ffmpeg child process can write into the app's
        # own temp dir but not to an arbitrary user location (the save-panel grant
        # belongs to this process, not the child); the move here runs in-process
        # and does carry that grant. It also avoids leaving a half-written file.
        ext = '.gif' if is_gif else '.mp4'
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        # Open the writer (mp4 via bundled ffmpeg; gif via pillow).
        try:
            if is_gif:
                writer = imageio.get_writer(tmp_path, mode='I', duration=1.0 / fps, loop=0)
            else:
                writer = imageio.get_writer(
                    tmp_path, format='FFMPEG', mode='I', fps=fps, codec='libx264',
                    macro_block_size=None, ffmpeg_log_level='error',
                    output_params=['-pix_fmt', 'yuv420p'])
        except Exception as exc:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            QMessageBox.warning(
                self, 'Record Video',
                'Could not start the video encoder.\n\nFor MP4 you may need the '
                f'imageio-ffmpeg package installed.\n\n{exc}')
            return

        saved_index = self.current_index
        self._recording = True
        self.record_start_btn.setEnabled(False)
        self.record_len_spin.setEnabled(False)
        self.record_button.setEnabled(False)
        self.record_slider.hide()
        self.record_progress.setRange(0, len(indices))
        self.record_progress.setValue(0)
        self.record_progress.setFormat('Recording… %p%')
        self.record_progress.show()
        frame_shape = None
        err = None
        try:
            for i, idx in enumerate(indices):
                self.current_index = int(np.clip(idx, 0, max(0, self.slice_slider.maximum())))
                self.update_image()
                QApplication.processEvents()
                frame = self._grab_frame_array()
                # Even dimensions for yuv420p; keep all frames the same size.
                if frame_shape is None:
                    h, w = frame.shape[:2]
                    frame_shape = (h - h % 2, w - w % 2)
                frame = frame[:frame_shape[0], :frame_shape[1]]
                writer.append_data(frame)
                self.record_progress.setValue(i + 1)
                QApplication.processEvents()
        except Exception as exc:
            err = exc
        finally:
            try:
                writer.close()
            except Exception:
                pass
            self._recording = False
            self.current_index = saved_index
            self.update_image()
            self._sync_slice_spin()
            self.record_start_btn.setEnabled(True)
            self.record_len_spin.setEnabled(True)
            self.record_button.setEnabled(True)
            self.record_progress.hide()
            if self._record_mode:
                self.record_slider.show()
        # Move the finished encode to the destination (in-process: keeps the
        # save-panel file grant under sandbox). Clean up the temp file on failure.
        if err is None:
            try:
                shutil.move(tmp_path, path)
            except Exception as exc:
                err = exc
        if err is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            QMessageBox.warning(self, 'Record Video',
                                f'Recording failed:\n{err}')
        else:
            QMessageBox.information(
                self, 'Record Video',
                f'Saved {len(indices)} frames to:\n{path}')

    # ── Capture (camera button) ───────────────────────────────────────────────
    def _camera_save(self):
        """Save the viewport image directly via pyqtgraph's Image File exporter
        (PNG/JPG/TIFF/…) — a plain file dialog, not the full Export dialog."""
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Image', '',
            'PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tif);;All Files (*)')
        if not path:
            return
        if not any(path.lower().endswith(e)
                   for e in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            path += '.png'
        try:
            import pyqtgraph.exporters as _pgexp
            _pgexp.ImageExporter(self.image_view.getView()).export(path)
        except Exception as exc:
            QMessageBox.warning(self, 'Save Image', f'Could not save the image:\n{exc}')

    def _camera_copy(self):
        """Copy the viewport image to the clipboard."""
        try:
            import pyqtgraph.exporters as _pgexp
            _pgexp.ImageExporter(self.image_view.getView()).export(copy=True)
        except Exception:
            # Fallback: grab the rendered widget straight to the clipboard.
            try:
                QApplication.clipboard().setPixmap(self.image_view.grab())
            except Exception as exc:
                QMessageBox.warning(self, 'Copy to Clipboard', f'Could not copy the image:\n{exc}')

    def map_to_volume_coordinates(self, x, y):
        if self.volume_data is None or self.volume_data.volume is None:
            return None
        w, h, d = self.volume_data.volume.shape
        if self.orientation == 'XY':
            return (int(x), int(y), self.current_index)
        if self.orientation == 'YZ':
            return (self.current_index, int(x), int(y))
        if self.orientation == 'XZ':
            return (int(x), self.current_index, int(y))
        return None


def _pick_modifier(ev):
    """Return True if the alignment pick modifier (Ctrl or Meta/Cmd) is held."""
    mods = ev.modifiers()
    return bool(mods & Qt.ControlModifier) or bool(mods & Qt.MetaModifier)


class _GLView(gl.GLViewWidget):
    """GLViewWidget with:
    - pan remapped from Ctrl/Cmd+drag  →  Shift+drag
    - Ctrl/Cmd+click emits pick_requested so the parent can do 3-D picking
    - Unlimited free rotation via a rotation matrix (no elevation clamp)
    """
    pick_requested = Signal(object)   # QPointF screen position

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pick_mode = False
        self._static = False   # PiP preview: camera fixed, no mouse interaction
        self._view_rot = self._rot_from_opts()

    def set_pick_mode(self, enabled):
        self._pick_mode = enabled

    # ── Rotation matrix helper (tripod overlay only) ─────────────────────────

    def _rot_from_opts(self):
        """Build 3×3 rotation matrix matching pyqtgraph's viewMatrix convention.

        Used only to orient the tripod overlay; the actual camera is driven by
        pyqtgraph's own viewMatrix() reading opts azimuth/elevation.
        """
        az = float(self.opts.get('azimuth',   45.0))
        el = float(self.opts.get('elevation', 30.0))
        el_rad = np.radians(el - 90.0)
        az_rad = np.radians(-(az + 90.0))
        Rx = np.array([
            [1, 0,               0              ],
            [0, np.cos(el_rad), -np.sin(el_rad) ],
            [0, np.sin(el_rad),  np.cos(el_rad) ],
        ], dtype=np.float64)
        Rz = np.array([
            [np.cos(az_rad), -np.sin(az_rad), 0],
            [np.sin(az_rad),  np.cos(az_rad), 0],
            [0,               0,              1],
        ], dtype=np.float64)
        return Rx @ Rz

    def sync_rot_from_opts(self):
        """Re-sync _view_rot (tripod) from the current azimuth/elevation opts."""
        self._view_rot = self._rot_from_opts()
        self.update()

    # ── Turntable orbit without the ±90° elevation clamp ─────────────────────

    def orbit(self, azim, elev):
        """Turntable rotation: horizontal spins around the world vertical axis,
        vertical tilts up/down. Elevation is wrapped, not clamped, so the volume
        can tumble over the top continuously in either direction."""
        # Past the pole the camera's up-vector flips, which makes a horizontal
        # drag spin the model the opposite visual way. Negate the azimuth step
        # while upside-down (cos(elevation) < 0) so horizontal drag stays
        # consistent regardless of how far the model has tumbled.
        if math.cos(math.radians(self.opts['elevation'])) < 0:
            azim = -azim
        self.opts['azimuth'] = (self.opts['azimuth'] + azim) % 360.0
        # Wrap elevation into (-180, 180] instead of clamping to [-90, 90].
        el = (self.opts['elevation'] + elev + 180.0) % 360.0 - 180.0
        self.opts['elevation'] = el
        self._view_rot = self._rot_from_opts()
        self.update()

    def mousePressEvent(self, ev):
        if self._static:
            ev.accept()
            return
        # Always track cursor position so drag handling works regardless.
        try:
            lpos = ev.position()
        except AttributeError:
            lpos = ev.localPos()
        self.mousePos = lpos

        if self._pick_mode and ev.button() == Qt.LeftButton and _pick_modifier(ev):
            self.pick_requested.emit(lpos)
            ev.accept()
            return          # do NOT call super() — that would start a Ctrl pan

        # For everything else let pyqtgraph handle it normally.
        # super().mousePressEvent only re-sets mousePos, which we already did,
        # so it's safe to call.
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._static:
            ev.accept()
            return
        try:
            lpos = ev.position()
        except AttributeError:
            lpos = ev.localPos()

        if not hasattr(self, 'mousePos'):
            self.mousePos = lpos
            return

        if ev.buttons() == Qt.LeftButton and (ev.modifiers() & Qt.ShiftModifier):
            diff = lpos - self.mousePos
            self.mousePos = lpos
            self.pan(diff.x(), diff.y(), 0, relative='view')
            return

        if ev.buttons() == Qt.LeftButton and _pick_modifier(ev):
            # After a Cmd+click pick, dragging with modifier still held → orbit
            diff = lpos - self.mousePos
            self.mousePos = lpos
            self.orbit(-diff.x(), diff.y())
            return

        super().mouseMoveEvent(ev)


class VolumeRender3D(QWidget):
    point_placed = Signal(tuple)
    planes_toggled = Signal(bool)   # coordinate-planes visibility toggle

    def __init__(self, parent=None, pip=False):
        super().__init__(parent)
        self._pip = pip   # picture-in-picture preview: no header/controls, fixed camera
        self._pip_fit_axes = None   # (horiz, vert, depth) volume axes for camera fit
        self.volume_data = None
        self.orientation = '3D'
        self.mode = 'Isosurface'
        self._show_planes = True
        self._alignment_mode = False
        self._alignment_points = []   # voxel-coord tuples
        self._alignment_gl_items = [] # live GL items for overlays
        self._render_factor = 1
        self._render_shape = (1, 1, 1)
        self._render_volume = None    # downsampled float32 array used for picking
        self._render_vol_min = 0.0
        self._render_vol_max = 1.0
        self._display_levels = None   # (lo, hi) window from the histogram, or None
        self._perm_volume = None      # permanent aligned display volume (non-destructive)
        self._quality = 'Default'     # 'Low' 256³, 'Default' 512³ (2×), 'High' 1024³ (4×)
        self._clips = {}              # axis(0/1/2) -> ('left'|'right', fraction)
        # Caches so slice-drag clipping just hides geometry/voxels rather than
        # recomputing the surface or re-normalising the volume each frame.
        self._iso_cache = None        # (verts, faces) from the FULL volume
        self._iso_cache_key = None
        self._showing_surface = False # a created STL surface is shown (not the volume)
        self._surface = None          # (verts, faces) of the shown surface
        self._surface_shape = None    # source-volume shape the surface verts live in
        self._surface_tri = None      # cached verts[faces] for the 2D outline
        self._surface_color = (0.78, 0.88, 1.0, 1.0)
        self._rgba_base = None        # full-volume RGBA (original alpha)
        self._rgba_cache_key = None
        self._plane_fracs = {}        # axis(0/1/2) -> slice fraction [0,1]
        # axis -> {in-plane axis: (lo,hi)} normalised visible extent of the
        # matching 2D viewport (zoom/pan/FOV); empty → full-volume extent.
        self._plane_rects = {}
        self._plane_items = []        # GL items for the coordinate planes + frames
        self.init_ui()
        # Debounce clip re-renders so dragging a slice stays responsive.
        self._clip_timer = QtCore.QTimer(self)
        self._clip_timer.setSingleShot(True)
        self._clip_timer.timeout.connect(lambda: self._render_scene(reset_camera=False))

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.iso_threshold_percent = 50
        self.iso_slider = QSlider(Qt.Horizontal)
        self.iso_slider.setRange(0, 100)
        self.iso_slider.setValue(self.iso_threshold_percent)
        self.iso_slider.setFixedWidth(90)
        self.iso_slider.valueChanged.connect(self.on_iso_slider_changed)

        self.iso_spinbox = QtWidgets.QSpinBox()
        self.iso_spinbox.setRange(0, 100)
        self.iso_spinbox.setValue(self.iso_threshold_percent)
        self.iso_spinbox.setSuffix('%')
        self.iso_spinbox.setFixedWidth(54)
        self.iso_spinbox.editingFinished.connect(
            lambda: self.iso_slider.setValue(self.iso_spinbox.value()))

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Isosurface', 'Phong Volume', 'Off'])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)

        header_bar = QWidget()
        header_bar.setFixedHeight(28)
        header = QHBoxLayout(header_bar)
        header.setContentsMargins(2, 2, 2, 2)
        header.setSpacing(4)
        header.addWidget(self.mode_combo)
        header.addSpacing(4)
        header.addWidget(QLabel('Isovalue:'))
        header.addWidget(self.iso_slider)
        header.addWidget(self.iso_spinbox)
        header.addSpacing(4)
        self.planes_button = QPushButton()
        self.planes_button.setCheckable(True)
        self.planes_button.setChecked(self._show_planes)
        self.planes_button.setFixedSize(24, 24)
        self.planes_button.setIconSize(QtCore.QSize(16, 16))
        self.planes_button.setIcon(_coord_axes_icon())
        self.planes_button.setToolTip('Show coordinate planes')
        self.planes_button.toggled.connect(self.planes_toggled)
        header.addWidget(self.planes_button)

        self.camera_button = _make_camera_button(self._camera_copy, self._camera_save)
        header.addWidget(self.camera_button)

        header.addStretch()
        self.maximize_button = QPushButton('▲')
        self.maximize_button.setFixedSize(24, 24)
        header.addWidget(self.maximize_button)
        self.header_bar = header_bar
        layout.addWidget(header_bar)

        self.gl_view = _GLView()
        self.gl_view.opts['distance'] = 200
        self.gl_view.opts['fov'] = 60
        self.gl_view.opts['azimuth'] = -45   # X right, consistent with 2D views
        self.gl_view.sync_rot_from_opts()    # initialise rotation matrix from the above opts
        layout.addWidget(self.gl_view, 1)

        self.gl_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.gl_view.customContextMenuRequested.connect(self._show_context_menu)

        self.tripod = TripodWidget('3D', parent=self.gl_view, gl_view=self.gl_view)
        self.gl_view.pick_requested.connect(self._on_gl_pick)
        self.gl_view.installEventFilter(self)

        # Overlay hint shown while clipping is active (mirrors the 2D
        # "Auto Min Max is on" note: same style, same top-right position).
        self.clip_hint = QLabel('Clipping is on', self.gl_view)
        self.clip_hint.setStyleSheet(
            'color: #FFD400; background: rgba(0,0,0,120);'
            'padding: 2px 6px; border-radius: 3px; font-size: 11px;')
        self.clip_hint.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.clip_hint.adjustSize()
        self.clip_hint.hide()

        self.status_text = QLabel('')
        self.status_text.setStyleSheet('color: red;')

        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(28)
        bottom = QHBoxLayout(bottom_bar)
        bottom.setContentsMargins(2, 2, 2, 2)
        bottom.addWidget(self.status_text, 1)
        self.bottom_bar = bottom_bar
        layout.addWidget(bottom_bar)

        # PiP preview: chrome-free, fixed camera, a thin frame to set it apart.
        if self._pip:
            self.header_bar.hide()
            self.bottom_bar.hide()
            self.gl_view._static = True
            self.gl_view.setContextMenuPolicy(Qt.NoContextMenu)
            self.tripod.hide()
            self.setStyleSheet('VolumeRender3D { border: 1px solid #888; }')

    def eventFilter(self, obj, event):
        if obj is self.gl_view:
            et = event.type()
            # PiP preview: fixed camera — swallow wheel/gesture zoom.
            if self._pip and et in (_EV_WHEEL, _EV_NATIVE_GESTURE):
                return True
            if et == _EV_RESIZE:
                self._reposition_tripod()
                self._reposition_clip_hint()
                if self._pip:
                    self._fit_pip_camera()   # re-fit on PiP resize (aspect change)
            elif et == _EV_MOUSE_MOVE:
                QtCore.QTimer.singleShot(0, self.tripod.update)
            elif et == _EV_WHEEL:
                if not (event.modifiers() & Qt.ControlModifier):
                    return True  # eat non-Ctrl wheel
                QtCore.QTimer.singleShot(0, self.tripod.update)
                # Ctrl held: fall through to GLViewWidget's own zoom
            elif et == _EV_NATIVE_GESTURE and _ZOOM_GESTURE_TYPE is not None:
                try:
                    if event.gestureType() == _ZOOM_GESTURE_TYPE:
                        factor = 1.0 + event.value()
                        if factor > 0:
                            self.gl_view.opts['distance'] = max(
                                1.0, self.gl_view.opts['distance'] / factor
                            )
                            self.gl_view.update()
                        return True
                except Exception:
                    pass
        return super().eventFilter(obj, event)

    def _reposition_tripod(self):
        m = 6
        self.tripod.move(m, self.gl_view.height() - self.tripod.height() - m)
        self.tripod.raise_()

    def _reposition_clip_hint(self):
        m = 6
        self.clip_hint.adjustSize()
        self.clip_hint.move(self.gl_view.width() - self.clip_hint.width() - m, m)
        self.clip_hint.raise_()

    def _fit_pip_camera(self):
        """Pull the camera in so the volume's bounding box fills the PiP view.

        Fits the front face (nearest under perspective) of the box to whichever
        screen dimension is more constraining, given the view's two visible axes.
        """
        if not self._pip or self._pip_fit_axes is None or self._render_shape is None:
            return
        h_ax, v_ax, d_ax = self._pip_fit_axes
        shape = self._render_shape
        if min(shape) <= 0:
            return
        half_w = shape[h_ax] / 2.0
        half_h = shape[v_ax] / 2.0
        depth  = shape[d_ax]
        gl = self.gl_view
        vp_w, vp_h = max(1, gl.width()), max(1, gl.height())
        thf = math.tan(math.radians(gl.opts['fov']) * 0.5)   # horizontal half-FOV
        # Visible half-extent at distance d: horizontal d*thf, vertical d*thf*(h/w).
        d_h = half_w / thf
        d_v = half_h * (vp_w / vp_h) / thf
        dist = max(d_h, d_v) + depth / 2.0      # +depth/2: nearest face under perspective
        gl.setCameraPosition(distance=dist * 1.05)   # tiny margin so it just touches

    # ── Alignment helpers ────────────────────────────────────────────────────

    def set_alignment_mode(self, active):
        self._alignment_mode = active
        self.gl_view.set_pick_mode(active)

    # ── Capture (camera button) ───────────────────────────────────────────────
    def _grab_image(self):
        """Render the current 3D view to a QImage (or None)."""
        try:
            return self.gl_view.grabFramebuffer()
        except Exception:
            return None

    def _camera_copy(self):
        img = self._grab_image()
        if img is not None and not img.isNull():
            QApplication.clipboard().setImage(img)
        else:
            QMessageBox.warning(self, 'Copy to Clipboard', 'Could not capture the 3D view.')

    def _camera_save(self):
        img = self._grab_image()
        if img is None or img.isNull():
            QMessageBox.warning(self, 'Save Image', 'Could not capture the 3D view.')
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Image', '',
            'PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tif);;All Files (*)')
        if not path:
            return
        if not any(path.lower().endswith(e)
                   for e in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
            path += '.png'
        if not img.save(path):
            QMessageBox.warning(self, 'Save Image', f'Could not save the image to:\n{path}')

    def _on_gl_pick(self, screen_pos):
        """Called by _GLView when a Cmd/Ctrl+click is detected."""
        pt = self._pick_volume_point(screen_pos)
        if pt is not None:
            self.point_placed.emit(pt)

    def set_alignment_overlays(self, points):
        self._alignment_points = list(points)
        self._rebuild_alignment_overlays()

    def clear_alignment_overlays(self):
        for item in self._alignment_gl_items:
            try:
                self.gl_view.removeItem(item)
            except Exception:
                pass
        self._alignment_gl_items = []
        self._alignment_points = []

    def _rebuild_alignment_overlays(self):
        """Re-create GL overlay items from stored alignment points."""
        for item in self._alignment_gl_items:
            try:
                self.gl_view.removeItem(item)
            except Exception:
                pass
        self._alignment_gl_items = []
        pts = self._alignment_points
        if not pts:
            return
        n = len(pts)

        # Individual point markers (yellow dots)
        colors = [(1.0, 1.0, 0.0, 1.0)] * 3 + [(1.0, 0.6, 0.0, 1.0)] * 2 + [(1.0, 0.3, 0.0, 1.0)]
        for i, pt in enumerate(pts):
            gl_pt = self._voxel_to_gl(pt).reshape(1, 3)
            color = colors[i] if i < len(colors) else (1.0, 1.0, 0.0, 1.0)
            try:
                dot = gl.GLScatterPlotItem(pos=gl_pt, color=color, size=10, pxMode=True)
                self.gl_view.addItem(dot)
                self._alignment_gl_items.append(dot)
            except Exception:
                pass

        # Plane rectangle after 3 points
        if n >= 3:
            items = self._make_plane_outline([self._voxel_to_gl(p) for p in pts[:3]])
            for item in items:
                self.gl_view.addItem(item)
                self._alignment_gl_items.append(item)

        # Line after 5 points
        if n >= 5:
            p0 = self._voxel_to_gl(pts[3])
            p1 = self._voxel_to_gl(pts[4])
            pos = np.array([p0, p1], dtype=np.float32)
            try:
                line = gl.GLLinePlotItem(pos=pos, color=(1.0, 0.6, 0.0, 1.0), width=3, antialias=True, mode='lines')
                self.gl_view.addItem(line)
                self._alignment_gl_items.append(line)
            except Exception:
                pass

        # Origin cross after 6 points
        if n >= 6:
            p = self._voxel_to_gl(pts[5])
            r = max(max(self._render_shape) * 0.04, 3.0)
            arms = np.array([
                p + np.array([r, 0, 0]), p - np.array([r, 0, 0]),
                p + np.array([0, r, 0]), p - np.array([0, r, 0]),
                p + np.array([0, 0, r]), p - np.array([0, 0, r]),
            ], dtype=np.float32)
            try:
                cross = gl.GLLinePlotItem(pos=arms, color=(1.0, 0.3, 0.0, 1.0), width=4, antialias=True, mode='lines')
                self.gl_view.addItem(cross)
                self._alignment_gl_items.append(cross)
            except Exception:
                pass

    def _make_plane_outline(self, gl_pts):
        """Return a list of GL items forming a yellow rectangle in the plane."""
        try:
            p0, p1, p2 = [np.array(p, dtype=np.float64) for p in gl_pts]
            u = p1 - p0
            if np.linalg.norm(u) < 1e-6:
                return []
            u /= np.linalg.norm(u)
            n = np.cross(p1 - p0, p2 - p0)
            if np.linalg.norm(n) < 1e-6:
                return []
            n /= np.linalg.norm(n)
            v = np.cross(n, u)
            v /= np.linalg.norm(v)
            centroid = np.mean([p0, p1, p2], axis=0)
            projs_u = [np.dot(p - centroid, u) for p in [p0, p1, p2]]
            projs_v = [np.dot(p - centroid, v) for p in [p0, p1, p2]]
            pad = max(max(self._render_shape) * 0.06, 5.0)
            hu = max(abs(x) for x in projs_u) + pad
            hv = max(abs(x) for x in projs_v) + pad
            corners = [
                centroid + u * hu + v * hv,
                centroid - u * hu + v * hv,
                centroid - u * hu - v * hv,
                centroid + u * hu - v * hv,
                centroid + u * hu + v * hv,
            ]
            pos = np.array(corners, dtype=np.float32)
            item = gl.GLLinePlotItem(pos=pos, color=(1.0, 1.0, 0.0, 0.9), width=2, antialias=True, mode='line_strip')
            return [item]
        except Exception:
            return []

    def set_geometry_overlay(self, points, geom_type, fit):
        """Show picked points (yellow dots) plus the fitted geometry (yellow
        wireframe) for the Geometry-based alignment. Reuses the alignment-overlay
        item list, so clear_alignment_overlays() tears it down."""
        self.clear_alignment_overlays()
        self._alignment_points = list(points)

        def add(item):
            if item is None:
                return
            try:
                self.gl_view.addItem(item)
                self._alignment_gl_items.append(item)
            except Exception:
                pass

        for pt in points:
            try:
                add(gl.GLScatterPlotItem(pos=self._voxel_to_gl(pt).reshape(1, 3),
                                         color=(1.0, 1.0, 0.0, 1.0), size=8, pxMode=True))
            except Exception:
                pass
        if fit is not None:
            for item in self._make_geometry_wire(geom_type, fit, points):
                add(item)

    def _gl_dir(self, voxel_dir):
        """A voxel-space direction in GL space (uniform downsample scale)."""
        d = np.asarray(voxel_dir, dtype=np.float64)
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else d

    @staticmethod
    def _perp_basis(axis):
        a = np.asarray(axis, dtype=np.float64)
        a = a / (np.linalg.norm(a) or 1.0)
        ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(a, ref); e1 /= (np.linalg.norm(e1) or 1.0)
        e2 = np.cross(a, e1)
        return e1, e2

    @staticmethod
    def _circle_strip(center, e1, e2, radius, n=48):
        ang = np.linspace(0.0, 2.0 * np.pi, n)
        pts = (center[None, :] + radius * (np.cos(ang)[:, None] * e1[None, :]
                                           + np.sin(ang)[:, None] * e2[None, :]))
        return pts.astype(np.float32)

    def _make_geometry_wire(self, geom_type, fit, points):
        """Yellow wireframe items for the fitted plane / cylinder / cone."""
        items = []
        try:
            factor = max(self._render_factor, 1)
            gpts = np.array([self._voxel_to_gl(p) for p in points], dtype=np.float64)
            pad = max(max(self._render_shape) * 0.05, 4.0)
            yellow = (1.0, 1.0, 0.0, 0.9)
            if geom_type == 'Plane':
                z = self._gl_dir(fit['normal'])
                c = self._voxel_to_gl(fit['point']).astype(np.float64)
                e1, e2 = self._perp_basis(z)
                du = (gpts - c) @ e1
                dv = (gpts - c) @ e2
                hu = float(np.max(np.abs(du))) + pad if len(gpts) else 10.0
                hv = float(np.max(np.abs(dv))) + pad if len(gpts) else 10.0
                corners = np.array([
                    c + e1 * hu + e2 * hv, c - e1 * hu + e2 * hv,
                    c - e1 * hu - e2 * hv, c + e1 * hu - e2 * hv,
                    c + e1 * hu + e2 * hv], dtype=np.float32)
                items.append(gl.GLLinePlotItem(pos=corners, color=yellow, width=2,
                                               antialias=True, mode='line_strip'))
            elif geom_type == 'Cylinder':
                a = self._gl_dir(fit['axis'])
                c = self._voxel_to_gl(fit['point']).astype(np.float64)
                r = float(fit['radius']) / factor
                e1, e2 = self._perp_basis(a)
                ts = (gpts - c) @ a if len(gpts) else np.array([-10.0, 10.0])
                tmin, tmax = float(ts.min()) - pad, float(ts.max()) + pad
                for tt in (tmin, tmax):
                    ring = self._circle_strip(c + tt * a, e1, e2, r)
                    items.append(gl.GLLinePlotItem(pos=ring, color=yellow, width=2,
                                                   antialias=True, mode='line_strip'))
                for k in range(6):
                    ang = 2.0 * np.pi * k / 6.0
                    dirv = np.cos(ang) * e1 + np.sin(ang) * e2
                    seg = np.array([c + tmin * a + r * dirv,
                                    c + tmax * a + r * dirv], dtype=np.float32)
                    items.append(gl.GLLinePlotItem(pos=seg, color=yellow, width=1,
                                                   antialias=True, mode='lines'))
            elif geom_type == 'Cone':
                a = self._gl_dir(fit['axis'])
                v = self._voxel_to_gl(fit['apex']).astype(np.float64)
                e1, e2 = self._perp_basis(a)
                ts = (gpts - v) @ a if len(gpts) else np.array([10.0, 40.0])
                tmax = float(ts.max()) + pad
                rbase = max(tmax, 0.0) * math.tan(float(fit['half_angle']))
                base_c = v + tmax * a
                ring = self._circle_strip(base_c, e1, e2, rbase)
                items.append(gl.GLLinePlotItem(pos=ring, color=yellow, width=2,
                                               antialias=True, mode='line_strip'))
                for k in range(8):
                    ang = 2.0 * np.pi * k / 8.0
                    edge = base_c + rbase * (np.cos(ang) * e1 + np.sin(ang) * e2)
                    seg = np.array([v, edge], dtype=np.float32)
                    items.append(gl.GLLinePlotItem(pos=seg, color=yellow, width=1,
                                                   antialias=True, mode='lines'))
        except Exception:
            pass
        return items

    def _voxel_to_gl(self, voxel_pt):
        """Convert voxel coords to GL world coords (accounting for downsampling)."""
        p = np.array(voxel_pt, dtype=np.float32) / max(self._render_factor, 1)
        offset = np.array(self._render_shape, dtype=np.float32) / 2.0
        return p - offset

    def _ray_from_screen(self, ndc_x, ndc_y):
        """Return (cam_origin, ray_dir) in GL world space for the given NDC coords.

        Primary path: uses pyqtgraph's viewMatrix() + projectionMatrix(region, vp)
        (correct for both euler and quaternion camera modes).
        Fallback: manual computation from opts['azimuth'/'elevation'] with the
        formulas that match pyqtgraph 0.14's viewMatrix convention exactly.
        """
        # ── Primary: derive ray from the actual GL matrices ──────────────────
        try:
            vm = self.gl_view.viewMatrix()
            vp = self.gl_view.getViewport()          # (x0, y0, w, h)
            try:
                pm = self.gl_view.projectionMatrix(vp, vp)   # pyqtgraph ≥ 0.13
            except TypeError:
                pm = self.gl_view.projectionMatrix()          # very old pyqtgraph

            mvp = pm * vm
            inv_result = mvp.inverted()
            mvp_inv = inv_result[0] if isinstance(inv_result, tuple) else inv_result

            def _unproject(ndcz):
                # Full 4×4 multiply + perspective division — QMatrix4x4.map(QVector3D)
                # may skip the division when w≠1, so we do it explicitly.
                vals = [0.0, 0.0, 0.0, 0.0]
                for ri in range(4):
                    row = mvp_inv.row(ri)
                    vals[ri] = (row.x() * ndc_x + row.y() * ndc_y
                                + row.z() * ndcz  + row.w())
                w = vals[3]
                if abs(w) < 1e-10:
                    raise ValueError('degenerate w')
                return np.array([vals[0] / w, vals[1] / w, vals[2] / w])

            near_pt = _unproject(-1.0)
            far_pt  = _unproject( 1.0)
            ray = far_pt - near_pt
            rlen = np.linalg.norm(ray)
            if rlen < 1e-10:
                raise ValueError('degenerate ray')
            return near_pt, ray / rlen
        except AttributeError:
            pass   # viewMatrix()/getViewport() absent — fall through
        except Exception:
            pass

        # ── Fallback: manual camera math matching pyqtgraph 0.14 euler convention
        # viewMatrix = T(0,0,-dist)*Rx(el-90)*R(-z, az+90)*T(-center)
        # Camera pos  = center + dist*(cos_el·cos_az, cos_el·sin_az, sin_el)
        try:
            opts  = self.gl_view.opts
            dist  = float(opts['distance'])
            az    = math.radians(float(opts.get('azimuth',   45.0)))
            el    = math.radians(float(opts.get('elevation', 30.0)))
            ctr   = opts['center']
            try:
                if callable(getattr(ctr, 'x', None)):
                    cx, cy, cz = float(ctr.x()), float(ctr.y()), float(ctr.z())
                elif hasattr(ctr, 'x'):
                    cx, cy, cz = float(ctr.x), float(ctr.y), float(ctr.z)
                else:
                    cx, cy, cz = float(ctr[0]), float(ctr[1]), float(ctr[2])
            except Exception:
                cx = cy = cz = 0.0

            cos_el, sin_el = math.cos(el), math.sin(el)
            cos_az, sin_az = math.cos(az), math.sin(az)

            cam   = np.array([cx + dist*cos_el*cos_az,
                              cy + dist*cos_el*sin_az,
                              cz + dist*sin_el])
            right = np.array([-sin_az,          cos_az,         0.0    ])
            up    = np.array([-sin_el*cos_az,  -sin_el*sin_az,  cos_el ])
            look  = np.array([-cos_el*cos_az,  -cos_el*sin_az, -sin_el ])

            w = self.gl_view.width()
            h = self.gl_view.height()
            tan_h = math.tan(math.radians(float(opts.get('fov', 60.0))) / 2.0)
            tan_v = tan_h * h / w   # pyqtgraph fov is horizontal; vert scales by h/w

            ray = look + right * (ndc_x * tan_h) + up * (ndc_y * tan_v)
            rlen = np.linalg.norm(ray)
            if rlen < 1e-10:
                return None, None
            return cam, ray / rlen
        except Exception:
            return None, None

    def _pick_volume_point(self, screen_pos):
        """Raymarch through the rendered volume and return voxel coords of the first
        visible surface hit (iso threshold in Isosurface mode, AABB entry otherwise)."""
        if self.volume_data is None or self.volume_data.volume is None:
            return None
        try:
            w = self.gl_view.width()
            h = self.gl_view.height()
            if w == 0 or h == 0:
                return None
            sx, sy = float(screen_pos.x()), float(screen_pos.y())
            ndc_x  =  (2.0 * sx / w) - 1.0
            ndc_y  = 1.0 - (2.0 * sy / h)

            cam, ray = self._ray_from_screen(ndc_x, ndc_y)
            if cam is None:
                return None

            # ── Ray-AABB intersection with the rendered (downsampled) volume box ──
            shape = np.array(self._render_shape, dtype=np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                t1 = np.where(np.abs(ray) > 1e-12, (-shape / 2.0 - cam) / ray, -1e18)
                t2 = np.where(np.abs(ray) > 1e-12, ( shape / 2.0 - cam) / ray,  1e18)
            t_near = float(np.max(np.minimum(t1, t2)))
            t_far  = float(np.min(np.maximum(t1, t2)))
            if t_near > t_far or t_far < 0:
                return None
            t_start = max(t_near, 0.0)

            # ── Isosurface mode: march along the ray at 0.5-voxel steps ─────────
            volume = self._render_volume
            if self.mode == 'Isosurface' and volume is not None:
                v_min = self._render_vol_min
                v_max = self._render_vol_max
                if v_max > v_min:
                    threshold = v_min + self.iso_threshold_percent / 100.0 * (v_max - v_min)
                    step = 0.5
                    ts = np.arange(t_start, t_far + step, step)
                    if ts.size > 0:
                        pts = cam + ray * ts[:, np.newaxis]
                        pts += shape / 2.0
                        xi = np.clip(pts[:, 0].astype(np.intp), 0, volume.shape[0] - 1)
                        yi = np.clip(pts[:, 1].astype(np.intp), 0, volume.shape[1] - 1)
                        zi = np.clip(pts[:, 2].astype(np.intp), 0, volume.shape[2] - 1)
                        vals = volume[xi, yi, zi]
                        hits = np.where(vals >= threshold)[0]
                        if hits.size > 0:
                            idx = hits[0]
                            factor = self._render_factor
                            raw_shape = self.volume_data.volume.shape
                            vx = int(np.clip(int(xi[idx]) * factor, 0, raw_shape[0] - 1))
                            vy = int(np.clip(int(yi[idx]) * factor, 0, raw_shape[1] - 1))
                            vz = int(np.clip(int(zi[idx]) * factor, 0, raw_shape[2] - 1))
                            return (vx, vy, vz)
                # Isosurface miss — fall through to AABB entry as best-effort

            # ── Phong Volume mode or iso miss: use AABB entry point ───────────────
            hit_gl  = cam + t_start * ray
            hit_vox = (hit_gl + shape / 2.0) * self._render_factor
            raw_shape = self.volume_data.volume.shape
            vx = int(np.clip(hit_vox[0], 0, raw_shape[0] - 1))
            vy = int(np.clip(hit_vox[1], 0, raw_shape[1] - 1))
            vz = int(np.clip(hit_vox[2], 0, raw_shape[2] - 1))
            return (vx, vy, vz)
        except Exception as exc:
            print('3D pick error:', exc)
            return None

    def on_mode_changed(self, text):
        self.mode = text
        if text == 'Off':
            self.iso_slider.setEnabled(False)
            self.iso_spinbox.setEnabled(False)
            self._showing_surface = False
            self._render_off()
            return
        if text == 'Surface Mesh':
            self.iso_slider.setEnabled(False)
            self.iso_spinbox.setEnabled(False)
            self._showing_surface = True
            # Place the surface in the isosurface's (mode-independent) frame.
            self._set_render_frame(self._surface_render_voxels())
            self._render_surface(reset_camera=False)   # keep the volume's view
            return
        iso_active = (text == 'Isosurface')
        self.iso_slider.setEnabled(iso_active)
        self.iso_spinbox.setEnabled(iso_active)
        self._showing_surface = False
        self.update_view()

    def on_iso_slider_changed(self, value):
        self.iso_threshold_percent = value
        self.iso_spinbox.blockSignals(True)
        self.iso_spinbox.setValue(value)
        self.iso_spinbox.blockSignals(False)
        if self.mode == 'Isosurface':
            self.update_view()

    def set_volume(self, volume_data):
        self.volume_data = volume_data
        self._perm_volume = None
        self._display_levels = None
        self._clips = {}
        self.update_view()

    def set_surface_available(self, available):
        """Add/remove the 'Surface Mesh' option in the renderer dropdown. It is
        only present while at least one surface exists under the volume."""
        i = self.mode_combo.findText('Surface Mesh')
        if available and i < 0:
            # Keep 'Off' last: insert Surface Mesh just before it.
            off = self.mode_combo.findText('Off')
            self.mode_combo.insertItem(off if off >= 0 else self.mode_combo.count(),
                                       'Surface Mesh')
        elif not available and i >= 0:
            if self.mode_combo.currentText() == 'Surface Mesh':
                self.clear_surface()                  # revert to the volume first
            i = self.mode_combo.findText('Surface Mesh')
            if i >= 0:
                self.mode_combo.removeItem(i)

    def show_surface(self, verts, faces, color=None, source_shape=None):
        """Load a surface mesh and switch the renderer to 'Surface Mesh'.

        ``source_shape`` is the shape of the (full-resolution) volume the surface
        verts live in; it places the mesh in the same coordinate frame as the
        volume/isosurface so the two coincide in space."""
        verts = np.asarray(verts, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        self._surface = (verts, faces)
        self._surface_shape = tuple(source_shape) if source_shape is not None else None
        # Precompute triangle vertices once for the fast 2D slice-outline.
        self._surface_tri = verts[faces] if len(faces) else None
        if color is not None:
            self._surface_color = color
        self.set_surface_available(True)
        self.mode = 'Surface Mesh'
        self._showing_surface = True
        self.iso_slider.setEnabled(False)
        self.iso_spinbox.setEnabled(False)
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentText('Surface Mesh')
        self.mode_combo.blockSignals(False)
        # Establish the isosurface's (mode-independent) render frame so the mesh
        # is placed/scaled identically every time — including after a project
        # reload, where the saved camera is stored against this exact frame.
        self._set_render_frame(self._surface_render_voxels())
        # Don't reset the camera — keep exactly the volume's view so switching
        # renderers doesn't move the object.
        self._render_surface(reset_camera=False)

    def clear_surface(self):
        """Stop showing a surface and return to the volume display."""
        self._showing_surface = False
        self._surface = None
        self._surface_tri = None
        if self.mode == 'Surface Mesh':
            self.mode = 'Isosurface'
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentText('Isosurface')
            self.mode_combo.blockSignals(False)
            self.iso_slider.setEnabled(True)
            self.iso_spinbox.setEnabled(True)
        self.update_view()

    def _render_surface(self, reset_camera=False):
        self.gl_view.clear()
        self._alignment_gl_items = []
        self._plane_items = []
        self.status_text.setText('')
        if not self._surface:
            return
        verts, faces = self._surface
        if verts is None or len(verts) == 0 or len(faces) == 0:
            self.status_text.setText('Empty surface')
            return
        verts = np.asarray(verts, dtype=np.float64)
        rshape = np.asarray(self._render_shape, dtype=np.float64)
        sshape = (np.asarray(self._surface_shape, dtype=np.float64)
                  if self._surface_shape is not None else None)
        if sshape is not None and np.all(sshape > 0) and np.all(rshape > 1):
            # Same coordinate frame as the volume/isosurface: map the full-res
            # source-voxel coords into the downsampled render frame, centred like
            # render_isosurface (which translates by -render_shape/2).
            gl_verts = (verts * (rshape / sshape) - rshape / 2.0).astype(np.float32)
        else:
            lo = verts.min(axis=0)
            hi = verts.max(axis=0)
            gl_verts = (verts - (lo + hi) / 2.0).astype(np.float32)
        md = gl.MeshData(vertexes=gl_verts, faces=np.asarray(faces))
        mesh = gl.GLMeshItem(meshdata=md, smooth=True, shader='shaded',
                             drawEdges=False, glOptions='opaque',
                             color=self._surface_color)
        self.gl_view.addItem(mesh)
        if reset_camera:
            self.gl_view.setCameraPosition(distance=max(self._render_shape) * 2.5)

    def set_levels(self, min_val, max_val):
        """Set the histogram window applied to the rendered volume."""
        self._display_levels = (float(min_val), float(max_val))

    def set_permanent_volume(self, vol):
        """Show a pre-aligned display volume without overwriting the data volume."""
        self._perm_volume = vol
        self.update_view()

    def clear_permanent_volume(self):
        self._perm_volume = None
        self.update_view()

    def _max_render_voxels(self):
        # Linear resolution per axis relative to Low: Default = 2×, High = 4×.
        base = {'Low': 256, 'Default': 512, 'High': 1024}.get(self._quality, 512)
        if self.mode == 'Isosurface':
            # Marching-cubes mesh size scales with surface area (~resolution²) and
            # can exhaust RAM on large, detailed volumes, so cap the isosurface
            # working resolution. The Phong volume mode keeps the full budget.
            base = min(base, 256)
        return base ** 3

    def _show_context_menu(self, pos):
        menu = QtWidgets.QMenu(self.gl_view)
        quality_menu = menu.addMenu('Quality')
        group = QtWidgets.QActionGroup(quality_menu)
        group.setExclusive(True)
        for label in ('Low', 'Default', 'High'):
            act = quality_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(self._quality == label)
            group.addAction(act)
            act.triggered.connect(lambda _checked=False, l=label: self._set_quality(l))
        menu.exec(self.gl_view.mapToGlobal(pos))

    def _set_quality(self, quality):
        if quality == self._quality:
            return
        self._quality = quality
        self.update_view()

    def set_clips(self, clips):
        """Set active clipping planes: {axis(0/1/2): ('left'|'right', fraction)}.

        Re-renders the cached volume (debounced) without disturbing the camera.
        """
        self._clips = dict(clips)
        # Clipping only affects the volume / isosurface render. While a Surface
        # Mesh is shown (or the renderer is Off) there is nothing to re-clip, and
        # the debounced re-render would otherwise rebuild the whole surface mesh
        # on every slice scroll / locked pan — recomputing normals + re-uploading
        # the VBO for a multi-million-triangle mesh — which freezes the UI. Skip
        # all of it in those modes.
        inactive_mode = self._showing_surface or self.mode == 'Off'
        active = bool(self._clips) and not inactive_mode
        self.clip_hint.setVisible(active)
        if active:
            self._reposition_clip_hint()
        if self._render_volume is None or inactive_mode:
            return
        if not self._clip_timer.isActive():
            self._clip_timer.start(80)

    # ── Coordinate planes (3D analogue of the 2D coordinate lines) ────────────
    # Plane colours match the 2D slice/line colours: the YZ plane (fixed X) is
    # red, the XZ plane (fixed Y) green, the XY plane (fixed Z) blue.
    _PLANE_COLORS = {0: (217, 74, 74), 1: (74, 181, 74), 2: (74, 144, 217)}

    def set_coord_planes(self, fracs, rects=None):
        """Show coordinate planes at the given slice fractions {axis: frac}.

        ``rects`` optionally gives each plane's in-plane extent as
        {axis: {in-plane axis: (lo, hi)}} in normalised [0,1] volume coords, so
        the plane tracks the zoom/pan/FOV of the matching 2D viewport.
        """
        self._plane_fracs = dict(fracs)
        self._plane_rects = dict(rects or {})
        self._draw_coord_planes()

    def set_planes_visible(self, visible):
        """Enable/disable the coordinate planes."""
        self._show_planes = bool(visible)
        _set_button_active(self.planes_button, bool(visible))
        self._draw_coord_planes()

    def _draw_coord_planes(self):
        for it in self._plane_items:
            try:
                self.gl_view.removeItem(it)
            except Exception:
                pass
        self._plane_items = []
        # 'Off' renderer hides everything for this object, navigation planes too.
        if self.mode == 'Off':
            return
        if not self._show_planes or self._render_volume is None or not self._plane_fracs:
            return
        shape = self._render_shape
        # A depth pre-pass (write only the nearest plane's depth, no colour)
        # followed by an equal-depth colour pass means exactly ONE plane — the
        # nearest at each pixel — colours that pixel. So intersecting planes
        # never blend (no red+green=yellow), regardless of viewing angle.
        # A depth-buffer clear isolates this from the volume/surface depth:
        #   Isosurface: planes drawn BEFORE the surface, then depth cleared, so
        #     the opaque surface always ends up in front of them.
        #   Phong: volume drawn first, depth cleared, then planes over it (faint,
        #     so the opaque volume still reads through).
        phong = self.mode != 'Isosurface'
        fill_alpha = 0.10 if phong else 0.22
        if phong:
            clear_dv, depth_dv, color_dv = 1, 2, 3          # after the volume (0)
        else:
            depth_dv, color_dv, clear_dv = -20, -15, -10    # before the surface (0)

        clear = _GLDepthClear()
        clear.setDepthValue(clear_dv)
        self.gl_view.addItem(clear)
        self._plane_items.append(clear)

        for ax, frac in self._plane_fracs.items():
            sa = shape[ax]
            pos = frac * (sa - 1) - sa / 2.0
            b, c = [i for i in (0, 1, 2) if i != ax]

            # In-plane extent: the visible window of the matching 2D viewport
            # (normalised [0,1]), defaulting to the full volume extent. Centred
            # on the volume so frac 0→-shape/2 and frac 1→+shape/2.
            rect = self._plane_rects.get(ax, {})
            blo_n, bhi_n = rect.get(b, (0.0, 1.0))
            clo_n, chi_n = rect.get(c, (0.0, 1.0))
            blo = (blo_n - 0.5) * shape[b]
            bhi = (bhi_n - 0.5) * shape[b]
            clo = (clo_n - 0.5) * shape[c]
            chi = (chi_n - 0.5) * shape[c]

            def corner(bv, cv):
                p = [0.0, 0.0, 0.0]
                p[ax] = pos
                p[b] = bv
                p[c] = cv
                return p

            verts = np.array([corner(blo, clo), corner(bhi, clo),
                              corner(bhi, chi), corner(blo, chi)], dtype=np.float32)
            faces = np.array([[0, 1, 2], [0, 2, 3]])
            r, g, bl = self._PLANE_COLORS[ax]
            color = (r / 255.0, g / 255.0, bl / 255.0, fill_alpha)

            # Pass 1: depth only (no colour) — records the nearest plane depth.
            depth_md = gl.MeshData(vertexes=verts, faces=faces)
            depth_plane = gl.GLMeshItem(meshdata=depth_md, smooth=False, drawEdges=False)
            depth_plane.setGLOptions(_PLANE_DEPTH_GLOPTS)
            depth_plane.setDepthValue(depth_dv)
            self.gl_view.addItem(depth_plane)
            self._plane_items.append(depth_plane)

            # Pass 2: colour, equal-depth test — only the nearest plane draws.
            color_md = gl.MeshData(vertexes=verts, faces=faces)
            color_plane = gl.GLMeshItem(meshdata=color_md, smooth=False,
                                        drawEdges=False, color=color)
            color_plane.setGLOptions(_PLANE_COLOR_GLOPTS)
            color_plane.setDepthValue(color_dv)
            self.gl_view.addItem(color_plane)
            self._plane_items.append(color_plane)

            # White frame, drawn on top (no depth test) so it shows through the
            # volume to convey the plane's full extent.
            loop = np.vstack([verts, verts[0]])
            frame = gl.GLLinePlotItem(pos=loop, color=(1, 1, 1, 1), width=2,
                                      antialias=True, mode='line_strip')
            frame.setGLOptions(_ONTOP_GLOPTS)
            frame.setDepthValue(20)
            self.gl_view.addItem(frame)
            self._plane_items.append(frame)

    def _clip_keep_mask(self, coords, shape):
        """Boolean keep-mask for points (Nx3 in voxel coords) under active clips.

        Clipping is display-only — it never alters the volume data, it just
        marks which points/voxels are shown.
        """
        keep = np.ones(len(coords), dtype=bool)
        for ax, (mode, frac) in self._clips.items():
            plane = frac * (shape[ax] - 1)
            if mode == 'left':        # hide below the plane
                keep &= coords[:, ax] >= plane
            elif mode == 'right':     # hide above the plane
                keep &= coords[:, ax] <= plane
        return keep

    def _render_off(self):
        """'Off' renderer: draw nothing for this object — no volume, isosurface,
        surface mesh, coordinate planes, or overlays appear in the 3D view."""
        self.gl_view.clear()
        self._alignment_gl_items = []
        self._plane_items = []
        self.status_text.setText('')

    def update_view(self):
        if self.mode == 'Off':
            self._render_off()
            return
        if self._showing_surface and self._surface is not None:
            self._render_surface(reset_camera=False)
            return
        if self.volume_data is None or self.volume_data.volume is None:
            self.gl_view.clear()
            self._alignment_gl_items = []
            self._plane_items = []
            self.status_text.setText('')
            self._render_factor = 1
            self._render_shape = (1, 1, 1)
            self._render_volume = None
            return
        self._set_render_frame(self._max_render_voxels())
        self._render_scene(reset_camera=True)

    def _set_render_frame(self, max_voxels):
        """Compute and cache the downsampled render volume + its shape/range for
        ``max_voxels`` (the world frame the 3D scene is placed in). Returns the
        render shape, or None if no volume is loaded."""
        if self.volume_data is None or self.volume_data.volume is None:
            return None
        # Use the permanent aligned display volume if one has been set.
        raw = self._perm_volume if self._perm_volume is not None else self.volume_data.volume
        # Downsample FIRST (keeps the peak RAM from astype small)
        if raw.size > max_voxels:
            factor = max(1, int(np.ceil((raw.size / max_voxels) ** (1.0 / 3.0))))
            raw = raw[::factor, ::factor, ::factor]
        else:
            factor = 1
        volume = raw if raw.dtype == np.float32 else raw.astype(np.float32)
        self._render_factor = factor
        self._render_shape = volume.shape
        self._render_volume = volume
        self._render_vol_min = float(np.min(volume))
        self._render_vol_max = float(np.max(volume))
        return self._render_shape

    def _surface_render_voxels(self):
        """Voxel budget for placing a created Surface Mesh. Deliberately the same
        budget the isosurface uses (its 256³ cap), so a surface sits in exactly
        the isosurface's world frame AND that frame never depends on which volume
        render last ran. Without this the surface scale (and the saved camera that
        is stored against it) would shift between Isosurface and Phong, which broke
        surfaces reloaded from a saved project."""
        base = {'Low': 256, 'Default': 512, 'High': 1024}.get(self._quality, 512)
        return min(base, 256) ** 3

    def _render_scene(self, reset_camera=False):
        """(Re)build the 3D scene from the cached render volume + active clips."""
        if self.mode == 'Off':
            self._render_off()
            return
        # A created surface owns the 3D view; never let a volume re-render (e.g.
        # the debounced clip-timer, which calls _render_scene directly, bypassing
        # update_view's guard) clobber it with the gray volume/isosurface.
        if self._showing_surface and self._surface is not None:
            self._render_surface(reset_camera=reset_camera)
            return
        self.gl_view.clear()
        self._alignment_gl_items = []
        self._plane_items = []        # clear() removed them; _draw re-adds below
        self.status_text.setText('')
        volume = self._render_volume
        if volume is None:
            return
        if self.mode == 'Isosurface':
            self.render_isosurface(volume)
        else:
            self.render_volume(volume)
        if reset_camera:
            self.gl_view.setCameraPosition(distance=max(self._render_shape) * 2.5)
        self._rebuild_alignment_overlays()
        self._draw_coord_planes()

    def render_isosurface(self, volume, translate=None):
        if measure is None:
            self.status_text.setText('Install scikit-image for ISO surface rendering')
            return
        if volume.ndim != 3 or any(dim < 2 for dim in volume.shape):
            self.status_text.setText('3D surface unavailable for this volume')
            return
        # Threshold is ALWAYS computed from the full volume's global range, so
        # clipping (which only hides geometry) never shifts the isovalue.
        volume_min, volume_max = self._render_vol_min, self._render_vol_max
        if volume_max <= volume_min:
            self.status_text.setText('No surface available for constant volume')
            return
        threshold = volume_min + self.iso_threshold_percent / 100.0 * (volume_max - volume_min)
        if not (volume_min < threshold < volume_max):
            self.status_text.setText('ISO threshold out of range')
            return

        # Build the full-volume mesh once per (volume, threshold); slice-drag
        # clipping then just re-filters the cached mesh.
        key = (id(volume), round(threshold, 6))
        if self._iso_cache_key != key:
            try:
                verts, faces, _normals, _ = measure.marching_cubes(volume, level=threshold)
            except MemoryError:
                self.status_text.setText('Surface too large to build at this threshold')
                return
            except Exception as exc:
                print('Volume surface error:', exc)
                self.status_text.setText('No iso-surface found at this threshold')
                return
            if len(verts) > 12_000_000:   # guard against a pathological mesh
                self.status_text.setText('Surface too complex to display — adjust the isovalue')
                self._iso_cache = None
                self._iso_cache_key = None
                return
            self._iso_cache = (verts, faces)
            self._iso_cache_key = key

        if self._iso_cache is None:
            return
        verts, faces = self._iso_cache

        # Hide (don't cut) the clipped part by dropping geometry beyond the plane.
        if self._clips:
            keep = self._clip_keep_mask(verts, volume.shape)
            if not keep.any():
                return
            if not keep.all():
                remap = np.cumsum(keep) - 1
                fmask = keep[faces].all(axis=1)
                faces = remap[faces[fmask]]
                verts = verts[keep]
            if len(faces) == 0:
                return

        meshdata = gl.MeshData(vertexes=verts, faces=faces)
        mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, shader='shaded',
                             drawEdges=False, glOptions='opaque')
        if translate is None:
            translate = (-volume.shape[0] / 2.0, -volume.shape[1] / 2.0, -volume.shape[2] / 2.0)
        mesh.translate(*translate)
        self.gl_view.addItem(mesh)

    def _build_rgba(self, volume):
        """Full-volume RGBA from the global window mapping (no clipping applied)."""
        vol_min, vol_max = self._render_vol_min, self._render_vol_max
        if vol_max <= vol_min:
            return None
        # Map the histogram window [lo, hi] to [0, 1]; values outside the window
        # clamp so they become fully transparent / fully opaque.
        if self._display_levels is not None:
            lo, hi = self._display_levels
        else:
            lo, hi = vol_min, vol_max
        if hi <= lo:
            hi = lo + 1e-6
        normalized = np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

        # Surface normals via volume gradient for Phong-style shading. Compute
        # one axis at a time, casting to float32, to limit peak RAM.
        gx = np.gradient(normalized, axis=0).astype(np.float32)
        gy = np.gradient(normalized, axis=1).astype(np.float32)
        gz = np.gradient(normalized, axis=2).astype(np.float32)
        gnorm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2).astype(np.float32) + 1e-8
        nx = gx / gnorm; del gx
        ny = gy / gnorm; del gy
        nz = gz / gnorm; del gz, gnorm
        diffuse = np.clip(nx * 0.577 + ny * 0.577 + nz * 0.577, 0.0, 1.0).astype(np.float32)
        del nx, ny, nz
        shading = np.clip(0.35 + 0.65 * diffuse, 0.0, 1.0).astype(np.float32)
        del diffuse

        rgba = np.zeros(normalized.shape + (4,), dtype=np.uint8)
        gray = np.clip(normalized * shading * 255.0, 0, 255).astype(np.uint8)
        del shading
        rgba[..., 0] = gray
        rgba[..., 1] = gray
        rgba[..., 2] = gray
        # Opacity ramp: a gamma < 1 boost makes the material read clearly/solidly
        # (dense voxels become fully opaque) while low intensity (air) stays
        # transparent.
        rgba[..., 3] = np.clip(np.power(normalized, 0.6) * 300.0, 0, 255).astype(np.uint8)
        return rgba

    def render_volume(self, volume, translate=None):
        try:
            if volume.ndim != 3:
                self.status_text.setText('Volume rendering requires a 3D scalar volume')
                return
            # Build the full-volume RGBA once per (volume, window); clipping then
            # just hides voxels by zeroing their alpha (data is untouched).
            key = (id(volume), self._display_levels)
            if self._rgba_cache_key != key or self._rgba_base is None:
                self._rgba_base = self._build_rgba(volume)
                self._rgba_cache_key = key
            if self._rgba_base is None:
                self.status_text.setText('Volume rendering requires a non-constant volume')
                return

            rgba = self._rgba_base
            if self._clips:
                rgba = rgba.copy()
                for ax, (mode, frac) in self._clips.items():
                    n = rgba.shape[ax]
                    idx = int(round(frac * (n - 1)))
                    sl = [slice(None)] * 3
                    if mode == 'left':        # hide below the plane
                        sl[ax] = slice(0, idx)
                    elif mode == 'right':     # hide above the plane
                        sl[ax] = slice(idx + 1, n)
                    rgba[tuple(sl) + (3,)] = 0     # alpha → 0 (future: semi-transparent)

            vol_item = gl.GLVolumeItem(rgba, sliceDensity=1, smooth=True,
                                       glOptions='translucent')
            if translate is None:
                translate = (-rgba.shape[0] / 2.0, -rgba.shape[1] / 2.0, -rgba.shape[2] / 2.0)
            vol_item.translate(*translate)
            self.gl_view.addItem(vol_item)
        except Exception as exc:
            print('Volume render failed:', exc)
            self.status_text.setText('Volume render failed')

    def update_preview(self, volume):
        """Rebuild the 3D view with a given volume array without resetting the camera."""
        if volume is None:
            return
        saved_opts = {k: v for k, v in self.gl_view.opts.items()}
        self.gl_view.clear()
        self._alignment_gl_items = []
        raw = volume if volume.dtype == np.float32 else volume.astype(np.float32)
        max_voxels = self._max_render_voxels()
        if raw.size > max_voxels:
            factor = max(1, int(np.ceil((raw.size / max_voxels) ** (1.0 / 3.0))))
            vol_ds = raw[::factor, ::factor, ::factor]
        else:
            vol_ds = raw
        # The render methods read the global range from these fields.
        self._render_vol_min = float(np.min(vol_ds))
        self._render_vol_max = float(np.max(vol_ds))
        if self.mode == 'Isosurface':
            self.render_isosurface(vol_ds)
        else:
            self.render_volume(vol_ds)
        # Restore camera (no setCameraPosition — keeps current view angle)
        for k, v in saved_opts.items():
            self.gl_view.opts[k] = v
        self.gl_view.update()


class SimpleAlignmentDialog(QDialog):
    """Non-modal Simple Alignment panel with real-time preview."""
    preview_changed   = Signal(object)   # emits (R 3x3, offset (3,)) on every slider move
    alignment_applied = Signal(object)   # emits (R 3x3, offset (3,)) on Apply
    alignment_cancelled = Signal()

    _PREVIEW_THROTTLE_MS = 40   # cap live-preview rate; always emits the final value

    def __init__(self, vol_shape, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Simple Alignment')
        self.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self._vol_shape = vol_shape
        self._preview_pending = False
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._on_preview_timeout)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(2, 48)

        def make_row(label, min_v, max_v, unit):
            lbl = QLabel(label)
            sl  = QSlider(Qt.Horizontal)
            sl.setMinimum(min_v)
            sl.setMaximum(max_v)
            sl.setValue(0)
            val_lbl = QLabel(f'0 {unit}')
            val_lbl.setFixedWidth(60)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return sl, lbl, val_lbl

        self.rot_x,   lbl_rx, self.val_rx = make_row('Rotate X:',    -180, 180, '°')
        self.rot_y,   lbl_ry, self.val_ry = make_row('Rotate Y:',    -180, 180, '°')
        self.rot_z,   lbl_rz, self.val_rz = make_row('Rotate Z:',    -180, 180, '°')
        self.trans_x, lbl_tx, self.val_tx = make_row('Translate X:', -200, 200, 'vx')
        self.trans_y, lbl_ty, self.val_ty = make_row('Translate Y:', -200, 200, 'vx')
        self.trans_z, lbl_tz, self.val_tz = make_row('Translate Z:', -200, 200, 'vx')

        rows = [
            (lbl_rx, self.rot_x,   self.val_rx, '°'),
            (lbl_ry, self.rot_y,   self.val_ry, '°'),
            (lbl_rz, self.rot_z,   self.val_rz, '°'),
            (lbl_tx, self.trans_x, self.val_tx, 'vx'),
            (lbl_ty, self.trans_y, self.val_ty, 'vx'),
            (lbl_tz, self.trans_z, self.val_tz, 'vx'),
        ]
        for r, (lbl, sl, val_lbl, unit) in enumerate(rows):
            grid.addWidget(lbl,     r, 0)
            grid.addWidget(sl,      r, 1)
            grid.addWidget(val_lbl, r, 2)
            sl.valueChanged.connect(self._on_slider_changed)
            sl.valueChanged.connect(
                lambda v, vl=val_lbl, u=unit: vl.setText(f'{v} {u}')
            )

        layout.addLayout(grid)

        btn_row = QHBoxLayout()
        self.apply_btn  = QPushButton('Apply')
        self.reset_btn  = QPushButton('Reset')
        self.cancel_btn = QPushButton('Cancel')
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.reset_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.cancel_btn)
        layout.addLayout(btn_row)

        self.apply_btn.clicked.connect(self._on_apply)
        self.reset_btn.clicked.connect(self._on_reset)
        self.cancel_btn.clicked.connect(self._on_cancel)

    def _current_transform(self):
        """Return (R 3x3, offset (3,)) for the current slider values."""
        rx = math.radians(self.rot_x.value())
        ry = math.radians(self.rot_y.value())
        rz = math.radians(self.rot_z.value())
        tx = float(self.trans_x.value())
        ty = float(self.trans_y.value())
        tz = float(self.trans_z.value())
        cx = math.cos(rx); sx = math.sin(rx)
        cy = math.cos(ry); sy = math.sin(ry)
        cz = math.cos(rz); sz = math.sin(rz)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float64)
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float64)
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float64)
        R   = Rz @ Ry @ Rx
        # Rotate around volume centre; translation is additional shift
        ctr = np.array(self._vol_shape, dtype=np.float64) / 2.0
        offset = ctr - R @ ctr + np.array([tx, ty, tz], dtype=np.float64)
        return R, offset

    def _on_slider_changed(self):
        # Throttle: emit immediately, then at most once per interval, always
        # delivering the latest value so the preview never lags behind a drag.
        if self._preview_timer.isActive():
            self._preview_pending = True
            return
        self._fire_preview()
        self._preview_timer.start(self._PREVIEW_THROTTLE_MS)

    def _fire_preview(self):
        self._preview_pending = False
        R, off = self._current_transform()
        self.preview_changed.emit((R, off))

    def _on_preview_timeout(self):
        if self._preview_pending:
            self._fire_preview()
            self._preview_timer.start(self._PREVIEW_THROTTLE_MS)

    def _on_apply(self):
        self._preview_timer.stop()
        R, off = self._current_transform()
        self.alignment_applied.emit((R, off))
        self.close()

    def _on_reset(self):
        for sl in (self.rot_x, self.rot_y, self.rot_z,
                   self.trans_x, self.trans_y, self.trans_z):
            sl.blockSignals(True)
            sl.setValue(0)
            sl.blockSignals(False)
        for vl, u in [(self.val_rx,'°'),(self.val_ry,'°'),(self.val_rz,'°'),
                      (self.val_tx,'vx'),(self.val_ty,'vx'),(self.val_tz,'vx')]:
            vl.setText(f'0 {u}')
        self._on_slider_changed()

    def _on_cancel(self):
        self._preview_timer.stop()
        self.alignment_cancelled.emit()
        self.close()

    def closeEvent(self, event):
        self.alignment_cancelled.emit()
        super().closeEvent(event)


# ── Geometry fitting (numpy-only least squares) ──────────────────────────────
# These power the Geometry-based alignment. They use only numpy.linalg (never
# scipy.linalg / scipy.optimize, which are excluded from the App Store bundle).

def _axis_from_angles(th, ph):
    st = math.sin(th)
    return np.array([st * math.cos(ph), st * math.sin(ph), math.cos(th)], dtype=np.float64)


def _angles_from_axis(a):
    a = np.asarray(a, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return 0.0, 0.0
    a = a / n
    th = math.acos(max(-1.0, min(1.0, a[2])))
    ph = math.atan2(a[1], a[0])
    return th, ph


def _lm_fit(residual_fn, params0, iters=60):
    """Levenberg–Marquardt least-squares with a finite-difference Jacobian
    (numpy only). Returns (params, cost) or (None, inf) on failure."""
    params = np.array(params0, dtype=np.float64)
    r = residual_fn(params)
    if r is None:
        return None, float('inf')
    cost = float(r @ r)
    lam = 1e-3
    eps = 1e-6
    for _ in range(iters):
        J = np.zeros((len(r), len(params)))
        for k in range(len(params)):
            dp = params.copy()
            dp[k] += eps
            rk = residual_fn(dp)
            if rk is None:
                return (params, cost)
            J[:, k] = (rk - r) / eps
        JTJ = J.T @ J
        JTr = J.T @ r
        improved = False
        for _try in range(10):
            try:
                step = np.linalg.solve(JTJ + lam * (np.diag(np.diag(JTJ)) + 1e-12 * np.eye(len(params))), -JTr)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            new = params + step
            rn = residual_fn(new)
            if rn is not None and float(rn @ rn) < cost:
                params, r, cost = new, rn, float(rn @ rn)
                lam = max(lam * 0.5, 1e-12)
                improved = True
                break
            lam *= 4.0
        if not improved or np.linalg.norm(step) < 1e-9:
            break
    return params, cost


def _fit_plane(pts):
    """Best-fit plane → {'point', 'normal'} or None (needs ≥3 points)."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return None
    c = pts.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vt[-1]
    nn = np.linalg.norm(normal)
    if nn < 1e-9:
        return None
    return {'type': 'plane', 'point': c, 'normal': normal / nn}


def _fit_cylinder(pts):
    """Least-squares cylinder → {'point' (on axis), 'axis', 'radius'} or None.
    Needs ≥6 points. Multi-start over the principal directions to beat the
    axis-orientation ambiguity."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 6:
        return None
    centroid = pts.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    def make_resid(c0):
        def resid(p):
            c = p[:3]
            a = _axis_from_angles(p[3], p[4])
            w = pts - c
            t = w @ a
            radial = w - np.outer(t, a)
            dist = np.linalg.norm(radial, axis=1)
            return dist - dist.mean()
        return resid

    best = None
    for a0 in (vt[0], vt[1], vt[2]):
        th, ph = _angles_from_axis(a0)
        resid = make_resid(centroid)
        params, cost = _lm_fit(resid, [centroid[0], centroid[1], centroid[2], th, ph])
        if params is None:
            continue
        a = _axis_from_angles(params[3], params[4])
        c = params[:3]
        c = c - np.dot(c - centroid, a) * a   # anchor the axis point near the data
        w = pts - c
        t = w @ a
        dist = np.linalg.norm(w - np.outer(t, a), axis=1)
        r = float(dist.mean())
        rms = float(np.sqrt(np.mean((dist - r) ** 2)))
        if r > 1e-6 and np.isfinite(rms) and (best is None or rms < best['rms']):
            best = {'type': 'cylinder', 'point': c, 'axis': a, 'radius': r, 'rms': rms}
    return best


def _fit_cone(pts):
    """Least-squares cone → {'apex', 'axis', 'half_angle'} or None. Needs ≥6
    points. ``axis`` points from the apex toward the open end."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 6:
        return None
    centroid = pts.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    extent = float(np.linalg.norm(pts - centroid, axis=1).max()) + 1e-6

    def resid(p):
        v = p[:3]
        a = _axis_from_angles(p[3], p[4])
        alpha = p[5]
        if not (1e-3 < alpha < math.pi / 2 - 1e-3):
            return None
        w = pts - v
        t = w @ a
        m = np.linalg.norm(w - np.outer(t, a), axis=1)
        # ~orthogonal distance to the cone surface
        return m * math.cos(alpha) - t * math.sin(alpha)

    best = None
    for sgn in (1.0, -1.0):
        a0 = sgn * vt[0]
        th, ph = _angles_from_axis(a0)
        # start the apex behind the data along the axis
        v0 = centroid - a0 * extent
        for alpha0 in (math.radians(20), math.radians(40)):
            params, cost = _lm_fit(resid, [v0[0], v0[1], v0[2], th, ph, alpha0])
            if params is None:
                continue
            r = resid(params)
            if r is None:
                continue
            rms = float(np.sqrt(np.mean(r ** 2)))
            a = _axis_from_angles(params[3], params[4])
            # orient axis from apex toward the points
            if np.dot(centroid - params[:3], a) < 0:
                a = -a
            alpha = float(params[5])
            if np.isfinite(rms) and (best is None or rms < best['rms']):
                best = {'type': 'cone', 'apex': params[:3].copy(),
                        'axis': a, 'half_angle': alpha, 'rms': rms}
    return best


def _frame_from_z(z, x_hint):
    """Right-handed frame (columns x, y, z) with the given z and an x as close
    to ``x_hint`` as possible (projected into the plane ⊥ z)."""
    z = np.asarray(z, dtype=np.float64)
    z = z / np.linalg.norm(z)
    x = np.asarray(x_hint, dtype=np.float64) - np.dot(x_hint, z) * z
    if np.linalg.norm(x) < 1e-6:
        alt = np.array([0.0, 1.0, 0.0]) if abs(z[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        x = alt - np.dot(alt, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    return np.column_stack([x, y, z]).astype(np.float64)


def _geometry_alignment_transform(geom_type, fit, pts):
    """Return (R, origin) for a geometry-based alignment, or None.

    Z is the plane normal / cylinder axis / cone axis. For a plane the in-plane
    X/Y come from the principal directions of the points projected onto the plane
    (its oriented bounding box). For a cylinder/cone X/Y are derived from the
    current object orientation (the world axis least parallel to Z), keeping the
    rotation about the axis minimal."""
    if fit is None:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    if geom_type == 'Plane':
        z = np.asarray(fit['normal'], dtype=np.float64)
        if z[2] < 0:                      # keep normal roughly toward +Z
            z = -z
        c = np.asarray(fit['point'], dtype=np.float64)
        centered = pts - c
        proj = centered - np.outer(centered @ z, z)
        try:
            _, _, vt = np.linalg.svd(proj, full_matrices=False)
            x_hint = vt[0]
        except np.linalg.LinAlgError:
            x_hint = np.array([1.0, 0.0, 0.0])
        R = _frame_from_z(z, x_hint)
        origin = c
    elif geom_type == 'Cylinder':
        z = np.asarray(fit['axis'], dtype=np.float64)
        if z[2] < 0:
            z = -z
        x_hint = np.array([0.0, 1.0, 0.0]) if abs(z[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        R = _frame_from_z(z, x_hint)
        c = np.asarray(fit['point'], dtype=np.float64)
        origin = c + np.dot(pts.mean(axis=0) - c, z) * z   # axis point nearest the data
    elif geom_type == 'Cone':
        z = np.asarray(fit['axis'], dtype=np.float64)
        if z[2] < 0:
            z = -z
        x_hint = np.array([0.0, 1.0, 0.0]) if abs(z[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        R = _frame_from_z(z, x_hint)
        origin = np.asarray(fit['apex'], dtype=np.float64)   # apex is the natural origin
    else:
        return None
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(origin)):
        return None
    return R, origin


_GEOM_MIN_POINTS = {'Plane': 3, 'Cylinder': 6, 'Cone': 6}


def _fit_geometry(geom_type, pts):
    if geom_type == 'Plane':
        return _fit_plane(pts)
    if geom_type == 'Cylinder':
        return _fit_cylinder(pts)
    if geom_type == 'Cone':
        return _fit_cone(pts)
    return None


def _compute_321_transform(plane_pts, line_pts, origin_pt):
    """Return (R, origin) for 3-2-1 alignment, or None if degenerate."""
    p0, p1, p2 = plane_pts.astype(np.float64)
    normal = np.cross(p1 - p0, p2 - p0)
    if np.linalg.norm(normal) < 1e-6:
        return None
    normal /= np.linalg.norm(normal)
    line_dir = (line_pts[1] - line_pts[0]).astype(np.float64)
    if np.linalg.norm(line_dir) < 1e-6:
        return None
    line_dir /= np.linalg.norm(line_dir)
    z_axis = normal
    x_axis = line_dir - np.dot(line_dir, z_axis) * z_axis  # project onto plane
    if np.linalg.norm(x_axis) < 1e-6:
        return None
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)
    return R, np.array(origin_pt, dtype=np.float64)


def _alignment_bbox(R, offset0, in_shape):
    """Recentre a scipy affine so the full rotated volume fits the output frame.

    The affine maps output→input as ``in = R @ out + offset0``. Returns
    ``(R, offset, out_shape)`` where ``out_shape`` is the bounding box of the
    rotated input (with 1-voxel padding) and ``offset`` places that box at the
    output origin. Used by both 3-2-1 (``offset0`` = picked origin point) and
    Simple Alignment (``offset0`` = the dialog's scipy offset)."""
    R       = np.asarray(R,       dtype=np.float64)
    offset0 = np.asarray(offset0, dtype=np.float64)
    in_shape = np.asarray(in_shape, dtype=np.float64)
    corners = np.array(
        [(i, j, k)
         for i in (0.0, in_shape[0] - 1)
         for j in (0.0, in_shape[1] - 1)
         for k in (0.0, in_shape[2] - 1)],
        dtype=np.float64,
    )
    # Output coords of each input corner: out = R^T @ (in - offset0) == (in - offset0) @ R
    u = (corners - offset0) @ R
    u_min = u.min(axis=0)
    u_max = u.max(axis=0)
    out_shape = tuple(max(1, int(np.ceil(u_max[i] - u_min[i])) + 2) for i in range(3))
    offset = offset0 + R @ (u_min - 1.0)
    return R, offset, out_shape


def _compose_alignment(R1, off1, R2, off2):
    """Compose a new output→display transform (R2, off2) under an existing
    display→input transform (R1, off1), giving a single output→input transform.

    in = R1 @ (R2 @ out + off2) + off1 = (R1 @ R2) @ out + (R1 @ off2 + off1)."""
    R1 = np.asarray(R1, dtype=np.float64)
    R2 = np.asarray(R2, dtype=np.float64)
    off1 = np.asarray(off1, dtype=np.float64)
    off2 = np.asarray(off2, dtype=np.float64)
    return R1 @ R2, R1 @ off2 + off1


class AlignmentDialog321(QDialog):
    """Non-modal step-by-step 3-2-1 alignment dialog."""

    _PICK_KEY = '⌘+click' if sys.platform == 'darwin' else 'Ctrl+click'
    _PHASES = [
        (3, 'Phase 1 — Plane',
         f'{_PICK_KEY} 3 points to define the alignment plane.'),
        (2, 'Phase 2 — Line',
         f'{_PICK_KEY} 2 points to define the reference direction.'),
        (1, 'Phase 3 — Origin',
         f'{_PICK_KEY} 1 point to define the origin.'),
    ]

    reset_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('3-2-1 Alignment')
        # Qt.Tool keeps the window floating above the parent app without
        # hiding when the user clicks back in the main viewport.
        self.setWindowFlags(
            Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint
        )
        self.setModal(False)
        self.points = []
        self._setup_ui()
        self._refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._phase_label = QLabel()
        self._phase_label.setAlignment(Qt.AlignCenter)
        font = self._phase_label.font()
        font.setPointSize(11)
        font.setBold(True)
        self._phase_label.setFont(font)
        layout.addWidget(self._phase_label)

        self._instruction_label = QLabel()
        self._instruction_label.setAlignment(Qt.AlignCenter)
        self._instruction_label.setWordWrap(True)
        layout.addWidget(self._instruction_label)

        self._dots_label = QLabel()
        self._dots_label.setAlignment(Qt.AlignCenter)
        dots_font = self._dots_label.font()
        dots_font.setPointSize(14)
        self._dots_label.setFont(dots_font)
        layout.addWidget(self._dots_label)

        layout.addSpacing(4)
        btn_row = QHBoxLayout()
        self._reset_btn = QPushButton('Reset')
        self._reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self._reset_btn)

        self._apply_btn = QPushButton('Apply')
        self._apply_btn.setDefault(True)
        self._apply_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._apply_btn)

        cancel_btn = QPushButton('Cancel')
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        self.setMinimumWidth(310)

    def _refresh(self):
        n = len(self.points)
        # Find which phase we're in
        cum = 0
        for idx, (count, name, instr) in enumerate(self._PHASES):
            if n < cum + count:
                self._phase_label.setText(name)
                self._instruction_label.setText(
                    f'{instr}\n({n - cum}/{count} placed)'
                )
                break
            cum += count
        else:
            self._phase_label.setText('Ready')
            self._instruction_label.setText('All 6 points placed.\nClick Apply to align.')

        # Build dot indicators for all three phases
        cum = 0
        parts = []
        for count, _, _ in self._PHASES:
            placed = min(max(n - cum, 0), count)
            parts.append('●' * placed + '○' * (count - placed))
            cum += count
        self._dots_label.setText('   '.join(parts))
        self._apply_btn.setEnabled(n >= 6)

    def add_point(self, point):
        if len(self.points) >= 6:
            return
        self.points.append(point)
        self._refresh()

    def _on_reset(self):
        self.points = []
        self._refresh()
        self.reset_requested.emit()

    def get_points(self):
        return list(self.points)


class GeometryAlignmentDialog(QDialog):
    """Non-modal Geometry-based alignment dialog: pick points on a Plane,
    Cylinder, or Cone surface; a Gauss (least-squares) fit drives the alignment.
    The fit/preview/Apply-enabled state is managed by MainWindow."""

    _PICK_KEY = '⌘+click' if sys.platform == 'darwin' else 'Ctrl+click'
    _MAX_POINTS = 50

    geometry_changed = Signal(str)
    reset_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Geometry-based Alignment')
        self.setWindowFlags(Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setModal(False)
        self.points = []
        self._setup_ui()
        self._refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        row = QHBoxLayout()
        row.addWidget(QLabel('Geometry:'))
        self.geom_combo = QComboBox()
        self.geom_combo.addItems(['Plane', 'Cylinder', 'Cone'])
        self.geom_combo.currentTextChanged.connect(self._on_geom_changed)
        row.addWidget(self.geom_combo, 1)
        layout.addLayout(row)

        self._instruction = QLabel()
        self._instruction.setWordWrap(True)
        self._instruction.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._instruction)

        self._status = QLabel()
        self._status.setAlignment(Qt.AlignCenter)
        sfont = self._status.font()
        sfont.setBold(True)
        self._status.setFont(sfont)
        layout.addWidget(self._status)

        layout.addSpacing(4)
        btn_row = QHBoxLayout()
        self._reset_btn = QPushButton('Reset')
        self._reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self._reset_btn)
        self._apply_btn = QPushButton('Apply')
        self._apply_btn.setDefault(True)
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._apply_btn)
        cancel_btn = QPushButton('Cancel')
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        self.setMinimumWidth(330)

    def geometry(self):
        return self.geom_combo.currentText()

    def _on_geom_changed(self, text):
        # Picked points are kept; MainWindow refits for the new geometry.
        self._refresh()
        self.geometry_changed.emit(text)

    def add_point(self, point):
        if len(self.points) >= self._MAX_POINTS:
            return
        self.points.append(point)
        self._refresh()

    def _on_reset(self):
        self.points = []
        self._refresh()
        self.reset_requested.emit()

    def get_points(self):
        return list(self.points)

    def set_apply_enabled(self, ok):
        self._apply_btn.setEnabled(bool(ok))

    def set_fit_status(self, text):
        self._status.setText(text)

    def _refresh(self):
        g = self.geometry()
        need = _GEOM_MIN_POINTS.get(g, 3)
        self._instruction.setText(
            f'{self._PICK_KEY} points on the {g.lower()} surface in the 3D view '
            f'(at least {need}, up to {self._MAX_POINTS}).\n{len(self.points)} placed.')


# Surface resolution presets → downsample factor per axis.
_SURFACE_PRESETS = {'High': 1, 'Medium': 2, 'Low': 4}
_SURFACE_COLOR = (0.78, 0.88, 1.0, 1.0)   # very light blue


def _write_stl(path, verts, faces):
    """Write a binary STL file from vertices + triangle faces (numpy only)."""
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    tris = verts[faces]                                  # (N, 3, 3)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, nlen, out=np.zeros_like(normals),
                        where=nlen > 0).astype(np.float32)
    n = int(tris.shape[0])
    rec = np.zeros(n, dtype=np.dtype([('n', '<f4', (3,)),
                                      ('v', '<f4', (3, 3)),
                                      ('a', '<u2')]))
    rec['n'] = normals
    rec['v'] = tris.astype(np.float32)
    with open(path, 'wb') as f:
        f.write(b'Voxels Viewer'.ljust(80, b'\x00'))
        f.write(np.array([n], dtype='<u4').tobytes())
        f.write(rec.tobytes())


def _read_stl(path):
    """Read a binary STL into (verts, faces). Each triangle keeps its own three
    vertices (no shared-vertex dedup), which is fine for display. Returns None
    if the file is missing or malformed."""
    try:
        with open(path, 'rb') as f:
            f.seek(80)
            head = f.read(4)
            if len(head) < 4:
                return None
            n = int(np.frombuffer(head, dtype='<u4')[0])
            data = f.read(n * 50)
        if n == 0 or len(data) < n * 50:
            return None
        rec = np.frombuffer(data, count=n, dtype=np.dtype(
            [('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('a', '<u2')]))
        verts = rec['v'].reshape(-1, 3).astype(np.float32)
        faces = np.arange(n * 3, dtype=np.int64).reshape(n, 3)
        return verts, faces
    except Exception:
        return None


class SurfaceCreationDialog(QDialog):
    """Pick a resolution preset for the surface mesh built from the isovalue."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Create Surface')
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Resolution preset:'))
        self.combo = QComboBox()
        self.combo.addItems(['Low', 'Medium', 'High'])
        self.combo.setCurrentText('Medium')
        layout.addWidget(self.combo)
        hint = QLabel('High ≈ one vertex per surface voxel.\n'
                      'Medium and Low use ~2× and ~4× fewer vertices per axis.')
        hint.setStyleSheet('color: gray;')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setMinimumWidth(320)

    def preset(self):
        return self.combo.currentText()


def _dtype_label(dt) -> str:
    """Human-readable data type, e.g. '16-bit int', '32-bit float'."""
    dt = np.dtype(dt)
    bits = dt.itemsize * 8
    if np.issubdtype(dt, np.floating):
        return f'{bits}-bit float'
    if np.issubdtype(dt, np.unsignedinteger):
        return f'{bits}-bit unsigned int'
    if np.issubdtype(dt, np.signedinteger):
        return f'{bits}-bit int'
    return str(dt)


def _format_bytes(nbytes: int) -> str:
    if nbytes >= 1024 ** 3:
        return f'{nbytes / 1024 ** 3:.2f} GB'
    return f'{nbytes / 1024 ** 2:.1f} MB'


def _suggest_isovalue_percent(histogram, bin_edges):
    """Suggest an isovalue, as a 0–100% position over the data range, placed
    midway between the two largest histogram peaks (typically the air and the
    material modes of a CT scan). Returns an int percent, or None if two peaks
    can't be identified."""
    if histogram is None or bin_edges is None:
        return None
    counts = np.asarray(histogram, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    if counts.size < 4 or edges.size != counts.size + 1:
        return None
    # Light smoothing so noise doesn't masquerade as peaks.
    k = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    k /= k.sum()
    sm = np.convolve(counts, k, mode='same')
    n = sm.size
    # Local maxima (interior + endpoints).
    peaks = [i for i in range(1, n - 1)
             if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > 0]
    if sm[0] > sm[1]:
        peaks.append(0)
    if sm[-1] > sm[-2]:
        peaks.append(n - 1)
    if len(peaks) < 2:
        return None
    peaks.sort(key=lambda i: sm[i], reverse=True)
    p1, p2 = peaks[0], peaks[1]
    centers = (edges[:-1] + edges[1:]) / 2.0
    mid = 0.5 * (centers[p1] + centers[p2])
    lo, hi = float(edges[0]), float(edges[-1])
    if hi - lo < 1e-9:
        return None
    pct = (mid - lo) / (hi - lo) * 100.0
    return int(round(min(100.0, max(0.0, pct))))


class HistogramView(QWidget):
    """Read-only volume histogram with plain gray bars (no window/mapping UI)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bins = None
        self._counts = None
        self._log = False
        self._bar_brush = pg.mkColor('lightgray')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget(background='w')
        self.plot_widget.setLabel('bottom', 'Intensity')
        self.plot_widget.setLabel('left', 'Count')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.hist_bar = pg.BarGraphItem(x=[], height=[], width=0.005,
                                        brush=self._bar_brush, pen=None)
        self.plot_widget.addItem(self.hist_bar)
        layout.addWidget(self.plot_widget)

    def set_histogram(self, bins, counts):
        if bins is None or counts is None:
            return
        self._bins = bins
        self._counts = counts
        self._render()

    def set_histogram_scale(self, log: bool):
        self._log = log
        self._render()

    def apply_theme(self, dark: bool):
        self.plot_widget.setBackground('#1e1e1e' if dark else 'w')
        self._bar_brush = pg.mkColor('#606060' if dark else 'lightgray')
        text_color = '#cccccc' if dark else '#000000'
        pen = pg.mkPen(color=text_color)
        for name in ('bottom', 'left'):
            ax = self.plot_widget.getAxis(name)
            ax.setPen(pen)
            ax.setTextPen(pen)
        self._render()

    def _render(self):
        if self._bins is None or self._counts is None:
            return
        bins, counts = self._bins, self._counts
        centers = (bins[:-1] + bins[1:]) / 2.0
        heights = np.log1p(counts.astype(float)) if self._log else counts.astype(float)
        self.hist_bar.setOpts(x=centers, height=heights,
                              width=(bins[1] - bins[0]) * 0.9,
                              brush=self._bar_brush, pen=None)
        self.plot_widget.setXRange(float(bins[0]), float(bins[-1]))


class VolumeInfoDialog(QDialog):
    """Volume histogram plus an info table."""

    def __init__(self, volume_data, preferences, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Volume Information')
        self.setMinimumWidth(360)
        self._build_ui(volume_data, preferences or {})

    def _build_ui(self, volume_data, preferences):
        layout = QVBoxLayout(self)

        theme = preferences.get('theme', 'Automatic')
        dark = _detect_os_dark() if theme == 'Automatic' else (theme == 'Dark')

        # ── Read-only gray histogram ──────────────────────────────────────────
        self.hist = HistogramView()
        self.hist.setMinimumHeight(220)
        hist = getattr(volume_data, 'histogram', None)
        edges = getattr(volume_data, 'bin_edges', None)
        if hist is not None and edges is not None:
            self.hist.set_histogram(edges, hist)
            self.hist.set_histogram_scale(
                preferences.get('histogram_scale', 'Logarithmic') == 'Logarithmic')
            self.hist.apply_theme(dark)
        layout.addWidget(self.hist, 1)

        # ── Info table ────────────────────────────────────────────────────────
        vol = volume_data.volume
        nx, ny, nz = (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2]))
        vx, vy, vz = (float(volume_data.voxel_size[0]),
                      float(volume_data.voxel_size[1]),
                      float(volume_data.voxel_size[2]))

        table = QtWidgets.QTableWidget(5, 4, self)
        table.setHorizontalHeaderLabels(['', 'X', 'Y', 'Z'])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)

        def put(r, c, text):
            item = QtWidgets.QTableWidgetItem(text)
            if c > 0:
                item.setTextAlignment(Qt.AlignCenter)
            table.setItem(r, c, item)

        put(0, 0, 'Voxel dimensions')
        put(0, 1, str(nx)); put(0, 2, str(ny)); put(0, 3, str(nz))
        put(1, 0, 'Physical dimensions (mm)')
        put(1, 1, f'{nx * vx:.4g}'); put(1, 2, f'{ny * vy:.4g}'); put(1, 3, f'{nz * vz:.4g}')
        put(2, 0, 'Voxel size (mm)')
        put(2, 1, f'{vx:.4g}'); put(2, 2, f'{vy:.4g}'); put(2, 3, f'{vz:.4g}')
        put(3, 0, 'Data type')
        put(3, 1, _dtype_label(volume_data.dtype))
        table.setSpan(3, 1, 1, 3)
        put(4, 0, 'Volume size')
        put(4, 1, _format_bytes(int(vol.nbytes)))
        table.setSpan(4, 1, 1, 3)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for c in (1, 2, 3):
            header.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        table.resizeRowsToContents()
        h = table.horizontalHeader().height() + sum(
            table.rowHeight(r) for r in range(table.rowCount())) + 2
        table.setFixedHeight(h)
        layout.addWidget(table)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f'{APP_NAME} {APP_VERSION}')
        self.resize(1600, 980)
        self.volume_data = VolumeData()
        self.preferences = _load_prefs()
        self._alignment_dialog = None
        self._geom_align_dialog = None   # Geometry-based alignment dialog
        self._geom_fit = None            # cached fit for the active geometry dialog
        self._surfaces = []              # created surface meshes (under Volume)
        self._surface_counter = 0
        self._project_meshes = []        # meshes promoted to top-level tree objects
        self._active_surface_rec = None  # surface record currently shown in 3D
        self._sync_locked = bool(self.preferences.get('sync_locked', True))
        self._syncing = False
        # In unlocked mode only the most-recently-moved slider's coordinate
        # plane is shown in 3D; it auto-hides after the same 3s timeout as the
        # 2D coordinate lines. (Locked mode always shows all three planes.)
        self._active_plane_axis = None
        self._plane_hide_timer = QtCore.QTimer(self)
        self._plane_hide_timer.setSingleShot(True)
        self._plane_hide_timer.timeout.connect(self._on_plane_hide_timeout)
        # Throttle the locked-mode viewport sync (re-slicing/re-centering the
        # other viewports) so a fast pan drag stays smooth — see
        # _on_viewport_range_changed.
        self._lock_sync_timer = QtCore.QTimer(self)
        self._lock_sync_timer.setSingleShot(True)
        self._lock_sync_timer.timeout.connect(self._do_lock_sync)
        self._pending_lock_source = None
        self._project_source = None
        self._project_path = None    # saved .voxels path; None ⇒ unsaved "New Project"
        self._volume_name = 'Volume' # name shown for the Volume node in the tree
        self._dirty = False          # unsaved changes to an opened project
        self._loading = False        # suppress dirty-marking during project restore
        # Single cumulative view-time alignment (output→input, scipy convention
        # in = R @ out + offset). Applied at display time; the voxel data is
        # never modified. _align_active is False when showing the raw volume.
        self._align_active = False
        self._align_R      = np.eye(3, dtype=np.float64)
        self._align_offset = np.zeros(3, dtype=np.float64)
        self._align_shape  = None   # output bounding-box shape (current display frame)
        self._aligned_cache = None  # whole volume pre-resampled into the active frame
        self._aligned_key   = None  # transform key the cache was built for
        # Alignment history shown under the project tree's Alignments folder.
        # Index 0 is always the non-removable "Initial Alignment" (the raw CT
        # scan frame). Each applied Simple/3-2-1 alignment appends a record
        # holding its absolute input→display transform; one is always active.
        self._alignments = []
        self._active_alignment = 0
        self._align_counters = {'simple': 0, '321': 0, 'geometry': 0}
        # Persistent measurements (project tree "Measurements"). Each record is a
        # dict with kind/orientation/slice/alignment/points/view_range/graphic;
        # the graphic exists only while the measurement is on its visible slice.
        self._measurements = []
        self._meas_counter = {'distance': 0, 'angle': 0, 'diameter': 0}
        # Persistent gray-value tools (project tree "Gray Values"). Same model
        # as measurements: each record carries kind/orientation/slice/alignment/
        # points/view_range/graphic.
        self._gray_values = []
        self._gray_counter = {'picker': 0, 'profile': 0}
        self.setup_ui()
        self.create_menu()
        self._apply_theme()
        self._restore_layout()
        self._update_title()

        # Debounce the (expensive) Phong 3D re-render while the window slider is
        # dragged — fires once the user pauses.
        self._levels_timer = QtCore.QTimer(self)
        self._levels_timer.setSingleShot(True)
        self._levels_timer.timeout.connect(self.view_3d.update_view)

        # Esc exits Auto Min Max mode from anywhere in the app.
        self._esc_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Escape), self)
        self._esc_shortcut.setContext(Qt.ApplicationShortcut)
        self._esc_shortcut.activated.connect(self._on_escape)

    def _on_escape(self):
        if self.left_panel.auto_minmax_btn.isChecked():
            self.left_panel.auto_minmax_btn.setChecked(False)

    # ── Project title / unsaved-changes tracking ──────────────────────────────
    def _project_display_name(self):
        if self._project_path:
            return os.path.splitext(os.path.basename(self._project_path))[0]
        return 'New Project'

    def _update_title(self):
        self.setWindowTitle(f'{APP_NAME} {APP_VERSION} - {self._project_display_name()}')

    def _mark_dirty(self):
        """Flag unsaved changes (ignored while a project is being restored)."""
        if not self._loading:
            self._dirty = True

    # ── Volume (project tree) name ────────────────────────────────────────────
    def _set_volume_name(self, name):
        """Set the Volume node name in state and in the tree (no dirty flag)."""
        self._volume_name = name or 'Volume'
        self.project_tree.set_volume_name(self._volume_name)

    def _on_volume_renamed(self, name):
        """User renamed the Volume node in the tree."""
        self._volume_name = name or 'Volume'
        self._mark_dirty()

    @staticmethod
    def _derive_volume_name(paths):
        """Smart Volume name from imported file(s); fall back to 'Volume'.

        A single file → its stem. A stack of slices/projections differs only by
        an index, so use the file-name part common to all of them (prefix or
        suffix), trimmed of separators and the varying digits.
        """
        stems = [os.path.splitext(os.path.basename(p))[0] for p in paths if p]
        if not stems:
            return 'Volume'
        if len(stems) == 1:
            return stems[0] or 'Volume'
        prefix = os.path.commonprefix(stems)
        suffix = os.path.commonprefix([s[::-1] for s in stems])[::-1]
        trim = ' _-.0123456789'
        prefix = prefix.strip(trim)
        suffix = suffix.strip(trim)
        candidate = prefix if len(prefix) >= len(suffix) else suffix
        return candidate or 'Volume'

    def _start_new_project(self):
        """Reset to an unsaved 'New Project' (e.g. after a fresh import)."""
        self._project_path = None
        self._dirty = False
        self._update_title()

    def _needs_save_prompt(self):
        # Prompt for an unsaved New Project (with content) or a modified open project.
        return self.volume_data.is_loaded() and (self._project_path is None or self._dirty)

    def closeEvent(self, event):
        if self._needs_save_prompt():
            box = QMessageBox(self)
            box.setIcon(QMessageBox.NoIcon)   # no question-mark icon
            box.setWindowTitle(APP_NAME)
            box.setText('Save the current project?')
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            box.setDefaultButton(QMessageBox.Yes)
            ans = box.exec()
            if ans == QMessageBox.Cancel:
                event.ignore()
                return
            if ans == QMessageBox.Yes and not self.save_voxels_project():
                event.ignore()    # save dialog cancelled / failed → don't quit
                return
        super().closeEvent(event)

    def setup_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        self.left_panel = BrightnessCurveWidget(self)
        self.left_panel.setMinimumWidth(200)
        self.left_panel.setMaximumWidth(420)
        self.left_panel.curve_changed.connect(self.on_curve_changed)

        # Project Structure tree above the histogram, sharing the sidebar 50/50.
        self.project_tree = ProjectStructureWidget(self)
        self.project_tree.properties_requested.connect(self.open_volume_information)
        self.project_tree.volume_renamed.connect(self._on_volume_renamed)
        self.project_tree.alignment_activate_requested.connect(self._activate_alignment)
        self.project_tree.alignment_remove_requested.connect(self._remove_alignment)
        self.project_tree.measurement_activate_requested.connect(self._goto_measurement)
        self.project_tree.measurement_remove_requested.connect(self._remove_measurement)
        self.project_tree.grayvalue_activate_requested.connect(self._goto_gray_value)
        self.project_tree.grayvalue_remove_requested.connect(self._remove_gray_value)
        self.project_tree.grayvalue_export_requested.connect(self._export_gray_value)
        self.project_tree.create_surface_requested.connect(self.open_create_surface)
        self.project_tree.surface_activate_requested.connect(self._goto_surface)
        self.project_tree.surface_remove_requested.connect(self._remove_surface)
        self.project_tree.surface_export_stl_requested.connect(self._export_surface_stl)
        self.project_tree.surface_promote_requested.connect(self._promote_surface)
        self.project_tree.projectmesh_activate_requested.connect(self._show_project_mesh)
        self.project_tree.projectmesh_export_stl_requested.connect(self._export_project_mesh_stl)
        self.project_tree.projectmesh_remove_requested.connect(self._remove_project_mesh)
        self.project_tree.set_volume_name(self._volume_name)

        self.left_splitter = QtWidgets.QSplitter(Qt.Vertical)
        self.left_splitter.addWidget(self.project_tree)
        self.left_splitter.addWidget(self.left_panel)
        self.left_splitter.setSizes([400, 400])   # 50% each
        self.left_splitter.setStretchFactor(0, 1)
        self.left_splitter.setStretchFactor(1, 1)
        self.left_splitter.setMinimumWidth(200)
        self.left_splitter.setMaximumWidth(420)

        grid_widget = QWidget()
        grid_widget.setMinimumWidth(800)
        grid_layout = QGridLayout(grid_widget)
        self.view_xy = SliceViewer('XY', self)
        self.view_yz = SliceViewer('YZ', self)
        self.view_xz = SliceViewer('XZ', self)
        self.view_3d = VolumeRender3D(self)
        # Shared picture-in-picture 3D preview, overlaid on a maximized 2D viewport.
        self.pip_3d = VolumeRender3D(self, pip=True)
        self.pip_3d.hide()

        grid_layout.addWidget(self.view_xy, 0, 0)
        grid_layout.addWidget(self.view_yz, 0, 1)
        grid_layout.addWidget(self.view_xz, 1, 0)
        grid_layout.addWidget(self.view_3d, 1, 1)
        grid_layout.setSpacing(2)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setRowStretch(0, 1)
        grid_layout.setRowStretch(1, 1)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)

        self.splitter = QtWidgets.QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.left_splitter)
        self.splitter.addWidget(grid_widget)
        self.splitter.setSizes([220, 1200])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.splitterMoved.connect(self._save_layout_prefs)
        main_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.setSpacing(2)
        main_layout.addWidget(self.splitter, 1)

        self.view_xy.maximize_button.clicked.connect(lambda: self.toggle_maximize(self.view_xy))
        self.view_yz.maximize_button.clicked.connect(lambda: self.toggle_maximize(self.view_yz))
        self.view_xz.maximize_button.clicked.connect(lambda: self.toggle_maximize(self.view_xz))
        self.view_3d.maximize_button.clicked.connect(lambda: self.toggle_maximize(self.view_3d))

        self.view_xy.point_placed.connect(self.on_point_placed)
        self.view_yz.point_placed.connect(self.on_point_placed)
        self.view_xz.point_placed.connect(self.on_point_placed)
        self.view_3d.point_placed.connect(self.on_point_placed)

        # Register a persistent record whenever a viewer creates a measurement
        # or a gray-value tool.
        for sv in (self.view_xy, self.view_yz, self.view_xz):
            sv.measurement_created.connect(
                lambda m, sv=sv: self._on_measurement_created(sv, m))
            sv.gray_value_created.connect(
                lambda g, sv=sv: self._on_gray_value_created(sv, g))

        # 3D render mode / isovalue changes count as project changes.
        self.view_3d.mode_combo.currentTextChanged.connect(lambda *_: self._mark_dirty())
        # Show the surface's 2D slice-outline only while the Surface Mesh renderer
        # is active (runs after view_3d.on_mode_changed has updated the state).
        self.view_3d.mode_combo.currentTextChanged.connect(
            lambda *_: self._refresh_surface_outline())
        self.view_3d.iso_slider.valueChanged.connect(lambda *_: self._mark_dirty())

        # 2D toggle: coordinate lines in the 2D viewports (linked across them).
        for v in (self.view_xy, self.view_yz, self.view_xz):
            v.planes_toggled.connect(self._on_coord_lines_toggled)
        # 3D toggle: coordinate planes in the 3D view only.
        self.view_3d.planes_toggled.connect(self._on_coord_planes_toggled)

        # Restore the saved visibility states (default: 2D lines on, 3D planes off).
        lines_on = bool(self.preferences.get('coord_lines_visible', True))
        planes_on = bool(self.preferences.get('coord_planes_visible', False))
        for sv in (self.view_xy, self.view_yz, self.view_xz):
            sv.planes_button.blockSignals(True)
            sv.planes_button.setChecked(lines_on)
            sv.planes_button.blockSignals(False)
            sv.set_coord_lines_enabled(lines_on)
        self.view_3d.planes_button.blockSignals(True)
        self.view_3d.planes_button.setChecked(planes_on)
        self.view_3d.planes_button.blockSignals(False)
        self.view_3d.set_planes_visible(planes_on)

        self.left_panel.auto_minmax_toggled.connect(self.view_xy.set_auto_mode)
        self.left_panel.auto_minmax_toggled.connect(self.view_yz.set_auto_mode)
        self.left_panel.auto_minmax_toggled.connect(self.view_xz.set_auto_mode)
        self.view_xy.region_selected.connect(self.left_panel.set_window_minmax)
        self.view_yz.region_selected.connect(self.left_panel.set_window_minmax)
        self.view_xz.region_selected.connect(self.left_panel.set_window_minmax)

        self.view_xy.axis_position_changed.connect(
            lambda axis, idx, total, sv=self.view_xy: self._on_axis_position_changed(sv, axis, idx, total))
        self.view_yz.axis_position_changed.connect(
            lambda axis, idx, total, sv=self.view_yz: self._on_axis_position_changed(sv, axis, idx, total))
        self.view_xz.axis_position_changed.connect(
            lambda axis, idx, total, sv=self.view_xz: self._on_axis_position_changed(sv, axis, idx, total))

        # Slice-video recording: one recorder at a time; the other two viewports
        # show the chosen range's two endpoints as coordinate lines.
        for v in (self.view_xy, self.view_yz, self.view_xz):
            v.record_mode_changed.connect(
                lambda on, sv=v: self._on_record_mode_changed(sv, on))
            v.record_range_changed.connect(
                lambda lo, hi, total, sv=v: self._on_record_range_changed(sv, lo, hi, total))

        meas_tool = self.preferences.get('measurement_tool', 'distance')
        for sv in (self.view_xy, self.view_yz, self.view_xz):
            vb = sv.image_view.getView().getViewBox()
            vb.sigRangeChanged.connect(
                lambda *args, sv=sv: self._on_viewport_range_changed(sv))
            sv.lock_clicked.connect(
                lambda sv=sv: self._on_lock_clicked(sv))
            sv.set_measurement_tool(meas_tool)
            sv.measurement_tool_changed.connect(self._on_measurement_tool_changed)
            sv.clip_changed.connect(self._update_clips)

        self.maximized_widget = None
        self._current_layout = 'Classic'
        self._layout_positions = {
            self.view_xy: (0, 0), self.view_yz: (0, 1),
            self.view_xz: (1, 0), self.view_3d: (1, 1),
        }

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        open_project_action = QtWidgets.QAction('Open Voxels Project...', self)
        save_project_action = QtWidgets.QAction('Save Voxels Project...', self)
        open_project_action.triggered.connect(self.open_voxels_project)
        save_project_action.triggered.connect(self.save_voxels_project)
        file_menu.addAction(open_project_action)
        file_menu.addAction(save_project_action)
        file_menu.addSeparator()
        import_menu = file_menu.addMenu('Import')
        import_slice_action = QtWidgets.QAction('Import Slice Files...', self)
        import_volume_action = QtWidgets.QAction('Import Volume...', self)
        import_slice_action.triggered.connect(self.import_slices)
        import_volume_action.triggered.connect(self.import_volume)
        import_menu.addAction(import_slice_action)
        import_menu.addAction(import_volume_action)
        file_menu.addSeparator()
        exit_action = QtWidgets.QAction('Exit', self)
        exit_action.setShortcut(QtGui.QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menubar.addMenu('&Edit')
        prefs_action = QtWidgets.QAction('Preferences...', self)
        prefs_action.triggered.connect(self.open_preferences)
        edit_menu.addAction(prefs_action)

        view_menu = menubar.addMenu('&View')
        layout_menu = view_menu.addMenu('Layout')
        classic_action = QtWidgets.QAction('Classic', self)
        engineering_action = QtWidgets.QAction('Engineering', self)
        classic_action.triggered.connect(lambda: self._apply_layout('Classic'))
        engineering_action.triggered.connect(lambda: self._apply_layout('Engineering'))
        layout_menu.addAction(classic_action)
        layout_menu.addAction(engineering_action)

        interp_menu = view_menu.addMenu('Voxels Interpolation')
        interp_group = QtWidgets.QActionGroup(interp_menu)
        interp_group.setExclusive(True)
        interp_on = bool(self.preferences.get('voxel_interpolation', False))
        for label, val in (('Off', False), ('On', True)):
            act = interp_menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(interp_on == val)
            interp_group.addAction(act)
            act.triggered.connect(lambda _c=False, v=val: self._on_interpolation_changed(v))
        for sv in self._slice_viewers():
            sv.set_interpolation(interp_on)

        operations_menu = menubar.addMenu('&Operations')
        alignment_menu = operations_menu.addMenu('Alignment')
        # Tripod icon on the Alignment submenu. Render it as a template/mask so
        # macOS tints it to the menu text colour (full contrast on light/dark),
        # with a thicker stroke so all three axes read at menu size.
        menu_tripod = _axes_tripod_icon('#000000', width=2.4)
        menu_tripod.setIsMask(True)
        alignment_menu.setIcon(menu_tripod)
        simple_align = QtWidgets.QAction('Simple Alignment...', self)
        simple_align.triggered.connect(self.open_simple_alignment)
        point_align = QtWidgets.QAction('3-2-1 Alignment...', self)
        point_align.triggered.connect(self.open_point_alignment)
        geom_align = QtWidgets.QAction('Geometry-based Alignment...', self)
        geom_align.triggered.connect(self.open_geometry_alignment)
        reset_align = QtWidgets.QAction('Reset Alignment', self)
        reset_align.triggered.connect(self.reset_alignment)
        alignment_menu.addAction(simple_align)
        alignment_menu.addAction(point_align)
        alignment_menu.addAction(geom_align)
        alignment_menu.addSeparator()
        alignment_menu.addAction(reset_align)
        operations_menu.addSeparator()
        surface_action = QtWidgets.QAction('Create Surface...', self)
        surface_action.triggered.connect(self.open_create_surface)
        operations_menu.addAction(surface_action)
        operations_menu.addSeparator()
        volume_info_action = QtWidgets.QAction('Volume Information...', self)
        volume_info_action.triggered.connect(self.open_volume_information)
        operations_menu.addAction(volume_info_action)

        help_menu = menubar.addMenu('&Help')
        about_action = QtWidgets.QAction('About...', self)
        about_action.triggered.connect(lambda: AboutDialog(self).exec())
        help_menu.addAction(about_action)

    def toggle_maximize(self, widget):
        grid_layout = widget.parentWidget().layout()
        viewers = [self.view_xy, self.view_yz, self.view_xz, self.view_3d]
        positions = self._layout_positions
        if self.maximized_widget is None:
            for v in viewers:
                grid_layout.removeWidget(v)
                if v is not widget:
                    v.hide()
            grid_layout.addWidget(widget, 0, 0, 2, 2)
            self.maximized_widget = widget
            widget.maximize_button.setText('▼')
            # A maximized 2D viewport gets a 3D PiP aligned to its orientation.
            if widget in (self.view_xy, self.view_yz, self.view_xz):
                self._show_pip_for(widget)
        else:
            if isinstance(self.maximized_widget, SliceViewer):
                self.maximized_widget.detach_pip()
            grid_layout.removeWidget(widget)
            for v in viewers:
                r, c = positions[v]
                grid_layout.addWidget(v, r, c)
                v.show()
            self.maximized_widget.maximize_button.setText('▲')
            self.maximized_widget = None
        self._save_layout_prefs()

    # Camera angles (azimuth, elevation) for each maximized 2D viewport's PiP.
    # Looking directions: XY side = from +Z (top, -90/90); XZ side = along +Y
    # (90/0). Per the chosen mapping: XY→XZ side, XZ→XY side, YZ→XZ side.
    _PIP_CAMERA = {'XY': (90.0, 0.0), 'XZ': (-90.0, 90.0), 'YZ': (90.0, 0.0)}
    # Volume axes (horizontal, vertical, depth) facing the camera for each PiP,
    # used to fit the bounding box to the view. XZ side → X,Z visible / Y depth;
    # XY side (top) → X,Y visible / Z depth.
    _PIP_FIT_AXES = {'XY': (0, 2, 1), 'XZ': (0, 1, 2), 'YZ': (0, 2, 1)}

    def _show_pip_for(self, viewer):
        """Configure and overlay the 3D PiP preview for a maximized 2D viewport."""
        pip, src = self.pip_3d, self.view_3d
        # Mirror what the main 3D view is rendering.
        pip.volume_data    = src.volume_data
        pip._perm_volume   = src._perm_volume
        pip.mode           = src.mode
        pip._quality       = src._quality
        pip._display_levels = src._display_levels
        pip.iso_threshold_percent = src.iso_threshold_percent
        # Mirror the Surface Mesh renderer too (otherwise the PiP falls back to
        # Phong). _render_surface places the mesh relative to the render frame, so
        # those fields must be copied before update_view (which early-returns for
        # a surface and would otherwise leave them stale).
        pip._showing_surface = src._showing_surface
        pip._surface       = src._surface
        pip._surface_shape = src._surface_shape
        pip._surface_color = src._surface_color
        pip._render_volume = src._render_volume
        pip._render_shape  = src._render_shape
        pip._render_factor = src._render_factor
        pip._render_vol_min = getattr(src, '_render_vol_min', 0.0)
        pip._render_vol_max = getattr(src, '_render_vol_max', 1.0)
        pip.update_view()
        # Lock the camera to the maximized viewport's orientation.
        az, el = self._PIP_CAMERA.get(viewer.orientation, (-45.0, 30.0))
        pip.gl_view.opts['azimuth'] = az
        pip.gl_view.opts['elevation'] = el
        pip.gl_view.sync_rot_from_opts()
        # Fit the camera so the volume's bounding box fills the PiP.
        pip._pip_fit_axes = self._PIP_FIT_AXES.get(viewer.orientation)
        pip.gl_view.update()
        # Show only this viewport's coordinate plane, moving with its slider.
        pip.set_planes_visible(True)
        viewer.attach_pip(pip)       # resizes the PiP → triggers a fit
        pip._fit_pip_camera()
        self._update_pip_plane()

    def _update_pip_plane(self):
        """Sync the PiP's single coordinate plane to the maximized viewport's slice."""
        w = self.maximized_widget
        if self._pip_inactive(w):
            return
        info = w.slice_fraction()
        if info is not None:
            self.pip_3d.set_coord_planes({info[0]: info[1]})

    def _pip_inactive(self, w):
        return (w is None or not isinstance(w, SliceViewer)
                or w._pip is not self.pip_3d)

    def _apply_layout(self, name):
        if name == self._current_layout:
            return
        if self.maximized_widget is not None:
            self.toggle_maximize(self.maximized_widget)
        if name == 'Engineering':
            new_positions = {
                self.view_xy: (0, 0), self.view_3d: (0, 1),
                self.view_xz: (1, 0), self.view_yz: (1, 1),
            }
        else:
            new_positions = {
                self.view_xy: (0, 0), self.view_yz: (0, 1),
                self.view_xz: (1, 0), self.view_3d: (1, 1),
            }
        grid_layout = self.view_xy.parentWidget().layout()
        for v in (self.view_xy, self.view_yz, self.view_xz, self.view_3d):
            grid_layout.removeWidget(v)
        for v, (r, c) in new_positions.items():
            grid_layout.addWidget(v, r, c)
            v.show()
        self._layout_positions = new_positions
        self._current_layout = name
        self._save_layout_prefs()

    def _on_interpolation_changed(self, enabled):
        for sv in self._slice_viewers():
            sv.set_interpolation(enabled)
        self.preferences['voxel_interpolation'] = bool(enabled)
        _save_prefs(self.preferences)

    def _save_import_dir(self, file_path: str) -> None:
        self.preferences['last_import_dir'] = os.path.dirname(os.path.abspath(file_path))
        _save_prefs(self.preferences)

    def _save_layout_prefs(self):
        self.preferences['splitter_sizes'] = self.splitter.sizes()
        self.preferences['maximized_viewport'] = (
            self.maximized_widget.orientation if self.maximized_widget else None
        )
        self.preferences['viewport_layout'] = self._current_layout
        self.preferences['sync_locked'] = self._sync_locked
        _save_prefs(self.preferences)

    def _restore_layout(self):
        sizes = self.preferences.get('splitter_sizes', [220, 1200])
        if sizes:
            self.splitter.setSizes(sizes)
        saved_layout = self.preferences.get('viewport_layout', 'Engineering')
        if saved_layout != self._current_layout:
            self._apply_layout(saved_layout)
        if self._sync_locked:
            for sv in self._slice_viewers():
                sv.set_locked(True)
                sv.set_lines_pinned(True)
        name = self.preferences.get('maximized_viewport')
        if name:
            viewport_map = {
                'XY': self.view_xy, 'YZ': self.view_yz,
                'XZ': self.view_xz, '3D': self.view_3d,
            }
            widget = viewport_map.get(name)
            if widget:
                self.toggle_maximize(widget)

    def _load_with_progress(self, fn, label='Loading...', total=0, title='Importing'):
        """Run fn(on_progress) in a background thread.
        Shows a progress dialog only if the load takes longer than 3 seconds.
        fn receives an on_progress(current, total) callable it may call for updates."""
        dlg = QProgressDialog(label, None, 0, total, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setValue(0)

        result_holder = [None]
        loop = QtCore.QEventLoop()

        timer = QtCore.QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(dlg.show)
        timer.start(3000)

        worker = _ImportWorker(fn)

        def _on_progress(cur, tot):
            # Adopt the loader's reported total so the bar fills correctly even
            # when it wasn't known up front (e.g. opening a project).
            if tot > 0 and dlg.maximum() != tot:
                dlg.setMaximum(tot)
            dlg.setValue(cur)

        worker.progress.connect(_on_progress)

        def _on_finished(r):
            result_holder[0] = r
            timer.stop()
            loop.quit()

        worker.finished.connect(_on_finished)
        worker.start()
        loop.exec()
        worker.wait()
        dlg.close()
        return result_holder[0]

    def import_slices(self):
        start_dir = self.preferences.get('last_import_dir', '')
        paths, _ = QFileDialog.getOpenFileNames(self, 'Import Slice Files', start_dir, 'Images (*.tif *.tiff *.raw);;All files (*)')
        if not paths:
            return
        self._save_import_dir(paths[0])
        sample = paths[0]
        ext = os.path.splitext(sample)[1].lower()
        raw_mode = ext == '.raw'
        initial_metadata = None
        if not raw_mode:
            initial_metadata = self.guess_tiff_metadata(paths)
        dialog = MetadataDialog(self, raw=raw_mode, initial_values=initial_metadata)
        if dialog.exec() != QDialog.Accepted:
            return
        dims, voxels, dtype, flip_z, big_endian = dialog.get_metadata()
        if raw_mode:
            volume = self._load_with_progress(
                lambda _: self.load_raw_volume(paths[0], dims, dtype, flip_z, big_endian),
                label=f'Loading {os.path.basename(paths[0])}...',
            )
        else:
            volume = self._load_with_progress(
                lambda cb: self.load_image_stack(paths, dtype, on_progress=cb),
                label=f'Loading {len(paths)} image file(s)...',
                total=len(paths),
            )
        if volume is None:
            QMessageBox.critical(self, 'Import Failed', 'Unable to load the selected slices.')
            return
        self.volume_data.set_volume(volume, voxels)
        self._reset_alignment_state()
        if raw_mode:
            self._project_source = {
                'type': 'raw_slices', 'files': [os.path.basename(p) for p in paths],
                'dims': list(dims), 'voxel_size': list(voxels),
                'dtype': str(np.dtype(dtype)), 'flip_z': flip_z, 'big_endian': big_endian,
            }
        else:
            self._project_source = {
                'type': 'tiff_slices', 'files': [os.path.basename(p) for p in paths],
                'voxel_size': list(voxels), 'dtype': str(np.dtype(dtype)),
            }
        self._set_volume_name(self._derive_volume_name(paths))
        self._reset_alignments()
        self._reset_measurements()
        self._reset_gray_values()
        self._reset_surfaces()
        self.update_views()
        self._auto_set_isovalue()
        self._start_new_project()

    def import_volume(self):
        start_dir = self.preferences.get('last_import_dir', '')
        path, _ = QFileDialog.getOpenFileName(self, 'Import Volume File', start_dir, 'Raw (*.raw);;TIFF (*.tif *.tiff);;All files (*)')
        if not path:
            return
        self._save_import_dir(path)
        ext = os.path.splitext(path)[1].lower()
        raw_mode = ext == '.raw'
        initial_metadata = None
        if not raw_mode:
            initial_metadata = self.guess_tiff_metadata([path])
        dialog = MetadataDialog(self, raw=raw_mode, initial_values=initial_metadata)
        if dialog.exec() != QDialog.Accepted:
            return
        dims, voxels, dtype, flip_z, big_endian = dialog.get_metadata()
        if raw_mode:
            volume = self._load_with_progress(
                lambda cb: self.load_raw_volume(path, dims, dtype, flip_z,
                                                big_endian, on_progress=cb),
                label=f'Loading {os.path.basename(path)}...',
            )
        else:
            volume = self._load_with_progress(
                lambda _: self.load_volume_tiff(path, dtype),
                label=f'Loading {os.path.basename(path)}...',
            )
        if volume is None:
            QMessageBox.critical(self, 'Import Failed', 'Unable to load the selected volume file.')
            return
        self.volume_data.set_volume(volume, voxels)
        self._reset_alignment_state()
        if raw_mode:
            self._project_source = {
                'type': 'raw_volume', 'file': os.path.basename(path),
                'dims': list(dims), 'voxel_size': list(voxels),
                'dtype': str(np.dtype(dtype)), 'flip_z': flip_z, 'big_endian': big_endian,
            }
        else:
            self._project_source = {
                'type': 'tiff_volume', 'file': os.path.basename(path),
                'voxel_size': list(voxels), 'dtype': str(np.dtype(dtype)),
            }
        self._set_volume_name(self._derive_volume_name([path]))
        self._reset_alignments()
        self._reset_measurements()
        self._reset_gray_values()
        self._reset_surfaces()
        self.update_views()
        self._auto_set_isovalue()
        self._start_new_project()

    # ── Project save / open ───────────────────────────────────────────────────

    def _camera_state(self):
        opts = self.view_3d.gl_view.opts
        ctr  = opts.get('center')
        try:
            if callable(getattr(ctr, 'x', None)):
                center = [float(ctr.x()), float(ctr.y()), float(ctr.z())]
            elif hasattr(ctr, '__getitem__'):
                center = [float(ctr[0]), float(ctr[1]), float(ctr[2])]
            else:
                center = [0.0, 0.0, 0.0]
        except Exception:
            center = [0.0, 0.0, 0.0]
        return {
            'distance':  float(opts.get('distance',  200)),
            'azimuth':   float(opts.get('azimuth',   -45)),
            'elevation': float(opts.get('elevation',  30)),
            'center':    center,
        }

    def _restore_camera(self, cam):
        gv = self.view_3d.gl_view
        cx, cy, cz = cam.get('center', [0.0, 0.0, 0.0])
        gv.opts['center']    = QtGui.QVector3D(cx, cy, cz)
        gv.opts['distance']  = float(cam.get('distance',  200))
        gv.opts['azimuth']   = float(cam.get('azimuth',   -45))
        gv.opts['elevation'] = float(cam.get('elevation',  30))
        gv.sync_rot_from_opts()
        gv.update()

    def save_voxels_project(self):
        """Save the project. Returns True on success, False if cancelled/failed."""
        if not self.volume_data.is_loaded() or self._project_source is None:
            QMessageBox.warning(self, 'Save Project',
                                'No volume loaded. Please import data first.')
            return False
        start_dir = self._project_path or self.preferences.get('last_import_dir', '')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Voxels Project', start_dir, 'Voxels Project (*.voxels)')
        if not path:
            return False
        if not path.endswith('.voxels'):
            path += '.voxels'

        def vp_state(viewer):
            xr, yr = viewer.image_view.getView().getViewBox().viewRange()
            return {'slice': viewer.current_index,
                    'view_range': [list(xr), list(yr)]}

        # Persist surface meshes as STL files alongside the .voxels project and
        # reference them from the JSON (geometry is not inlined into the JSON).
        surface_entries, mesh_entries = self._serialize_meshes(path)

        panel = self.left_panel
        # Store the single cumulative view-time alignment (output→input affine).
        align = None
        if self._align_active:
            align = {
                'R':         self._align_R.tolist(),
                'offset':    self._align_offset.tolist(),
                'out_shape': list(self._align_shape),
            }
        project = {
            'version': 1,
            'app_version': APP_VERSION_FULL,
            'source': self._project_source,
            'volume_name': self._volume_name,
            'viewports': {
                'xy': vp_state(self.view_xy),
                'yz': vp_state(self.view_yz),
                'xz': vp_state(self.view_xz),
            },
            'histogram': {
                'window_min': panel.window_min(),
                'window_max': panel.window_max(),
                'scale': self.preferences.get('histogram_scale', 'Logarithmic'),
            },
            'alignment': align,
            'alignments': [self._serialize_alignment(a) for a in self._alignments],
            'active_alignment': self._active_alignment,
            'measurements': [self._serialize_measurement(r) for r in self._measurements],
            'gray_values': [self._serialize_gray_value(r) for r in self._gray_values],
            'surfaces': surface_entries,
            'project_meshes': mesh_entries,
            'sync_locked': self._sync_locked,
            'render': {
                'mode': self.view_3d.mode,
                'iso_threshold_percent': self.view_3d.iso_threshold_percent,
            },
            'camera': self._camera_state(),
        }
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(project, f, indent=2)
        except OSError as exc:
            QMessageBox.critical(self, 'Save Failed', str(exc))
            return False
        self._project_path = path
        self._dirty = False
        self._save_import_dir(path)
        self._update_title()
        return True

    def open_voxels_project(self):
        start_dir = self.preferences.get('last_import_dir', '')
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Voxels Project', start_dir, 'Voxels Project (*.voxels)')
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                project = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, 'Open Failed',
                                 f'Could not read project file:\n{exc}')
            return

        # Front-compatibility: warn if the project was saved by a newer build.
        # (Older projects open fine — back compatibility is preserved.)
        proj_ver = project.get('app_version')
        if proj_ver and _parse_version(proj_ver) > _parse_version(APP_VERSION_FULL):
            ans = QMessageBox.warning(
                self, 'Newer Project Version',
                f'This project was created with {APP_NAME} {proj_ver}, which is '
                f'newer than this build ({APP_VERSION_FULL}).\n\n'
                f'It may not open correctly. Open it anyway?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                return

        project_dir = os.path.dirname(os.path.abspath(path))
        load_err = []

        def _do_load(on_progress):
            try:
                return self._load_volume_from_source(
                    project['source'], project_dir, on_progress=on_progress)
            except Exception as exc:
                load_err.append(exc)
                return None

        result = self._load_with_progress(
            _do_load,
            label=f'Opening {os.path.basename(path)}...',
            title='Opening Project',
        )
        if result is None:
            msg = str(load_err[0]) if load_err else 'Unknown error'
            QMessageBox.critical(self, 'Open Failed',
                                 f'Could not load volume:\n{msg}')
            return

        volume, voxels = result
        self._loading = True   # suppress dirty-marking while restoring state
        self.volume_data.set_volume(volume, tuple(voxels))
        self._project_source = project['source']
        self._set_volume_name(project.get('volume_name', 'Volume'))
        self._save_import_dir(path)

        # Restore the alignment as a view-time transform (no resample). Handles
        # the current single-transform dict, the legacy list of 3-2-1 ops, and
        # the legacy non-destructive dict. update_views() pushes it to the views.
        self._reset_alignment_state()
        self._restore_alignment(project.get('alignment'))
        self._restore_alignments(project)
        self._restore_measurements(project)
        self._restore_gray_values(project)
        self._reset_surfaces()   # clear the previous project's surfaces

        self.update_views()

        # Restore viewport state
        for key, viewer in [('xy', self.view_xy), ('yz', self.view_yz),
                             ('xz', self.view_xz)]:
            vp = project.get('viewports', {}).get(key, {})
            if 'slice' in vp:
                viewer.slice_slider.setValue(int(vp['slice']))
            if 'view_range' in vp:
                xr, yr = vp['view_range']
                viewer._auto_range_pending = False
                viewer.image_view.getView().getViewBox().setRange(
                    xRange=xr, yRange=yr, padding=0)

        # Restore histogram settings
        hist = project.get('histogram', {})
        win_min = hist.get('window_min')
        win_max = hist.get('window_max')
        if win_min is not None and win_max is not None:
            self.left_panel.set_window_minmax(win_min, win_max)
        scale = hist.get('scale')
        if scale:
            self.preferences['histogram_scale'] = scale
            self.left_panel.set_histogram_scale(scale == 'Logarithmic')

        # Restore lock state
        saved_lock = project.get('sync_locked', False)
        if saved_lock != self._sync_locked:
            self._on_lock_clicked(self.view_xy)

        # Restore surface meshes from the STLs saved beside the project. Must
        # come after update_views() (so the 3D render shape is set for placement)
        # and before the render-mode restore below (so the 'Surface Mesh' renderer
        # is available if the project was saved in that mode).
        self._restore_meshes(project, project_dir)

        # Restore 3D render mode and isovalue
        render = project.get('render', {})
        if 'iso_threshold_percent' in render:
            self.view_3d.iso_slider.setValue(int(render['iso_threshold_percent']))
        if 'mode' in render:
            self.view_3d.mode_combo.setCurrentText(render['mode'])

        # Restore 3D camera — must come last, after all update_view() calls
        cam = project.get('camera')
        if cam:
            self._restore_camera(cam)

        # Restore complete: this is now a clean, saved project.
        self._loading = False
        self._project_path = path
        self._dirty = False
        self._update_title()

        # A surface mesh built during the synchronous open can paint as a blocky
        # artifact ("a weird cube") because the 3D GL context isn't fully current
        # yet — the user otherwise has to toggle the renderer to force a clean
        # rebuild. Do that toggle automatically once the event loop is running and
        # the GL view has been shown/painted, keeping the restored camera.
        if self.view_3d._showing_surface:
            QtCore.QTimer.singleShot(0, self._rerender_restored_surface)

    def _rerender_restored_surface(self):
        """Rebuild the surface mesh in a now-valid GL context (see open_voxels_project)."""
        v3 = self.view_3d
        if not (v3._showing_surface and v3._surface is not None):
            return
        v3._set_render_frame(v3._surface_render_voxels())
        v3._render_surface(reset_camera=False)
        v3.gl_view.update()
        self._refresh_surface_outline()

    def _load_volume_from_source(self, src, project_dir, on_progress=None):
        """Load a volume from a project source dict. Returns (ndarray, voxel_size)."""
        def absp(name):
            return os.path.join(project_dir, name)

        src_type = src['type']
        dtype    = np.dtype(src.get('dtype', 'uint8'))
        voxels   = tuple(src.get('voxel_size', (1.0, 1.0, 1.0)))

        if src_type == 'raw_slices':
            vol = self.load_raw_volume(
                absp(src['files'][0]), tuple(src['dims']), dtype,
                src.get('flip_z', False), src.get('big_endian', False),
                on_progress=on_progress)
            return vol, voxels

        if src_type == 'tiff_slices':
            vol = self.load_image_stack([absp(f) for f in src['files']], dtype,
                                        on_progress=on_progress)
            return vol, voxels

        if src_type == 'raw_volume':
            vol = self.load_raw_volume(
                absp(src['file']), tuple(src['dims']), dtype,
                src.get('flip_z', False), src.get('big_endian', False),
                on_progress=on_progress)
            return vol, voxels

        if src_type == 'tiff_volume':
            vol = self.load_volume_tiff(absp(src['file']), dtype)
            return vol, voxels

        raise ValueError(f'Unknown source type: {src_type!r}')

    # ── Raw / TIFF loaders ────────────────────────────────────────────────────

    def load_raw_volume(self, path, dims, dtype, flip_z, big_endian=False, on_progress=None):
        try:
            count = int(np.prod(dims))
            file_dtype = np.dtype(dtype).newbyteorder('>') if big_endian else np.dtype(dtype)
            if on_progress is None:
                with open(path, 'rb') as rawfile:
                    data = np.fromfile(rawfile, dtype=file_dtype, count=count)
            else:
                # Read in Z-slice chunks so on_progress(current, total) fires
                # throughout. Progress is reported in slices (small ints) to
                # avoid overflowing the int signal on very large volumes.
                nz, voxels_per_slice = dims[2], dims[1] * dims[0]
                chunk_z = max(1, nz // 100)        # ≈ 100 progress ticks
                data = np.empty(count, dtype=file_dtype)
                read = 0
                with open(path, 'rb') as rawfile:
                    for z0 in range(0, nz, chunk_z):
                        z1 = min(z0 + chunk_z, nz)
                        block = np.fromfile(rawfile, dtype=file_dtype,
                                            count=(z1 - z0) * voxels_per_slice)
                        if block.size == 0:
                            break
                        data[read:read + block.size] = block
                        read += block.size
                        on_progress(z1, nz)
            if data.size != count:
                return None
            volume = data.reshape((dims[2], dims[1], dims[0]))
            if flip_z:
                volume = np.flip(volume, axis=0)
            return np.ascontiguousarray(volume.transpose(2, 1, 0), dtype=np.dtype(dtype).newbyteorder('='))
        except Exception as exc:
            print('Raw load error:', exc)
            return None

    def guess_tiff_metadata(self, paths):
        if not paths:
            return None
        first_path = paths[0]
        try:
            if tifffile:
                image = tifffile.imread(first_path)
            else:
                image = imageio.imread(first_path)
            image = np.asarray(image)
            if image.ndim == 4 and image.shape[3] in (3, 4):
                image = self.convert_color_to_grayscale(image)
            elif image.ndim == 3 and image.shape[2] in (3, 4):
                image = self.convert_color_to_grayscale(image)

            if image.ndim == 2:
                height, width = image.shape
                depth = len(paths)
            elif image.ndim == 3:
                if len(paths) == 1:
                    depth, height, width = image.shape
                else:
                    # Treat each selected file as one slice and use first page shape.
                    height, width = image.shape[:2]
                    depth = len(paths)
            else:
                return None

            dtype_name = self.guess_dtype_name(image)
            return {
                'width': int(width),
                'height': int(height),
                'depth': int(depth),
                'dtype': dtype_name,
                'voxel_x': 1.0,
                'voxel_y': 1.0,
                'voxel_z': 1.0,
            }
        except Exception:
            return None

    def guess_dtype_name(self, image):
        if image is None:
            return '8-bit unsigned'
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            return '32-bit float'
        if np.issubdtype(image.dtype, np.signedinteger):
            min_val = int(np.min(image))
            max_val = int(np.max(image))
            if min_val >= -128 and max_val <= 127:
                return '8-bit signed'
            return '16-bit signed'
        if np.issubdtype(image.dtype, np.unsignedinteger):
            max_val = int(np.max(image))
            if max_val <= 255:
                return '8-bit unsigned'
            return '16-bit unsigned'
        return '8-bit unsigned'

    def convert_color_to_grayscale(self, image):
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[2] in (3, 4):
            rgb = image[..., :3].astype(np.float32)
            gray = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32))
            if np.issubdtype(image.dtype, np.integer):
                gray = np.round(gray).astype(image.dtype)
            return gray
        if image.ndim == 4 and image.shape[3] in (3, 4):
            rgb = image[..., :3].astype(np.float32)
            gray = np.dot(rgb, np.array([0.299, 0.587, 0.114], dtype=np.float32))
            if np.issubdtype(image.dtype, np.integer):
                gray = np.round(gray).astype(image.dtype)
            return gray
        return image

    def ensure_volume_scalar(self, volume):
        if volume is None:
            return None
        volume = np.asarray(volume)
        if volume.ndim == 4:
            if volume.shape[3] in (3, 4):
                volume = np.mean(volume[..., :3], axis=3)
            else:
                volume = np.mean(volume, axis=-1)
        if volume.ndim == 3:
            return volume
        if volume.ndim == 2:
            return volume[:, :, np.newaxis]
        return None

    def load_image_stack(self, paths, dtype, on_progress=None):
        try:
            paths = sorted(paths)
            stack = []
            total = len(paths)
            for i, path in enumerate(paths):
                image = self.load_image(path, dtype)
                if on_progress:
                    on_progress(i + 1, total)
                if image is None:
                    return None
                if image.ndim == 3 and image.shape[2] not in (3, 4):
                    # Multi-page TIFF files may return a 3D array of pages.
                    for page in image:
                        if page.ndim != 2:
                            return None
                        stack.append(page.astype(dtype))
                elif image.ndim == 2:
                    stack.append(image)
                else:
                    return None
            if not stack:
                return None
            volume = np.stack(stack, axis=2)   # (rows, cols, slices) = (Y, X, Z)
            return volume.transpose(1, 0, 2)   # -> (X, Y, Z)
        except Exception as exc:
            print('Stack load error:', exc)
            return None

    def load_volume_tiff(self, path, dtype):
        try:
            if tifffile:
                volume = tifffile.imread(path)
            else:
                volume = imageio.imread(path)
            volume = self.convert_color_to_grayscale(volume)
            volume = self.ensure_volume_scalar(volume)
            if volume is None:
                return None
            # tifffile 3-D: (pages, rows, cols) = (Z, Y, X) -> (X, Y, Z)
            if volume.ndim == 3:
                volume = volume.transpose(2, 1, 0)
            return volume.astype(dtype)
        except Exception as exc:
            print('TIFF load error:', exc)
            return None

    def load_image(self, path, dtype):
        try:
            if tifffile and path.lower().endswith(('.tif', '.tiff')):
                image = tifffile.imread(path)
            else:
                image = imageio.imread(path)
            image = self.convert_color_to_grayscale(image)
            return image.astype(dtype)
        except Exception as exc:
            print('Image load error:', exc)
            return None

    def _auto_set_isovalue(self):
        """On import, guess a good 3D isovalue from the histogram — midway between
        its two largest peaks (air vs material) — and set the slider to it."""
        pct = _suggest_isovalue_percent(
            getattr(self.volume_data, 'histogram', None),
            getattr(self.volume_data, 'bin_edges', None))
        if pct is not None:
            self.view_3d.iso_slider.setValue(int(pct))

    def update_views(self):
        self.view_xy.set_volume(self.volume_data)
        self.view_yz.set_volume(self.volume_data)
        self.view_xz.set_volume(self.volume_data)
        self.view_3d.set_volume(self.volume_data)
        # set_volume clears any permanent transform; re-push the active alignment
        # so it survives a volume refresh.
        if self._align_active and self.volume_data.is_loaded():
            self._set_view_alignment(self._align_R, self._align_offset, self._align_shape)
        if self.volume_data.histogram is not None:
            self.left_panel.set_histogram(self.volume_data.bin_edges, self.volume_data.histogram)
            self.left_panel.set_data_range(
                float(self.volume_data.bin_edges[0]),
                float(self.volume_data.bin_edges[-1]),
                integer=np.issubdtype(self.volume_data.dtype, np.integer),
            )
        self._update_coord_planes()
        # Refresh the PiP if a 2D viewport is maximized (e.g. volume changed).
        if isinstance(self.maximized_widget, SliceViewer) and \
                self.maximized_widget._pip is self.pip_3d:
            self._show_pip_for(self.maximized_widget)
        # Re-display the persistent measurements / gray-value tools that belong
        # on the current slices / active alignment (volume/alignment may change).
        self._sync_measurements()
        self._sync_gray()

    def on_curve_changed(self, curve):
        if not self.volume_data.is_loaded() or len(curve) < 4:
            return
        min_val = curve[1][0]
        max_val = curve[2][0]
        self._mark_dirty()
        for viewer in (self.view_xy, self.view_yz, self.view_xz):
            viewer.set_levels(min_val, max_val)   # fast LUT update, no re-slice
        self.view_3d.set_levels(min_val, max_val)
        # Only the Phong volume render depends on the window. The isosurface and
        # a created Surface Mesh both ignore it entirely, so re-rendering them on
        # a window change is pure wasted work — and rebuilding a large surface
        # mesh on every Auto Min Max tick makes the UI crawl. Re-render (debounced)
        # only when the Phong volume is actually on screen.
        if self.view_3d.mode != 'Isosurface' and not self.view_3d._showing_surface:
            self._levels_timer.start(200)

    def open_preferences(self):
        dialog = PreferencesDialog(self, self.preferences)
        if dialog.exec() != QDialog.Accepted:
            return
        self.preferences = dialog.get_preferences()
        self._apply_theme()
        _save_prefs(self.preferences)

    def _apply_theme(self):
        theme = self.preferences.get('theme', 'Automatic')
        dark = _detect_os_dark() if theme == 'Automatic' else (theme == 'Dark')
        app = QApplication.instance()
        if dark:
            app.setPalette(_make_dark_palette())
        else:
            app.setPalette(app.style().standardPalette())
        self.left_panel.apply_theme(dark)
        self.left_panel.set_histogram_scale(self.preferences.get('histogram_scale', 'Logarithmic') == 'Logarithmic')
        self.project_tree.apply_theme(dark)

    def open_volume_information(self):
        if not self.volume_data.is_loaded():
            QMessageBox.warning(self, 'Volume Information', 'Please import a volume first.')
            return
        dlg = VolumeInfoDialog(self.volume_data, self.preferences, self)
        dlg.exec()

    def open_simple_alignment(self):
        if not self.volume_data.is_loaded():
            QMessageBox.warning(self, 'Simple Alignment', 'Please import a volume first.')
            return
        if hasattr(self, '_simple_align_dialog') and self._simple_align_dialog is not None:
            self._simple_align_dialog.raise_()
            return
        # The voxel data is never mutated (alignment is view-time), so we can
        # reference the volume directly instead of copying it.
        orig = self.volume_data.volume
        self._simple_align_orig_vol = orig
        # Pre-build a small volume once for the live 3D preview (orig never
        # changes while the dialog is open), so each slider move only pays for a
        # tiny resample instead of re-downsampling the full volume every time.
        f = max(1, int(np.ceil((orig.size / 96 ** 3) ** (1.0 / 3.0))))
        self._simple_preview_vol3d = orig[::f, ::f, ::f].astype(np.float32)
        self._simple_preview_factor3d = f
        dlg = SimpleAlignmentDialog(self.volume_data.volume.shape, self)
        self._simple_align_dialog = dlg
        dlg.preview_changed.connect(self._on_simple_preview)
        dlg.alignment_applied.connect(self._on_simple_applied)
        dlg.alignment_cancelled.connect(self._on_simple_cancelled)
        dlg.show()
        dlg.raise_()

    def _on_simple_preview(self, transform):
        R, offset = transform
        orig = getattr(self, '_simple_align_orig_vol', None)
        if orig is None:
            return
        for sv in self._slice_viewers():
            sv.set_preview_transform(R, offset, orig)
        # Fast low-res 3D preview using the cached downsampled volume.
        vol_ds = getattr(self, '_simple_preview_vol3d', None)
        if _scipy_affine_transform is not None and vol_ds is not None:
            # offset is in orig voxel space; downsampled space is orig / factor
            off_ds = offset / self._simple_preview_factor3d
            try:
                prev_ds = _scipy_affine_transform(
                    vol_ds, R, offset=off_ds,
                    output_shape=vol_ds.shape, order=1, mode='constant', cval=0.0,
                )
                self.view_3d.update_preview(prev_ds)
            except Exception:
                pass

    def _on_simple_applied(self, transform):
        R, offset = transform
        orig = getattr(self, '_simple_align_orig_vol', None)
        self._simple_align_dialog = None
        self._simple_align_orig_vol = None
        self._simple_preview_vol3d = None
        for sv in self._slice_viewers():
            sv.clear_preview_transform()
        if orig is None or _scipy_affine_transform is None:
            QMessageBox.warning(self, 'Simple Alignment',
                                'scipy is required to apply the transformation.')
            return
        # Simple Alignment is defined on the raw volume (its preview samples the
        # raw volume), so it replaces any existing view-time alignment. The
        # dialog's (R, offset) is the pre-bounding-box scipy offset.
        R_b, off_b, out_shape = _alignment_bbox(R, offset, orig.shape)
        self._set_view_alignment(R_b, off_b, out_shape)
        self._record_alignment('simple')

    def _on_simple_cancelled(self):
        self._simple_align_dialog = None
        self._simple_align_orig_vol = None
        self._simple_preview_vol3d = None
        for sv in self._slice_viewers():
            sv.clear_preview_transform()
        self.update_views()

    def _slice_viewers(self):
        """Return all SliceViewer instances."""
        viewers = []
        for attr in ('view_xy', 'view_yz', 'view_xz'):
            sv = getattr(self, attr, None)
            if sv is not None:
                viewers.append(sv)
        return viewers

    # ── Slice-video record coordination ───────────────────────────────────────
    _RECORD_AXIS = {'XY': 'Z', 'YZ': 'X', 'XZ': 'Y'}

    def _on_record_mode_changed(self, source, on):
        """Enforce a single recorder and refresh the range lines on the others."""
        if on:
            for sv in self._slice_viewers():
                if sv is not source and sv._record_mode:
                    sv.set_record_mode(False)
        else:
            for sv in self._slice_viewers():
                sv.clear_record_lines()

    def _on_record_range_changed(self, source, low, high, total):
        """Show the recording viewport's range as two lines on the other two."""
        axis = self._RECORD_AXIS.get(source.orientation)
        for sv in self._slice_viewers():
            if sv is source:
                sv.clear_record_lines()
            elif axis is not None:
                sv.show_record_lines(axis, low, high, total)

    def _display_shape(self):
        """Dimensions of the frame currently shown in the slice viewers.

        With a view-time alignment this is the aligned bounding-box shape (shared
        by all three viewports), otherwise the raw volume shape. The lock-mode
        crosshair / slider sync must use these display dimensions, not the raw
        volume shape, or the viewports drift out of sync when aligned."""
        if self._align_shape is not None:
            return tuple(int(s) for s in self._align_shape)
        if self.volume_data is not None and self.volume_data.volume is not None:
            return self.volume_data.volume.shape
        return None

    def _on_axis_position_changed(self, source_viewer, axis, index, total):
        # While a locked-mode sync is running, the slice images are updated by
        # on_slice_changed directly; skip the per-slice overlay/measurement
        # cascade here — _do_lock_sync does one consolidated update instead.
        if self._syncing:
            return
        self._mark_dirty()
        # The moving viewport changed slice: re-display the persistent
        # measurements / gray-value tools on the new slice (hide the others).
        self._sync_measurements_for(source_viewer)
        self._sync_gray_for(source_viewer)
        # A slice move updates any active clipping plane + the 3D coord planes.
        # In unlocked mode, only the moving viewport's plane is shown in 3D, and
        # it auto-hides after the same timeout as the 2D coordinate lines.
        if not self._sync_locked:
            info = source_viewer.slice_fraction()
            self._active_plane_axis = info[0] if info is not None else None
            self._plane_hide_timer.start(3000)
        else:
            self._plane_hide_timer.stop()
        self._update_clips()
        self._update_coord_planes()
        # Keep the PiP's coordinate plane in step with the maximized slider.
        if source_viewer is self.maximized_widget:
            self._update_pip_plane()
        vol_shape = self._display_shape()
        if vol_shape is None:
            return
        src_range = source_viewer.image_view.getView().getViewBox().viewRange()
        for viewer in self._slice_viewers():
            seg = self._compute_seg_range(source_viewer.orientation, src_range,
                                          viewer.orientation, vol_shape)
            viewer.show_axis_line(axis, index, total, seg)

        # In locked mode, re-center every viewport on its own intersection.
        if not self._sync_locked or self._syncing:
            return
        self._syncing = True
        try:
            vb0 = self._slice_viewers()[0].image_view.getView().getViewBox()
            xr0, yr0 = vb0.viewRange()
            self._center_all_on_intersection(xr0[1] - xr0[0], yr0[1] - yr0[0])
        finally:
            self._syncing = False

    def _update_clips(self):
        """Collect each 2D viewport's clip state and push them to the 3D view."""
        clips = {}
        for sv in self._slice_viewers():
            info = sv.clip_info()
            if info is not None:
                ax, mode, frac = info
                clips[ax] = (mode, frac)
        self.view_3d.set_clips(clips)

    def _update_coord_planes(self):
        """Push each 2D viewport's slice position to the 3D coordinate planes."""
        vol_shape = self._display_shape()
        fracs = {}
        rects = {}
        for sv in self._slice_viewers():
            info = sv.slice_fraction()
            if info is not None:
                fracs[info[0]] = info[1]
                if vol_shape is not None:
                    rects[info[0]] = self._plane_extent_norm(sv, vol_shape)
        # Unlocked: show only the plane whose slider was last moved; show none
        # before any move and after the auto-hide timeout (mirrors the 2D lines).
        if not self._sync_locked:
            if self._active_plane_axis is None:
                fracs = {}
            else:
                fracs = {k: v for k, v in fracs.items() if k == self._active_plane_axis}
        rects = {k: rects[k] for k in fracs if k in rects}
        self.view_3d.set_coord_planes(fracs, rects)

    def _plane_extent_norm(self, viewer, vol_shape):
        """Normalised [0,1] visible extent of ``viewer``'s plane along each of
        its two in-plane volume axes → {axis: (lo, hi)}.

        Mirrors the 2D plot↔volume mapping (plot-x = raw, plot-y = flipped):
          XY → X, Y ;  YZ → Y, Z ;  XZ → X, Z.
        """
        nx, ny, nz = vol_shape
        (px0, px1), (py0, py1) = viewer.image_view.getView().getViewBox().viewRange()

        def nrm(lo, hi, n):
            lo, hi = (lo, hi) if lo <= hi else (hi, lo)
            return (max(0.0, lo / n), min(1.0, hi / n))

        o = viewer.orientation
        if o == 'XY':   # plot-x = X, plot-y = flipped-Y
            return {0: nrm(px0, px1, nx),
                    1: nrm(ny - 1 - py1, ny - 1 - py0, ny)}
        if o == 'YZ':   # plot-x = Y, plot-y = flipped-Z
            return {1: nrm(px0, px1, ny),
                    2: nrm(nz - 1 - py1, nz - 1 - py0, nz)}
        if o == 'XZ':   # plot-x = X, plot-y = flipped-Z
            return {0: nrm(px0, px1, nx),
                    2: nrm(nz - 1 - py1, nz - 1 - py0, nz)}
        return {}

    def _on_plane_hide_timeout(self):
        """Auto-hide the unlocked single 3D plane (in sync with the 2D lines)."""
        self._active_plane_axis = None
        self._update_coord_planes()

    def _on_coord_lines_toggled(self, visible):
        """Link the 2D coordinate-line toggles and apply to all 2D viewports."""
        for sv in self._slice_viewers():
            b = sv.planes_button
            if b.isChecked() != visible:
                b.blockSignals(True)
                b.setChecked(visible)
                b.blockSignals(False)
            sv.set_coord_lines_enabled(visible)
        # Draw the lines immediately on enable (don't wait for a slice/zoom event).
        if visible:
            self._show_all_coord_lines()
        self.preferences['coord_lines_visible'] = bool(visible)
        _save_prefs(self.preferences)

    def _on_coord_planes_toggled(self, visible):
        """Show/hide the 3D coordinate planes and remember the choice."""
        self.view_3d.set_planes_visible(visible)
        self.preferences['coord_planes_visible'] = bool(visible)
        _save_prefs(self.preferences)

    def _on_measurement_tool_changed(self, kind):
        """Persist the chosen measurement tool and keep all viewports in sync."""
        self.preferences['measurement_tool'] = kind
        _save_prefs(self.preferences)
        for sv in self._slice_viewers():
            sv.set_measurement_tool(kind)

    def _on_lock_clicked(self, source_viewer):
        self._mark_dirty()
        self._sync_locked = not self._sync_locked
        self.preferences['sync_locked'] = self._sync_locked
        _save_prefs(self.preferences)
        # Reset the unlocked single-plane focus: locked shows all planes;
        # unlocked shows none until a slider is moved (then auto-hides).
        self._active_plane_axis = None
        self._plane_hide_timer.stop()
        self._update_coord_planes()
        for sv in self._slice_viewers():
            sv.set_locked(self._sync_locked)
            sv.set_lines_pinned(self._sync_locked)
        if self._sync_locked:
            src_range = source_viewer.image_view.getView().getViewBox().viewRange()
            xr, yr = src_range
            W, H = xr[1] - xr[0], yr[1] - yr[0]
            self._syncing = True
            try:
                self._sync_slices_to_center(source_viewer, src_range)
                self._center_all_on_intersection(W, H)
            finally:
                self._syncing = False
            self._show_all_coord_lines()

    def _show_all_coord_lines(self):
        sources = [
            (self.view_xy, 'Z', 2),
            (self.view_yz, 'X', 0),
            (self.view_xz, 'Y', 1),
        ]
        vol_shape = self._display_shape()
        if vol_shape is None:
            return
        for src, axis, ax_idx in sources:
            index = src.current_index
            total = vol_shape[ax_idx]
            src_range = src.image_view.getView().getViewBox().viewRange()
            for viewer in self._slice_viewers():
                seg = self._compute_seg_range(src.orientation, src_range,
                                              viewer.orientation, vol_shape)
                viewer.show_axis_line(axis, index, total, seg)

    _LOCK_SYNC_MS = 50   # throttle for the heavy locked-mode cross-viewport sync

    def _on_viewport_range_changed(self, source_viewer):
        # Re-entrant range changes (from _center_all_on_intersection during a
        # sync) are ignored — _do_lock_sync handles everything in one pass.
        if self._syncing:
            return
        self._mark_dirty()
        # Cheap, live, every event (locked or not): the source viewport's
        # crosshair line across the viewports + the 3D coordinate planes. This is
        # the same light work unlocked mode does, so the panned viewport stays as
        # responsive as in unlocked mode.
        self._update_range_overlays(source_viewer, restart_hide=not self._sync_locked)
        if not self._sync_locked:
            return
        # Locked: keep the panned viewport's own crosshair pinned to the view
        # centre live (cheap), so it doesn't lag behind the pan while the heavier
        # orthogonal re-slice/re-centre runs on a throttled timer off this path.
        self._update_source_crosshair_live(source_viewer)
        self._pending_lock_source = source_viewer
        if not self._lock_sync_timer.isActive():
            self._lock_sync_timer.start(self._LOCK_SYNC_MS)

    def _update_source_crosshair_live(self, sv):
        """Position the panned viewport's two crosshair lines at its current view
        centre (locked mode), tracking the pan continuously rather than waiting
        for the throttled slice re-sync."""
        shape = self._display_shape()
        if shape is None:
            return
        nx, ny, nz = shape
        (x0, x1), (y0, y1) = sv.image_view.getView().getViewBox().viewRange()
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        # Vertical line at plot-x = cx; horizontal line at plot-y = cy; segments
        # span the visible viewport so it reads as a full centred crosshair.
        if sv.orientation == 'XY':
            sv.show_axis_line('X', cx, nx, (y0, y1))
            sv.show_axis_line('Y', ny - 1 - cy, ny, (x0, x1))
        elif sv.orientation == 'YZ':
            sv.show_axis_line('Y', cx, ny, (y0, y1))
            sv.show_axis_line('Z', nz - 1 - cy, nz, (x0, x1))
        elif sv.orientation == 'XZ':
            sv.show_axis_line('X', cx, nx, (y0, y1))
            sv.show_axis_line('Z', nz - 1 - cy, nz, (x0, x1))

    def _update_range_overlays(self, source_viewer, restart_hide=False):
        """Show this viewport's coordinate line across the viewports and refresh
        the 3D coordinate planes (the cheap per-pan overlay work)."""
        vol_shape = self._display_shape()
        if vol_shape is None:
            return
        axis   = {'XY': 'Z', 'YZ': 'X', 'XZ': 'Y'}[source_viewer.orientation]
        ax_idx = {'XY': 2, 'YZ': 0, 'XZ': 1}[source_viewer.orientation]
        src_range = source_viewer.image_view.getView().getViewBox().viewRange()
        for viewer in self._slice_viewers():
            seg = self._compute_seg_range(source_viewer.orientation, src_range,
                                          viewer.orientation, vol_shape)
            viewer.show_axis_line(axis, source_viewer.current_index,
                                  vol_shape[ax_idx], seg)
        if restart_hide and not self._sync_locked:
            self._active_plane_axis = ax_idx
            self._plane_hide_timer.start(3000)
        self._update_coord_planes()

    def _do_lock_sync(self):
        """Re-slice/re-center the other viewports to the pan source's center, then
        refresh overlays ONCE (throttled entry — see _on_viewport_range_changed).

        While ``_syncing`` is set, the slice/range cascades triggered below are
        short-circuited in their handlers, so all the overlay, clip, coordinate-
        line and measurement work happens exactly once here per tick."""
        sv = self._pending_lock_source
        if sv is None or not self._sync_locked or self._syncing:
            return
        before = (self.view_xy.current_index, self.view_yz.current_index,
                  self.view_xz.current_index)
        self._syncing = True
        try:
            src_range = sv.image_view.getView().getViewBox().viewRange()
            xr, yr = src_range
            W, H = xr[1] - xr[0], yr[1] - yr[0]
            self._sync_slices_to_center(sv, src_range)
            self._center_all_on_intersection(W, H, exclude=sv)
        finally:
            self._syncing = False
        # Overlays whose extent tracks the view (cheap) refresh every tick.
        self._show_all_coord_lines()
        self._update_coord_planes()
        if isinstance(self.maximized_widget, SliceViewer):
            self._update_pip_plane()
        # Re-sync measurements / gray tools ONLY for the viewports whose slice
        # actually changed. Rebuilding ROI graphics is expensive, so never touch
        # the source viewport (its slice doesn't change during its own in-plane
        # pan) or any viewport that didn't move — that churn was the main cause
        # of locked-mode pan sluggishness with measurements present.
        after = (self.view_xy.current_index, self.view_yz.current_index,
                 self.view_xz.current_index)
        changed = [v for v, b, a in zip((self.view_xy, self.view_yz, self.view_xz),
                                        before, after) if b != a]
        if changed:
            self._update_clips()
            for v in changed:
                self._sync_measurements_for(v)
                self._sync_gray_for(v)

    def _center_all_on_intersection(self, W, H, exclude=None):
        """Set every locked viewport's range to W×H centered on its own line
        intersection. ``exclude`` skips a viewport — used to leave the viewport
        the user is actively panning untouched (re-ranging it mid-drag fights
        pyqtgraph's own pan handling and makes panning feel sluggish; it is
        already at the right place, since the crosshair is derived from it).

        Intersection plot coords (each viewport's horizontal axis = raw, vertical = flipped):
          XY: (x=x_idx, y=ny-1-y_idx)   — X-line × Y-line
          YZ: (x=y_idx, y=nz-1-z_idx)   — Y-line × Z-line
          XZ: (x=x_idx, y=nz-1-z_idx)   — X-line × Z-line
        """
        shape = self._display_shape()
        if shape is None:
            return
        nx, ny, nz = shape
        x_idx = self.view_yz.current_index   # X crosshair (YZ traverses X)
        y_idx = self.view_xz.current_index   # Y crosshair (XZ traverses Y)
        z_idx = self.view_xy.current_index   # Z crosshair (XY traverses Z)
        hw, hh = W / 2.0, H / 2.0
        centers = {
            self.view_xy: (x_idx,        ny - 1 - y_idx),
            self.view_yz: (y_idx,        nz - 1 - z_idx),
            self.view_xz: (x_idx,        nz - 1 - z_idx),
        }
        for viewer, (cx, cy) in centers.items():
            if viewer is exclude:
                continue
            viewer.image_view.getView().getViewBox().setRange(
                xRange=(cx - hw, cx + hw), yRange=(cy - hh, cy + hh), padding=0)

    def _sync_slices_to_center(self, source_viewer, src_range):
        """Move slice sliders in the other viewports to match the center of source_viewer.

        Coordinate mappings (each viewport's horizontal axis has no flip, vertical is [::-1]):
          XY: plot-x = X_raw,  plot-y → Y_raw = ny-1-plot_y   (traverses Z)
          YZ: plot-x = Y_raw,  plot-y → Z_raw = nz-1-plot_y   (traverses X)
          XZ: plot-x = X_raw,  plot-y → Z_raw = nz-1-plot_y   (traverses Y)
        """
        shape = self._display_shape()
        if shape is None:
            return
        nx, ny, nz = shape
        xr, yr = src_range
        cx = (xr[0] + xr[1]) / 2.0
        cy = (yr[0] + yr[1]) / 2.0

        def set_slider(slider, val, hi):
            slider.setValue(int(np.clip(round(val), 0, hi)))

        if source_viewer.orientation == 'XY':
            set_slider(self.view_yz.slice_slider, cx,          nx - 1)  # X
            set_slider(self.view_xz.slice_slider, ny - 1 - cy, ny - 1)  # Y
        elif source_viewer.orientation == 'YZ':
            set_slider(self.view_xz.slice_slider, cx,          ny - 1)  # Y
            set_slider(self.view_xy.slice_slider, nz - 1 - cy, nz - 1)  # Z
        elif source_viewer.orientation == 'XZ':
            set_slider(self.view_yz.slice_slider, cx,          nx - 1)  # X
            set_slider(self.view_xy.slice_slider, nz - 1 - cy, nz - 1)  # Z

    def _compute_seg_range(self, src_orient, src_range, tgt_orient, vol_shape):
        """Return (a, b) solid-segment range in tgt_orient's plot coordinates.

        Each 2D viewport maps its horizontal axis directly (no flip) and its
        vertical axis with [::-1] (so plot_y = dim-1 - raw_index).

        XY: x=X (no flip), y=flipped-Y  — traverses Z
        YZ: x=Y (no flip), y=flipped-Z  — traverses X
        XZ: x=X (no flip), y=flipped-Z  — traverses Y
        """
        nx, ny, nz = vol_shape
        px = src_range[0]  # [x_min, x_max] in source plot-x
        py = src_range[1]  # [y_min, y_max] in source plot-y

        if src_orient == 'XY':        # axis = Z; px=X range, py=flipped-Y range
            if tgt_orient == 'YZ':
                # Z-line is horizontal in YZ; perpendicular = Y (no flip in YZ).
                # XY plot-y → Y_raw = ny-1-plot_y
                return (ny - 1 - py[1], ny - 1 - py[0])
            if tgt_orient == 'XZ':
                # Z-line is horizontal in XZ; perpendicular = X (no flip).
                return (px[0], px[1])

        elif src_orient == 'YZ':      # axis = X; px=Y range, py=flipped-Z range
            if tgt_orient == 'XY':
                # X-line is vertical in XY; perpendicular = Y (flipped in XY).
                # YZ px = Y_raw; XY plot-y = ny-1-Y_raw
                return (ny - 1 - px[1], ny - 1 - px[0])
            if tgt_orient == 'XZ':
                # X-line is vertical in XZ; perpendicular = Z (flipped in both).
                # Both flip Z identically, so plot-y coords map 1-to-1.
                return (py[0], py[1])

        elif src_orient == 'XZ':      # axis = Y; px=X range, py=flipped-Z range
            if tgt_orient == 'XY':
                # Y-line is horizontal in XY; perpendicular = X (no flip).
                return (px[0], px[1])
            if tgt_orient == 'YZ':
                # Y-line is vertical in YZ; perpendicular = Z (flipped in both).
                return (py[0], py[1])

        return None

    def open_point_alignment(self):
        if not self.volume_data.is_loaded():
            QMessageBox.warning(self, '3-2-1 Alignment', 'Please import a volume first.')
            return
        if self._alignment_dialog is not None:
            self._alignment_dialog.raise_()
            return
        dlg = AlignmentDialog321(self)
        self._alignment_dialog = dlg
        self.view_3d.set_alignment_mode(True)
        dlg.reset_requested.connect(lambda: self.view_3d.clear_alignment_overlays())
        dlg.finished.connect(self._on_alignment_finished)
        dlg.show()
        dlg.raise_()

    def _on_alignment_finished(self, result):
        dlg = self._alignment_dialog
        self._alignment_dialog = None
        self.view_3d.set_alignment_mode(False)
        self.view_3d.clear_alignment_overlays()
        if result != QDialog.Accepted or dlg is None:
            return
        pts = dlg.get_points()
        if len(pts) < 6:
            return
        plane_pts = np.array(pts[:3], dtype=np.float32)
        line_pts  = np.array(pts[3:5], dtype=np.float32)
        origin_pt = np.array(pts[5],   dtype=np.float32)
        result_transform = _compute_321_transform(plane_pts, line_pts, origin_pt)
        if result_transform is None:
            QMessageBox.warning(self, '3-2-1 Alignment',
                                'Could not compute alignment — points may be collinear.')
            return
        R, origin = result_transform
        self._apply_321(R, origin)

    def open_geometry_alignment(self):
        if not self.volume_data.is_loaded():
            QMessageBox.warning(self, 'Geometry-based Alignment',
                                'Please import a volume first.')
            return
        if self._geom_align_dialog is not None:
            self._geom_align_dialog.raise_()
            return
        dlg = GeometryAlignmentDialog(self)
        self._geom_align_dialog = dlg
        self.view_3d.set_alignment_mode(True)
        dlg.reset_requested.connect(self._on_geom_align_changed)
        dlg.geometry_changed.connect(lambda *_: self._on_geom_align_changed())
        dlg.finished.connect(self._on_geom_align_finished)
        dlg.show()
        dlg.raise_()
        self._on_geom_align_changed()

    def _on_geom_align_changed(self):
        """Refit the selected geometry to the picked points, update the 3D
        preview, and enable Apply only when the fit succeeds."""
        dlg = self._geom_align_dialog
        if dlg is None:
            return
        pts = dlg.get_points()
        g = dlg.geometry()
        need = _GEOM_MIN_POINTS.get(g, 3)
        fit = _fit_geometry(g, pts) if len(pts) >= need else None
        self._geom_fit = fit
        self.view_3d.set_geometry_overlay(pts, g, fit)
        if len(pts) < need:
            dlg.set_fit_status(f'Need at least {need} points for a {g.lower()} fit.')
            dlg.set_apply_enabled(False)
        elif fit is None:
            dlg.set_fit_status('Fit failed — add more or better-spread points.')
            dlg.set_apply_enabled(False)
        else:
            rms = fit.get('rms')
            extra = f'   (RMS {rms:.2f} vox)' if rms is not None else ''
            dlg.set_fit_status(f'{g} fit ✓{extra}')
            dlg.set_apply_enabled(True)

    def _on_geom_align_finished(self, result):
        dlg = self._geom_align_dialog
        self._geom_align_dialog = None
        self._geom_fit = None
        self.view_3d.set_alignment_mode(False)
        self.view_3d.clear_alignment_overlays()
        if result != QDialog.Accepted or dlg is None:
            return
        pts = dlg.get_points()
        g = dlg.geometry()
        fit = _fit_geometry(g, pts)
        tf = (_geometry_alignment_transform(g, fit, np.array(pts, dtype=np.float64))
              if fit is not None else None)
        if tf is None:
            QMessageBox.warning(self, 'Geometry-based Alignment',
                                'Could not compute an alignment from the picked points.')
            return
        R, origin = tf
        self._apply_new_alignment(R, origin, 'Geometry-based Alignment', kind='geometry')

    # ── Alignment history (project tree) ──────────────────────────────────────
    def _reset_alignments(self):
        """Start a fresh history with just the Initial Alignment (active)."""
        self._alignments = [{
            'name': 'Initial Alignment', 'kind': 'initial',
            'R': None, 'offset': None, 'out_shape': None,
        }]
        self._active_alignment = 0
        self._align_counters = {'simple': 0, '321': 0, 'geometry': 0}
        self._refresh_alignment_tree()

    def _refresh_alignment_tree(self):
        entries = [{'name': a['name'], 'initial': a['kind'] == 'initial'}
                   for a in self._alignments]
        self.project_tree.set_alignments(entries, self._active_alignment)

    def _record_alignment(self, kind):
        """Snapshot the just-applied cumulative transform as a new history entry
        and make it the active one."""
        labels = {'simple': 'Simple Alignment', '321': '3-2-1 Alignment',
                  'geometry': 'Geometry Alignment'}
        self._align_counters[kind] = self._align_counters.get(kind, 0) + 1
        rec = {
            'name': f"{labels.get(kind, 'Alignment')} {self._align_counters[kind]}",
            'kind': kind,
            'R': self._align_R.copy(),
            'offset': self._align_offset.copy(),
            'out_shape': tuple(int(s) for s in self._align_shape),
        }
        self._alignments.append(rec)
        self._active_alignment = len(self._alignments) - 1
        self._refresh_alignment_tree()
        self._sync_measurements()
        self._sync_gray()

    def _show_raw_alignment(self):
        """Drop the view-time alignment and show the raw (Initial) volume."""
        old_R, old_offset = self._align_R, self._align_offset
        self._reset_alignment_state()
        for sv in self._slice_viewers():
            sv.clear_permanent_transform()
        self.view_3d.clear_permanent_volume()
        # Carry the child surface meshes back into the raw (Initial) frame.
        self._remap_surfaces(old_R, old_offset)
        self.update_views()

    def _activate_alignment(self, index):
        """Make the history entry at ``index`` the active alignment."""
        if index < 0 or index >= len(self._alignments):
            return
        rec = self._alignments[index]
        self._active_alignment = index
        if rec['kind'] == 'initial':
            self._show_raw_alignment()
        else:
            self._set_view_alignment(rec['R'], rec['offset'], rec['out_shape'])
        self._mark_dirty()
        self._refresh_alignment_tree()
        self._sync_measurements()
        self._sync_gray()

    def _remove_alignment(self, index):
        """Delete a non-initial history entry; re-activate a neighbour if it was
        the active one. Measurements and gray-value tools created in this
        alignment are dependent and are removed with it (after confirmation)."""
        if index <= 0 or index >= len(self._alignments):
            return  # index 0 (Initial) is never removable
        dep_meas = [r for r in self._measurements if r['alignment'] == index]
        dep_gray = [r for r in self._gray_values if r['alignment'] == index]
        n_dep = len(dep_meas) + len(dep_gray)
        if n_dep:
            parts = []
            if dep_meas:
                parts.append(f"{len(dep_meas)} measurement(s)")
            if dep_gray:
                parts.append(f"{len(dep_gray)} gray-value tool(s)")
            ans = QMessageBox.warning(
                self, 'Remove Alignment',
                f"{' and '.join(parts)} were created in "
                f"“{self._alignments[index]['name']}” and will also be "
                f"removed.\n\nRemove the alignment and its dependent items?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ans != QMessageBox.Yes:
                return
        was_active = (index == self._active_alignment)
        # Drop dependent measurements / gray values (and any live graphics).
        for rec in dep_meas:
            if rec.get('graphic') is not None:
                rec['graphic'].remove()
            self._measurements.remove(rec)
        for rec in dep_gray:
            if rec.get('graphic') is not None:
                rec['graphic'].remove()
            self._gray_values.remove(rec)
        # Shift alignment indices on the survivors.
        for rec in self._measurements + self._gray_values:
            if rec['alignment'] > index:
                rec['alignment'] -= 1
        del self._alignments[index]
        if was_active:
            self._activate_alignment(index - 1)   # fall back to the previous entry
        else:
            if self._active_alignment > index:
                self._active_alignment -= 1
            self._mark_dirty()
            self._refresh_alignment_tree()
            self._sync_measurements()
            self._sync_gray()
        self._refresh_measurement_tree()
        self._refresh_gray_tree()

    def _serialize_alignment(self, a):
        return {
            'name': a['name'], 'kind': a['kind'],
            'R': a['R'].tolist() if a['R'] is not None else None,
            'offset': a['offset'].tolist() if a['offset'] is not None else None,
            'out_shape': list(a['out_shape']) if a['out_shape'] is not None else None,
        }

    def _restore_alignments(self, project):
        """Rebuild the alignment history from an opened project (or reconstruct a
        minimal history for legacy files that stored only the active transform)."""
        saved = project.get('alignments')
        if saved:
            self._alignments = []
            for a in saved:
                self._alignments.append({
                    'name': a.get('name', 'Alignment'),
                    'kind': a.get('kind', 'simple'),
                    'R': np.array(a['R'], dtype=np.float64) if a.get('R') is not None else None,
                    'offset': np.array(a['offset'], dtype=np.float64) if a.get('offset') is not None else None,
                    'out_shape': tuple(a['out_shape']) if a.get('out_shape') is not None else None,
                })
            self._active_alignment = int(project.get('active_alignment', 0))
            self._align_counters = {'simple': 0, '321': 0, 'geometry': 0}
            for a in self._alignments:
                if a['kind'] in self._align_counters:
                    self._align_counters[a['kind']] += 1
        else:
            self._alignments = [{
                'name': 'Initial Alignment', 'kind': 'initial',
                'R': None, 'offset': None, 'out_shape': None,
            }]
            self._align_counters = {'simple': 0, '321': 0, 'geometry': 0}
            if self._align_active:
                self._align_counters['simple'] = 1
                self._alignments.append({
                    'name': 'Restored Alignment', 'kind': 'simple',
                    'R': self._align_R.copy(), 'offset': self._align_offset.copy(),
                    'out_shape': tuple(int(s) for s in self._align_shape),
                })
                self._active_alignment = 1
            else:
                self._active_alignment = 0
        self._refresh_alignment_tree()

    # ── Measurements (project tree) ───────────────────────────────────────────
    def _viewer_for_orientation(self, orientation):
        return {'XY': self.view_xy, 'YZ': self.view_yz,
                'XZ': self.view_xz}.get(orientation)

    def _reset_measurements(self):
        for rec in self._measurements:
            if rec.get('graphic') is not None:
                rec['graphic'].remove()
        self._measurements = []
        self._meas_counter = {'distance': 0, 'angle': 0, 'diameter': 0}
        self._refresh_measurement_tree()

    def _refresh_measurement_tree(self):
        self.project_tree.set_measurements(
            [{'name': r['name'], 'kind': r['kind']} for r in self._measurements])

    def _on_measurement_created(self, viewer, m):
        """A viewer just made a new measurement graphic — register a persistent
        record for it (tagged with the current slice and active alignment)."""
        kind = getattr(m, 'kind', 'distance')
        label = {'distance': 'Distance', 'angle': 'Angle',
                 'diameter': 'Diameter'}.get(kind, 'Measurement')
        self._meas_counter[kind] = self._meas_counter.get(kind, 0) + 1
        vr = viewer.image_view.getView().getViewBox().viewRange()
        rec = {
            'name': f'{label} {self._meas_counter[kind]}',
            'kind': kind,
            'orientation': viewer.orientation,
            'slice': viewer.current_index,
            'alignment': self._active_alignment,
            'points': m.geometry(),
            'view_range': [list(vr[0]), list(vr[1])],
            'graphic': m,
        }
        m._record = rec
        self._measurements.append(rec)
        self._connect_measurement_drag(m, rec)
        self._refresh_measurement_tree()
        self._mark_dirty()

    def _connect_measurement_drag(self, m, rec):
        """Keep the record's geometry in step with interactive drags."""
        def _sync_geom():
            rec['points'] = m.geometry()
            self._mark_dirty()
        m.roi.sigRegionChanged.connect(_sync_geom)

    def _spawn_measurement_graphic(self, viewer, rec):
        m = viewer.build_measurement(rec['kind'], rec['points'])
        if m is None:
            return
        m._record = rec
        rec['graphic'] = m
        self._connect_measurement_drag(m, rec)

    def _sync_measurements_for(self, viewer):
        """Show exactly the measurements that belong on this viewer's current
        slice + active alignment; hide (but keep) the rest."""
        viewer.clear_measurements()
        if not self.volume_data.is_loaded():
            return
        if getattr(self, '_simple_align_dialog', None) is not None:
            return   # don't show measurements during an alignment preview
        for rec in self._measurements:
            if rec['orientation'] != viewer.orientation:
                continue
            if rec['alignment'] != self._active_alignment:
                continue
            if viewer.current_index != rec['slice']:
                continue
            self._spawn_measurement_graphic(viewer, rec)

    def _sync_measurements(self):
        for sv in self._slice_viewers():
            self._sync_measurements_for(sv)

    def _goto_measurement(self, index):
        """Navigate back to where a measurement was created (activating its
        alignment and slice) so it becomes visible again."""
        if index < 0 or index >= len(self._measurements):
            return
        rec = self._measurements[index]
        if rec['alignment'] != self._active_alignment:
            self._activate_alignment(rec['alignment'])
        viewer = self._viewer_for_orientation(rec['orientation'])
        if viewer is None:
            return
        # Setting the slider drives on_slice_changed → measurement re-sync. If the
        # slice is already current, sync explicitly below.
        viewer.slice_slider.setValue(int(rec['slice']))
        vr = rec.get('view_range')
        if vr is not None:
            viewer.image_view.getView().getViewBox().setRange(
                xRange=vr[0], yRange=vr[1], padding=0)
        self._sync_measurements_for(viewer)

    def _remove_measurement(self, index):
        if index < 0 or index >= len(self._measurements):
            return
        rec = self._measurements.pop(index)
        if rec.get('graphic') is not None:
            rec['graphic'].remove()
        self._refresh_measurement_tree()
        self._mark_dirty()

    def _serialize_measurement(self, r):
        return {
            'name': r['name'], 'kind': r['kind'],
            'orientation': r['orientation'], 'slice': int(r['slice']),
            'alignment': int(r['alignment']), 'points': r['points'],
            'view_range': r.get('view_range'),
        }

    def _restore_measurements(self, project):
        self._measurements = []
        self._meas_counter = {'distance': 0, 'angle': 0, 'diameter': 0}
        for mr in project.get('measurements', []) or []:
            kind = mr.get('kind', 'distance')
            self._measurements.append({
                'name': mr.get('name', 'Measurement'), 'kind': kind,
                'orientation': mr.get('orientation', 'XY'),
                'slice': int(mr.get('slice', 0)),
                'alignment': int(mr.get('alignment', 0)),
                'points': mr.get('points'),
                'view_range': mr.get('view_range'),
                'graphic': None,
            })
            if kind in self._meas_counter:
                self._meas_counter[kind] += 1
        self._refresh_measurement_tree()

    # ── Gray-value tools (project tree) ───────────────────────────────────────
    def _reset_gray_values(self):
        for rec in self._gray_values:
            if rec.get('graphic') is not None:
                rec['graphic'].remove()
        self._gray_values = []
        self._gray_counter = {'picker': 0, 'profile': 0}
        self._refresh_gray_tree()

    def _refresh_gray_tree(self):
        self.project_tree.set_gray_values(
            [{'name': r['name'], 'kind': r['kind']} for r in self._gray_values])

    def _on_gray_value_created(self, viewer, g):
        kind = getattr(g, 'kind', 'picker')
        label = {'picker': 'Picker', 'profile': 'Profile'}.get(kind, 'Gray Value')
        self._gray_counter[kind] = self._gray_counter.get(kind, 0) + 1
        vr = viewer.image_view.getView().getViewBox().viewRange()
        rec = {
            'name': f'{label} {self._gray_counter[kind]}',
            'kind': kind,
            'orientation': viewer.orientation,
            'slice': viewer.current_index,
            'alignment': self._active_alignment,
            'points': g.geometry(),
            'view_range': [list(vr[0]), list(vr[1])],
            'graphic': g,
        }
        g._record = rec
        self._gray_values.append(rec)
        self._connect_gray_drag(g, rec)
        self._refresh_gray_tree()
        self._mark_dirty()

    def _connect_gray_drag(self, g, rec):
        def _sync_geom():
            rec['points'] = g.geometry()
            self._mark_dirty()
        g.changed_signal().connect(_sync_geom)

    def _spawn_gray_graphic(self, viewer, rec):
        g = viewer.build_gray_value(rec['kind'], rec['points'])
        if g is None:
            return
        g._record = rec
        rec['graphic'] = g
        self._connect_gray_drag(g, rec)

    def _sync_gray_for(self, viewer):
        viewer.clear_gray_values()
        if not self.volume_data.is_loaded():
            return
        if getattr(self, '_simple_align_dialog', None) is not None:
            return
        for rec in self._gray_values:
            if rec['orientation'] != viewer.orientation:
                continue
            if rec['alignment'] != self._active_alignment:
                continue
            if viewer.current_index != rec['slice']:
                continue
            self._spawn_gray_graphic(viewer, rec)

    def _sync_gray(self):
        for sv in self._slice_viewers():
            self._sync_gray_for(sv)

    def _goto_gray_value(self, index):
        if index < 0 or index >= len(self._gray_values):
            return
        rec = self._gray_values[index]
        if rec['alignment'] != self._active_alignment:
            self._activate_alignment(rec['alignment'])
        viewer = self._viewer_for_orientation(rec['orientation'])
        if viewer is None:
            return
        viewer.slice_slider.setValue(int(rec['slice']))
        vr = rec.get('view_range')
        if vr is not None:
            viewer.image_view.getView().getViewBox().setRange(
                xRange=vr[0], yRange=vr[1], padding=0)
        self._sync_gray_for(viewer)

    def _remove_gray_value(self, index):
        if index < 0 or index >= len(self._gray_values):
            return
        rec = self._gray_values.pop(index)
        if rec.get('graphic') is not None:
            rec['graphic'].remove()
        self._refresh_gray_tree()
        self._mark_dirty()

    def _export_gray_value(self, index):
        """Export a gray-value Profile's samples to CSV (moved here from the
        2D viewport right-click). Navigates to the profile first so its samples
        are freshly computed, then exports from the live graphic."""
        if index < 0 or index >= len(self._gray_values):
            return
        rec = self._gray_values[index]
        if rec['kind'] != 'profile':
            return
        self._goto_gray_value(index)   # ensure it's on-screen and sampled
        g = rec.get('graphic')
        if g is None or not hasattr(g, 'export_csv'):
            QMessageBox.warning(self, 'Export Gray Value Profile',
                                'Could not access the profile data.')
            return
        g.export_csv()

    def _serialize_gray_value(self, r):
        return {
            'name': r['name'], 'kind': r['kind'],
            'orientation': r['orientation'], 'slice': int(r['slice']),
            'alignment': int(r['alignment']), 'points': r['points'],
            'view_range': r.get('view_range'),
        }

    def _restore_gray_values(self, project):
        self._gray_values = []
        self._gray_counter = {'picker': 0, 'profile': 0}
        for gr in project.get('gray_values', []) or []:
            kind = gr.get('kind', 'picker')
            self._gray_values.append({
                'name': gr.get('name', 'Gray Value'), 'kind': kind,
                'orientation': gr.get('orientation', 'XY'),
                'slice': int(gr.get('slice', 0)),
                'alignment': int(gr.get('alignment', 0)),
                'points': gr.get('points'),
                'view_range': gr.get('view_range'),
                'graphic': None,
            })
            if kind in self._gray_counter:
                self._gray_counter[kind] += 1
        self._refresh_gray_tree()

    # ── Surfaces (project tree) ───────────────────────────────────────────────
    def _reset_surfaces(self):
        self._surfaces = []
        self._project_meshes = []
        self._surface_counter = 0
        self._active_surface_rec = None
        self.view_3d.clear_surface()
        self._refresh_surface_tree()
        self._refresh_project_meshes_tree()
        self._refresh_surface_outline()

    def _serialize_meshes(self, project_path):
        """Write each surface / project mesh as a binary STL next to the project
        file and return (surfaces, project_meshes) lists of JSON-safe entries that
        reference those files. Geometry lives in the STLs, not the .voxels JSON."""
        project_dir = os.path.dirname(os.path.abspath(project_path))
        base = os.path.splitext(os.path.basename(project_path))[0]

        def dump(records, suffix):
            entries = []
            for i, rec in enumerate(records):
                fname = f'{base}_{suffix}_{i + 1}.stl'
                try:
                    _write_stl(os.path.join(project_dir, fname),
                               rec['verts'], rec['faces'])
                except Exception:
                    continue
                shape = rec.get('shape')
                entries.append({
                    'name': rec.get('name', 'Surface'), 'file': fname,
                    'preset': rec.get('preset'), 'level': rec.get('level'),
                    'shape': list(shape) if shape is not None else None,
                })
            return entries

        return dump(self._surfaces, 'surface'), dump(self._project_meshes, 'mesh')

    def _restore_meshes(self, project, project_dir):
        """Reload surfaces / project meshes from the STLs saved beside the
        project, rebuild the tree, and show the last surface in 3D."""
        self._surfaces = []
        self._project_meshes = []
        self._surface_counter = 0

        def load(entries):
            recs = []
            for entry in entries or []:
                fname = entry.get('file')
                if not fname:
                    continue
                mesh = _read_stl(os.path.join(project_dir, fname))
                if mesh is None:
                    continue
                verts, faces = mesh
                shape = entry.get('shape')
                recs.append({
                    'name': entry.get('name', 'Surface'),
                    'verts': verts, 'faces': faces,
                    'preset': entry.get('preset'), 'level': entry.get('level'),
                    'shape': tuple(shape) if shape is not None else None,
                })
            return recs

        self._surfaces = load(project.get('surfaces'))
        self._project_meshes = load(project.get('project_meshes'))
        self._surface_counter = len(self._surfaces)
        self._refresh_surface_tree()
        self._refresh_project_meshes_tree()
        if self._surfaces:
            self._show_surface_in_3d(self._surfaces[-1])
        else:
            self.view_3d.clear_surface()
            self._refresh_surface_outline()

    def _refresh_surface_tree(self):
        self.project_tree.set_surfaces([{'name': s['name']} for s in self._surfaces])
        # 'Surface Mesh' renderer is only offered while a surface exists.
        self.view_3d.set_surface_available(len(self._surfaces) > 0)

    def _refresh_project_meshes_tree(self):
        self.project_tree.set_project_meshes(
            [{'name': m['name']} for m in self._project_meshes])

    def _save_mesh_stl(self, rec):
        """Prompt for a path and write ``rec``'s mesh as a binary STL."""
        default = (rec.get('name') or 'surface').replace(' ', '_') + '.stl'
        path, _ = QFileDialog.getSaveFileName(self, 'Export As STL', default,
                                              'STL Mesh (*.stl);;All Files (*)')
        if not path:
            return
        if not path.lower().endswith('.stl'):
            path += '.stl'
        try:
            _write_stl(path, rec['verts'], rec['faces'])
        except Exception as exc:
            QMessageBox.warning(self, 'Export As STL', f'Could not write the STL file:\n{exc}')

    def _export_surface_stl(self, index):
        if 0 <= index < len(self._surfaces):
            self._save_mesh_stl(self._surfaces[index])

    def _promote_surface(self, index):
        """Add a copy of a surface as a top-level mesh object in the tree."""
        if not (0 <= index < len(self._surfaces)):
            return
        src = self._surfaces[index]
        self._project_meshes.append({
            'name': src['name'], 'verts': src['verts'], 'faces': src['faces'],
            'preset': src.get('preset'), 'level': src.get('level'),
            'shape': src.get('shape'),
        })
        self._refresh_project_meshes_tree()
        self._mark_dirty()

    def _show_project_mesh(self, index):
        if 0 <= index < len(self._project_meshes):
            m = self._project_meshes[index]
            self.view_3d.show_surface(m['verts'], m['faces'], _SURFACE_COLOR,
                                      source_shape=m.get('shape'))
            self._refresh_surface_outline()

    def _export_project_mesh_stl(self, index):
        if 0 <= index < len(self._project_meshes):
            self._save_mesh_stl(self._project_meshes[index])

    def _remove_project_mesh(self, index):
        if not (0 <= index < len(self._project_meshes)):
            return
        self._project_meshes.pop(index)
        self._refresh_project_meshes_tree()
        self._mark_dirty()

    def open_create_surface(self):
        if not self.volume_data.is_loaded():
            QMessageBox.warning(self, 'Create Surface', 'Please import a volume first.')
            return
        if measure is None:
            QMessageBox.warning(self, 'Create Surface',
                                'scikit-image is required to build a surface mesh.')
            return
        dlg = SurfaceCreationDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        rec = self._build_surface(dlg.preset())
        if rec is None:
            return
        self._surface_counter += 1
        rec['name'] = f'Surface {self._surface_counter}'
        self._surfaces.append(rec)
        self._show_surface_in_3d(rec)
        self._refresh_surface_tree()
        self._mark_dirty()

    def _build_surface(self, preset):
        """Marching cubes on the raw volume at the current isovalue and ``preset``
        resolution (the sharp, high-quality mesh, exactly as before), then rigidly
        transform it into the active alignment frame so it sits on the displayed
        isosurface. The rotation/translation is rigid, so the mesh quality is
        unchanged. Returns {'verts','faces','preset','level','shape'} or None."""
        factor = _SURFACE_PRESETS.get(preset, 2)
        vol = self.volume_data.volume
        vmin, vmax = float(np.min(vol)), float(np.max(vol))
        if vmax <= vmin:
            QMessageBox.warning(self, 'Create Surface', 'The volume has no contrast.')
            return None
        level = vmin + self.view_3d.iso_threshold_percent / 100.0 * (vmax - vmin)
        if not (vmin < level < vmax):
            QMessageBox.warning(self, 'Create Surface',
                                'The current isovalue is out of range — adjust it first.')
            return None

        def build(_on_progress):
            ds = vol[::factor, ::factor, ::factor]
            ds = ds if ds.dtype == np.float32 else ds.astype(np.float32)
            try:
                verts, faces, _n, _v = measure.marching_cubes(ds, level=level)
            except (MemoryError, ValueError, RuntimeError) as exc:
                return ('error', str(exc))
            if factor > 1:
                verts = verts * float(factor)
            return (verts.astype(np.float32), faces.astype(np.int32))

        result = self._load_with_progress(
            build, label=f'Building {preset.lower()}-resolution surface…',
            title='Create Surface')
        if result is None or isinstance(result[0], str):   # None or ('error', msg)
            QMessageBox.warning(self, 'Create Surface',
                                'Could not build a surface at this isovalue.')
            return None
        verts, faces = result
        if len(verts) == 0 or len(faces) == 0:
            QMessageBox.warning(self, 'Create Surface',
                                'No surface exists at the current isovalue.')
            return None
        # Apply the current alignment transform afterwards (rigid → no quality
        # loss). The cumulative transform maps display(output)→input as
        # in = R @ out + offset, so raw verts map to out = (in - offset) @ R.
        if self._align_active:
            verts = ((verts.astype(np.float64) - self._align_offset)
                     @ self._align_R).astype(np.float32)
            shape = tuple(int(s) for s in self._align_shape)
        else:
            shape = tuple(int(s) for s in vol.shape)
        return {'verts': verts, 'faces': faces, 'preset': preset, 'level': level,
                'shape': shape}

    def _show_surface_in_3d(self, rec):
        self._active_surface_rec = rec
        self.view_3d.show_surface(rec['verts'], rec['faces'], _SURFACE_COLOR,
                                  source_shape=rec.get('shape'))
        self._refresh_surface_outline()

    def _remap_surfaces(self, old_R, old_offset):
        """Re-express every child surface mesh in the new active-alignment display
        frame after the alignment changed from (old_R, old_offset) to the current
        one, so the surfaces stay registered with the volume.

        Verts are stored in the active alignment's output frame, related to the
        raw input volume by ``in = out @ R.T + offset`` (R is a rotation). So we
        map each vertex back to raw input coords through the old transform, then
        forward into the new display frame."""
        if not self._surfaces:
            return
        old_R   = np.asarray(old_R,      dtype=np.float64)
        old_off = np.asarray(old_offset, dtype=np.float64)
        new_R   = np.asarray(self._align_R,      dtype=np.float64)
        new_off = np.asarray(self._align_offset, dtype=np.float64)
        new_shape = (tuple(int(s) for s in self._align_shape)
                     if self._align_shape is not None else None)
        for rec in self._surfaces:
            v = rec.get('verts')
            rec['shape'] = new_shape
            if v is None or len(v) == 0:
                continue
            raw = v.astype(np.float64) @ old_R.T + old_off
            rec['verts'] = ((raw - new_off) @ new_R).astype(np.float32)
        # Re-show the active surface so the 3D mesh + 2D outline pick up the move.
        if self.view_3d._showing_surface and self._active_surface_rec in self._surfaces:
            self._show_surface_in_3d(self._active_surface_rec)

    def _refresh_surface_outline(self):
        """Push the active surface's slice-intersection outline to the 2D views
        (or clear it when no surface is shown)."""
        v3 = self.view_3d
        tri = getattr(v3, '_surface_tri', None)
        if v3._showing_surface and tri is not None and v3._surface_shape is not None:
            for sv in self._slice_viewers():
                sv.set_surface_outline(tri, v3._surface_shape)
        else:
            for sv in self._slice_viewers():
                sv.clear_surface_outline()

    def _goto_surface(self, index):
        if 0 <= index < len(self._surfaces):
            self._show_surface_in_3d(self._surfaces[index])

    def _remove_surface(self, index):
        if not (0 <= index < len(self._surfaces)):
            return
        self._surfaces.pop(index)
        self._refresh_surface_tree()   # updates 'Surface Mesh' availability
        if self._surfaces:
            self._show_surface_in_3d(self._surfaces[-1])
        else:
            self.view_3d.clear_surface()   # no surfaces left → back to the volume
            self._refresh_surface_outline()
        self._mark_dirty()

    # ── View-time alignment state ────────────────────────────────────────────
    def _reset_alignment_state(self):
        """Reset the cumulative alignment to identity for the loaded volume.

        Does not touch the viewers directly — the following update_views() shows
        the raw volume (set_volume already clears any permanent transform)."""
        self._align_active = False
        self._align_R      = np.eye(3, dtype=np.float64)
        self._align_offset = np.zeros(3, dtype=np.float64)
        self._aligned_cache = self._aligned_key = None   # free any cached frame
        if self.volume_data is not None and self.volume_data.volume is not None:
            self._align_shape = self.volume_data.volume.shape
        else:
            self._align_shape = None

    def _restore_alignment(self, align):
        """Set the cumulative alignment from a saved project (sets state only;
        update_views() pushes it to the viewers). Handles the current
        single-transform dict, the legacy list of 3-2-1 ops, and the legacy
        non-destructive dict without a stored output shape."""
        if not align or not self.volume_data.is_loaded():
            return
        in_shape = self.volume_data.volume.shape
        if isinstance(align, list):
            # Legacy stack of 3-2-1 ops, each picked in the then-current frame.
            R_cum, off_cum = np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
            shape_cum = in_shape
            for op in align:
                R2b, off2b, out_shape = _alignment_bbox(
                    np.array(op['R'], dtype=np.float64),
                    np.array(op['origin'], dtype=np.float64), shape_cum)
                R_cum, off_cum = _compose_alignment(R_cum, off_cum, R2b, off2b)
                shape_cum = out_shape
            R, offset, out_shape = R_cum, off_cum, shape_cum
        elif isinstance(align, dict):
            R = np.array(align['R'], dtype=np.float64)
            offset = np.array(align['offset'], dtype=np.float64)
            out_shape = align.get('out_shape')
            if out_shape is None:
                # Legacy dict: offset is already recentred; reconstruct a shape
                # large enough to hold the rotated volume.
                corners = np.array(
                    [(i, j, k)
                     for i in (0.0, in_shape[0] - 1)
                     for j in (0.0, in_shape[1] - 1)
                     for k in (0.0, in_shape[2] - 1)], dtype=np.float64)
                out = (corners - offset) @ R
                out_shape = tuple(max(1, int(np.ceil(out[:, i].max())) + 2)
                                  for i in range(3))
        else:
            return
        self._align_active = True
        self._align_R      = np.asarray(R, dtype=np.float64)
        self._align_offset = np.asarray(offset, dtype=np.float64)
        self._align_shape  = tuple(int(s) for s in out_shape)

    # Pre-resample the whole aligned volume into a shared cache when the output
    # frame is at most this many float32 bytes (~1.5 GB). Above this we fall back
    # to per-slice sampling so huge volumes don't blow up memory.
    _MAX_ALIGNED_CACHE_BYTES = 1_500_000_000

    def _aligned_volume_cache(self, R, offset, out_shape):
        """Full volume resampled into the aligned output frame (float32), cached
        and reused while the transform is unchanged. Returns None if scipy is
        missing or the frame is too large (caller then samples per slice)."""
        key = (R.tobytes(), offset.tobytes(), tuple(out_shape))
        if self._aligned_key == key and self._aligned_cache is not None:
            return self._aligned_cache
        vol = self.volume_data.volume
        if vol is None or _scipy_affine_transform is None:
            self._aligned_cache = self._aligned_key = None
            return None
        if int(np.prod(out_shape)) * 4 > self._MAX_ALIGNED_CACHE_BYTES:
            self._aligned_cache = self._aligned_key = None
            return None
        try:
            aligned = _scipy_affine_transform(
                vol, R, offset=offset, output_shape=tuple(out_shape),
                order=1, mode='constant', cval=0.0, output=np.float32)
        except Exception:
            self._aligned_cache = self._aligned_key = None
            return None
        self._aligned_cache = aligned
        self._aligned_key = key
        return aligned

    def _set_view_alignment(self, R, offset, out_shape):
        """Make (R, offset, out_shape) the active cumulative alignment and push it
        to every viewport as a non-destructive, view-time transform."""
        self._mark_dirty()
        old_R, old_offset = self._align_R, self._align_offset
        self._align_active = True
        self._align_R      = np.asarray(R,      dtype=np.float64)
        self._align_offset = np.asarray(offset, dtype=np.float64)
        self._align_shape  = tuple(int(s) for s in out_shape)
        # Resample the whole aligned volume once (shared by all viewers) so 2D
        # pan/scroll stays fast; viewers fall back to per-slice sampling if None.
        aligned = self._aligned_volume_cache(self._align_R, self._align_offset,
                                             self._align_shape)
        for sv in self._slice_viewers():
            sv.set_permanent_transform(self._align_R, self._align_offset,
                                       self._align_shape, aligned)
        self._apply_alignment_to_3d(self._align_R, self._align_offset)
        # Carry the child surface meshes into the new alignment frame (after the
        # 3D volume is re-aligned, so the re-shown surface is placed correctly).
        self._remap_surfaces(old_R, old_offset)

    def _apply_new_alignment(self, R2, offset0, title, kind='321'):
        """Compose a freshly-picked transform (in the currently displayed frame)
        onto the cumulative alignment and apply it view-time.

        ``offset0`` is the pre-bounding-box scipy offset (a picked origin point
        for 3-2-1/geometry, or the dialog offset for Simple Alignment)."""
        if _scipy_affine_transform is None:
            QMessageBox.warning(self, title,
                                'scipy is required to apply the transformation.')
            return
        display_shape = self._align_shape
        if display_shape is None:
            return
        # Bounding-box-recentre the new transform within the current display frame.
        R2, off2, out_shape = _alignment_bbox(R2, offset0, display_shape)
        # Compose under the existing display→input transform.
        R, offset = _compose_alignment(self._align_R, self._align_offset, R2, off2)
        self._set_view_alignment(R, offset, out_shape)
        self._record_alignment(kind)

    def reset_alignment(self):
        """User action: drop the view-time alignment and show the raw volume
        (i.e. re-activate the Initial Alignment)."""
        if not self.volume_data.is_loaded():
            return
        if self._alignments:
            self._activate_alignment(0)
        else:
            self._mark_dirty()
            self._show_raw_alignment()

    def _apply_321(self, R, origin_pt):
        # Points were picked in the currently displayed frame; origin_pt is the
        # pre-bounding-box scipy offset for this transform.
        self._apply_new_alignment(R, origin_pt, '3-2-1 Alignment')

    def _apply_alignment_to_3d(self, R, offset):
        """Render a downsampled, view-time aligned copy in the 3D viewport.

        Only a downsampled copy is resampled (capped well under the render
        budget), so this stays interactive even for large volumes."""
        orig   = self.volume_data.volume
        if orig is None:
            return
        factor = max(1, int(np.ceil((orig.size / 256 ** 3) ** (1.0 / 3.0))))
        vol_ds = orig[::factor, ::factor, ::factor]
        # The cumulative affine is in full-res voxels; in downsampled space the
        # rotation is unchanged and the offset scales by 1/factor.
        R_ds, off_ds, out_shape_ds = _alignment_bbox(
            R, np.asarray(offset, dtype=np.float64) / factor, vol_ds.shape)
        try:
            resampled = _scipy_affine_transform(
                vol_ds, R_ds, offset=off_ds,
                output_shape=out_shape_ds,
                order=1, mode='constant', cval=0.0,
            )
            self.view_3d.set_permanent_volume(resampled)
        except Exception as exc:
            print('3D alignment failed:', exc)

    def on_point_placed(self, point):
        dlg = self._alignment_dialog
        if dlg is not None and dlg.isVisible():
            dlg.add_point(point)
            self.view_3d.set_alignment_overlays(dlg.points)
            return
        gdlg = self._geom_align_dialog
        if gdlg is not None and gdlg.isVisible():
            gdlg.add_point(point)
            self._on_geom_align_changed()


def main():
    # High-DPI support — required on Windows (e.g. 150% scaling) so icons and
    # other pixmaps render at the correct size instead of being clipped.
    # These must be set before the QApplication is constructed. They are
    # harmless / no-ops on macOS (Retina is handled natively) and on Qt6.
    try:
        QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:
        pass
    for attr in ('AA_EnableHighDpiScaling', 'AA_EnableHighDpiPixmaps'):
        try:
            QtCore.QCoreApplication.setAttribute(getattr(Qt, attr), True)
        except Exception:
            pass

    QtCore.QCoreApplication.setApplicationName('Voxels Viewer')
    QtCore.QCoreApplication.setOrganizationName('Voxels Viewer')
    # On macOS the application menu name (About/Quit) is read from the process
    # bundle name before Qt builds the menu — which is "Python" for a plain
    # script. Override it via the Cocoa bundle so it reads "Voxels Viewer".
    # Requires pyobjc (pip install pyobjc-framework-Cocoa); harmless if absent.
    if sys.platform == 'darwin':
        try:
            from Foundation import NSBundle
            bundle = NSBundle.mainBundle()
            info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if info is not None:
                info['CFBundleName'] = 'Voxels Viewer'
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
