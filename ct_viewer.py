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
APP_BUILD = 12
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
<p style="color:gray;">Copyright &copy; 2026 Stepan Rumyantsev. All rights reserved.</p>

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
            self.setLow(value)
        elif self._drag == 'high':
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
        span_col = (pal.color(QtGui.QPalette.Highlight) if enabled
                    else pal.color(QtGui.QPalette.Mid))
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


def _measure_icon(kind: str) -> QtGui.QIcon:
    """Monochrome symbolic icon for a measurement tool, drawn at 1× and 2×
    into a 20×20 logical-pixel canvas (matches the lock-icon style)."""
    icon = QtGui.QIcon()
    for dpr in (1.0, 2.0):
        S_phys = round(20 * dpr)
        px = QtGui.QPixmap(S_phys, S_phys)
        px.setDevicePixelRatio(dpr)
        px.fill(Qt.transparent)

        p = QtGui.QPainter(px)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.scale(S_phys / 20.0, S_phys / 20.0)

        col = QtGui.QColor('#cccccc')
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
        pts = []
        for _name, sp in self.roi.getSceneHandlePositions():
            vp = self.vb.mapSceneToView(sp)
            pts.append(np.array([vp.x(), vp.y()], dtype=float))
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


def _grayvalue_icon(kind: str) -> QtGui.QIcon:
    """Monochrome symbolic icon for a gray-value tool (picker / profile)."""
    icon = QtGui.QIcon()
    for dpr in (1.0, 2.0):
        S_phys = round(20 * dpr)
        px = QtGui.QPixmap(S_phys, S_phys)
        px.setDevicePixelRatio(dpr)
        px.fill(Qt.transparent)
        p = QtGui.QPainter(px)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.scale(S_phys / 20.0, S_phys / 20.0)
        pen = QtGui.QPen(QtGui.QColor('#cccccc'), 1.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
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

        # Right-click the line (or curve) exports the profile to CSV.
        self._orig_roi_click = self.roi.mouseClickEvent
        self.roi.mouseClickEvent = self._on_line_click
        self.curve.setClickable(True, width=8)
        self.curve.mouseClickEvent = self._on_curve_click

        self.roi.sigRegionChanged.connect(self.update)
        self._profile = None    # (distance_mm, values, px, py) for export
        self.update()

    def _endpoints(self):
        pts = []
        for _n, sp in self.roi.getSceneHandlePositions():
            vp = self.vb.mapSceneToView(sp)
            pts.append(np.array([vp.x(), vp.y()], dtype=float))
        return pts

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

    def _on_line_click(self, ev):
        if ev.button() == Qt.RightButton:
            ev.accept()
            self._export_csv()
        else:
            self._orig_roi_click(ev)

    def _on_curve_click(self, ev):
        if ev.button() == Qt.RightButton:
            ev.accept()
            self._export_csv()

    def _export_csv(self):
        if self._profile is None:
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
        self._display_levels = None
        self._auto_range_pending = True
        self._preview_R      = None   # 3×3 rotation for live Simple Alignment preview
        self._preview_offset = None   # (3,) offset
        self._preview_vol    = None   # original volume (not resampled)
        self._perm_R      = None      # permanent non-destructive alignment transform
        self._perm_offset = None
        self._perm_shape  = None      # output bounding-box shape for the permanent transform
        self._axis_lines  = {}   # axis -> pg.InfiniteLine
        self._axis_timers = {}   # axis -> QTimer
        self._lines_pinned = False
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
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(28)
        bottom = QHBoxLayout(bottom_bar)
        bottom.setContentsMargins(2, 2, 2, 2)
        bottom.addWidget(self.slice_slider)
        layout.addWidget(bottom_bar)

    def eventFilter(self, obj, event):
        gv = self.image_view.ui.graphicsView
        if obj is self.image_view and event.type() == _EV_RESIZE:
            self._reposition_tripod()
            self._reposition_auto_hint()
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

    def set_locked(self, locked):
        self.lock_button.setIcon(_lock_icon(locked))

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
        self._measurements.append(m)

    def clear_measurements(self):
        for m in self._measurements:
            m.remove()
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
        self._gray_items.append(obj)

    def clear_gray_values(self):
        for g in self._gray_items:
            g.remove()
        self._gray_items = []

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

    def _apply_slider_range(self, recenter=True):
        """Set the slider maximum from the active frame's traversal depth."""
        shape = self._active_shape()
        if shape is None:
            self.slice_slider.setMaximum(0)
            return
        depth = shape[self._traversal_axis()]
        if recenter:
            self.current_index = depth // 2
        else:
            self.current_index = int(np.clip(self.current_index, 0, depth - 1))
        self.slice_slider.setMaximum(depth - 1)
        self.slice_slider.setValue(self.current_index)

    def set_volume(self, volume_data):
        self.volume_data = volume_data
        self._display_levels = None
        self._auto_range_pending = True
        self._perm_R = None
        self._perm_offset = None
        self._perm_shape = None
        self.clear_measurements()
        self.clear_gray_values()
        if volume_data is None or not volume_data.is_loaded():
            self.image_view.clear()
            self.slice_slider.setMaximum(0)
            return
        self._apply_slider_range(recenter=True)
        self.update_image()

    def on_slice_changed(self, value):
        self.current_index = value
        self.clear_measurements()
        self.clear_gray_values()
        self.update_image()
        shape = self._active_shape()
        if shape is not None:
            ax_idx = self._traversal_axis()
            axis   = {'XY': 'Z', 'YZ': 'X', 'XZ': 'Y'}[self.orientation]
            self.axis_position_changed.emit(axis, value, shape[ax_idx])

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

    def set_permanent_transform(self, R, offset, out_shape):
        """Set a persistent non-destructive alignment transform (survives preview cycles).

        ``out_shape`` is the bounding-box shape of the aligned output frame; the
        viewer samples / scrolls over this frame rather than the raw volume.
        """
        self._perm_R      = np.array(R,      dtype=np.float64)
        self._perm_offset = np.array(offset, dtype=np.float64)
        self._perm_shape  = tuple(int(s) for s in out_shape)
        self.clear_measurements()
        self.clear_gray_values()
        self._auto_range_pending = True
        self._apply_slider_range(recenter=True)
        self.update_image()

    def clear_permanent_transform(self):
        self._perm_R      = None
        self._perm_offset = None
        self._perm_shape  = None
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

    def show_axis_line(self, axis, index, total, seg_range=None):
        # Compute angle + position for this viewport's coordinate space.
        # Vertical (angle=90): the axis is the horizontal dimension here, no flip.
        # Horizontal (angle=0): the axis is the vertical dimension, always flipped ([::-1]).
        if self.orientation == 'XY':
            if axis == 'X':
                angle, pos = 90, index
            elif axis == 'Y':
                angle, pos = 0, total - 1 - index
            else:
                return  # Z is this viewport's own traversal axis
        elif self.orientation == 'YZ':
            if axis == 'Y':
                angle, pos = 90, index
            elif axis == 'Z':
                angle, pos = 0, total - 1 - index
            else:
                return  # X is this viewport's own traversal axis
        elif self.orientation == 'XZ':
            if axis == 'X':
                angle, pos = 90, index
            elif axis == 'Z':
                angle, pos = 0, total - 1 - index
            else:
                return  # Y is this viewport's own traversal axis
        else:
            return

        color = _AXIS_LINE_COLORS[axis]
        pen_dash  = pg.mkPen(color=color, width=1, style=Qt.DashLine)
        pen_solid = pg.mkPen(color=color, width=2, style=Qt.SolidLine)

        # Remove any existing items for this axis before redrawing.
        view = self.image_view.getView()
        if axis in self._axis_lines:
            for item in self._axis_lines[axis]:
                view.removeItem(item)

        # Always use InfiniteLine for the dashed full extent — it never
        # influences ViewBox auto-range, so the image scale is preserved.
        items = [pg.InfiniteLine(pos=pos, angle=angle, pen=pen_dash)]

        if seg_range is not None:
            a, b = min(seg_range), max(seg_range)
            if angle == 0:  # horizontal at y=pos
                seg_item = pg.PlotDataItem([a, b], [pos, pos], pen=pen_solid)
            else:           # vertical at x=pos
                seg_item = pg.PlotDataItem([pos, pos], [a, b], pen=pen_solid)
            items.append(seg_item)

        for item in items:
            view.addItem(item, ignoreBounds=True)
        self._axis_lines[axis] = items

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
        if axis in self._axis_lines:
            view = self.image_view.getView()
            for item in self._axis_lines.pop(axis):
                view.removeItem(item)
        if axis in self._axis_timers:
            self._axis_timers.pop(axis).stop()

    def set_lines_pinned(self, pinned):
        self._lines_pinned = pinned
        if pinned:
            for timer in self._axis_timers.values():
                timer.stop()
        else:
            # Stop timers and remove all lines immediately.
            for timer in list(self._axis_timers.values()):
                timer.stop()
            self._axis_timers.clear()
            view = self.image_view.getView()
            for items in list(self._axis_lines.values()):
                for item in items:
                    view.removeItem(item)
            self._axis_lines.clear()

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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.volume_data = None
        self.orientation = '3D'
        self.mode = 'Isosurface'
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
        self.init_ui()

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
        self.mode_combo.addItems(['Isosurface', 'Phong Volume'])
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
        header.addStretch()
        self.maximize_button = QPushButton('▲')
        self.maximize_button.setFixedSize(24, 24)
        header.addWidget(self.maximize_button)
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

        self.status_text = QLabel('')
        self.status_text.setStyleSheet('color: red;')

        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(28)
        bottom = QHBoxLayout(bottom_bar)
        bottom.setContentsMargins(2, 2, 2, 2)
        bottom.addWidget(self.status_text, 1)
        layout.addWidget(bottom_bar)

    def eventFilter(self, obj, event):
        if obj is self.gl_view:
            et = event.type()
            if et == _EV_RESIZE:
                self._reposition_tripod()
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

    # ── Alignment helpers ────────────────────────────────────────────────────

    def set_alignment_mode(self, active):
        self._alignment_mode = active
        self.gl_view.set_pick_mode(active)

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
        iso_active = (text == 'Isosurface')
        self.iso_slider.setEnabled(iso_active)
        self.iso_spinbox.setEnabled(iso_active)
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
        self.update_view()

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

    def update_view(self):
        self.gl_view.clear()
        self._alignment_gl_items = []
        self.status_text.setText('')
        if self.volume_data is None or self.volume_data.volume is None:
            self._render_factor = 1
            self._render_shape = (1, 1, 1)
            self._render_volume = None
            return
        # Use permanent aligned display volume if one has been set.
        raw = self._perm_volume if self._perm_volume is not None else self.volume_data.volume
        # Downsample FIRST (keeps the peak RAM from astype small)
        max_voxels = self._max_render_voxels()
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
        if self.mode == 'Isosurface':
            self.render_isosurface(volume)
        else:
            self.render_volume(volume)
        distance = max(raw.shape) * 2.5
        self.gl_view.setCameraPosition(distance=distance)
        self._rebuild_alignment_overlays()

    def render_isosurface(self, volume):
        if measure is None:
            self.status_text.setText('Install scikit-image for ISO surface rendering')
            return
        if volume.ndim != 3 or any(dim < 2 for dim in volume.shape):
            self.status_text.setText('3D surface unavailable for this volume')
            return
        volume_min = float(np.min(volume))
        volume_max = float(np.max(volume))
        if volume_max <= volume_min:
            self.status_text.setText('No surface available for constant volume')
            return

        threshold = volume_min + self.iso_threshold_percent / 100.0 * (volume_max - volume_min)
        if not (volume_min < threshold < volume_max):
            self.status_text.setText('ISO threshold out of range')
            return

        try:
            verts, faces, normals, _ = measure.marching_cubes(volume, level=threshold)
        except MemoryError:
            self.status_text.setText('Surface too large to build at this threshold')
            return
        except Exception as exc:
            print('Volume surface error:', exc)
            self.status_text.setText('No iso-surface found at this threshold')
            return

        # Guard against a pathological surface (huge mesh → RAM blow-up / hang).
        if len(verts) > 12_000_000:
            self.status_text.setText(
                'Surface too complex to display — adjust the isovalue')
            return

        meshdata = gl.MeshData(vertexes=verts, faces=faces)
        mesh = gl.GLMeshItem(meshdata=meshdata, smooth=True, shader='shaded', drawEdges=False, glOptions='opaque')
        mesh.translate(-volume.shape[0] / 2.0, -volume.shape[1] / 2.0, -volume.shape[2] / 2.0)
        self.gl_view.addItem(mesh)

    def render_volume(self, volume):
        try:
            if volume.ndim != 3:
                self.status_text.setText('Volume rendering requires a 3D scalar volume')
                return
            vol_min = float(np.min(volume))
            vol_max = float(np.max(volume))
            if vol_max <= vol_min:
                self.status_text.setText('Volume rendering requires a non-constant volume')
                return

            # Map the histogram window [lo, hi] to [0, 1]; values outside the
            # window clamp so they become fully transparent / fully opaque.
            if self._display_levels is not None:
                lo, hi = self._display_levels
            else:
                lo, hi = vol_min, vol_max
            if hi <= lo:
                hi = lo + 1e-6
            normalized = np.clip((volume - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

            # Estimate surface normals via volume gradient for Phong-style shading.
            # Compute one axis at a time and cast to float32 immediately to avoid
            # np.gradient's float64 promotion holding 3× full-volume arrays at once.
            gx = np.gradient(normalized, axis=0).astype(np.float32)
            gy = np.gradient(normalized, axis=1).astype(np.float32)
            gz = np.gradient(normalized, axis=2).astype(np.float32)
            gnorm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2).astype(np.float32) + 1e-8
            nx = gx / gnorm; del gx
            ny = gy / gnorm; del gy
            nz = gz / gnorm; del gz, gnorm

            # Diffuse term from a fixed overhead-diagonal light (1,1,1) normalised
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
            # Alpha ramps with intensity so air/background stays transparent
            rgba[..., 3] = np.clip(normalized * 220.0, 0, 220).astype(np.uint8)
            del normalized, gray

            vol_item = gl.GLVolumeItem(
                rgba,
                sliceDensity=1,
                smooth=True,
                glOptions='translucent',
            )
            vol_item.translate(-rgba.shape[0] / 2.0, -rgba.shape[1] / 2.0, -rgba.shape[2] / 2.0)
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
        self._sync_locked = bool(self.preferences.get('sync_locked', True))
        self._syncing = False
        self._project_source = None
        # Single cumulative view-time alignment (output→input, scipy convention
        # in = R @ out + offset). Applied at display time; the voxel data is
        # never modified. _align_active is False when showing the raw volume.
        self._align_active = False
        self._align_R      = np.eye(3, dtype=np.float64)
        self._align_offset = np.zeros(3, dtype=np.float64)
        self._align_shape  = None   # output bounding-box shape (current display frame)
        self.setup_ui()
        self.create_menu()
        self._apply_theme()
        self._restore_layout()

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

    def setup_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        self.left_panel = BrightnessCurveWidget(self)
        self.left_panel.setMinimumWidth(200)
        self.left_panel.setMaximumWidth(420)
        self.left_panel.curve_changed.connect(self.on_curve_changed)

        grid_widget = QWidget()
        grid_widget.setMinimumWidth(800)
        grid_layout = QGridLayout(grid_widget)
        self.view_xy = SliceViewer('XY', self)
        self.view_yz = SliceViewer('YZ', self)
        self.view_xz = SliceViewer('XZ', self)
        self.view_3d = VolumeRender3D(self)

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
        self.splitter.addWidget(self.left_panel)
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

        meas_tool = self.preferences.get('measurement_tool', 'distance')
        for sv in (self.view_xy, self.view_yz, self.view_xz):
            vb = sv.image_view.getView().getViewBox()
            vb.sigRangeChanged.connect(
                lambda *args, sv=sv: self._on_viewport_range_changed(sv))
            sv.lock_clicked.connect(
                lambda sv=sv: self._on_lock_clicked(sv))
            sv.set_measurement_tool(meas_tool)
            sv.measurement_tool_changed.connect(self._on_measurement_tool_changed)

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

        operations_menu = menubar.addMenu('&Operations')
        alignment_menu = operations_menu.addMenu('Alignment')
        simple_align = QtWidgets.QAction('Simple Alignment...', self)
        simple_align.triggered.connect(self.open_simple_alignment)
        point_align = QtWidgets.QAction('3-2-1 Alignment...', self)
        point_align.triggered.connect(self.open_point_alignment)
        reset_align = QtWidgets.QAction('Reset Alignment', self)
        reset_align.triggered.connect(self.reset_alignment)
        alignment_menu.addAction(simple_align)
        alignment_menu.addAction(point_align)
        alignment_menu.addSeparator()
        alignment_menu.addAction(reset_align)
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
        else:
            grid_layout.removeWidget(widget)
            for v in viewers:
                r, c = positions[v]
                grid_layout.addWidget(v, r, c)
                v.show()
            self.maximized_widget.maximize_button.setText('▲')
            self.maximized_widget = None
        self._save_layout_prefs()

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
        self.update_views()

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
        self.update_views()

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
        if not self.volume_data.is_loaded() or self._project_source is None:
            QMessageBox.warning(self, 'Save Project',
                                'No volume loaded. Please import data first.')
            return
        start_dir = self.preferences.get('last_import_dir', '')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Voxels Project', start_dir, 'Voxels Project (*.voxels)')
        if not path:
            return
        if not path.endswith('.voxels'):
            path += '.voxels'

        def vp_state(viewer):
            xr, yr = viewer.image_view.getView().getViewBox().viewRange()
            return {'slice': viewer.current_index,
                    'view_range': [list(xr), list(yr)]}

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
        self.volume_data.set_volume(volume, tuple(voxels))
        self._project_source = project['source']
        self._save_import_dir(path)

        # Restore the alignment as a view-time transform (no resample). Handles
        # the current single-transform dict, the legacy list of 3-2-1 ops, and
        # the legacy non-destructive dict. update_views() pushes it to the views.
        self._reset_alignment_state()
        self._restore_alignment(project.get('alignment'))

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

    def on_curve_changed(self, curve):
        if not self.volume_data.is_loaded() or len(curve) < 4:
            return
        min_val = curve[1][0]
        max_val = curve[2][0]
        for viewer in (self.view_xy, self.view_yz, self.view_xz):
            viewer.set_levels(min_val, max_val)   # fast LUT update, no re-slice
        self.view_3d.set_levels(min_val, max_val)
        # The isosurface ignores the window; only the Phong volume needs a
        # re-render, and that's expensive — debounce it so it runs once the user
        # stops dragging rather than on every slider tick.
        if self.view_3d.mode != 'Isosurface':
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

    def _on_measurement_tool_changed(self, kind):
        """Persist the chosen measurement tool and keep all viewports in sync."""
        self.preferences['measurement_tool'] = kind
        _save_prefs(self.preferences)
        for sv in self._slice_viewers():
            sv.set_measurement_tool(kind)

    def _on_lock_clicked(self, source_viewer):
        self._sync_locked = not self._sync_locked
        self.preferences['sync_locked'] = self._sync_locked
        _save_prefs(self.preferences)
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
            self._show_all_locked_lines()

    def _show_all_locked_lines(self):
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

    def _on_viewport_range_changed(self, source_viewer):
        # Zoom/pan in one viewport surfaces this viewport's coordinate line in
        # the others (and refreshes its solid segment). Always show — in
        # unlocked mode this triggers their appearance and the auto-hide timer;
        # in locked mode the lines are pinned, so this just keeps segments live.
        vol_shape = self._display_shape()
        if vol_shape is not None:
            axis   = {'XY': 'Z', 'YZ': 'X', 'XZ': 'Y'}[source_viewer.orientation]
            ax_idx = {'XY': 2, 'YZ': 0, 'XZ': 1}[source_viewer.orientation]
            src_range = source_viewer.image_view.getView().getViewBox().viewRange()
            for viewer in self._slice_viewers():
                seg = self._compute_seg_range(source_viewer.orientation, src_range,
                                              viewer.orientation, vol_shape)
                viewer.show_axis_line(axis, source_viewer.current_index,
                                     vol_shape[ax_idx], seg)

        # Propagate zoom/pan to all viewports when locked (centered on intersection).
        if not self._sync_locked or self._syncing:
            return
        self._syncing = True
        src_range = source_viewer.image_view.getView().getViewBox().viewRange()
        xr, yr = src_range
        W, H = xr[1] - xr[0], yr[1] - yr[0]
        try:
            self._sync_slices_to_center(source_viewer, src_range)
            self._center_all_on_intersection(W, H)
        finally:
            self._syncing = False

    def _center_all_on_intersection(self, W, H):
        """Set every locked viewport's range to W×H centered on its own line intersection.

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

    # ── View-time alignment state ────────────────────────────────────────────
    def _reset_alignment_state(self):
        """Reset the cumulative alignment to identity for the loaded volume.

        Does not touch the viewers directly — the following update_views() shows
        the raw volume (set_volume already clears any permanent transform)."""
        self._align_active = False
        self._align_R      = np.eye(3, dtype=np.float64)
        self._align_offset = np.zeros(3, dtype=np.float64)
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

    def _set_view_alignment(self, R, offset, out_shape):
        """Make (R, offset, out_shape) the active cumulative alignment and push it
        to every viewport as a non-destructive, view-time transform."""
        self._align_active = True
        self._align_R      = np.asarray(R,      dtype=np.float64)
        self._align_offset = np.asarray(offset, dtype=np.float64)
        self._align_shape  = tuple(int(s) for s in out_shape)
        for sv in self._slice_viewers():
            sv.set_permanent_transform(self._align_R, self._align_offset, self._align_shape)
        self._apply_alignment_to_3d(self._align_R, self._align_offset)

    def _apply_new_alignment(self, R2, offset0, title):
        """Compose a freshly-picked transform (in the currently displayed frame)
        onto the cumulative alignment and apply it view-time.

        ``offset0`` is the pre-bounding-box scipy offset (a picked origin point
        for 3-2-1, or the dialog offset for Simple Alignment)."""
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

    def reset_alignment(self):
        """User action: drop the view-time alignment and show the raw volume."""
        if not self.volume_data.is_loaded():
            return
        self._reset_alignment_state()
        for sv in self._slice_viewers():
            sv.clear_permanent_transform()
        self.view_3d.clear_permanent_volume()
        self.update_views()

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
