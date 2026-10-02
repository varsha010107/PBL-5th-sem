import sys
import os
import csv
import cv2
import numpy as np
import torch

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QMessageBox,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QComboBox,
    QStatusBar,
    QFrame,
    QSizePolicy,
    QScrollArea
)

from PyQt5.QtGui import (
    QPixmap,
    QImage,
    QPainter,
    QPen
)

from PyQt5.QtCore import Qt

SRC_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
from defect_engine import analyze_defect

# =========================================================
# PROJECT PATHS
# =========================================================

BASE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../.."
    )
)

MODEL_PATH = os.path.join(
    BASE,
    "models",
    "day10",
    "spatial_localizer_v2.pth"
)

SRC_PATH = os.path.join(
    BASE,    "src"
)

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


# =========================================================
# IMPORT AI MODEL
# =========================================================

try:

    from heatmap_model import SpatialLocalizer

    AI_AVAILABLE = True

except Exception as e:

    print("AI model import error:", e)

    AI_AVAILABLE = False


# =========================================================
# MAIN APPLICATION
# =========================================================

class SemiconductorLocalizer(QMainWindow):

    def __init__(self):

        super().__init__()

        self.reference_path = None
        self.search_path = None
        self.fabricated_path = None

        # -------------------------------------------------
        # IMAGE VIEWER ZOOM
        # -------------------------------------------------
        self.viewer_zoom = 1.0
        self.viewer_fit_mode = True
        self.viewer_pixmap = None

        self.last_prediction = None

        # Last successful Spatial AI localization
        self.last_localization_x = None
        self.last_localization_y = None
        # Image zoom
        self.zoom_factor = 1.0
        self.current_pixmap = None

        self.ai_model = None

        self.load_ai_model()

        self.setWindowTitle(
            "AI Semiconductor Pattern Localizer"
        )

        self.setGeometry(
            100,
            100,
            1250,
            800
        )

        self.create_ui()


    # =====================================================
    # LOAD AI MODEL
    # =====================================================

    def load_ai_model(self):

        if not AI_AVAILABLE:

            return

        if not os.path.exists(MODEL_PATH):

            print(
                "AI model not found:",
                MODEL_PATH
            )

            return

        try:

            self.ai_model = SpatialLocalizer()

            self.ai_model.load_state_dict(
                torch.load(
                    MODEL_PATH,
                    map_location="cpu"
                )
            )

            self.ai_model.eval()

            print("AI model loaded successfully")

        except Exception as e:

            print(
                "AI model loading failed:",
                e
            )

            self.ai_model = None


    # =====================================================
    # USER INTERFACE
    # =====================================================


    def create_ui(self):

        # =================================================
        # CENTRAL WIDGET
        # =================================================

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(18, 15, 18, 15)
        main_layout.setSpacing(12)


        # =================================================
        # GLOBAL STYLE
        # =================================================

        self.setStyleSheet("""
            QMainWindow {
                background-color: #11151a;
            }

            QWidget {
                color: #e8edf2;
                font-family: Arial;
            }

            QGroupBox {
                border: 1px solid #39434d;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px;
                font-weight: bold;
                color: #9fd3ff;
                background-color: #181e24;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                background-color: #181e24;
            }

            QPushButton {
                background-color: #26313b;
                border: 1px solid #4b5965;
                border-radius: 6px;
                padding: 10px;
                color: #ffffff;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #34424e;
                border: 1px solid #6ea8d7;
            }

            QPushButton:pressed {
                background-color: #1d2730;
            }

            QComboBox {
                background-color: #202831;
                border: 1px solid #4b5965;
                border-radius: 5px;
                padding: 8px;
                color: white;
            }

            QLabel {
                color: #e8edf2;
            }

            QStatusBar {
                background-color: #0c1014;
                color: #8fa3b5;
            }
        """)


        # =================================================
        # HEADER
        # =================================================

        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #18212a;
                border: 1px solid #34414d;
                border-radius: 10px;
            }
        """)

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 10, 18, 10)


        title_layout = QVBoxLayout()

        title = QLabel("FINLOC AI")

        title.setStyleSheet("""
            font-size: 25px;
            font-weight: bold;
            color: #ffffff;
        """)

        subtitle = QLabel(
            "AI-Powered Semiconductor Pattern Localization"
        )

        subtitle.setStyleSheet("""
            font-size: 12px;
            color: #91a4b5;
        """)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header_layout.addLayout(title_layout)

        header_layout.addStretch()


        # AI STATUS

        ai_status = QLabel(
            "●  AI MODEL READY"
            if self.ai_model is not None
            else "●  AI MODEL OFFLINE"
        )

        ai_status.setStyleSheet("""
            QLabel {
                color: #72d572;
                font-weight: bold;
                padding: 8px 14px;
                border: 1px solid #3c6945;
                border-radius: 6px;
                background-color: #17251b;
            }
        """)

        header_layout.addWidget(ai_status)


        version = QLabel("v1.0")

        version.setStyleSheet("""
            color: #718291;
            padding-left: 12px;
        """)

        header_layout.addWidget(version)

        main_layout.addWidget(header)

        # =================================================
        # MAIN CONTENT
        # =================================================

        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        main_layout.addLayout(content_layout)


        # =================================================
        # LEFT CONTROL PANEL
        # =================================================

        control_panel = QVBoxLayout()
        control_panel.setSpacing(7)


        # =================================================
        # RIGHT ANALYSIS PANEL
        # =================================================

        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)


        # =================================================
        # ADD PANELS TO MAIN CONTENT
        # =================================================

        content_layout.addLayout(
            control_panel,
            0
        )

        content_layout.addLayout(
            right_panel,
            1
        )
        
        

        # =================================================
        # REFERENCE IMAGE
        # =================================================

        reference_group = QGroupBox(
            "REFERENCE PATTERN"
        )

        reference_layout = QVBoxLayout()

        self.reference_label = QLabel(
            "No reference image loaded"
        )

        self.reference_label.setAlignment(
            Qt.AlignCenter
        )

        self.reference_label.setMinimumHeight(65)

        self.reference_label.setStyleSheet("""
            QLabel {
                color: #dce6ee;
                background-color: #111820;
                border: 1px dashed #40505c;
                border-radius: 5px;
                padding: 10px;
                font-size: 12px;
            }
        """)

        reference_layout.addWidget(
            self.reference_label
        )


        reference_button = QPushButton(
            "LOAD REFERENCE IMAGE"
        )

        reference_button.clicked.connect(
            self.load_reference
        )

        reference_layout.addWidget(
            reference_button
        )

        reference_group.setLayout(
            reference_layout
        )

        control_panel.addWidget(
            reference_group
        )


        # =================================================
        # SEARCH IMAGE
        # =================================================

        search_group = QGroupBox(
            "WIDE-FIELD SEARCH IMAGE"
        )

        search_layout = QVBoxLayout()

        self.search_label = QLabel(
            "No search image loaded"
        )

        self.search_label.setAlignment(
            Qt.AlignCenter
        )

        self.search_label.setMinimumHeight(65)

        self.search_label.setStyleSheet("""
            QLabel {
                color: #dce6ee;
                background-color: #111820;
                border: 1px dashed #40505c;
                border-radius: 5px;
                padding: 10px;
                font-size: 12px;
            }
        """)

        search_layout.addWidget(
            self.search_label
        )


        search_button = QPushButton(
            "LOAD SEARCH IMAGE"
        )

        search_button.clicked.connect(
            self.load_search
        )

        search_layout.addWidget(
            search_button
        )

        search_group.setLayout(
            search_layout
        )

        control_panel.addWidget(
            search_group
        )


        # =================================================
        # METHOD
        # =================================================

        method_group = QGroupBox(
            "LOCALIZATION ENGINE"
        )

        method_layout = QVBoxLayout()

        self.method_combo = QComboBox()

        self.method_combo.addItems([
            "Template Matching",
            "Spatial AI"
        ])

        self.method_combo.setStyleSheet("""
            QComboBox {
                background-color: #1b2731;
                color: #ffffff;
                border: 1px solid #4b6070;
                border-radius: 4px;
                padding: 8px;
                min-height: 20px;
            }

            QComboBox:hover {
                border: 1px solid #6da9cc;
            }

            QComboBox::drop-down {
                border-left: 1px solid #4b6070;
                width: 28px;
            }

            QComboBox QAbstractItemView {
                background-color: #111820;
                color: #ffffff;
                selection-background-color: #285f7a;
                selection-color: #ffffff;
                border: 1px solid #4b6070;
                padding: 4px;
            }
        """)

        # -------------------------------------------------
        # COMBO BOX STYLE
        # -------------------------------------------------

        self.method_combo.setStyleSheet("""
            QComboBox {
                background-color: #202a33;
                color: #f0f4f7;
                border: 1px solid #465662;
                border-radius: 5px;
                padding: 8px 10px;
                font-size: 12px;
            }

            QComboBox:hover {
                border: 1px solid #6aaed6;
            }

            QComboBox:focus {
                border: 1px solid #6aaed6;
            }

            QComboBox::drop-down {
                background-color: #202a33;
                border-left: 1px solid #465662;
                width: 28px;
            }

            QComboBox QAbstractItemView {
                background-color: #182129;
                color: #f0f4f7;
                border: 1px solid #465662;
                selection-background-color: #286887;
                selection-color: #ffffff;
                padding: 4px;
            }
        """)

        method_layout.addWidget(
            self.method_combo
        )

        method_group.setLayout(
            method_layout
        )

        control_panel.addWidget(
            method_group
        )
        # =================================================
        # LOCALIZE BUTTON
        # =================================================

        self.localize_button = QPushButton(
            "▶  LOCALIZE PATTERN"
        )

        self.localize_button.setMinimumHeight(
            55
        )

        self.localize_button.setStyleSheet("""
            QPushButton {
                background-color: #245b78;
                border: 1px solid #4e91b5;
                border-radius: 7px;
                font-size: 14px;
                font-weight: bold;
                color: white;
            }

            QPushButton:hover {
                background-color: #31779d;
            }

            QPushButton:pressed {
                background-color: #1c465d;
            }
        """)

        self.localize_button.clicked.connect(
            self.localize
        )

        control_panel.addWidget(
            self.localize_button
        )


        # =================================================
        # RESULT PANEL
        # =================================================

        result_group = QGroupBox(
            "LOCALIZATION RESULT"
        )

        result_layout = QVBoxLayout()
        result_layout.setContentsMargins(8, 8, 8, 8)

        self.result_label = QLabel(
            "Waiting for localization..."
        )

        self.result_label.setWordWrap(
            True
        )

        self.result_label.setAlignment(
            Qt.AlignTop | Qt.AlignLeft
        )

        self.result_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        self.result_label.setMinimumHeight(105)
        self.result_label.setMaximumHeight(125)
        result_group.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Fixed
        )

        self.result_label.setStyleSheet("""
            QLabel {
                background-color: #10151a;
                border-radius: 6px;
                padding: 8px;
                color: #b8c6d1;
            }
        """)

        result_layout.addWidget(
            self.result_label
        )

        result_group.setLayout(
            result_layout
        )

        


        # =================================================
        # FABRICATION INSPECTION
        # =================================================

        fabrication_group = QGroupBox(
            "FABRICATION INSPECTION"
        )

        fabrication_layout = QVBoxLayout()


        self.fabricated_label = QLabel(
            "No fabricated image loaded"
        )

        self.fabricated_label.setAlignment(
            Qt.AlignCenter
        )
        self.fabricated_label.setWordWrap(
            True
        )

        self.fabricated_label.setMinimumHeight(
            35
        )

        self.fabricated_label.setMaximumHeight(
            40
        )
        self.fabricated_label.setStyleSheet("""
            QLabel {
                background-color: #10151a;
                border: 1px solid #394650;
                border-radius: 5px;
                padding: 8px;
                color: #b8c6d1;
            }
        """)

        fabrication_layout.addWidget(
            self.fabricated_label
        )


        fabricated_button = QPushButton(
            "LOAD FABRICATED IMAGE"
        )

        fabricated_button.clicked.connect(
            self.load_fabricated
        )

        fabrication_layout.addWidget(
            fabricated_button
        )


        self.inspect_button = QPushButton(
            "INSPECT FABRICATION"
        )

        self.inspect_button.setMinimumHeight(
            34
        )
        self.inspect_button.setMaximumHeight(
            36
        )

        self.inspect_button.clicked.connect(
            self.inspect_fabrication
        )

        self.inspect_button.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 13px;
            }
        """)

        fabrication_layout.addWidget(
            self.inspect_button
        )


        self.fabrication_result = QLabel(
            "Inspection: Waiting..."
        )
    

        self.fabrication_result.setWordWrap(
            True
        )

        self.fabrication_result.setAlignment(
            Qt.AlignTop | Qt.AlignLeft
        )

        self.fabrication_result.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        self.fabrication_result.setMinimumHeight(100)
        self.fabrication_result.setMaximumHeight(125)
        fabrication_group.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Fixed
        )

        self.fabrication_result.setStyleSheet("""
            QLabel {
                background-color: #10151a;
                border-radius: 6px;
                padding: 8px;
                color: #b8c6d1;
            }
        """)

        fabrication_layout.addWidget(
            self.fabrication_result
        )


        fabrication_group.setLayout(
            fabrication_layout
        )
        # =================================================
        # RIGHT RESULTS ROW
        # =================================================

        result_row = QHBoxLayout()
        result_row.setSpacing(8)
        result_row.setContentsMargins(0, 0, 0, 0)

        result_row.addWidget(
            result_group,
            1
        )

        result_row.addWidget(
            fabrication_group,
            1
        )

        right_panel.addLayout(
            result_row,
            0
        )
        

        # =================================================
        # RIGHT IMAGE VIEWER
        # =================================================

        viewer_frame = QFrame()

        viewer_frame.setStyleSheet("""
            QFrame {
                background-color: #0d1115;
                border: 1px solid #394650;
                border-radius: 8px;
            }
        """)

        viewer_layout = QVBoxLayout(
            viewer_frame
        )

        viewer_layout.setContentsMargins(
            10, 10, 10, 10
        )


        viewer_title = QLabel(
            "PATTERN INSPECTION VIEW"
        )

        viewer_title.setAlignment(
            Qt.AlignCenter
        )

        viewer_title.setStyleSheet("""
            font-size: 15px;
            font-weight: bold;
            color: #9fd3ff;
            padding: 6px;
        """)

        viewer_layout.addWidget(
            viewer_title
        )


        # -------------------------------------------------
        # ZOOM CONTROLS
        # -------------------------------------------------

        zoom_layout = QHBoxLayout()

        zoom_out_button = QPushButton("−")
        zoom_reset_button = QPushButton("100%")
        zoom_in_button = QPushButton("+")
        zoom_fit_button = QPushButton("FIT")

        for button in (
            zoom_out_button,
            zoom_reset_button,
            zoom_in_button,
            zoom_fit_button
        ):

            button.setMinimumHeight(32)

            button.setStyleSheet("""
                QPushButton {
                    background-color: #202b34;
                    border: 1px solid #3b4b58;
                    border-radius: 5px;
                    color: #d8e5ee;
                    font-weight: bold;
                    padding: 4px 14px;
                }

                QPushButton:hover {
                    background-color: #29404f;
                    border: 1px solid #5b91b5;
                }

                QPushButton:pressed {
                    background-color: #16232c;
                }
            """)

            zoom_layout.addWidget(button)

        zoom_layout.addStretch()

        self.zoom_label = QLabel("FIT")

        self.zoom_label.setAlignment(
            Qt.AlignCenter
        )

        self.zoom_label.setMinimumWidth(70)

        self.zoom_label.setStyleSheet("""
            QLabel {
                color: #9fd3ff;
                font-weight: bold;
                padding: 4px;
            }
        """)

        zoom_layout.addWidget(
            self.zoom_label
        )

        viewer_layout.addLayout(
            zoom_layout
        )

        # -------------------------------------------------
        # SCROLLABLE IMAGE VIEWER
        # -------------------------------------------------

        self.image_scroll = QScrollArea()

        self.image_scroll.setWidgetResizable(
            False
        )

        self.image_scroll.setAlignment(
            Qt.AlignCenter
        )

        self.image_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #080b0e;
                border: 1px solid #26313a;
            }

            QScrollBar:horizontal,
            QScrollBar:vertical {
                background: #111820;
            }

            QScrollBar::handle:horizontal,
            QScrollBar::handle:vertical {
                background: #3b5262;
                border-radius: 4px;
            }
        """)

        self.image_viewer = QLabel(
            "Load a search image to begin inspection"
        )

        self.image_viewer.setAlignment(
            Qt.AlignCenter
        )

        self.image_viewer.setStyleSheet("""
            QLabel {
                background-color: #080b0e;
                color: #596875;
            }
        """)

        self.image_viewer.setMinimumSize(
            400,
            300
        )

        self.image_scroll.setWidget(
            self.image_viewer
        )

        viewer_layout.addWidget(
            self.image_scroll,
            1
        )

        # -------------------------------------------------
        # ZOOM BUTTON CONNECTIONS
        # -------------------------------------------------

        zoom_out_button.clicked.connect(
            self.zoom_out
        )

        zoom_reset_button.clicked.connect(
            self.zoom_reset
        )

        zoom_in_button.clicked.connect(
            self.zoom_in
        )

        zoom_fit_button.clicked.connect(
            self.zoom_fit
        )
        # =================================================
        # ADD PATTERN VIEW TO RIGHT PANEL
        # =================================================

        right_panel.addWidget(
            viewer_frame,
            1
        )


        # =================================================
        # STATUS BAR
        # =================================================

        self.status = QStatusBar()

        self.setStatusBar(
            self.status
        )

        self.status.showMessage(
            "Ready — Load reference and search images"
        )

    # =====================================================
    # PERFORMANCE SUMMARY
    # =====================================================

    def get_performance_summary(self):

        csv_path = os.path.join(
            "results",
            "day20",
            "integrated_results.csv"
        )

        if not os.path.exists(csv_path):

            return (
                "TEST: --  |  PASS: --  |  FAIL: --"
            )

        try:

            total = 0
            passed = 0
            failed = 0

            with open(
                csv_path,
                "r",
                newline=""
            ) as file:

                reader = csv.DictReader(file)

                for row in reader:

                    total += 1

                    if row["status"].strip().upper() == "PASS":
                        passed += 1

                    else:
                        failed += 1

            if total > 0:

                pass_rate = (
                    passed / total
                ) * 100

            else:

                pass_rate = 0


            return (
                f"TEST: {total}  |  "
                f"PASS: {passed}  |  "
                f"FAIL: {failed}  |  "
                f"PASS RATE: {pass_rate:.1f}%"
            )


        except Exception:

            return (
                "TEST: --  |  "
                "PASS: --  |  "
                "FAIL: --"
            )

    # =====================================================
    # LOAD REFERENCE
    # =====================================================
    def load_reference(self):

        dialog = QFileDialog(
            self,
            "Select Reference Image"
        )

        dialog.setOption(
            QFileDialog.DontUseNativeDialog,
            True
        )

        dialog.setNameFilter(
            "PNG Images (*.png)"
        )

        dialog.setFileMode(
            QFileDialog.ExistingFile
        )

        dialog.setStyleSheet("""
            QFileDialog {
                background-color: #111820;
                color: #d8e2ea;
            }

            QFileDialog QLabel {
                color: #b8c6d1;
            }

            QFileDialog QTreeView,
            QFileDialog QListView {
                background-color: #0d1115;
                color: #d8e2ea;
                border: 1px solid #394650;
                selection-background-color: #285f7a;
                selection-color: white;
            }

            QFileDialog QLineEdit {
                background-color: #18222b;
                color: #ffffff;
                border: 1px solid #394650;
                padding: 5px;
            }

            QFileDialog QPushButton {
                background-color: #24333f;
                color: #ffffff;
                border: 1px solid #4b6070;
                padding: 7px 15px;
                border-radius: 4px;
            }

            QFileDialog QPushButton:hover {
                background-color: #31566d;
            }

            QFileDialog QComboBox {
                background-color: #18222b;
                color: #ffffff;
                border: 1px solid #394650;
                padding: 5px;
            }
        """)

        if dialog.exec_():

            selected_files = dialog.selectedFiles()

            if selected_files:

                filename = selected_files[0]

                self.reference_path = filename

                self.reference_label.setText(
                    os.path.basename(filename)
                )

                self.status.showMessage(
                    "Reference image loaded"
                )
    
     # =====================================================
    # LOAD FABRICATED IMAGE
    # =====================================================
    def load_fabricated(self):

        dialog = QFileDialog(
            self,
            "Select Fabricated Image"
        )

        dialog.setOption(
            QFileDialog.DontUseNativeDialog,
            True
        )

        dialog.setNameFilter(
            "PNG Images (*.png)"
        )

        dialog.setFileMode(
            QFileDialog.ExistingFile
        )

        dialog.setStyleSheet("""
            QFileDialog {
                background-color: #111820;
                color: #d8e2ea;
            }

            QFileDialog QLabel {
                color: #b8c6d1;
            }

            QFileDialog QTreeView,
            QFileDialog QListView {
                background-color: #0d1115;
                color: #d8e2ea;
                border: 1px solid #394650;
                selection-background-color: #285f7a;
                selection-color: white;
            }

            QFileDialog QLineEdit {
                background-color: #18222b;
                color: #ffffff;
                border: 1px solid #394650;
                padding: 5px;
            }

            QFileDialog QPushButton {
                background-color: #24333f;
                color: #ffffff;
                border: 1px solid #4b6070;
                padding: 7px 15px;
                border-radius: 4px;
            }

            QFileDialog QPushButton:hover {
                background-color: #31566d;
            }

            QFileDialog QComboBox {
                background-color: #18222b;
                color: #ffffff;
                border: 1px solid #394650;
                padding: 5px;
            }
        """)

        if dialog.exec_():

            selected_files = dialog.selectedFiles()

            if selected_files:

                filename = selected_files[0]

                self.fabricated_path = filename

                self.fabricated_label.setText(
                    os.path.basename(filename)
                )

                self.status.showMessage(
                    "Fabricated image loaded"
                )
    # =====================================================
    # FABRICATION INSPECTION
    # =====================================================

    def inspect_fabrication(self):

        if self.reference_path is None:

            QMessageBox.warning(
                self,
                "Missing Reference",
                "Please load a reference image first."
            )

            return


        if self.fabricated_path is None:

            QMessageBox.warning(
                self,
                "Missing Fabricated Image",
                "Please load a fabricated image first."
            )

            return


        try:

            reference = cv2.imread(
                self.reference_path,
                cv2.IMREAD_GRAYSCALE
            )

            fabricated = cv2.imread(
                self.fabricated_path,
                cv2.IMREAD_GRAYSCALE
            )


            if reference is None:

                raise RuntimeError(
                    "Unable to read reference image"
                )


            if fabricated is None:

                raise RuntimeError(
                    "Unable to read fabricated image"
                )


            # -------------------------------------------------
            # GET LOCALIZED POSITION
            # -------------------------------------------------

            if (
                not hasattr(self, "last_localization_x")
                or
                not hasattr(self, "last_localization_y")
                or
                self.last_localization_x is None
                or
                self.last_localization_y is None
            ):

                raise RuntimeError(
                    "Please localize the pattern before "
                    "performing fabrication inspection."
                )


            x = int(self.last_localization_x)
            y = int(self.last_localization_y)


            # -------------------------------------------------
            # REFERENCE SIZE
            # -------------------------------------------------

            reference_height, reference_width = (
                reference.shape
            )


            # -------------------------------------------------
            # CHECK ROI BOUNDS
            # -------------------------------------------------

            if (
                x < 0
                or
                y < 0
                or
                x + reference_width > fabricated.shape[1]
                or
                y + reference_height > fabricated.shape[0]
            ):

                raise RuntimeError(
                    "Localized pattern is outside fabricated image bounds"
                )


            # -------------------------------------------------
            # EXTRACT LOCALIZED FABRICATED PATTERN
            # -------------------------------------------------

            fabricated_roi = fabricated[
                y:y + reference_height,
                x:x + reference_width
            ]


            # -------------------------------------------------
            # VERIFY SAME SIZE
            # -------------------------------------------------

            if fabricated_roi.shape != reference.shape:

                raise RuntimeError(
                    f"Localized ROI size mismatch: "
                    f"{reference.shape} vs {fabricated_roi.shape}"
                )


            # -------------------------------------------------
            # ANALYZE ONLY THE LOCALIZED PATTERN
            # -------------------------------------------------

            result = analyze_defect(
                reference,
                fabricated_roi
            )


            status = result["status"]

            defect_type = result["defect_type"]

            confidence = result["confidence"]

            diff_percentage = result[
                "defect_percentage"
            ]

            changed_pixels = result[
                "changed_pixels"
            ]

            regions = len(
                result["regions"]
            )
            detected_regions = result["regions"]

            if status == "PASS":

                result_text = (
                    "FABRICATION: PASS\n\n"
                    "Defect: GOOD\n"
                    f"Difference: {diff_percentage:.4f}%\n"
                    f"Changed Pixels: {changed_pixels}\n"
                    f"Regions: {regions}\n"
                    f"Confidence: {confidence:.2f}%"
                )

                self.fabrication_result.setStyleSheet("""
                    QLabel {
                        background-color: #0d2b18;
                        border: 2px solid #00dd66;
                        border-radius: 6px;
                        padding: 10px;
                        color: #00ff77;
                        font-weight: bold;
                    }
                """)


            else:

                result_text = (
                    "FABRICATION: FAIL\n\n"
                    f"Defect: {defect_type}\n"
                    f"Difference: {diff_percentage:.4f}%\n"
                    f"Changed Pixels: {changed_pixels}\n"
                    f"Regions: {regions}\n"
                    f"Confidence: {confidence:.2f}%"
                )

                self.fabrication_result.setStyleSheet("""
                    QLabel {
                        background-color: #351010;
                        border: 2px solid #ff3030;
                        border-radius: 6px;
                        padding: 10px;
                        color: #ff5050;
                        font-weight: bold;
                    }
                """)
            # -----------------------------------------------------
            # DEFECT VISUALIZATION
            # -----------------------------------------------------

            if status == "FAIL" and detected_regions:

                self.show_fabrication_overlay(
                    self.fabricated_path,
                    detected_regions,
                    x,
                    y
                )


            self.fabrication_result.setText(
                result_text
            )


            self.status.showMessage(
                "Fabrication inspection complete"
            )


        except Exception as e:

            self.fabrication_result.setText(
                "Inspection failed:\n"
                + str(e)
            )

            self.status.showMessage(
                "Fabrication inspection failed"
            )
    # =====================================================
    # SHOW FABRICATION DEFECT OVERLAY
    # =====================================================
    def show_fabrication_overlay(
        self,
        image_path,
        regions,
        offset_x,
        offset_y
    ):

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            return

        # -------------------------------------------------
        # CONVERT GRAYSCALE TO RGB
        # -------------------------------------------------

        display_image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

        # -------------------------------------------------
        # FILTER VALID DEFECT REGIONS
        # -------------------------------------------------

        valid_regions = []

        for region in regions:

            x = region["x"] + offset_x
            y = region["y"] + offset_y
            w = region["w"]
            h = region["h"]
            area = int(region["area"])

            # Remove tiny noise regions
            if area < 30:
                continue

            # Remove extremely thin noise
            if w < 3 or h < 3:
                continue

            valid_regions.append(region)

        # -------------------------------------------------
        # DRAW DEFECT BOUNDING BOXES
        # -------------------------------------------------

        for index, region in enumerate(
            valid_regions,
            start=1
        ):

            x = region["x"] + offset_x
            y = region["y"] + offset_y
            w = region["w"]
            h = region["h"]

            # RED bounding box
            cv2.rectangle(
                display_image,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                4
            )

            # Defect number
            cv2.putText(
                display_image,
                f"DEFECT {index}",
                (x, max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

        # -------------------------------------------------
        # DEFECT COUNT
        # -------------------------------------------------

        cv2.putText(
            display_image,
            f"DEFECTS DETECTED: {len(valid_regions)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        # -------------------------------------------------
        # CONVERT TO QT IMAGE
        # -------------------------------------------------

        height, width, channels = (
            display_image.shape
        )

        bytes_per_line = (
            channels * width
        )

        qimage = QImage(
            display_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(
            qimage.copy()
        )

        # -------------------------------------------------
        # FIT IMAGE TO VIEWER
        # -------------------------------------------------

        pixmap = pixmap.scaled(
            self.image_viewer.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Store original pixmap for zoom operations
        self.current_pixmap = QPixmap.fromImage(
            qimage.copy()
        )

        self.zoom_factor = 1.0

        self.zoom_fit()


    # =====================================================
    # IMAGE ZOOM FUNCTIONS
    # =====================================================

    def update_zoom_display(self):

        if self.current_pixmap is None:
            return

        original_width = self.current_pixmap.width()
        original_height = self.current_pixmap.height()

        new_width = max(
            1,
            int(original_width * self.zoom_factor)
        )

        new_height = max(
            1,
            int(original_height * self.zoom_factor)
        )

        zoomed_pixmap = self.current_pixmap.scaled(
            new_width,
            new_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_viewer.setPixmap(
            zoomed_pixmap
        )

        self.image_viewer.resize(
            zoomed_pixmap.size()
        )

        self.zoom_label.setText(
            f"{int(self.zoom_factor * 100)}%"
        )


    def zoom_in(self):

        if self.current_pixmap is None:
            return

        self.zoom_factor *= 1.25

        if self.zoom_factor > 5.0:
            self.zoom_factor = 5.0

        self.update_zoom_display()


    def zoom_out(self):

        if self.current_pixmap is None:
            return

        self.zoom_factor /= 1.25

        if self.zoom_factor < 0.25:
            self.zoom_factor = 0.25

        self.update_zoom_display()


    def zoom_reset(self):

        if self.current_pixmap is None:
            return

        self.zoom_factor = 1.0

        self.update_zoom_display()


    def zoom_fit(self):

        if self.current_pixmap is None:
            return

        viewport_size = self.image_scroll.viewport().size()

        image_width = self.current_pixmap.width()
        image_height = self.current_pixmap.height()

        if image_width <= 0 or image_height <= 0:
            return

        scale_x = viewport_size.width() / image_width
        scale_y = viewport_size.height() / image_height

        self.zoom_factor = min(
            scale_x,
            scale_y
        )

        # Prevent excessively tiny images
        if self.zoom_factor <= 0:
            self.zoom_factor = 1.0

        self.update_zoom_display()


    
    # =====================================================
    # LOAD SEARCH
    # =====================================================

    def load_search(self):

        dialog = QFileDialog(
            self,
            "Select Search Image"
        )

        dialog.setOption(
            QFileDialog.DontUseNativeDialog,
            True
        )

        dialog.setNameFilter(
            "PNG Images (*.png)"
        )

        dialog.setFileMode(
            QFileDialog.ExistingFile
        )

        dialog.setStyleSheet("""
            QFileDialog {
                background-color: #111820;
                color: #d8e2ea;
            }

            QFileDialog QLabel {
                color: #b8c6d1;
            }

            QFileDialog QTreeView,
            QFileDialog QListView {
                background-color: #0d1115;
                color: #d8e2ea;
                border: 1px solid #394650;
                selection-background-color: #285f7a;
                selection-color: white;
            }

            QFileDialog QLineEdit {
                background-color: #18222b;
                color: #ffffff;
                border: 1px solid #394650;
                padding: 5px;
            }

            QFileDialog QPushButton {
                background-color: #24333f;
                color: #ffffff;
                border: 1px solid #4b6070;
                padding: 7px 15px;
                border-radius: 4px;
            }

            QFileDialog QPushButton:hover {
                background-color: #31566d;
            }

            QFileDialog QComboBox {
                background-color: #18222b;
                color: #ffffff;
                border: 1px solid #394650;
                padding: 5px;
            }
        """)

        if dialog.exec_():

            selected_files = dialog.selectedFiles()

            if selected_files:

                filename = selected_files[0]

                self.search_path = filename

                self.search_label.setText(
                    os.path.basename(filename)
                )

                self.display_image(
                    filename
                )

                self.status.showMessage(
                    "Search image loaded"
                )
    # =====================================================
    # DISPLAY IMAGE
    # =====================================================

    # =====================================================
    # DISPLAY IMAGE
    # =====================================================

    # =====================================================
    # IMAGE VIEWER ZOOM CONTROLS
    # =====================================================

    def update_zoom_display(self):

        if self.viewer_fit_mode:

            self.zoom_label.setText(
                "FIT"
            )

        else:

            self.zoom_label.setText(
                f"{int(self.viewer_zoom * 100)}%"
            )


    def set_viewer_pixmap(self, pixmap):

        if pixmap is None or pixmap.isNull():
            return

        self.viewer_pixmap = pixmap

        if self.viewer_fit_mode:

            viewport_size = (
                self.image_scroll.viewport().size()
            )

            scaled = pixmap.scaled(
                viewport_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.image_viewer.setPixmap(
                scaled
            )

            self.image_viewer.resize(
                scaled.size()
            )

        else:

            width = int(
                pixmap.width() *
                self.viewer_zoom
            )

            height = int(
                pixmap.height() *
                self.viewer_zoom
            )

            scaled = pixmap.scaled(
                width,
                height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.image_viewer.setPixmap(
                scaled
            )

            self.image_viewer.resize(
                scaled.size()
            )

        self.update_zoom_display()


    def zoom_in(self):

        if self.viewer_pixmap is None:
            return

        self.viewer_fit_mode = False

        self.viewer_zoom = min(
            self.viewer_zoom * 1.25,
            8.0
        )

        self.set_viewer_pixmap(
            self.viewer_pixmap
        )


    def zoom_out(self):

        if self.viewer_pixmap is None:
            return

        self.viewer_fit_mode = False

        self.viewer_zoom = max(
            self.viewer_zoom / 1.25,
            0.25
        )

        self.set_viewer_pixmap(
            self.viewer_pixmap
        )


    def zoom_reset(self):

        if self.viewer_pixmap is None:
            return

        self.viewer_fit_mode = False
        self.viewer_zoom = 1.0

        self.set_viewer_pixmap(
            self.viewer_pixmap
        )


    def zoom_fit(self):

        if self.viewer_pixmap is None:
            return

        self.viewer_fit_mode = True

        self.set_viewer_pixmap(
            self.viewer_pixmap
        )


    def display_image(
        self,
        filename,
        rectangle=None,
        matched=True
    ):

        image = cv2.imread(filename)

        if image is None:
            return

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------------------------------
        # DRAW LOCALIZATION RESULT
        # -------------------------------------------------

        if rectangle is not None:

            x, y, w, h = rectangle

            # GREEN = matched
            # RED   = not matched
            if matched:
                box_color = (0, 255, 0)
            else:
                box_color = (255, 0, 0)

            cv2.rectangle(
                image,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                box_color,
                8
            )


            # Draw center point
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            cv2.circle(
                image,
                (cx, cy),
                15,
                box_color,
                -1
            )

        # -------------------------------------------------
        # CONVERT TO QT IMAGE
        # -------------------------------------------------

        height, width, channels = image.shape

        bytes_per_line = channels * width

        qimage = QImage(
            image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qimage)

        self.set_viewer_pixmap(
            pixmap
        )
    # =====================================================
    # TEMPLATE MATCHING
    # =====================================================

    def template_matching(self):

        reference = cv2.imread(
            self.reference_path,
            cv2.IMREAD_GRAYSCALE
        )

        search = cv2.imread(
            self.search_path,
            cv2.IMREAD_GRAYSCALE
        )

        if reference is None or search is None:
            raise RuntimeError(
                "Unable to read images"
            )

        # Reference must fit inside search image
        if (
            reference.shape[0] > search.shape[0]
            or reference.shape[1] > search.shape[1]
        ):
            raise RuntimeError(
                "Reference image is larger than search image"
            )

        result = cv2.matchTemplate(
            search,
            reference,
            cv2.TM_CCOEFF_NORMED
        )

        _, max_score, _, max_location = cv2.minMaxLoc(
            result
        )

        x, y = max_location

        h, w = reference.shape

        return (
            x,
            y,
            w,
            h,
            max_score
        )


    # =====================================================
    # SPATIAL AI
    # =====================================================

    def spatial_ai(self):

        # -------------------------------------------------
        # CHECK AI MODEL
        # -------------------------------------------------

        if self.ai_model is None:

            raise RuntimeError(
                "AI model is not available"
            )


        # -------------------------------------------------
        # LOAD SEARCH IMAGE
        # -------------------------------------------------

        image = cv2.imread(
            self.search_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            raise RuntimeError(
                "Unable to read search image"
            )


        # Original search-image dimensions
        original_height, original_width = image.shape


        # -------------------------------------------------
        # LOAD REFERENCE IMAGE
        # -------------------------------------------------

        reference = cv2.imread(
            self.reference_path,
            cv2.IMREAD_GRAYSCALE
        )

        if reference is None:

            raise RuntimeError(
                "Unable to read reference image"
            )


        reference_height, reference_width = (
            reference.shape
        )


        # -------------------------------------------------
        # RESIZE REFERENCE IMAGE FOR AI
        # -------------------------------------------------

        ai_reference = cv2.resize(
            reference,
            (128, 128),
            interpolation=cv2.INTER_AREA
        )

        ai_reference = (
            ai_reference.astype(np.float32)
            / 255.0
        )


        # -------------------------------------------------
        # RESIZE SEARCH IMAGE FOR AI
        # -------------------------------------------------

        ai_search = cv2.resize(
            image,
            (256, 256),
            interpolation=cv2.INTER_AREA
        )

        ai_search = (
            ai_search.astype(np.float32)
            / 255.0
        )


        # -------------------------------------------------
        # CREATE AI TENSORS
        # -------------------------------------------------

        reference_tensor = torch.tensor(
            ai_reference,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)


        search_tensor = torch.tensor(
            ai_search,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        # -------------------------------------------------
        # AI HEATMAP
        # -------------------------------------------------

        with torch.no_grad():

            heatmap = self.ai_model(
                reference_tensor,
                search_tensor
            )

        heatmap = heatmap[0, 0]


        # -------------------------------------------------
        # FIND MULTIPLE AI CANDIDATES
        # -------------------------------------------------

        heatmap_cpu = (
            heatmap.detach().cpu()
        )

        heatmap_np = (
            heatmap_cpu.numpy()
        )


        # Find local maxima in the heatmap.
        #
        # This prevents us from using only one
        # maximum cell.

        pooled = torch.nn.functional.max_pool2d(
            heatmap_cpu.unsqueeze(0).unsqueeze(0),
            kernel_size=5,
            stride=1,
            padding=2
        )[0, 0]


        local_maxima = (
            heatmap_cpu == pooled
        )


        candidate_indices = (
            torch.nonzero(
                local_maxima,
                as_tuple=False
            )
        )


        candidates = []


        for item in candidate_indices:

            hy = int(item[0])
            hx = int(item[1])

            score = float(
                heatmap_np[
                    hy,
                    hx
                ]
            )

            candidates.append(
                (
                    score,
                    hx,
                    hy
                )
            )


        # Highest AI scores first

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )


        # Keep the strongest candidates.
        #
        # 50 gives the verification stage enough
        # opportunity to recover the correct location.

        candidates = candidates[:50]


        # -------------------------------------------------
        # VERIFY AI CANDIDATES
        # -------------------------------------------------

        best_x = None
        best_y = None
        best_confidence = -1.0
        best_verification = -1.0


        for confidence, hx, hy in candidates:

            # ---------------------------------------------
            # Convert heatmap coordinate -> AI image
            # ---------------------------------------------

            x_256 = hx * 4
            y_256 = hy * 4


            # ---------------------------------------------
            # Convert AI image -> ORIGINAL IMAGE
            # ---------------------------------------------

            candidate_x = (
                x_256 *
                original_width /
                256.0
            )

            candidate_y = (
                y_256 *
                original_height /
                256.0
            )


            candidate_x = int(
                round(candidate_x)
            )

            candidate_y = int(
                round(candidate_y)
            )


            # ---------------------------------------------
            # Keep candidate inside image
            # ---------------------------------------------

            max_x = (
                original_width -
                reference_width
            )

            max_y = (
                original_height -
                reference_height
            )


            if max_x < 0 or max_y < 0:

                raise RuntimeError(
                    "Reference image is larger than search image"
                )


            candidate_x = max(
                0,
                min(
                    candidate_x,
                    max_x
                )
            )

            candidate_y = max(
                0,
                min(
                    candidate_y,
                    max_y
                )
            )


            # ---------------------------------------------
            # LOCAL REFINEMENT
            #
            # Search around the AI prediction.
            # This gives sub-grid accuracy instead of
            # accepting the coarse 64x64 heatmap position.
            # ---------------------------------------------

            search_radius = 120


            roi_x1 = max(
                0,
                candidate_x - search_radius
            )

            roi_y1 = max(
                0,
                candidate_y - search_radius
            )

            roi_x2 = min(
                original_width,
                candidate_x +
                reference_width +
                search_radius
            )

            roi_y2 = min(
                original_height,
                candidate_y +
                reference_height +
                search_radius
            )


            roi = image[
                roi_y1:roi_y2,
                roi_x1:roi_x2
            ]


            # ROI must be larger than reference

            if (
                roi.shape[0] < reference_height
                or
                roi.shape[1] < reference_width
            ):

                continue


            verification_map = (
                cv2.matchTemplate(
                    roi,
                    reference,
                    cv2.TM_CCOEFF_NORMED
                )
            )


            (
                _,
                verification_score,
                _,
                verification_location
            ) = cv2.minMaxLoc(
                verification_map
            )


            refined_x = (
                roi_x1 +
                verification_location[0]
            )

            refined_y = (
                roi_y1 +
                verification_location[1]
            )


            # ---------------------------------------------
            # KEEP BEST VERIFIED CANDIDATE
            #
            # Verification is the final decision.
            # ---------------------------------------------

            if verification_score > best_verification:

                best_x = refined_x
                best_y = refined_y

                best_confidence = (
                    confidence
                )

                best_verification = (
                    float(
                        verification_score
                    )
                )


        # -------------------------------------------------
        # MAKE SURE A CANDIDATE WAS FOUND
        # -------------------------------------------------

        if best_x is None:

            raise RuntimeError(
                "Spatial AI could not generate a valid candidate"
            )


        # -------------------------------------------------
        # RETURN FINAL RESULT
        # -------------------------------------------------

        return (
            best_x,
            best_y,
            best_confidence,
            best_verification,
            reference_width,
            reference_height
        )


    # =====================================================
    # LOCALIZE
    # =====================================================
    # =====================================================
    # LOCALIZE
    # =====================================================

    def localize(self):
        self.last_localization_x = None
        self.last_localization_y = None

        # -------------------------------------------------
        # CHECK REFERENCE
        # -------------------------------------------------

        if self.reference_path is None:

            QMessageBox.warning(
                self,
                "Reference Image Missing",
                "Please select a reference image first."
            )

            self.status.showMessage(
                "Reference image not selected"
            )

            return


        # -------------------------------------------------
        # CHECK SEARCH
        # -------------------------------------------------

        if self.search_path is None:

            QMessageBox.warning(
                self,
                "Search Image Missing",
                "Please select a search image first."
            )

            self.status.showMessage(
                "Search image not selected"
            )

            return


        # -------------------------------------------------
        # CHECK SAME IMAGE
        # -------------------------------------------------

        if os.path.abspath(
            self.reference_path
        ) == os.path.abspath(
            self.search_path
        ):

            QMessageBox.critical(
                self,
                "Invalid Image Selection",
                "The same image cannot be used as both\n"
                "Reference Image and Search Image.\n\n"
                "Please select two different images."
            )

            self.status.showMessage(
                "Invalid selection: same image used twice"
            )

            return


        method = self.method_combo.currentText()


        try:

            # =================================================
            # TEMPLATE MATCHING
            # =================================================

            if method == "Template Matching":

                x, y, w, h, score = (
                    self.template_matching()
                )

                threshold = 0.70

                if score >= threshold:
                    status = "SUCCESS"
                    matched = True

                    # Save successful localization
                    self.last_localization_x = int(x)
                    self.last_localization_y = int(y)

                else:
                    status = "PATTERN NOT FOUND"
                    matched = False

                    # Clear previous localization
                    self.last_localization_x = None
                    self.last_localization_y = None


                reference = cv2.imread(
                    self.reference_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if reference is None:
                    raise RuntimeError(
                        "Unable to read reference image"
                    )

                reference_height, reference_width = reference.shape

                self.display_image(
                    self.search_path,
                    (
                        x,
                        y,
                        reference_width,
                        reference_height
                    ),
                    matched
                )


                self.result_label.setText(
                    f"Method: Template Matching\n"
                    f"Predicted X: {x}\n"
                    f"Predicted Y: {y}\n"
                    f"Match Score: {score:.4f}\n"
                    f"Threshold: {threshold:.2f}\n"
                    f"Status: {status}"
                )


                if matched:

                    self.status.showMessage(
                        "Template localization successful"
                    )

                else:

                    self.status.showMessage(
                        "Template localization: pattern not found"
                    )


            # =================================================
            # SPATIAL AI
            # =================================================

            elif method == "Spatial AI":

                (
                    x,
                    y,
                    confidence,
                    verification_score,
                    reference_width,
                    reference_height
                ) = self.spatial_ai()
                self.last_localization_x = int(x)
                self.last_localization_y = int(y)

                threshold = 0.70

                if verification_score >= threshold:
                    status = "SUCCESS"
                    matched = True
                else:
                    status = "PATTERN NOT FOUND"
                    matched = False


                reference = cv2.imread(
                    self.reference_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if reference is None:
                    raise RuntimeError(
                        "Unable to read reference image"
                    )

                reference_height, reference_width = reference.shape

                self.display_image(
                    self.search_path,
                    (
                        x,
                        y,
                        reference_width,
                        reference_height
                    ),
                    matched
                )


                self.result_label.setText(
                    f"Method: Spatial AI\n"
                    f"Predicted X: {x:.2f}\n"
                    f"Predicted Y: {y:.2f}\n"
                    f"Heatmap Score: {confidence:.4f}\n"
                    f"Reference Verification: "
                    f"{verification_score:.4f}\n"
                    f"Threshold: {threshold:.2f}\n"
                    f"Status: {status}"
                )


                if matched:

                    self.status.showMessage(
                        "Spatial AI localization successful"
                    )

                else:

                    self.status.showMessage(
                        "Spatial AI localization: pattern not found"
                    )


        except Exception as e:

            self.result_label.setText(
                "Localization failed:\n"
                + str(e)
            )

            self.status.showMessage(
                "Localization failed"
            )

            QMessageBox.critical(
                self,
                "Localization Error",
                str(e)
            )
   


# =========================================================
# APPLICATION START
# =========================================================

if __name__ == "__main__":

    app = QApplication(
        sys.argv
    )

    window = SemiconductorLocalizer()

    window.show()

    sys.exit(
        app.exec_()
    )
