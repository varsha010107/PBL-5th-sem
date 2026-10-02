import sys
import os

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
    QSizePolicy
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
    "spatial_localizer.pth"
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

        self.last_prediction = None

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
        performance = QLabel(
            "TEST: 100  |  ACCURACY: 100%"
        )

        performance.setStyleSheet("""
            QLabel {
                color: #72d572;
                font-weight: bold;
                padding: 8px 12px;
                border: 1px solid #3c6945;
                border-radius: 6px;
                background-color: #17251b;
                margin-right: 8px;
            }
        """)

        header_layout.addWidget(
            performance
        )

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
        content_layout.setSpacing(15)

        main_layout.addLayout(content_layout)


        # =================================================
        # LEFT CONTROL PANEL
        # =================================================

        control_panel = QVBoxLayout()
        control_panel.setSpacing(5)


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

        self.result_label = QLabel(
            "Waiting for localization..."
        )

        self.result_label.setWordWrap(
            True
        )

        self.result_label.setAlignment(
            Qt.AlignTop
        )

        self.result_label.setMinimumHeight(
            105
        )

        self.result_label.setMaximumHeight(
            115
        )
        self.result_label.setStyleSheet("""
            QLabel {
                background-color: #10151a;
                border-radius: 6px;
                padding: 12px;
                color: #b8c6d1;
            }
        """)

        result_layout.addWidget(
            self.result_label
        )

        result_group.setLayout(
            result_layout
        )

        control_panel.addWidget(
            result_group
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
            38
        )
        self.inspect_button.setMaximumHeight(
            40
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
        self.fabrication_result.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Fixed
        )

        self.fabrication_result.setWordWrap(
            True

        )

        self.fabrication_result.setMinimumHeight(
            82
        )

        self.fabrication_result.setMaximumHeight(
            90
        )

        self.fabrication_result.setStyleSheet("""
            QLabel {
                background-color: #10151a;
                border-radius: 6px;
                padding: 10px;
                color: #b8c6d1;
            }
        """)

        fabrication_layout.addWidget(
            self.fabrication_result
        )


        fabrication_group.setLayout(
            fabrication_layout
        )

        control_panel.addWidget(
            fabrication_group
        )


        control_panel.addStretch()

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


        self.image_viewer = QLabel(
            "Load a search image to begin inspection"
        )

        self.image_viewer.setAlignment(
            Qt.AlignCenter
        )

        self.image_viewer.setMinimumSize(
            750,
            600
        )

        self.image_viewer.setStyleSheet("""
            QLabel {
                background-color: #080b0e;
                border: 1px solid #26313a;
                color: #596875;
            }
        """)

        viewer_layout.addWidget(
            self.image_viewer
        )


        # =================================================
        # ADD PANELS
        # =================================================

        content_layout.addLayout(
            control_panel,
            1
        )

        content_layout.addWidget(
            viewer_frame,
            3
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


            result = analyze_defect(
                reference,
                fabricated
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
                detected_regions
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

    def show_fabrication_overlay(self, image_path, regions):

        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            return

        # Convert grayscale to RGB
        display_image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

        # Draw every detected defect region
        for index, region in enumerate(regions, start=1):

            x = int(region["x"])
            y = int(region["y"])
            w = int(region["w"])
            h = int(region["h"])

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

        # Convert to Qt image
        height, width, channels = display_image.shape

        bytes_per_line = channels * width

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

        # Fit inside viewer
        pixmap = pixmap.scaled(
            self.image_viewer.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.image_viewer.setPixmap(
            pixmap
        )
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

        self.image_viewer.setPixmap(
            pixmap.scaled(
                self.image_viewer.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
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
        # RESIZE SEARCH IMAGE FOR AI
        # -------------------------------------------------

        ai_image = cv2.resize(
            image,
            (256, 256),
            interpolation=cv2.INTER_AREA
        )


        ai_image = (
            ai_image.astype(
                np.float32
            ) / 255.0
        )


        tensor = torch.tensor(
            ai_image,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)


        # -------------------------------------------------
        # AI HEATMAP
        # -------------------------------------------------

        with torch.no_grad():

            heatmap = self.ai_model(
                tensor
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
            best_verification
        )


    # =====================================================
    # LOCALIZE
    # =====================================================
    # =====================================================
    # LOCALIZE
    # =====================================================

    def localize(self):

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
                else:
                    status = "PATTERN NOT FOUND"
                    matched = False


                self.display_image(
                    self.search_path,
                    (
                        x,
                        y,
                        w,
                        h
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

                x, y, confidence, verification_score = (
                    self.spatial_ai()
                )


                threshold = 0.70

                if verification_score >= threshold:
                    status = "SUCCESS"
                    matched = True
                else:
                    status = "PATTERN NOT FOUND"
                    matched = False


                self.display_image(
                    self.search_path,
                    (
                        x,
                        y,
                        500,
                        500
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
