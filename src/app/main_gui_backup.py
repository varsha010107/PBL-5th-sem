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
    QFrame
)

from PyQt5.QtGui import (
    QPixmap,
    QImage,
    QPainter,
    QPen
)

from PyQt5.QtCore import Qt


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

        central = QWidget()

        self.setCentralWidget(
            central
        )

        main_layout = QHBoxLayout(
            central
        )


        # -------------------------------------------------
        # LEFT PANEL
        # -------------------------------------------------

        control_panel = QVBoxLayout()


        title = QLabel(
            "SEMICONDUCTOR\n"
            "PATTERN LOCALIZER"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        title.setStyleSheet(
            """
            font-size: 18px;
            font-weight: bold;
            padding: 15px;
            """
        )

        control_panel.addWidget(
            title
        )


        # -------------------------------------------------
        # REFERENCE
        # -------------------------------------------------

        reference_group = QGroupBox(
            "Reference Pattern"
        )

        reference_layout = QVBoxLayout()


        self.reference_label = QLabel(
            "No reference image loaded"
        )

        self.reference_label.setAlignment(
            Qt.AlignCenter
        )

        self.reference_label.setMinimumHeight(
            80
        )

        reference_layout.addWidget(
            self.reference_label
        )


        reference_button = QPushButton(
            "Load Reference"
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


        # -------------------------------------------------
        # SEARCH
        # -------------------------------------------------

        search_group = QGroupBox(
            "Wide-Field Search Image"
        )

        search_layout = QVBoxLayout()


        self.search_label = QLabel(
            "No search image loaded"
        )

        self.search_label.setAlignment(
            Qt.AlignCenter
        )

        self.search_label.setMinimumHeight(
            80
        )

        search_layout.addWidget(
            self.search_label
        )


        search_button = QPushButton(
            "Load Search Image"
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


        # -------------------------------------------------
        # METHOD
        # -------------------------------------------------

        method_group = QGroupBox(
            "Localization Method"
        )

        method_layout = QVBoxLayout()


        self.method_combo = QComboBox()

        self.method_combo.addItems(
            [
                "Template Matching",
                "Spatial AI"
            ]
        )

        method_layout.addWidget(
            self.method_combo
        )


        method_group.setLayout(
            method_layout
        )

        control_panel.addWidget(
            method_group
        )


        # -------------------------------------------------
        # LOCALIZE BUTTON
        # -------------------------------------------------

        self.localize_button = QPushButton(
            "LOCALIZE PATTERN"
        )

        self.localize_button.setMinimumHeight(
            55
        )

        self.localize_button.clicked.connect(
            self.localize
        )

        self.localize_button.setStyleSheet(
            """
            font-weight: bold;
            font-size: 14px;
            """
        )

        control_panel.addWidget(
            self.localize_button
        )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        result_group = QGroupBox(
            "Localization Result"
        )

        result_layout = QVBoxLayout()


        self.result_label = QLabel(
            "Result: Waiting..."
        )

        self.result_label.setWordWrap(
            True
        )

        result_layout.addWidget(
            self.result_label
        )


        result_group.setLayout(
            result_layout
        )

        control_panel.addWidget(
            result_group
        )


        control_panel.addStretch()


        # -------------------------------------------------
        # IMAGE VIEWER
        # -------------------------------------------------

        viewer_layout = QVBoxLayout()


        viewer_title = QLabel(
            "SEARCH IMAGE VIEWER"
        )

        viewer_title.setAlignment(
            Qt.AlignCenter
        )

        viewer_title.setStyleSheet(
            """
            font-size: 16px;
            font-weight: bold;
            padding: 8px;
            """
        )

        viewer_layout.addWidget(
            viewer_title
        )


        self.image_viewer = QLabel(
            "Load a search image to begin"
        )

        self.image_viewer.setAlignment(
            Qt.AlignCenter
        )

        self.image_viewer.setFrameShape(
            QFrame.Box
        )

        self.image_viewer.setMinimumSize(
            750,
            650
        )

        viewer_layout.addWidget(
            self.image_viewer
        )


        # -------------------------------------------------
        # ADD PANELS
        # -------------------------------------------------

        main_layout.addLayout(
            control_panel,
            1
        )

        main_layout.addLayout(
            viewer_layout,
            3
        )


        # -------------------------------------------------
        # STATUS BAR
        # -------------------------------------------------

        self.status = QStatusBar()

        self.setStatusBar(
            self.status
        )

        self.status.showMessage(
            "Ready"
        )


    # =====================================================
    # LOAD REFERENCE
    # =====================================================

    def load_reference(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Image",
            "",
            "PNG Images (*.png)"
        )

        if filename:

            self.reference_path = filename

            self.reference_label.setText(
                os.path.basename(filename)
            )

            self.status.showMessage(
                "Reference image loaded"
            )


    # =====================================================
    # LOAD SEARCH
    # =====================================================

    def load_search(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Search Image",
            "",
            "PNG Images (*.png)"
        )

        if filename:

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

    def display_image(
        self,
        filename,
        rectangle=None
    ):

        image = cv2.imread(
            filename
        )

        if image is None:

            return

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # Draw prediction rectangle

        if rectangle is not None:

            x, y, w, h = rectangle

            cv2.rectangle(
                image,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                (255, 0, 0),
                8
            )

            # Center point

            cx = int(
                x + w / 2
            )

            cy = int(
                y + h / 2
            )

            cv2.circle(
                image,
                (cx, cy),
                15,
                (255, 0, 0),
                -1
            )


        height, width, channels = (
            image.shape
        )

        bytes_per_line = (
            channels * width
        )


        qimage = QImage(
            image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )


        pixmap = QPixmap.fromImage(
            qimage
        )


        pixmap = pixmap.scaled(
            self.image_viewer.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )


        self.image_viewer.setPixmap(
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


        result = cv2.matchTemplate(
            search,
            reference,
            cv2.TM_CCOEFF_NORMED
        )


        _, max_score, _, max_location = (
            cv2.minMaxLoc(result)
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

        if self.ai_model is None:

            raise RuntimeError(
                "AI model is not available"
            )


        image = cv2.imread(
            self.search_path,
            cv2.IMREAD_GRAYSCALE
        )


        if image is None:

            raise RuntimeError(
                "Unable to read search image"
            )


        image = cv2.resize(
            image,
            (256, 256)
        )


        image = (
            image.astype(
                np.float32
            ) / 255.0
        )


        tensor = torch.tensor(
            image
        ).unsqueeze(0).unsqueeze(0)


        with torch.no_grad():

            heatmap = self.ai_model(
                tensor
            )


        heatmap = heatmap[0, 0]


        index = torch.argmax(
            heatmap
        )


        hy, hx = np.unravel_index(
            index.item(),
            heatmap.shape
        )


        # Heatmap = 64x64
        # Search input = 256x256

        x_256 = hx * 4
        y_256 = hy * 4


        # Convert back to original 2000x2000

        x = x_256 * 2000.0 / 256.0
        y = y_256 * 2000.0 / 256.0


        confidence = heatmap[
            hy,
            hx
        ].item()
        # -------------------------------------------------
        # VERIFY AI PREDICTION USING SELECTED REFERENCE
        # -------------------------------------------------

        reference = cv2.imread(
            self.reference_path,
            cv2.IMREAD_GRAYSCALE
        )

        search_original = cv2.imread(
            self.search_path,
            cv2.IMREAD_GRAYSCALE
        )

        if reference is None or search_original is None:
            raise RuntimeError(
                "Unable to read verification images"
            )

        rh, rw = reference.shape

        # AI prediction is the approximate top-left
        # location of the target.
        x_int = int(x)
        y_int = int(y)

        # Search within a tolerance window around
        # the AI prediction.
        margin = 100

        x1 = max(
            0,
            x_int - margin
        )

        y1 = max(
            0,
            y_int - margin
        )

        x2 = min(
            search_original.shape[1],
            x_int + rw + margin
        )

        y2 = min(
            search_original.shape[0],
            y_int + rh + margin
        )

        verification_region = search_original[
            y1:y2,
            x1:x2
        ]

        # The verification region must be large enough
        # to contain the reference.
        if (
            verification_region.shape[0] < rh
            or verification_region.shape[1] < rw
        ):
            raise RuntimeError(
                "Verification region too small"
            )

        verification = cv2.matchTemplate(
            verification_region,
            reference,
            cv2.TM_CCOEFF_NORMED
        )

        _, verification_score, _, verification_location = (
            cv2.minMaxLoc(verification)
        )

        # Refined location from local verification.
        refined_x = x1 + verification_location[0]
        refined_y = y1 + verification_location[1]

        return (
            refined_x,
            refined_y,
            confidence,
            verification_score
        )


    # =====================================================
    # LOCALIZE
    # =====================================================

    def localize(self):


        # =================================================
        # CHECK REFERENCE IMAGE
        # =================================================

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


        # =================================================
        # CHECK SEARCH IMAGE
        # =================================================

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


        # =================================================
        # CHECK IMAGE ROLES
        # =================================================

        reference_name = os.path.basename(
            self.reference_path
        ).lower()

        search_name = os.path.basename(
            self.search_path
        ).lower()


        # Search image selected as Reference
        if reference_name.startswith("search_"):

            QMessageBox.critical(
                self,
                "Invalid Reference Image",
                "A SEARCH image was selected as the "
                "REFERENCE image.\n\n"
                "Please select a file starting with:\n"
                "reference_"
            )

            self.status.showMessage(
                "ERROR: Search image used as Reference"
            )

            return


        # Reference image selected as Search
        if search_name.startswith("reference_"):

            QMessageBox.critical(
                self,
                "Invalid Search Image",
                "A REFERENCE image was selected as the "
                "SEARCH image.\n\n"
                "Please select a file starting with:\n"
                "search_"
            )

            self.status.showMessage(
                "ERROR: Reference image used as Search"
            )

            return


        # Same physical file selected twice
        if os.path.abspath(
            self.reference_path
        ) == os.path.abspath(
            self.search_path
        ):

            QMessageBox.critical(
                self,
                "Invalid Image Selection",
                "Reference and Search images "
                "cannot be the same file."
            )

            self.status.showMessage(
                "ERROR: Same image selected twice"
            )

            return


        # =================================================
        # START LOCALIZATION
        # =================================================

        method = (
            self.method_combo.currentText()
        )

        try:

            # ---------------------------------------------
            # TEMPLATE MATCHING
            # ---------------------------------------------

            if method == "Template Matching":

                x, y, w, h, score = (
                    self.template_matching()
                )


                self.display_image(
                    self.search_path,
                    (
                        x,
                        y,
                        w,
                        h
                    )
                )

                threshold = 0.70
                if score >= threshold:
                    status = "SUCCESS"
                else:
                    status = "PATTERN NOT FOUND"
                self.result_label.setText(
                    f"Method: Template Matching\n"
                    f"Predicted X: {x}\n"
                    f"Predicted Y: {y}\n"
                    f"Match Score: {score:.4f}\n"
                    f"Threshold: {threshold:.2f}\n"
                    f"Status: {status}"
                )


                self.status.showMessage(
                    "Template localization complete"
                )


            # ---------------------------------------------
            # SPATIAL AI
            # ---------------------------------------------

            else:

                x, y, confidence, verification_score = (
                    self.spatial_ai()
                )


                # Reference dimensions
                # Original reference = 500x500

                self.display_image(
                    self.search_path,
                    (
                        x,
                        y,
                        500,
                        500
                    )
                )

                threshold = 0.70
                if verification_score >= threshold:
                    status = "SUCCESS"
                else:
                    status = "PATTERN NOT FOUND"
                self.result_label.setText(
                    f"Method: Spatial AI\n"
                    f"Predicted X: {x:.2f}\n"
                    f"Predicted Y: {y:.2f}\n"
                    f"Heatmap Score: {confidence:.4f}\n"
                    f"Reference Verification: {verification_score:.4f}\n"
                    f"Threshold: {threshold:.2f}\n"
                    f"Status: {status}"
                )


                self.status.showMessage(
                    "Spatial AI localization complete"
                )


        except Exception as e:

            self.result_label.setText(
                "Localization failed:\n"
                + str(e)
            )

            self.status.showMessage(
                "Localization failed"


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
