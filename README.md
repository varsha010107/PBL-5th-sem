# FINLOC AI — AI Semiconductor Pattern Localizer

**FINLOC AI** is an AI-assisted semiconductor inspection prototype designed to localize semiconductor layout patterns, verify their position, and perform localized fabrication-defect inspection.

The system combines **traditional computer vision Template Matching** with a trained **reference-conditioned Spatial AI model** and an interactive PyQt5 GUI.

---

## 🚀 Project Overview

Semiconductor inspection requires accurately locating a target pattern within a larger image before detailed inspection can be performed.

FINLOC AI follows the pipeline:

**Reference Pattern → Pattern Localization → Position Verification → Fabrication Inspection → Defect Visualization**

The system supports two localization approaches:

1. **Template Matching**
2. **Spatial AI**

The detected X-Y coordinates are then reused for localized fabrication inspection.

---

## 🎯 Objectives

- Automatically locate semiconductor patterns in larger images.
- Compare traditional computer vision with AI-based localization.
- Generate X-Y coordinates of the detected pattern.
- Provide confidence and verification information.
- Inspect the localized region for fabrication differences.
- Visualize detected defects using bounding boxes.
- Provide an interactive GUI for semiconductor pattern inspection.
- Build a reproducible prototype suitable for semiconductor inspection research and education.

---

## 🧠 System Architecture

```text
                 ┌──────────────────────┐
                 │   Reference Image    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Localization Engine │
                 └──────────┬───────────┘
                            │
                 ┌──────────┴───────────┐
                 │                      │
                 ▼                      ▼
        ┌────────────────┐     ┌─────────────────┐
        │ Template       │     │   Spatial AI    │
        │ Matching       │     │     Model       │
        └───────┬────────┘     └────────┬────────┘
                │                       │
                └───────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Position Verification│
                 │       X, Y           │
                 │ Confidence / Score   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Fabrication Image    │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Localized ROI        │
                 │ Defect Inspection    │
                 └──────────┬───────────┘
                            ▼
                 ┌──────────────────────┐
                 │ Defect Visualization │
                 │  Red Bounding Boxes  │
                 └──────────────────────┘
🔬 Localization Methods
1. Template Matching

Template Matching is the traditional computer-vision baseline.

The reference image is treated as a template and searched against the larger image.

The system obtains:

X coordinate
Y coordinate
Width
Height
Matching score

A configurable similarity threshold determines whether the pattern is considered successfully localized.

This approach does not require model training.

2. Spatial AI

FINLOC AI also uses a trained reference-conditioned Spatial AI model.

The model receives the reference pattern and search image and generates a spatial response indicating probable locations of the target pattern.

The highest-confidence candidate is selected and subsequently verified against the reference.

The localization pipeline is:

Reference + Search Image
          │
          ▼
   Spatial AI Model
          │
          ▼
   Spatial Response
          │
          ▼
 Highest-confidence
      location
          │
          ▼
     X-Y coordinates
          │
          ▼
      Verification

The resulting coordinates are stored and reused during fabrication inspection.

🏭 Fabrication Inspection

After localization, FINLOC AI can inspect the corresponding region of a fabricated image.

Instead of comparing unrelated regions, the system:

Obtains the localized X-Y position.
Uses the reference pattern dimensions.
Extracts the corresponding ROI from the fabricated image.
Compares the reference and fabricated ROI.
Detects significant pixel/structural differences.
Visualizes detected regions using bounding boxes.
Localized X,Y
     │
     ▼
Extract Fabricated ROI
     │
     ▼
Compare with Reference
     │
     ▼
Difference Analysis
     │
     ▼
Defect Detection
     │
     ▼
Visual Defect Boxes
🖥️ Graphical User Interface

FINLOC AI provides an interactive GUI built using PyQt5.

The interface supports:

Reference image selection
Search image selection
Fabricated image selection
Template Matching
Spatial AI localization
Localization result display
X-Y coordinate reporting
Confidence/verification information
Fabrication inspection
Defect visualization
Image zoom controls
FIT / 100% / zoom-in / zoom-out controls
📊 Model Information

The project contains multiple model checkpoints developed during experimentation.

Important model files include:

models/
├── day6/
│   └── finfet_localizer.pth
├── day7/
│   └── spatial_localizer.pth
└── day10/
    ├── spatial_localizer.pth
    └── spatial_localizer_v2.pth

The current Spatial AI GUI uses the reference-conditioned model:

models/day10/spatial_localizer_v2.pth

The model was developed using synthetic semiconductor pattern data for the project prototype.

📁 Project Structure
Applied_Materials_PBL/
│
├── data/
│   └── Reference and search image resources
│
├── models/
│   ├── day6/
│   ├── day7/
│   └── day10/
│
├── notebooks/
│   └── Experimental notebooks
│
├── references/
│   └── Reference material
│
├── results/
│   └── Experimental results
│
├── src/
│   ├── app/
│   │   └── main.py
│   │
│   ├── heatmap_model.py
│   ├── defect_engine.py
│   ├── train_day10.py
│   └── other development scripts
│
├── finloc
├── install_finloc.sh
├── requirements.txt
├── README.md
└── .gitignore

Large generated datasets and the Python virtual environment are intentionally excluded from this repository using .gitignore.

⚙️ Technologies Used
Technology	Purpose
Python	Core development
PyTorch	Spatial AI model
OpenCV	Image processing and Template Matching
NumPy	Numerical processing
PyQt5	Graphical user interface
Linux/Ubuntu	Development environment
Git/GitHub	Version control
📦 Installation
1. Clone the repository
git clone https://github.com/varsha010107/PBL-5th-sem.git
cd PBL-5th-sem
2. Run the installation script
chmod +x install_finloc.sh
./install_finloc.sh

If you prefer manual installation:

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
🖥️ Running FINLOC AI

Activate the environment:

source .venv/bin/activate

Set the Python source path:

export PYTHONPATH="$PWD/src:$PYTHONPATH"

Run the application:

python src/app/main.py

If the finloc launcher has been installed:

finloc
🛠️ Linux Qt Dependencies

On some Ubuntu systems, PyQt5 may require additional X11/Qt dependencies.

Install them using:

sudo apt update

sudo apt install -y \
libxcb-xinerama0 \
libxcb-cursor0 \
libxcb-icccm4 \
libxcb-image0 \
libxcb-keysyms1 \
libxcb-render-util0 \
libxcb-randr0 \
libxcb-shape0 \
libxcb-xfixes0 \
libxcb-sync1 \
libxkbcommon-x11-0

FINLOC AI uses:

opencv-python-headless

rather than the GUI-enabled OpenCV package to avoid conflicts between OpenCV's bundled Qt libraries and PyQt5.

🔍 Typical Workflow
Step 1 — Select Reference

Load the semiconductor pattern that needs to be located.

Step 2 — Select Search Image

Load a larger image containing multiple patterns or candidate regions.

Step 3 — Choose Localization Method

Select either:

Template Matching

or

Spatial AI
Step 4 — Localize

FINLOC AI identifies the probable target position and displays the localization result.

Step 5 — Verify

The detected region is checked against the reference pattern.

Step 6 — Fabrication Inspection

Load the fabricated image and run fabrication inspection.

Step 7 — Visualize Defects

Detected differences are displayed using bounding boxes on the fabricated image.

📐 Output

FINLOC AI can provide:

X Coordinate
Y Coordinate
Confidence
Verification Score
Reference Width
Reference Height
Defect Regions

Example conceptual output:

Localization Status : SUCCESS

X                  : ...
Y                  : ...
Confidence         : ...
Verification Score : ...

Fabrication Status : INSPECTION COMPLETE

Detected Regions   : ...
🧪 Experimental Dataset

The project uses synthetic semiconductor-pattern imagery for controlled experimentation.

The dataset contains:

Reference semiconductor patterns
Search images
Multiple pattern placements
Spatial localization examples
Fabrication comparison examples

The large generated dataset is excluded from the GitHub repository to keep the repository lightweight.

📈 Development Progress
Day 1–2
  ↓
Synthetic semiconductor pattern generation

Day 3–4
  ↓
Traditional Template Matching

Day 5
  ↓
AI localization dataset development

Day 6
  ↓
Coordinate regression experiments

Day 7
  ↓
Spatial localization / heatmap model

Day 8–9
  ↓
Reference and sample dataset expansion

Day 10
  ↓
Reference-conditioned Spatial AI

Final Integration
  ↓
Localization + Verification
  ↓
Fabrication Inspection
  ↓
GUI + Visualization
🧩 Key Features
Traditional + AI Localization

FINLOC AI allows comparison between a classical computer-vision approach and a trained Spatial AI approach.

Reference-conditioned Localization

The current Spatial AI model uses the reference pattern as part of the localization process.

Coordinate Reuse

The detected X-Y position is reused during fabrication inspection.

Localized Inspection

Inspection focuses on the corresponding region instead of treating the entire fabricated image as one undifferentiated comparison.

Interactive Visualization

The GUI provides visual feedback for localization and detected fabrication differences.

🎓 Project Scope

FINLOC AI is developed as an academic Applied Materials / Semiconductor Inspection PBL prototype.

The project demonstrates the integration of:

Computer Vision
        +
Deep Learning
        +
Spatial Localization
        +
Image Verification
        +
Fabrication Inspection
        +
Interactive Visualization

The current implementation uses synthetic data and is intended as a research/educational prototype rather than a production semiconductor manufacturing inspection system.

🔮 Future Development

Potential future extensions include:

Real semiconductor wafer/image datasets
SEM image integration
More complex layout structures
Multi-scale localization
Rotation and scale-aware detection
Improved defect classification
Automated defect severity estimation
Larger and more diverse training datasets
Model acceleration using GPU/edge hardware
Integration with semiconductor inspection workflows
Experimental evaluation on real fabrication data
👥 Project

FINLOC AI — AI Semiconductor Pattern Localizer

Developed as an academic project focused on AI-assisted semiconductor pattern localization and fabrication inspection.
