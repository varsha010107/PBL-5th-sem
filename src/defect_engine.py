import cv2
import numpy as np


# ============================================================
# SMART FABRICATION DEFECT ENGINE
# ============================================================

# Pixel-difference threshold.
# Small intensity variations below this are treated as noise.
PIXEL_DIFF_THRESHOLD = 10

# Minimum connected component area considered meaningful.
MIN_REGION_AREA = 8

# Particle detection
PARTICLE_DIFF_PIXELS = 30
PARTICLE_MAX_DIFF = 15
PARTICLE_MAX_AREA = 500

# Structural defect thresholds.
# defect_percentage is already expressed in %.
STRUCTURAL_DIFF_PIXELS = 150
STRUCTURAL_PERCENTAGE = 15.0

# Strong overall difference.
STRONG_DIFFERENCE_PERCENTAGE = 30.0


def analyze_defect(reference, fabricated):

    # --------------------------------------------------------
    # Validate images
    # --------------------------------------------------------

    if reference is None or fabricated is None:
        raise ValueError(
            "Unable to read inspection images"
        )

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    if len(reference.shape) == 3:
        reference = cv2.cvtColor(
            reference,
            cv2.COLOR_BGR2GRAY
        )

    if len(fabricated.shape) == 3:
        fabricated = cv2.cvtColor(
            fabricated,
            cv2.COLOR_BGR2GRAY
        )

    # --------------------------------------------------------
    # Validate size AFTER grayscale conversion
    # --------------------------------------------------------

    if reference.shape != fabricated.shape:
        raise ValueError(
            f"Image size mismatch: "
            f"{reference.shape} vs {fabricated.shape}"
        )

    # --------------------------------------------------------
    # Absolute pixel difference
    # --------------------------------------------------------

    diff = cv2.absdiff(
        reference,
        fabricated
    )

    # --------------------------------------------------------
    # Threshold small intensity variations
    # --------------------------------------------------------

    changed_mask = (
        diff > PIXEL_DIFF_THRESHOLD
    ).astype(
        np.uint8
    ) * 255

    # --------------------------------------------------------
    # Morphological noise suppression
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    # Remove isolated single-pixel noise
    changed_mask = cv2.morphologyEx(
        changed_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Connect nearby pixels belonging to the same defect
    changed_mask = cv2.morphologyEx(
        changed_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------------
    # Changed pixel statistics
    # --------------------------------------------------------

    changed_pixels = int(
        np.count_nonzero(changed_mask)
    )

    total_pixels = (
        reference.shape[0] *
        reference.shape[1]
    )

    defect_percentage = (
        changed_pixels /
        total_pixels *
        100.0
    )

    max_difference = int(
        diff.max()
    )

    mean_difference = float(
        diff.mean()
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    num_labels, component_labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            changed_mask,
            8
        )
    )

    regions = []

    for i in range(1, num_labels):

        area = int(
            stats[i, cv2.CC_STAT_AREA]
        )

        if area < MIN_REGION_AREA:
            continue

        x = int(
            stats[i, cv2.CC_STAT_LEFT]
        )

        y = int(
            stats[i, cv2.CC_STAT_TOP]
        )

        w = int(
            stats[i, cv2.CC_STAT_WIDTH]
        )

        h = int(
            stats[i, cv2.CC_STAT_HEIGHT]
        )

        regions.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area
        })

    # Largest regions first
    regions.sort(
        key=lambda r: r["area"],
        reverse=True
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    status = "PASS"
    defect_type = "GOOD"
    confidence = 0.0

    # ========================================================
    # PARTICLE DEFECT
    # ========================================================

    if changed_pixels >= PARTICLE_DIFF_PIXELS:

        if (
            len(regions) > 0
            and max_difference >= PARTICLE_MAX_DIFF
        ):

            largest_area = regions[0]["area"]

            if largest_area < PARTICLE_MAX_AREA:

                status = "FAIL"
                defect_type = "PARTICLE"

                confidence = min(
                    99.0,
                    60.0
                    + defect_percentage * 2.0
                    + max_difference * 0.10
                )

    # ========================================================
    # STRUCTURAL DEFECT
    # ========================================================

    if status == "PASS":

        if (
            changed_pixels >= STRUCTURAL_DIFF_PIXELS
            and defect_percentage >= STRUCTURAL_PERCENTAGE
        ):

            status = "FAIL"
            defect_type = "FABRICATION DEFECT"

            confidence = min(
                99.0,
                70.0
                + defect_percentage * 1.0
            )

    # ========================================================
    # STRONG OVERALL DIFFERENCE
    # ========================================================

    if status == "PASS":

        if defect_percentage >= STRONG_DIFFERENCE_PERCENTAGE:

            status = "FAIL"
            defect_type = "FABRICATION DEFECT"

            confidence = min(
                99.0,
                80.0
                + defect_percentage * 0.5
            )

    # ========================================================
    # PASS CONFIDENCE
    # ========================================================

    if status == "PASS":

        confidence = max(
            0.0,
            min(
                99.0,
                100.0 - defect_percentage
            )
        )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "status": status,
        "defect_type": defect_type,
        "confidence": confidence,
        "defect_percentage": defect_percentage,
        "changed_pixels": changed_pixels,
        "max_difference": max_difference,
        "mean_difference": mean_difference,
        "regions": regions,
        "diff_image": diff,
        "mask": changed_mask
    }
