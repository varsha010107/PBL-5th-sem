import cv2
import numpy as np


# ============================================================
# SMART FABRICATION DEFECT ENGINE
# ============================================================

PARTICLE_DIFF_PIXELS = 30
PARTICLE_MAX_DIFF = 5

STRUCTURAL_DIFF_PIXELS = 150
STRUCTURAL_PERCENTAGE = 0.15


def analyze_defect(reference, fabricated):

    # --------------------------------------------------------
    # Validate images
    # --------------------------------------------------------

    if reference is None or fabricated is None:
        raise ValueError("Unable to read inspection images")

    if reference.shape != fabricated.shape:
        raise ValueError(
            f"Image size mismatch: "
            f"{reference.shape} vs {fabricated.shape}"
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
    # Absolute difference
    # --------------------------------------------------------

    diff = cv2.absdiff(
        reference,
        fabricated
    )

    changed_mask = (
        diff > 2
    ).astype(
        np.uint8
    ) * 255

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

        if area < 2:
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

    # --------------------------------------------------------
    # PARTICLE
    # --------------------------------------------------------

    if changed_pixels >= PARTICLE_DIFF_PIXELS:

        if (
            len(regions) > 0
            and max_difference >= PARTICLE_MAX_DIFF
        ):

            largest_area = regions[0]["area"]

            if largest_area < 500:

                status = "FAIL"
                defect_type = "PARTICLE"

                confidence = min(
                    99.0,
                    60.0
                    + defect_percentage * 100
                    + max_difference * 0.10
                )

    # --------------------------------------------------------
    # STRUCTURAL DEFECT
    # --------------------------------------------------------

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
                + defect_percentage * 40
            )

    # --------------------------------------------------------
    # STRONG DIFFERENCE
    # --------------------------------------------------------

    if status == "PASS":

        if defect_percentage >= 0.30:

            status = "FAIL"
            defect_type = "FABRICATION DEFECT"

            confidence = min(
                99.0,
                80.0
                + defect_percentage * 20
            )

    # --------------------------------------------------------
    # PASS confidence
    # --------------------------------------------------------

    if status == "PASS":

        confidence = max(
            0.0,
            min(
                99.0,
                100.0 - defect_percentage * 100
            )
        )

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
