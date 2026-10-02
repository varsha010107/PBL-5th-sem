import cv2
import numpy as np
import os
import random


OUT_DIR = "data/test_images"
os.makedirs(OUT_DIR, exist_ok=True)


def create_semiconductor_pattern(
    filename,
    add_defects=False
):

    # Large inspection image
    H, W = 1000, 1400

    # Slightly gray microscope background
    image = np.random.normal(
        235, 7, (H, W)
    ).astype(np.uint8)

    # Smooth background
    image = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    # -------------------------------------------------
    # CHIP / METAL STRUCTURES
    # -------------------------------------------------

    for row in range(5):

        y = 100 + row * 170

        for col in range(7):

            x = 80 + col * 185

            # rectangular device region
            cv2.rectangle(
                image,
                (x, y),
                (x + 130, y + 110),
                random.randint(40, 80),
                2
            )

            # metal interconnects
            for k in range(4):

                yy = y + 15 + k * 25

                cv2.line(
                    image,
                    (x + 5, yy),
                    (x + 125, yy),
                    random.randint(40, 75),
                    3
                )

            # vertical structures
            for k in range(4):

                xx = x + 15 + k * 30

                cv2.line(
                    image,
                    (xx, y + 5),
                    (xx, y + 105),
                    random.randint(40, 75),
                    3
                )

            # contact/via
            cv2.circle(
                image,
                (x + 65, y + 55),
                8,
                30,
                -1
            )

            cv2.circle(
                image,
                (x + 65, y + 55),
                13,
                70,
                2
            )

    # -------------------------------------------------
    # SMALL RANDOM FEATURES
    # -------------------------------------------------

    for _ in range(150):

        x = random.randint(30, W - 30)
        y = random.randint(30, H - 30)

        r = random.randint(2, 6)

        cv2.circle(
            image,
            (x, y),
            r,
            random.randint(60, 130),
            -1
        )

    # -------------------------------------------------
    # FABRICATION DEFECTS
    # -------------------------------------------------

    if add_defects:

        # Particle 1
        cv2.circle(
            image,
            (500, 300),
            12,
            20,
            -1
        )

        # Missing metal region
        cv2.rectangle(
            image,
            (850, 500),
            (900, 540),
            235,
            -1
        )

        # Extra particle
        cv2.circle(
            image,
            (1050, 700),
            18,
            10,
            -1
        )

        # Broken interconnect
        cv2.rectangle(
            image,
            (250, 760),
            (330, 775),
            235,
            -1
        )

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------

    cv2.imwrite(
        os.path.join(OUT_DIR, filename),
        image
    )


# Reference image
create_semiconductor_pattern(
    "reference_0001.png",
    add_defects=False
)

# Fabricated image
create_semiconductor_pattern(
    "fabricated_0001.png",
    add_defects=True
)

print("Test semiconductor images generated.")
