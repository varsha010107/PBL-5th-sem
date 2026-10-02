import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

random.seed(123)
np.random.seed(123)

REF_SIZE = 500
SEARCH_SIZE = 2000

SPLITS = {
    "train": 800,
    "val": 100,
    "test": 100
}


def create_finfet():
    img = Image.new("L", (REF_SIZE, REF_SIZE), 255)
    draw = ImageDraw.Draw(img)

    # Random FinFET geometry
    fins = random.randint(6, 11)
    gates = random.randint(4, 7)

    fin_width = random.randint(7, 13)
    gate_width = random.randint(7, 13)

    x_spacing = random.randint(28, 42)
    y_spacing = random.randint(45, 65)

    x0 = random.randint(55, 85)
    y0 = random.randint(65, 100)

    # Vertical fins
    for i in range(fins):
        x = x0 + i * x_spacing

        draw.rectangle(
            [
                x,
                y0,
                x + fin_width,
                y0 + (gates - 1) * y_spacing + 25
            ],
            fill=random.randint(20, 60)
        )

    # Horizontal gates
    for j in range(gates):
        y = y0 + j * y_spacing

        draw.rectangle(
            [
                x0 - 15,
                y,
                x0 + (fins - 1) * x_spacing + fin_width + 15,
                y + gate_width
            ],
            fill=random.randint(45, 90)
        )

    # Central feature
    cx = x0 + ((fins - 1) * x_spacing) // 2
    cy = y0 + ((gates - 1) * y_spacing) // 2

    radius = random.randint(16, 30)

    draw.ellipse(
        [
            cx - radius,
            cy - radius,
            cx + radius,
            cy + radius
        ],
        outline=random.randint(10, 45),
        width=random.randint(3, 6)
    )

    # Rotation
    angle = random.uniform(-6, 6)

    img = img.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=255
    )

    # Blur
    if random.random() < 0.5:
        img = img.filter(
            ImageFilter.GaussianBlur(
                random.uniform(0.2, 0.8)
            )
        )

    # Sensor/image noise
    arr = np.array(img).astype(np.float32)

    noise = np.random.normal(
        0,
        random.uniform(2, 8),
        arr.shape
    )

    arr = np.clip(
        arr + noise,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(arr)


def create_search(reference, target_x, target_y):

    search = Image.new(
        "L",
        (SEARCH_SIZE, SEARCH_SIZE),
        255
    )

    draw = ImageDraw.Draw(search)

    # Background structures
    for _ in range(random.randint(100, 180)):

        x = random.randint(0, SEARCH_SIZE - 40)
        y = random.randint(0, SEARCH_SIZE - 40)

        w = random.randint(5, 50)
        h = random.randint(5, 65)

        gray = random.randint(175, 245)

        draw.rectangle(
            [
                x,
                y,
                x + w,
                y + h
            ],
            fill=gray
        )

    # Insert target
    search.paste(
        reference,
        (target_x, target_y)
    )

    # Search-image noise
    arr = np.array(search).astype(np.float32)

    noise = np.random.normal(
        0,
        random.uniform(1, 5),
        arr.shape
    )

    arr = np.clip(
        arr + noise,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(arr)


def save_sample(split, number):

    base = f"dataset/day9_ai/{split}"

    reference = create_finfet()

    target_x = random.randint(
        0,
        SEARCH_SIZE - REF_SIZE
    )

    target_y = random.randint(
        0,
        SEARCH_SIZE - REF_SIZE
    )

    search = create_search(
        reference,
        target_x,
        target_y
    )

    filename = f"{number:04d}"

    reference.save(
        f"{base}/reference/reference_{filename}.png"
    )

    search.save(
        f"{base}/search/search_{filename}.png"
    )

    with open(
        f"{base}/labels/label_{filename}.txt",
        "w"
    ) as f:

        f.write(
            f"{target_x},{target_y}\n"
        )

    return target_x, target_y


print("=" * 70)
print("DAY 9 - FULL VARIED AI DATASET GENERATION")
print("=" * 70)

total = 0

for split, count in SPLITS.items():

    print()
    print(f"Generating {split.upper()} dataset: {count} samples")

    for i in range(1, count + 1):

        x, y = save_sample(
            split,
            i
        )

        total += 1

        if i % 100 == 0 or i == count:
            print(
                f"{split}: {i}/{count}"
            )

print()
print("=" * 70)
print("DAY 9 DATASET GENERATION COMPLETE")
print("=" * 70)

print(f"Total samples : {total}")
print("Train         : 800")
print("Validation    : 100")
print("Test          : 100")
print()
print("Dataset:")
print("dataset/day9_ai/")
