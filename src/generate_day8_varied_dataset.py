import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

random.seed(42)
np.random.seed(42)

REF_SIZE = 500
SEARCH_SIZE = 2000
N = 50

REF_DIR = "dataset/day8_reference"
SEARCH_DIR = "dataset/day8_search"
GT_DIR = "dataset/day8_ground_truth"

os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(SEARCH_DIR, exist_ok=True)
os.makedirs(GT_DIR, exist_ok=True)


def create_finfet():
    """Create a varied synthetic semiconductor/FinFET-like pattern."""

    img = Image.new("L", (REF_SIZE, REF_SIZE), 255)
    draw = ImageDraw.Draw(img)

    # Random structural parameters
    fins = random.randint(6, 11)
    gates = random.randint(4, 7)

    fin_width = random.randint(7, 13)
    gate_width = random.randint(7, 13)

    x_spacing = random.randint(28, 42)
    y_spacing = random.randint(45, 65)

    x0 = 70
    y0 = 90

    # Vertical fins
    for i in range(fins):
        x = x0 + i * x_spacing
        draw.rectangle(
            [x, y0, x + fin_width, y0 + (gates - 1) * y_spacing + 20],
            fill=random.randint(25, 55)
        )

    # Horizontal gates
    for j in range(gates):
        y = y0 + j * y_spacing
        draw.rectangle(
            [x0 - 15, y, x0 + (fins - 1) * x_spacing + fin_width + 15,
             y + gate_width],
            fill=random.randint(55, 85)
        )

    # Optional circular feature / marker
    cx = x0 + ((fins - 1) * x_spacing) // 2
    cy = y0 + ((gates - 1) * y_spacing) // 2

    radius = random.randint(18, 28)

    draw.ellipse(
        [cx - radius, cy - radius,
         cx + radius, cy + radius],
        outline=random.randint(10, 40),
        width=random.randint(3, 6)
    )

    # Small random rotation
    angle = random.uniform(-5, 5)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC,
                     expand=False, fillcolor=255)

    # Small blur
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(
            random.uniform(0.2, 0.8)
        ))

    # Noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, random.uniform(2, 8), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def create_search_image(reference, target_x, target_y):
    """Place reference into a large noisy search field."""

    search = Image.new("L", (SEARCH_SIZE, SEARCH_SIZE), 255)
    draw = ImageDraw.Draw(search)

    # Background fabrication-like noise structures
    for _ in range(random.randint(80, 150)):
        x = random.randint(0, SEARCH_SIZE - 30)
        y = random.randint(0, SEARCH_SIZE - 30)

        w = random.randint(5, 45)
        h = random.randint(5, 60)

        gray = random.randint(180, 245)

        draw.rectangle(
            [x, y, x + w, y + h],
            fill=gray
        )

    # Paste target
    search.paste(reference, (target_x, target_y))

    # Slight global noise
    arr = np.array(search).astype(np.float32)
    noise = np.random.normal(0, random.uniform(1, 5), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


print("=" * 65)
print("DAY 8 - VARIED FINFET DATASET GENERATION")
print("=" * 65)

for i in range(1, N + 1):

    print(f"Generating sample {i}/{N}")

    reference = create_finfet()

    # Keep target completely inside 2000x2000 search image
    target_x = random.randint(0, SEARCH_SIZE - REF_SIZE)
    target_y = random.randint(0, SEARCH_SIZE - REF_SIZE)

    search = create_search_image(
        reference,
        target_x,
        target_y
    )

    name = f"{i:03d}"

    reference.save(
        f"{REF_DIR}/reference_{name}.png"
    )

    search.save(
        f"{SEARCH_DIR}/search_{name}.png"
    )

    with open(
        f"{GT_DIR}/ground_truth_{name}.txt",
        "w"
    ) as f:
        f.write(f"target_x={target_x}\n")
        f.write(f"target_y={target_y}\n")

    print(f"  Target = ({target_x}, {target_y})")

print("=" * 65)
print("DAY 8 DATASET COMPLETE")
print("=" * 65)
print(f"Reference images : {N}")
print(f"Search images    : {N}")
print(f"Ground truths    : {N}")

