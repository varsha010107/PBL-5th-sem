from PIL import Image, ImageDraw
import glob
import math

files = sorted(
    glob.glob(
        "dataset/day9_ai/train/reference/*.png"
    )
)[:50]

thumb = 180
cols = 5
rows = math.ceil(len(files) / cols)

sheet = Image.new(
    "RGB",
    (cols * thumb, rows * thumb),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, filename in enumerate(files):

    img = Image.open(filename).convert("RGB")
    img.thumbnail((160, 160))

    x = (i % cols) * thumb
    y = (i // cols) * thumb

    sheet.paste(
        img,
        (x + 10, y + 5)
    )

    draw.text(
        (x + 10, y + 165),
        f"{i+1:03d}",
        fill="black"
    )

sheet.save(
    "results/day9/reference_samples.png"
)

print(
    "Saved: results/day9/reference_samples.png"
)
