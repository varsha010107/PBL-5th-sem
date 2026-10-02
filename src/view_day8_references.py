from PIL import Image, ImageDraw
import glob
import math

files = sorted(glob.glob("dataset/day8_reference/*.png"))

thumb_size = 200
cols = 5
rows = math.ceil(len(files) / cols)

sheet = Image.new(
    "RGB",
    (cols * thumb_size, rows * thumb_size),
    "white"
)

for i, f in enumerate(files):

    img = Image.open(f).convert("RGB")
    img.thumbnail((180, 180))

    x = (i % cols) * thumb_size
    y = (i // cols) * thumb_size

    sheet.paste(img, (x + 10, y + 10))

    draw = ImageDraw.Draw(sheet)
    draw.text(
        (x + 10, y + 185),
        f"{i+1:03d}",
        fill="black"
    )

sheet.save("results/day8/reference_contact_sheet.png")

print("Saved:")
print("results/day8/reference_contact_sheet.png")
