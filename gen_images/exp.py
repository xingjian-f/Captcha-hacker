from PIL import Image
from PIL import ImageFont, ImageDraw

image = Image.new('1', (100, 100), 'white')
draw = ImageDraw.Draw(image)

# use a bitmap font
# font = ImageFont.load("arial.pil")

# draw.text((10, 10), "hello", font=font)

# use a truetype font
font = ImageFont.truetype("arial.ttf")
help(font)
draw.text((10, 25), "w", font=font)

image.show()