#from https://www.geeksforgeeks.org/python/create-transparent-png-image-with-python-pillow/
from PIL import Image

img = Image.open('blackEnd.JPG')
rgba = img.convert("RGBA")
datas = rgba.getdata()

newData = []
for item in datas:
    if item[0] == 0 and item[1] == 0 and item[2] == 0:  # finding black colour by its RGB value
        # storing a transparent value when we find a black colour
        newData.append((205, 205, 205, 255))
    else:
        newData.append(item)  # other colours remain unchanged

rgba.putdata(newData)
rgba.save("image205.png", "PNG")