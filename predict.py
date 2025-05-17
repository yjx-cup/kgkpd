from PIL import Image
from kgkpd import KGKPD

model = KGKPD()
while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        r_image = model.detect_image(image)
        r_image.show()
