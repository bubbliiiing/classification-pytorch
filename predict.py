from PIL import Image

from classification import Classification

classfication = Classification()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        continue
    else:
        class_name = classfication.detect_image(image)
        print(class_name)
