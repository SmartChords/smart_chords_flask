import cv2
import numpy as np
from PIL import Image
# from PIL import ImageDraw
from PIL import ImageChops
# from PIL import ImageFont


def normalize(image):
    return (255. - image) / 255.


def resize(image, height):
    width = int(float(height * image.shape[1]) / image.shape[0])
    sample_img = cv2.resize(image, (width, height))
    return sample_img


def findNextWhiteLine(im, start_line, color):
    whiteLine = Image.new(im.mode, (im.width, 1), color)
    whitebytes = whiteLine.tobytes()
    for y in range (start_line, im.height):
        line = im.crop((0, y, im.width, y+1))
        #line_bytes = line.tobytes()
        if line.tobytes() == whiteLine.tobytes():
            return y
    return -1

def findNextBlackLine(im, start_line, color):
    whiteLine = Image.new(im.mode, (im.width, 1), color)
    whitebytes = whiteLine.tobytes()
    for y in range (start_line, im.height):
        line = im.crop((0, y, im.width, y+1))
        #line_bytes = line.tobytes
        if line.tobytes() != whiteLine.tobytes():
            return y
    return -1

def partitionImage(im, color):
    next_white_line = 0
    next_black_line = 0
    array_img = []
    more = True
    while more:
        next_black_line = findNextBlackLine(im, next_black_line, color)
        if next_black_line != -1:
            next_white_line = findNextWhiteLine(im, next_black_line, color)
            if next_white_line != -1:
                sub_img = im.crop((0, next_black_line, im.width, next_white_line -1))
                array_img.append(sub_img)
                next_black_line = next_white_line;
            else:
                sub_img = im.crop((0, next_black_line, im.width, im.height))
                array_img.append(sub_img)
                more = False
        else:
            more = False

            if len(array_img) == 0:
                array_img.append(im)
    return array_img

#crop function, used by the partition (horizontal) algorithm
def crop(im, white):
    bg = Image.new(im.mode, im.size, white)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

#algorithm that splits (partitions) an image horizontally based on white lines
# note that the 2nd paramtere is actually the color passed, so the splitting can
# be done accordin to any color line (though we use white, which is 255).
#also, this is a recursive function, and returns an array of images. If no horizontal w
#white lines were found in the original image, it will return an array with one entry, the original image.
def split(im, white):
    # Is there a horizontal white line?
    whiteLine = Image.new(im.mode, (im.width, 1), white)
    print ("Image Width = " + str(im.width))
    for y in range(im.height):
        line = im.crop((0, y, im.width, y+1))
        if line.tobytes() == whiteLine.tobytes():
            # There is a white line
            # So we can split the image into two
            # and for efficiency, crop it
            ims = [
                crop(im.crop((0, 0, im.width, y)), white),
                crop(im.crop((0, y+1, im.width, im.height)), white)
            ]
            # Now, because there may be white lines within the two subimages
            # Call split again, making this recursive
            return [sub_im for im in ims for sub_im in split(im, white)]
    return[im]

# The starting point of the split algorithm. We call this
# function with the input image that we would like to split horizontally
# this function, as is, assumes that the splitting is done based on the color
# of the very first pixel in the image (pixel at 0,0). With a withe background,
# this is almost always white, but we could hard code this if necessary
def trim(im):
    # You have defined the pixel at (0, 0) as white in your code
    white = im.getpixel((0,0))

    # Initial crop
    im = crop(im, white)
    if im:
        return split(im, white)
    else:
        print("No image detected")

# def isMusicalImage(image_to_convert):
#     indicator_img = Image.open("indicator.png").convert('L')
#     indicator_w = indicator_img.width
#     indicator_h = indicator_img.height
#     image_h = image_to_convert.height
#     ratio = float((image_h / indicator_h))
#     resize_w = int(indicator_w * ratio)
#     resize_h = int(indicator_h * ratio)
#     resize_img = cv2.resize(np.array(indicator_img), (resize_w, resize_h))
#
#
#     result = cv2.matchTemplate(np.array(image_to_convert),resize_img,cv2.TM_CCOEFF_NORMED)
#     threshold = 0.8
#     flag = False
#     probability = np.amax(result)
#     if np.amax(result) > threshold:
#         flag = True
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#     if (min_loc < max_loc):
#         flag = True
#     unraveled = np.unravel_index(result.argmax(),result.shape)
#
#     return flag
