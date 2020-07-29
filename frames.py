import cv2
import re
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

def subimg_location(haystack, needle):
    arr_h = np.asarray(haystack)
    arr_n = np.asarray(needle)

    sub_array = np.array_split(arr_h, needle.size)

    y_h, x_h = arr_h.shape[:2]
    y_n, x_n = arr_n.shape[:2]

    xstop = x_h - x_n + 1
    ystop = y_h - y_n + 1

    matches = []
    for xmin in range(0, xstop):
        for ymin in range(0, ystop):
            xmax = xmin + x_n
            ymax = ymin + y_n

            arr_s = arr_h[ymin:ymax, xmin:xmax]     # Extract subimage
            arr_t = (arr_s == arr_n)                # Create test matrix
            hits = np.count_nonzero(arr_t == True)
            misses = np.count_nonzero(arr_t == False)
            total = hits + misses
            percentage = float(hits/total)
            print (percentage)
            if (percentage > 0.9):
                matches.append(xmin, ymin)
            if arr_t.all():                         # Only consider exact matches
                matches.append((xmin,ymin))

    return matches

def hasMusicalKey(image_to_convert):
    matches = []
    indicator_img = Image.open("indicator.png").convert('L')
    subimg_location(image_to_convert, indicator_img)
    image_to_convert = crop(image_to_convert, 255)
    indicator_w = indicator_img.width
    indicator_h = indicator_img.height
    image_h = image_to_convert.height
    ratio = float((image_h / indicator_h))
    resize_w = int(indicator_w * ratio)
    resize_h = int(indicator_h * ratio)
    resize_img = indicator_img.resize((resize_w, resize_h), Image.ANTIALIAS)
    resize_img = crop (resize_img, 255)
    slices = (int) (image_to_convert.width / resize_img.width)
    if ((int) (image_to_convert.width % resize_img.width) != 0):
        slices = slices + 1
    arr_h = np.asarray(image_to_convert)
    arr_n = np.asarray(resize_img)

    sub_images = np.array_split(arr_h, resize_img.size)
    for arr_s in sub_images:
        arr_t = (arr_s == arr_n)                # Create test matrix
        hits = np.count_nonzero(arr_t == True)
        misses = np.count_nonzero(arr_t == False)
        total = hits + misses
        percentage = float(hits/total)
        print (percentage)
        if (percentage > 0.9):
            matches.append(xmin, ymin)

    return matches

def isMusicalImage(image_to_convert):
     #hasMusicalKey(image_to_convert)
     indicator_img = Image.open("indicator.png").convert('L')
     indicator_w = indicator_img.width
     indicator_h = indicator_img.height
     image_h = image_to_convert.height
     ratio = float((image_h / indicator_h))
     resize_w = int(indicator_w * ratio)
     resize_h = int(indicator_h * ratio)
     resize_img = cv2.resize(np.array(indicator_img), (resize_w, resize_h))
     needle = indicator_img.resize((resize_w, resize_h), Image.ANTIALIAS)
     #if needle != None:
         #matches = subimg_location(image_to_convert, needle)
#
#
     #image_to_convert = crop(image_to_convert, 255)
     result = cv2.matchTemplate(np.array(image_to_convert),resize_img,cv2.TM_CCOEFF_NORMED)
     ## [normalize]
     cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
     confidence = 0.95
     match_indices = np.arange(result.size)[(result>confidence).flatten()]
     print (np.unravel_index(match_indices,result.shape))

     threshold = 0.8
     flag = False
     probability = np.amax(result)
     if np.amax(result) > threshold:
         flag = True
     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
     if (min_loc < max_loc):
         flag = True
#     unraveled = np.unravel_index(result.argmax(),result.shape)
#
     return flag


# def label_frame_with_chords_images(frame, chords, coords):
    # FROM HERE, chords WILL BE AN ARRAY OF NAMES THAT WE WILL USE TO QUERY THE LOOKUP TABLE
    # # IT WILL LOOK SOMETHING LIKE THIS: ['c#-min', 'g#-min', 'D-Maj7', 'D-Maj7', 'c#-min', 'B-Maj7']
    # # OR LIKE THIS: ['', '', 'c#-min', 'g#-min', '', 'D-Maj7', 'D-Maj7', 'c#-min', 'B-Maj7']


    # ToDo - I am looking for output like this from the algoritms.  This is just canned data.
    # # I am guessing this might change.  -RTW

    # frames = ["Frame-1.png", "Frame-2.png", "Frame-3.png", "Frame-4.png"]
    # chords = ["A-Chord.png", "Gmaj7-Chord.png", "G-Chord.png", "Em7-Chord.png", "E7-Chord.png", "B7-Chord.png"]
    # coords = [137, 300, 400, 640, 750, 910]

    # This is the padding between frames and between the chord chart and the frame
    # chordPad = 30
    # framePad = 25
    # chordWidth = 50
    # chordHeight = 70

    # Curser will be the Y-Row in the image at any time.
    # cursorY = 0

    # Create a new annotated image to display
    # TODO - Get the cumulative width and height
    # aImg = Image.new('RGB', size = (970, 1000), color = (255, 255, 255))

    # Process and paste each frame into the annotated image:
    # for frame in frames:
    #     # Open the frame
    #     frameImg = Image.open("./static/testframes/" + frame).convert('L')
    #
    #     # Paste the chords for this frame into annotated.png
    #     cursorY += chordPad
    #     for i in range(len(chords)):
    #         # Open the chord image
    #         chordImg = Image.open("./static/img/chords/" + chords[i]).convert('L')
    #
    #         # Paste the chord[i] at coords[i]
    #         Image.Image.paste(aImg, chordImg, (coords[i], cursorY))
    #
    #         # Close the chord image
    #         chordImg.close()
    #
    #     # Set the cursorY value
    #     cursorY += (chordHeight + framePad)
    #
    #     # Paste the frame
    #     Image.Image.paste(aImg, frameImg, (0, cursorY))
    #
    #     # Move the cursorY down the frame height
    #     cursorY += frameImg.size[1]
    #
    #     # Close the frame
    #     frameImg.close()
    #
    # # Resize and save the annotated image
    # basewidth = 600
    # wpercent = (basewidth/float(aImg.size[0]))
    # hsize = int((float(aImg.size[1])*float(wpercent)))
    # aImg = aImg.resize((basewidth, hsize), Image.ANTIALIAS)
    #
    # aImg.save("./static/img/downloads/annotated.png")
#     return flag

