import os
from flask import Flask, render_template, request, flash, redirect, jsonify, send_from_directory, send_file, make_response, session
from werkzeug.utils import secure_filename
from forms import ContactForm
from flask_mail import Message, Mail
from functools import wraps, update_wrapper
from tensorflow.python.framework import ops
from tensorflow.python.training import saver as saver_lib
import os
import notes

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageChops
from PIL import ImageFont


# mail = Mail()

app = Flask(__name__)
app.config.from_object('config.Config')
#app.config.from_object(os.environ['APP_SETTINGS'])
app.secret_key = 'development key'

# mail.init_app(app)


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

def isMusicalImage(image_to_convert):
    indicator_img = Image.open("indicator.png").convert('L')
    indicator_w = indicator_img.width
    indicator_h = indicator_img.height
    image_h = image_to_convert.height
    ratio = float((image_h / indicator_h))
    resize_w = int(indicator_w * ratio)
    resize_h = int(indicator_h * ratio)
    resize_img = cv2.resize(np.array(indicator_img), (resize_w, resize_h))


    result = cv2.matchTemplate(np.array(image_to_convert),resize_img,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    flag = False
    probability = np.amax(result)
    if np.amax(result) > threshold:
        flag = True
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if (min_loc < max_loc):
        flag = True
    unraveled = np.unravel_index(result.argmax(),result.shape)
    print (unraveled)
    



    return flag

def doConversion(image_to_convert):
      #isMusicalImage(image_to_convert)
      img = image_to_convert
      image = image_to_convert;
      #image = Image.open(img).convert('L')
      image = np.array(image)
      image = resize(image, HEIGHT)
      image = normalize(image)
      image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

      seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]
      prediction = sess.run(decoded,
                            feed_dict={
                                input: image,
                                seq_len: seq_lengths,
                                rnn_keep_prob: 1.0,
                            })
      str_predictions = sparse_tensor_to_strs(prediction)

      array_of_notes = []

      for w in str_predictions[0]:
          array_of_notes.append(int2word[w])
      notes=[]
      for i in array_of_notes:
          if i[0:5]=="note-":
              if not i[6].isdigit():
                  notes.append(i[5:7])
              else:
                  notes.append(i[5])

      #img = Image.open(img).convert('L')
      size = (img.size[0], int(img.size[1]*1.5))
      layer = Image.new('RGB', size, (255,255,255))
      layer.paste(img, box=None)
      img_arr = np.array(layer)
      height = int(img_arr.shape[0])
      width = int(img_arr.shape[1])
      # print(img_arr.shape[0])
      draw = ImageDraw.Draw(layer)
      # font = ImageFont.truetype(<font-file>, <font-size>)
      font = ImageFont.truetype("Aaargh.ttf", 20)
      # draw.text((x, y),"Sample Text",(r,g,b))
      j = width / 9
      for i in notes:
          draw.text((j, height-40), i, (0,0,0), font=font)
          j+= (width / (len(notes) + 4))

      return layer      


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'fileToUpload' not in request.files:
            flash("No file to upload")
            return redirect(request.url)

        image = request.files['fileToUpload']
        if image.filename == "":
            flash("No file to upload")
            return redirect(request.url)

        if image and allowed_image(image.filename):
            file_name = secure_filename(image.filename)
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], file_name))
            return render_template('preview.html', filename=file_name)
        else:
            flash("Incorrect file type")
            return redirect(request.url)


    else:
        return render_template('index.html')


def allowed_image(filename):
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route('/upload/<filename>')
def display_upload(filename):
	return send_from_directory(app.config['IMAGE_UPLOADS'], filename)

@app.route('/download/<download>')
def display_download(download):
	return send_from_directory(app.config['IMAGE_DOWNLOADS'], filename)



@app.route('/preview/<filename>', methods=['GET'])
def preview(filename):
    return render_template("preview.html", filename=filename)


@app.route('/annotated')
def annotated():
    return render_template('annotated.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    form = ContactForm()
    if form.validate_on_submit():
        # TODO: SET UP EMAILING FUNCTIONALITY
        return render_template('contact.html', success=True)


    return render_template('contact.html', form=form)

@app.route('/help')
def help():
    return render_template('help.html')

##############################################################
############ START OF IMAGE PROCESSOR  #######################
def sparse_tensor_to_strs(sparse_tensor):
    indices = sparse_tensor[0][0]
    values = sparse_tensor[0][1]
    dense_shape = sparse_tensor[0][2]

    strs = [[] for i in range(dense_shape[0])]
    string = []
    ptr = 0
    b = 0

    for idx in range(len(indices)):
        if indices[idx][0] != b:
            strs[b] = string
            string = []
            b = indices[idx][0]

        string.append(values[ptr])
        ptr = ptr + 1

    strs[b] = string
    return strs

######### IMAGE HELPERS #########
def normalize(image):
    return (255. - image) / 255.


def resize(image, height):
    width = int(float(height * image.shape[1]) / image.shape[0])
    sample_img = cv2.resize(image, (width, height))
    return sample_img
##################################


voc_file = "vocabulary_semantic.txt"
model = "Semantic-Model/semantic_model.meta"

ops.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()
# Read the dictionary
dict_file = open(voc_file, 'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
    word_idx = len(int2word)
    int2word[word_idx] = word
dict_file.close()

######### THIS SECTION RESTORES THE IMAGE RECOGNITION MODEL FROM Semantic-Model/semantic_model.meta #########
# Restore weights
tf.compat.v1.disable_eager_execution()
saver = saver_lib.import_meta_graph(model)
saver.restore(sess, model[:-5])
#######################################

graph = tf.compat.v1.get_default_graph()

input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
# logits = tf.get_collection("logits")[0]
logits = tf.compat.v1.get_collection("logits")[0]
# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

def send_img(filename):
    return send_from_directory(app.config['IMAGE_UPLOADS'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        converted_array = []

        filename = request.form['preview-image']

 
        #img = send_img(filename)
        #img.direct_passthrough = False
        f= filename
        image1 = Image.open(f).convert('L')
        
        #we assume the color at pixel 0,0 is the color of the background
        color = image1.getpixel((0,0))
        #horiz_images = trim(image1) # this splits the image horizontally, based on white lines
        horiz_images = partitionImage(image1, color) # this splits the image horizontally, based on white lines
        index = 0;
        #// for practice, we save extracted images in the same directory as app.py
        # NOTE ALSO< THAT AS THE CODE STANDS NOW, THE INPUT IMAGE MUST ALSO BE IN THE
        #APP.PY directory. THE SPLITTING IS DONE AFTER YOU SELECT AND SUBMIT an IMAGE
        for i in horiz_images: 
            fname = str(index) +".png"
            #i.save(fname)
            #converted = doConversion(fname)
            converted = doConversion(i)
            #converted.save(str(index) +"_converted.png")
            converted_array.append(converted)
            index = index + 1
        
        converted_height = 0
        for c in converted_array:
            converted_height = converted_height + c.height
        
        combined_width = converted_array[0].width
        dst = Image.new('L', (combined_width, converted_height))
        h_index = 0
        for c in converted_array: #this loop stictches back the parts, and saves the result
            dst.paste(c, (0, h_index))
            h_index = h_index + c.height

        dst_name = "converted_" + filename
        dst.save(dst_name)

        return render_template('annotated.html')
        

        # image = Image.open(img).convert('L')
        # image = np.array(image)
        # image = resize(image, HEIGHT)
        # image = normalize(image)
        # image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)
        #
        # seq_lengths = [image.shape[2] / WIDTH_REDUCTION]
        # prediction = sess.run(decoded,
        #                       feed_dict={
        #                           input: image,
        #                           seq_len: seq_lengths,
        #                           rnn_keep_prob: 1.0,
        #                       })
        # str_predictions = sparse_tensor_to_strs(prediction)
        # array_of_notes = []
        #
        # for w in str_predictions[0]:
        #     array_of_notes.append(int2word[w])
        #
        # notes = []
        # for i in array_of_notes:
        #     if i[0:4] == "key-":
        #         notes.append(i)
        #     if i[0:5] == "note-":
        #         notes.append(i.split("-")[1])
        #     if i == 'BAR':
        #         notes.append(i)
        #
        # # build_model_input(notes)
        # # chords = get_chord_predictions()
        #
        # # FROM HERE, chords WILL BE AN ARRAY OF NAMES THAT WE WILL USE TO QUERY THE LOOKUP TABLE
        # # IT WILL LOOK SOMETHING LIKE THIS: ['c#-min' 'g#-min' 'D-Maj7' 'D-Maj7' 'c#-min' 'B-Maj7']
        # # QUERY THE TABLE AND GET AN ARRAY OF THE CORRESPONDING DB ROWS
        #
        # ######### THIS SECTION WRITES TO THE IMAGE #########
        # img = Image.open(img).convert('L')
        # size = (img.size[0], int(img.size[1] * 1.5))
        # layer = Image.new('RGB', size, (255, 255, 255))
        # layer.paste(img, box=None)
        # img_arr = np.array(layer)
        # height = int(img_arr.shape[0])
        # width = int(img_arr.shape[1])
        # draw = ImageDraw.Draw(layer)
        # # font = ImageFont.truetype(<font-file>, <font-size>)
        # font = ImageFont.truetype("Aaargh.ttf", 20)
        # # draw.text((x, y),"Sample Text",(r,g,b))
        # j = width / 9
        # for i in notes: # for i in chord_images:
        #     ##########INSTEAD OF draw.text() HERE, WE WANT IT TO PASTE THE CHORD CHARTS #########
        #     draw.text((j, height - 40), i, (0, 0, 0), font=font)
        #     j += (width / (len(notes) + 4))
        # layer.save("static/img/download/annotated.png")
        # return render_template('annotated.html')

 
if __name__ == '__main__':
    app.run()
