import os
import os.path 
from os import path
from flask import Flask, render_template, request, flash, redirect, jsonify, send_from_directory, send_file, make_response, session
from werkzeug.utils import secure_filename
from forms import ContactForm
from flask_mail import Message, Mail
from functools import wraps, update_wrapper
from tensorflow.python.framework import ops
from tensorflow.python.training import saver as saver_lib
from notes import build_model_input, get_chord_predictions
from frames import partitionImage, resize, normalize#, label_frame_with_chords_images

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

# mail.init_app(app)

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

@app.route('/uploads/<filename>')
def display_upload(filename):
	return send_from_directory(app.config['IMAGE_UPLOADS'], filename)

@app.route('/downloads/<download>')
def display_download(download):
	return send_from_directory(app.config['IMAGE_DOWNLOADS'], download)

@app.route('/preview/<filename>', methods=['GET'])
def preview(filename):
    return render_template("preview.html", filename=filename)

@app.route('/annotated', methods=['GET'])
def annotated():
    return render_template('annotated.html', download='converted_annotatedtest.png')

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
logits = tf.compat.v1.get_collection("logits")[0]
# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)

def send_img(filename):
    return send_from_directory(app.config['IMAGE_UPLOADS'], filename)


def doConversion(image_to_convert, chords_list):
#     #isMusicalImage(image_to_convert)
    # img = image_to_convert
#     image = image_to_convert;
    # image = Image.open(img).convert('L')
    # image = np.array(image)
    # image = resize(image, HEIGHT)
    # image = normalize(image)
#     image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)
#
#     seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]
#     prediction = sess.run(decoded,
#                         feed_dict={
#                             input: image,
#                             seq_len: seq_lengths,
#                             rnn_keep_prob: 1.0,
#                         })
#     str_predictions = sparse_tensor_to_strs(prediction)
#
#     array_of_notes = []
#
#     for w in str_predictions[0]:
#       array_of_notes.append(int2word[w])
#     notes=[]
#     for i in array_of_notes:
#       if i[0:5]=="note-":
#           if not i[6].isdigit():
#               notes.append(i[5:7])
#           else:
#               notes.append(i[5])

    img = image_to_convert
    img = Image.open(img).convert('L')
    size = (img.size[0], int(img.size[1]*1.5))
    layer = Image.new('RGB', size, (255,255,255))
    layer.paste(img, box=None)
    img_arr = np.array(layer)
    height = int(img_arr.shape[0])
    width = int(img_arr.shape[1])
    # print(img_arr.shape[0])
    draw = ImageDraw.Draw(layer)
    font = ImageFont.truetype("Aaargh.ttf", 20)
    j = width / 5
    for i in chords_list:
      draw.text((j, height-40), i, (0,0,0), font=font)
      j+= (width / (len(chords_list) + 4))

    return layer

def get_notes_from_frame(img):
    image = np.array(img)
    image = resize(image, HEIGHT)
    image = normalize(image)
    image = np.asarray(image).reshape(1, image.shape[0], image.shape[1], 1)

    seq_lengths = [image.shape[2] / WIDTH_REDUCTION]
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

    notes = []
    for i in array_of_notes:
        if i[0:4] == "key-":
            notes.append(i)
        if i[0:5] == "note-":
            notes.append(i.split("-")[1])
        if i == 'BAR':
            notes.append(i)

    return notes


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        filename = request.form['preview-image']

        image1 = Image.open(filename).convert('L')

        #we assume the color at pixel 0,0 is the color of the background
        color = image1.getpixel((0,0))
        #horiz_images = trim(image1) # this splits the image horizontally, based on white lines
        horiz_images = partitionImage(image1, color) # this splits the image horizontally, based on white lines
        #// for practice, we save extracted images in the same directory as app.py
        # NOTE ALSO< THAT AS THE CODE STANDS NOW, THE INPUT IMAGE MUST ALSO BE IN THE
        #APP.PY directory. THE SPLITTING IS DONE AFTER YOU SELECT AND SUBMIT an IMAGE
        converted_array = []
        chords_dict = {}
        index = 0
        for i in horiz_images:
            fname = str(index) +".png"
            i.save(fname)

            # for i in range(index - 1):
            frame_image = Image.open(fname).convert('L')
            w, h = frame_image.size
            if w < 500 or h < 50:
                print(f"image too small in {index}")
                converted_array.append(frame_image)
                chords_dict[index] = []
                index = index + 1
                continue

            #notes is an array of note values that ideally would look something like this: ['key-3.821', '0.664', '0.883', '0.664', '0.415', '0.498','0.581', 'BAR', '0.581', '0.664', '0.581', '0.249', '0.332', '0.415', '0.498', 'BAR', '0.581', '0.664', '0.883', '0.996', '0.883', '0.664', '0.581', '0.415']

            notes = get_notes_from_frame(frame_image)
            print('notes from frame ', index)
            print(notes)
            if len(notes) == 0:
                print(f"no notes in {index}")
                converted_array.append(frame_image)
                chords_dict[index] = []
                index = index + 1
                continue
            else:
                build_model_input(notes, index)
                c = get_chord_predictions(index)
                chords_dict[index] = c

            print(chords_dict)

            converted = doConversion(fname, chords_dict[index])
            converted.save(str(index) +"_converted.png")
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
            
        for n in range(index):
            os.remove( str(n) + ".png" )
            if ( os.path.exists(str(n) + "_converted.png") ) :
                os.remove( str(n) + "_converted.png" )  
            else : 
                None

        dst_name = "converted_" + filename
        dst.save(dst_name)
        # dst.save(os.path.join(app.config["IMAGE_DOWNLOADS"], dst_name))

        print(dst_name)
        print('SHOULD RENDER TEMPLATE HERE')

        # download = "annotated.png"
        return render_template('annotated.html', download=dst_name)


if __name__ == '__main__':
    app.run()
