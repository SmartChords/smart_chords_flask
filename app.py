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
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

            return render_template('preview.html', filename=filename)
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
    
        filename = request.form['preview-image']
        img = send_img(filename)
        img.direct_passthrough = False
        
        """
        f = request.files['file']
        img = f
        image = Image.open(img).convert('L')
        image = np.array(image)
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

        # build_model_input(notes)
        # chords = get_chord_predictions()

        # FROM HERE, chords WILL BE AN ARRAY OF NAMES THAT WE WILL USE TO QUERY THE LOOKUP TABLE
        # IT WILL LOOK SOMETHING LIKE THIS: ['c#-min' 'g#-min' 'D-Maj7' 'D-Maj7' 'c#-min' 'B-Maj7']
        # QUERY THE TABLE AND GET AN ARRAY OF THE CORRESPONDING DB ROWS
        """
    
        frames = 4
        chordPad = 30
        framePad = 25
            
        chordA = Image.open("img/A-Chord.png").convert('L')
        chordG = Image.open("img/G-Chord.png").convert('L')
        frame = Image.open("img/Frame-1.png").convert('L')        
        
        aImgWidth = 907
        aImgHeight = 900
        
        # Create the ne wimage with the chord image height, frame height and padding.
        aImg = Image.new('RGB', size = (aImgWidth, aImgHeight), color = (255, 255, 255))
        
        # Process each frame with the chord fingerings.
        Image.Image.paste(aImg, chordA, (120, 30))
        Image.Image.paste(aImg, chordG, (485, 30))
        Image.Image.paste(aImg, frame, (0, 200))

        Image.Image.paste(aImg, chordA, (120, 359))
        Image.Image.paste(aImg, chordG, (485, 359))
        Image.Image.paste(aImg, frame, (0, 529))              
        
        # Save the annotated image for display to the user.
        aImg.save("img/annotated.png");
        
        return "Hello World"
        
        #return render_template('annotated.html')

        """
        ######### THIS SECTION WRITES TO THE IMAGE #########
        #img = Image.open(img).convert('L')
        #size = (img.size[0], int(img.size[1] * 1.5))
        #layer = Image.new('RGB', size, (255, 255, 255))
        #layer.paste(img, box=None)
        #img_arr = np.array(layer)
        #height = int(img_arr.shape[0])
        #width = int(img_arr.shape[1])
        #draw = ImageDraw.Draw(layer)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        #font = ImageFont.truetype("Aaargh.ttf", 20)
        # draw.text((x, y),"Sample Text",(r,g,b))
        #j = width / 9
        #for i in notes: # for i in chord_images:
        #    ##########INSTEAD OF draw.text() HERE, WE WANT IT TO PASTE THE CHORD CHARTS #########
        #    draw.text((j, height - 40), i, (0, 0, 0), font=font)
        #    j += (width / (len(notes) + 4))
        #layer.save("static/img/download/annotated.png")
        #return render_template('annotated.html')
        """

if __name__ == '__main__':
    app.run()
