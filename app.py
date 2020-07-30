import os
from os import path
from flask import Flask, render_template, request, flash, redirect, send_from_directory, send_file, make_response, jsonify
from werkzeug.utils import secure_filename
from forms import ContactForm
from flask_mail import Message, Mail
from tensorflow.python.framework import ops
from tensorflow.python.training import saver as saver_lib
from notes import build_model_input, get_chord_predictions
from frames import partitionImage, resize, normalize, isMusicalImage, createWhiteImage, getStartofBlack#, label_frame_with_chords_images

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageChops
from PIL import ImageFont

mail = Mail()

app = Flask(__name__)

app.config.from_object('config.Config')

mail.init_app(app)

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

@app.route('/error')
def throw_error():
    flash("How embarassing. Either something went wrong on our end or your file was no bueno. Please try again.")
    return redirect('/')

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

@app.route('/annotated/<download>', methods=['GET'])
def annotated(download):
    return render_template('annotated.html', download=download)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
  form = ContactForm()

  if request.method == 'POST':
    if form.validate() == False:
      flash('All fields are required.')
      return render_template('contact.html', form=form)
    else:
      msg = Message(form.subject.data, sender='contact@smartchords.com', recipients=[form.email.data])
      msg.body = """
      From: %s &lt;%s&gt;
      %s
      """ % (form.name.data, form.email.data, form.message.data)
      mail.send(msg)

      return render_template('contact.html', success=True)

  elif request.method == 'GET':
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

def resize_chord(img):
    basewidth = 50
    #wpercent = (basewidth/float(img.size[0]))
    #hsize = int((float(aImg.size[1])*float(wpercent)))
    img = img.resize((basewidth, 70), Image.ANTIALIAS)
    return img
 
def add_chord_label(image_to_convert, chords_list):
    img = image_to_convert
    img = Image.open(img).convert('L')
    
    # Grow the image height to make room for the chord image.
    size = (img.size[0], int(img.size[1]+90))
    layer = Image.new('RGB', size, (255,255,255))
    draw = ImageDraw.Draw(layer)
    
    # Flag "drawImage": set this to True to draw chord images, to False to draw text names
    drawImage = True
    if (drawImage) :
        w, h = layer.size
        
        # Kludge to get rid of some padding - TODO clean this up.
        w -= 240
        j = 120
        for i in chords_list:
            # Paste each chord image to the layer.        
            # dst.save(os.path.join(app.config["IMAGE_DOWNLOADS"], dst_name))
            
            # If chord file exists, then draw it
            if ( os.path.exists("./static/img/chords/" + str(i) + ".png") ) :
                chord = Image.open("./static/img/chords/" + str(i) + ".png")
                chord = resize_chord(chord)
                chord.save("./static/img/chords/" + str(i) + ".png")
                layer.paste(chord, ( j, 10 ))
                j += int( w / len(chords_list) )
                chord.close()
            
            # Else draw the text of the chord name
            else :            
                font = ImageFont.truetype("Aaargh.ttf", 20)
                draw.text((j, 30), i, (0,0,0), font=font)
                j += int( w / len(chords_list) )
            
        # Paste the frame layer to the bottom of the image.   
        layer.paste(img, (0, 90))
    
    else :
        img_arr = np.array(layer)
        height = int(img_arr.shape[0])
        width = int(img_arr.shape[1])
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
        file_name_full_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)

        image1 = Image.open(file_name_full_path).convert('L')

        #we assume the color at pixel 0,0 is the color of the background
        color = image1.getpixel((0,0))
        #horiz_images = trim(image1) # this splits the image horizontally, based on white lines
        horiz_images, white_images = partitionImage(image1, color) # this splits the image horizontally, based on white lines
        #// for practice, we save extracted images in the same directory as app.py
        # NOTE ALSO< THAT AS THE CODE STANDS NOW, THE INPUT IMAGE MUST ALSO BE IN THE
        #APP.PY directory. THE SPLITTING IS DONE AFTER YOU SELECT AND SUBMIT an IMAGE
        converted_array = []
        chords_dict = {}
        index = 0
        for i in horiz_images:
            fname = str(index) +".png"
            i.save(fname)
            left, upper = getStartofBlack(i, color)
            print ("black coordinates  = " + str(left) + ", " + str(upper))

            # for i in range(index - 1):
            frame_image = Image.open(fname).convert('L')
            #isMusic = isMusicalImage(frame_image)
            w, h = frame_image.size
            if w < 500 or h < 65:
                converted_array.append(frame_image)
                chords_dict[index] = []
                index = index + 1
                continue

            notes = get_notes_from_frame(frame_image)
            if len(notes) == 0:
                converted_array.append(frame_image)
                chords_dict[index] = []
                index = index + 1
                continue
            else:
                build_model_input(notes, index)
                c = get_chord_predictions(index)
                chords_dict[index] = c

            converted = add_chord_label(fname, chords_dict[index])
            converted.save(str(index) +"_converted.png")
            converted_array.append(converted)
            index = index + 1

        converted_height = 0
        for c in converted_array:
            converted_height = converted_height + c.height
            
        for w in white_images:
            converted_height = converted_height + w.height
            

        combined_width = converted_array[0].width
        dst = Image.new('L', (combined_width, converted_height))
        
        #this loop stictches back the parts, and saves the result
        h_index = 0
        white_index = 0
        for c in converted_array:
            dst.paste(white_images[white_index], (0, h_index))
            h_index = h_index + white_images[white_index].height
            white_index = white_index + 1
            dst.paste(c, (0, h_index))
            h_index = h_index + c.height

        dst_name = "converted_" + filename
        dst.save(os.path.join(app.config["IMAGE_DOWNLOADS"], dst_name))
        
        # Flag "removeFiles" set this to True to cleanup after transcribing
        removeFiles = True
        if (removeFiles) :
            for n in range(index):
                os.remove( str(n) + ".png" )
                if ( os.path.exists(str(n) + "_converted.png") ) :
                    os.remove( str(n) + "_converted.png" )
                else :
                    None

                if ( os.path.exists("chord_data/" + str(n) + ".csv") ):
                    os.remove("chord_data/" + str(n) + ".csv")
                else:
                    None

        return render_template('annotated.html', download=dst_name)


if __name__ == '__main__':
    app.run()
