import os
from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename
from forms import ContactForm
from flask_mail import Message, Mail

# mail = Mail()

# IMAGE_UPLOADS = "/static/img/uploads"
# ALLOWED_IMAGE_EXTENSIONS = ["JPEG", "JPG", "PNG"]

app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])
app.secret_key = 'development key'

# app.config["MAIL_SERVER"] = "smtp.gmail.com"
# app.config["MAIL_PORT"] = 465
# app.config["MAIL_USE_SSL"] = True
# app.config["MAIL_USERNAME"] = 'contact@example.com'
# app.config["MAIL_PASSWORD"] = 'your-password'
#
# mail.init_app(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print(request.files)
        if 'fileToUpload' not in request.files:
            flash("No file to upload")
            return redirect(request.url)

        image = request.files['fileToUpload']
        if image.filename == "":
            flash("No file to upload")
            return redirect(request.url)

        if image and allowed_image(image.filename):
            filename = secure_filename(image.filename)
            print(os.path.join(app.config["IMAGE_UPLOADS"], filename))
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

            return render_template('preview.html', filename=filename)
        else:
            flash("Incorrect file type")
            return redirect(request.url)

        # return render_template("preview.html", filename=filename)

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


app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='img/uploads/' + filename), code=301)

@app.route('/preview', methods=['GET', 'POST'])
def preview():
    return render_template("preview.html")

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    form = ContactForm()
    if form.validate_on_submit():
        # TODO: SET UP EMAILING FUNCTIONALITY
        # return render_template("index.html")
        # msg = Message(form.subject.data, sender='contact@example.com', recipients=['your_email@example.com'])
        # msg.body = """
        # From: %s &lt;%s&gt;
        # %s
        # """ % (form.name.data, form.email.data, form.message.data)
        # mail.send(msg)
        return render_template('contact.html', success=True)


    return render_template('contact.html', form=form)

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/load.html')
def load():
    return render_template('load.html')

if __name__ == '__main__':
    app.run()
