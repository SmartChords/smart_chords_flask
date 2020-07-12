from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField, validators
from wtforms.validators import DataRequired, InputRequired
from wtforms.fields.html5 import EmailField

class ContactForm(FlaskForm):
  name = StringField("Name",  [InputRequired("Please enter your name.")])
  email = EmailField('Email address', [validators.DataRequired(), validators.Email()])
  subject = StringField("Subject",  [InputRequired("Please enter a subject.")])
  message = TextAreaField("Message", [InputRequired("Please enter a message.")])
  submit = SubmitField("Send")
