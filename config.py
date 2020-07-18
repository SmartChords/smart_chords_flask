import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SECRET_KEY = 'todo-change-this'
    IMAGE_UPLOADS = './static/img/uploads'
    IMAGE_DOWNLOADS = './static/img/downloads'
    ALLOWED_IMAGE_EXTENSIONS = ["JPEG", "JPG", "PNG"]
    MAIL_SERVER = "smtp.gmail.com"
    MAIL_PORT = 465
    MAIL_USE_SSL = True
    MAIL_USERNAME = 'contact@example.com'
    MAIL_PASSWORD = 'your-password'


class ProductionConfig(Config):
    DEBUG = False


class StagingConfig(Config):
    DEVELOPMENT = True
    DEBUG = True



class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
