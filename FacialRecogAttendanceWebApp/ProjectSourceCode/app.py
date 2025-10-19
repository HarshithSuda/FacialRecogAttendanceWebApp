import os
import logging

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from sqlalchemy.orm import DeclarativeBase

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

# Create the SQLAlchemy instance
db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///attendance.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Configure file upload settings
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["STUDENTS_FOLDER"] = os.path.join("static", "students")
app.config["PROCESSED_FOLDER"] = os.path.join("static", "processed")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload size

# Create upload directories if they don't exist
for folder in [app.config["UPLOAD_FOLDER"], app.config["STUDENTS_FOLDER"], app.config["PROCESSED_FOLDER"]]:
    os.makedirs(folder, exist_ok=True)

# Initialize the app with the db
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# Create all tables
with app.app_context():
    import models  # Import the models here to avoid circular imports
    db.create_all()

# Load user for login manager
@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(int(user_id))