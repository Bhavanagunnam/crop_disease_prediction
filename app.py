from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
import os
import base64
import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key'

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    uploads = db.relationship('PredictionHistory', backref='user', lazy=True)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_data = db.Column(db.Text, nullable=False)
    disease_class = db.Column(db.String(255), nullable=False)
    confidence_percent = db.Column(db.Float, nullable=False)
    pesticide_recommendation = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

model = load_model('crop_disease_model.h5')

CLASS_NAMES = [
    'Pepper_bell__Bacterial_spot',
    'Pepper_bell__healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Sepitoria_leaf_spot',
    'Tomato_Spider_mites_2spots',
    'Tomato_Target_Spot',
    'Tomato_Tomato_YellowLeafCurl_Virus',
    'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy'
]

PESTICIDES = {
    'Pepper_bell__Bacterial_spot': 'Use copper-based bactericides and crop rotation.',
    'Pepper_bell__healthy': 'No disease detected. No treatment required.',
    'Potato___Early_blight': 'Use chlorothalonil or mancozeb fungicides.',
    'Potato___Late_blight': 'Apply preventive fungicides and resistant varieties.',
    'Potato___healthy': 'No disease detected. No treatment required.',
    'Tomato_Bacterial_spot': 'Use copper-based bactericides, remove infected leaves.',
    'Tomato_Early_blight': 'Spray chlorothalonil or mancozeb regularly.',
    'Tomato_Late_blight': 'Use appropriate fungicides like cyazofamid.',
    'Tomato_Leaf_Mold': 'Use fungicide sprays containing chlorothalonil.',
    'Tomato_Sepitoria_leaf_spot': 'Apply fungicides with pyraclostrobin.',
    'Tomato_Spider_mites_2spots': 'Apply miticides and insecticidal soaps.',
    'Tomato_Target_Spot': 'Use preventive fungicides promptly.',
    'Tomato_Tomato_YellowLeafCurl_Virus': 'Control whitefly vectors and remove infected plants.',
    'Tomato_Tomato_mosaic_virus': 'Use certified disease-free seeds, control aphids.',
    'Tomato_healthy': 'No disease detected. No treatment required.'
}

DISEASE_DETAILS = {
    'Pepper_bell__Bacterial_spot': {
        'description': 'Bacterial spot causes small, water-soaked lesions on leaves, stems, and fruit that turn dark and scabby.',
        'symptoms': 'Dark angular spots on leaves, leaf drop, and fruit blemishes.',
        'remedies': 'Use copper-based bactericides and remove infected plant debris.',
        'prevention': 'Rotate crops, avoid overhead watering, and use certified disease-free seeds.',
        'resources': 'https://ipm.cahnr.uconn.edu/bacterial-leaf-spot-in-peppers/'
    },
    'Pepper_bell__healthy': {
        'description': 'The plant appears healthy with no signs of disease.',
        'symptoms': 'No visible symptoms.',
        'remedies': 'No treatment required.',
        'prevention': 'Maintain good agricultural practices.',
        'resources': ''
    },
    'Potato___Early_blight': {
        'description': 'Early blight causes dark brown spots on leaves surrounded by concentric rings.',
        'symptoms': 'Leaf spots enlarge, causing leaf dieback and reduced yield.',
        'remedies': 'Apply fungicides like chlorothalonil or mancozeb.',
        'prevention': 'Practice crop rotation and remove infected plant material.',
        'resources': 'https://vegpath.plantpath.wisc.edu/diseases/potato-early-blight/'
    },
    'Potato___Late_blight': {
        'description': 'Late blight is a serious disease causing dark lesions on leaves and stems, leading to rapid decay.',
        'symptoms': 'Water-soaked spots on leaves, stem lesions, rotting tubers.',
        'remedies': 'Apply preventive fungicides such as copper compounds.',
        'prevention': 'Use resistant varieties and avoid excessive irrigation.',
        'resources': 'https://ipm.cahnr.uconn.edu/early-blight-and-late-blight-of-potato/'
    },
    'Potato___healthy': {
        'description': 'The potato plant is healthy with no disease symptoms.',
        'symptoms': 'No visible symptoms.',
        'remedies': 'No treatment required.',
        'prevention': 'Maintain crop health through best practices.',
        'resources': ''
    },
    'Tomato_Bacterial_spot': {
        'description': 'Bacterial spot on tomato causes brown spots on leaves and fruit.',
        'symptoms': 'Leaf spots, fruit lesions, defoliation.',
        'remedies': 'Use copper-based bactericides and remove infected leaves.',
        'prevention': 'Avoid overhead irrigation and practice crop rotation.',
        'resources': 'https://extension.umn.edu/disease-management/bacterial-spot-tomato-and-pepper'
    },
    'Tomato_Early_blight': {
        'description': 'Early blight causes concentric rings on leaf spots.',
        'symptoms': 'Yellowing leaves with brown spots.',
        'remedies': 'Fungicide sprays like mancozeb or chlorothalonil.',
        'prevention': 'Crop rotation and removal of infected debris.',
        'resources': 'https://extension.umn.edu/disease-management/early-blight-tomato-and-potato'
    },
    'Tomato_Late_blight': {
        'description': 'Late blight causes dark lesions on leaves and stems.',
        'symptoms': 'Rapid leaf damage and stem cankers.',
        'remedies': 'Use fungicides like cyazofamid.',
        'prevention': 'Use resistant varieties and avoid wet conditions.',
        'resources': 'https://ipm.cahnr.uconn.edu/early-blight-and-late-blight-of-potato/'
    },
    'Tomato_Leaf_Mold': {
        'description': 'Leaf mold causes yellow spots on upper leaf surfaces and mold on undersides.',
        'symptoms': 'Yellowing and curling of leaves.',
        'remedies': 'Fungicide sprays containing chlorothalonil.',
        'prevention': 'Ensure good air circulation and avoid overhead watering.',
        'resources': 'https://agritech.tnau.ac.in/crop_protection/tomato_diseases_6.html'
    },
    'Tomato_Sepitoria_leaf_spot': {
        'description': 'Septoria leaf spot causes small circular spots on leaves.',
        'symptoms': 'Leaf defoliation leading to reduced yield.',
        'remedies': 'Handle with fungicides containing pyraclostrobin.',
        'prevention': 'Remove infected leaves and crop debris.',
        'resources': 'https://extension.umn.edu/disease-management/bacterial-spot-tomato-and-pepper'
    },
    'Tomato_Spider_mites_2spots': {
        'description': 'Damage by spider mites causing two spots on affected areas.',
        'symptoms': 'Yellowing leaves with small spots.',
        'remedies': 'Use miticides and insecticidal soaps.',
        'prevention': 'Control weeds and maintain adequate irrigation.',
        'resources': 'https://ipm.ucanr.edu/agriculture/tomato/tomato-yellow-leaf-curl/'
    },
    'Tomato_Target_Spot': {
        'description': 'Target spot causes large brown lesions with concentric rings.',
        'symptoms': 'Large leaf spots that cause defoliation.',
        'remedies': 'Prompt use of preventive fungicides.',
        'prevention': 'Crop rotation and crop residue management.',
        'resources': 'https://extension.umn.edu/disease-management/bacterial-spot-tomato-and-pepper'
    },
    'Tomato_Tomato_YellowLeafCurl_Virus': {
        'description': 'A viral disease causing yellowing and curling of leaves.',
        'symptoms': 'Leaf curl, yellowing, stunted growth.',
        'remedies': 'Control whitefly vector and remove infected plants.',
        'prevention': 'Use resistant varieties and reflective mulches.',
        'resources': 'https://agriculture.vic.gov.au/biosecurity/plant-diseases/vegetable-diseases/tomato-yellow-leaf-curl-virus'
    },
    'Tomato_Tomato_mosaic_virus': {
        'description': 'Mosaic virus causes mottled leaves with light and dark green patches.',
        'symptoms': 'Leaf deformation and stunted plants.',
        'remedies': 'Use certified disease-free seeds and control aphids.',
        'prevention': 'Sanitize tools and remove infected plants.',
        'resources': 'https://ipm.ucanr.edu/agriculture/tomato/tomato-yellow-leaf-curl/#gsc.tab=0'
    },
    'Tomato_healthy': {
        'description': 'Healthy tomato plant with no disease symptoms.',
        'symptoms': 'No visible symptoms.',
        'remedies': 'No treatment required.',
        'prevention': 'Follow good agricultural practices.',
        'resources': ''
    },
}

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        new_user = User(username=username, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        flash('Account created, please login')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    PredictionHistory.query.filter_by(user_id=current_user.id).delete()
    db.session.commit()
    flash('Your prediction history has been cleared.')
    return redirect(url_for('index'))

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    error = None
    if request.method == "POST":
        try:
            files = request.files.getlist("files")
            if not files or files[0].filename == '':
                error = "Please upload at least one image file."
                user_history = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.timestamp.desc()).all()
                # Prepare details
                history_with_details = []
                for item in user_history:
                    detail = DISEASE_DETAILS.get(item.disease_class, {}) if item.disease_class != 'Unknown or not a leaf image' else {}
                    history_with_details.append({'item': item, 'details': detail})
                return render_template("index.html", error=error, history=history_with_details)

            for file in files:
                image = Image.open(file.stream).convert('RGB')
                img_array = preprocess_image(image)
                preds = model.predict(img_array)
                confidence = float(np.max(preds[0])) * 100
                predicted_class = CLASS_NAMES[np.argmax(preds[0])]

                if confidence < 70:
                    predicted_class = "Unknown or not a leaf image"
                    pesticide = "Unable to predict. Please upload a crop leaf image only."
                    details = {}
                else:
                    pesticide = PESTICIDES.get(predicted_class, "No pesticide recommendation available.")
                    details = DISEASE_DETAILS.get(predicted_class, {})

                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()

                new_history_entry = PredictionHistory(
                    user_id=current_user.id,
                    image_data=img_str,
                    disease_class=predicted_class,
                    confidence_percent=round(confidence, 2),
                    pesticide_recommendation=pesticide
                )
                db.session.add(new_history_entry)
                db.session.commit()

        except Exception as e:
            logging.exception("Error during prediction")
            error = f"An error occurred: {str(e)}"

    user_history = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.timestamp.desc()).all()
    history_with_details = []
    for item in user_history:
        detail = DISEASE_DETAILS.get(item.disease_class, {}) if item.disease_class != 'Unknown or not a leaf image' else {}
        history_with_details.append({'item': item, 'details': detail})

    return render_template("index.html", error=error, history=history_with_details)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
