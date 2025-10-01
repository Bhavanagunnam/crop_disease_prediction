# crop_disease_prediction
Crop Disease Detection Web Application
Overview
This project is a web application that detects diseases in crop leaves from images using a Convolutional Neural Network (CNN). It provides accurate disease classification, along with detailed descriptions, symptoms, remedies, and prevention tips for each disease. The project includes a user authentication system and persistent prediction history.

Features
Upload crop leaf images and get disease predictions.

Detailed disease information including symptoms and remedies.

User authentication with signup/login/logout.

Prediction history with image preview and disease details.

Confidence threshold to filter uncertain predictions.

Responsive UI built with Flask and Bootstrap.

Technologies
Python 3.x

Flask (Web framework)

TensorFlow / Keras (CNN model training and inference)

SQLAlchemy & SQLite (Database)

PIL (Image processing)

Bootstrap (Frontend styling)

Matplotlib (Training visualization)

Project Structure
text
/project-folder
  ├── app.py                  # Flask application code
  ├── crop_disease_model.h5   # Trained CNN model file
  ├── dataset/                # Dataset for training CNN
      ├── train/
      └── validation/
  ├── templates/              # HTML templates
  ├── static/                 # Static assets like CSS, images
  ├── README.md               # This documentation file
  ├── requirements.txt        # Python dependencies
  ├── users.db                # SQLite database file (runtime)
Setup Instructions
Clone the repo:

bash
git clone <your-repo-url>
cd project-folder
Install dependencies:

bash
pip install -r requirements.txt
Train the CNN model (optional, if you want to retrain):

bash
python train_model.py  # Script including the CNN training code
Run the Flask app:

bash
python app.py
Open http://localhost:5000 in your browser to use the app.

CNN Model
Custom CNN architecture with 3 convolutional layers and 1 dense layer.

Uses data augmentation (rotation, zoom, flips) during training.

Achieves high accuracy on crop disease classification.

Disease Information
Each disease class includes:

Detailed textual description

Visible symptoms

Suggested remedies

Prevention tips

External resource links for more details

This enriches user understanding and aids practical crop management.

Contribution
Feel free to fork, modify, and contribute enhancements, such as:

Additional diseases and crop types,

Improved user interface,

Multi-language support,

Integration with weather APIs,

Mobile app version.





