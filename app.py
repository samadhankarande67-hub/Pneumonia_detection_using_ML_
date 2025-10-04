import os
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set the folder for uploaded files
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the pre-trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.h5")
model = load_model(MODEL_PATH)

@app.route("/", methods=["GET"])
def index():
    """Render the home page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload():
    """Handle image upload and prediction."""
    try:
        # Check if an image was uploaded
        if "image" not in request.files:
            return render_template("display.html", prediction="No file uploaded. Please upload an image.")

        # Save the uploaded file
        f = request.files["image"]
        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(file_path)

        # Preprocess the image for prediction
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Make the prediction
        predictions = model.predict(x)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Define class labels
        class_labels = ["No Pneumonia", "Pneumonia"]
        prediction = f"Prediction: {class_labels[predicted_class]}"

        # Render the result page
        return render_template(
            "display.html", prediction=prediction, uploaded_file=f"/{file_path}"
        )
    except Exception as e:
        return render_template("display.html", prediction=f"Error: {e}")

@app.route("/about")
def about():
    """Render the about page."""
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
