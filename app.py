from flask import Flask, request, render_template
import os
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

model = load_model("trained.h5")  

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (300, 300))
    img = img / 255.0
    img = img.reshape(1, 300, 300, 3)
    prediction_prob = model.predict(img)
    if prediction_prob >= 0.5:
        return "Pneumonia"
    else:
        return "Normal"

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("home.html", message="No file part")
        file = request.files["file"]
       
        if file.filename == "":
            return render_template("home.html", message="No selected file")
        if file: 
            file_path = os.path.join("static", file.filename)
            file.save(file_path)
            prediction = predict_image(file_path)
            return render_template("result.html", prediction=prediction, image_path=file_path)
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
