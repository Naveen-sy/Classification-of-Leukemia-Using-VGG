from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
import numpy as np
import cv2
import pickle
import imutils


from flask import Flask, render_template, request, send_file, Response
from PIL import Image, ImageOps
import io

from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import os


app = Flask(__name__)

# Load Model:-
model1 = load_model("C:/Users/sivas/OneDrive/Desktop/Website Final1/TRAINING_EXPERIENCE.h5")
mlb = pickle.loads(open("C:/Users/sivas/OneDrive/Desktop/Website Final1/MLB.PICKLE", "rb").read())



def predict_label(img_path):
    image = cv2.imread(img_path)
    #image = cv2.equalizeHist(image)
    output = imutils.resize(image,width=400)
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    proba = model1.predict(image)[0]
    print(proba)
    idxs = np.argsort(proba)[::-1][:1]
    for (i, j) in enumerate(idxs):
            label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
            (mlb.classes_[j])
            return label


@app.route('/')
def home():
    return render_template('home.html')


# routes
""" @app.route("/process", methods=['GET', 'POST'])
def home_page():
	return render_template("detect.html") """


@app.route('/upload')
def upload():
    return render_template('enhance.html')

@app.route('/enhance', methods=['POST'])
def enhance():
    # Get uploaded image data
    file = request.files['image']
    img_bytes = file.read()

    # Open image using PIL
    img = Image.open(io.BytesIO(img_bytes))

    # Enhance image using histogram equalization
    img = ImageOps.equalize(img)
    img.save('C:/Users/sivas/OneDrive/Desktop/Website Final1/static/enhanced_image.jpg')

    # Convert image to bytes buffer
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    # Return enhanced image as HTTP response
    return send_file(img_bytes, mimetype='image/jpeg', as_attachment=False)




@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
     # Connect to the database
    conn = sqlite3.connect('C:/Users/sivas/OneDrive/Desktop/Website Final1/skss.db')
    cursor = conn.cursor()
    
    # Execute the SQL query to get the last entered details details
    cursor.execute("SELECT name, age, dob, gender, bloodGroup, phone, address FROM details ORDER BY id DESC LIMIT 1")
    details = cursor.fetchone()
    
    # Close the database connection
    cursor.close()
    conn.close()          
    
    if request.method == 'POST':
        img = request.files['image']
        img_path = "C:/Users/sivas/OneDrive/Desktop/Website Final1/static/enhanced_image.jpg"
        p = predict_label(img_path)
                
    return render_template("enhance1.html", prediction=p, img_path=img_path, details=details)

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        age = request.form['age']
        dob = request.form['dob']
        gender = request.form['gender']
        bloodGroup = request.form['bloodGroup']
        phone = request.form['phone']
        address = request.form['address']
        # Get image file
        """ image = request.files['image']
        # Save image file to static folder with custom filename
        filename = f"{name}_{image.filename}"
        image.save(os.path.join(app.static_folder, 'images', filename)) """
        # Save data to database
        conn = sqlite3.connect('C:/Users/sivas/OneDrive/Desktop/Website Final1/skss.db')
        
        c = conn.cursor()
        c.execute("INSERT INTO details (name, age, dob, gender, bloodGroup, phone, address) VALUES (?, ?, ?, ?, ?, ?, ?)", (name, age, dob, gender, bloodGroup, phone, address))
        conn.commit()
        conn.close()

        # Redirect to thank you page
        return redirect(url_for('upload'))



if __name__ == '__main__':
    app.run(debug=True)


""" if __name__ =='__main__':
	
	app.run()
 """