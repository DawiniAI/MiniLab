from flask import Flask,jsonify,render_template,request
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import random
import os
from werkzeug.utils import secure_filename
from PIL import Image
from keras.utils import img_to_array




## Breast Cancer Medicial | · Breast Histopathology Images
breast_json_file = open('models/breast/breast-model.json', 'r')
loaded_breast_model_json = breast_json_file.read()
breast_json_file.close()
breast_Model = model_from_json(loaded_breast_model_json)
breast_Model.load_weights("models/breast/breast-model.h5")
breast_Model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

 

 
## Bone  | Bone Break Classifier Using CNN
bone_json_file = open('models/bone/bone-model.json', 'r')
loaded_bone_model_json = bone_json_file.read()
bone_json_file.close()
bone_Model = model_from_json(loaded_bone_model_json)
bone_Model.load_weights("models/bone/bone-model.h5")
bone_Model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['acc'])


# Preparing and pre-processing the image
def preprocess_img(img_path,width,height):
    op_img = Image.open(img_path)
    img_resize = op_img.resize((width , height))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, width, height, 3)
    return img_reshape
 

class Api:

    def __init__(self,model):
        self.model = model
    def predict_breast(self,image) :
        prediction = self.model.predict(preprocess_img(image,25,25))
        return {
            'cancer_percentage':round(prediction[0][1]*100,2),
            'not_cancer_percentage':round(prediction[0][0]*100,2)
               } 
    def predict_bone(self,image):
        prediction = self.model.predict(preprocess_img(image,200,200))
        return {
            'avulsion_fracture':round(prediction[0][0]*100,2),
            'comminuted_fracture':round(prediction[0][1]*100,2),
            'compression_crush_fracture':round(prediction[0][2]*100,2),
            'fracture_dislocation':round(prediction[0][3]*100,2),
            'greenstick_fracture':round(prediction[0][4]*100,2),
            'hairline_fracture':round(prediction[0][5]*100,2),
            'impacted_dislocation':round(prediction[0][6]*100,2),
            'intra-articluar_fracture':round(prediction[0][7]*100,2),
            'longitudinal_fracture':round(prediction[0][8]*100,2),
            'oblique_dislocation':round(prediction[0][9]*100,2),
            'pathological_fracture':round(prediction[0][10]*100,2),
            'spiral_fracture':round(prediction[0][11]*100,2)
        }



### Creating our API & connected with HTML files ###
app = Flask(__name__)
@app.route("/")
def home():
    return {'response':200}

@app.route("/breast/predict",methods=['GET', 'POST'])
def breast_predict():
   if request.method == "POST":

        if request.files:
            image = request.files["img"].stream
            api = Api(breast_Model)
            return api.predict_breast(image)



@app.route("/bone/predict",methods=['GET', 'POST'])
def bone_predict():
   if request.method == "POST":
        if request.files:
            image = request.files["img"].stream
            api = Api(bone_Model)
            return api.predict_bone(image)


app.run(debug=True)

