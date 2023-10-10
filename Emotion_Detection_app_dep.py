import os
import cv2
from flask import Flask, request, redirect, render_template, flash
from flask import send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers
import tensorflow.keras as keras
from keras.callbacks import EarlyStopping

import numpy as np


classes = ["怒り","うんざり","恐怖","幸せ","素","悲しい","驚き"]
image_size = 224

UPLOAD_FOLDER = "./static/uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('DenseNet_nonSwin_batch32_size224_layer0_epoch50_color.hdf5')#学習済みモデルをロード

preprocess_fun = keras.applications.densenet.preprocess_input

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            #受け取った画像を読み込み、np形式に変換
            img = cv2.imread(filepath)
            b,g,r = cv2.split(img) 
            img = cv2.merge([r,g,b])
            img = cv2.resize(img, (image_size,image_size))
            img = keras.applications.densenet.preprocess_input(img)
            #変換したデータをモデルに渡して予測する
            result = model.predict(np.array([img])/255.)[0]
            predicted = np.argmax(result)
            pred_answer = classes[predicted] + " に分類されました"

            return render_template("index.html",answer=pred_answer,original_name=filename)#,render_template('index.html', filebinary=filepath.encode())

    return render_template("index.html",answer="")

#@app.route('/uploads/<filename>')
# ファイルを表示する
#def uploaded_file(filename):
    #return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)