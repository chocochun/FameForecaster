from flask import Flask, request, render_template, send_from_directory
import json
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from flaskexample import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request
from flaskexample.a_Model import ModelIt
from flaskexample.opencv_image import Img_feature
import pickle
import os

__author__ = 'Minchun'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Python code to connect to Postgres
# You may need to modify this based on your OS, 
# as detailed in the postgres dev setup materials.
#username = 'postgres'
#password = 'choco1234'     # change this
#host     = 'localhost'
#port     = '5432'            # default port that postgres listens on
#dbname  = 'birth_db'
#db = create_engine( 'postgresql://{}:{}@{}:{}/{}'.format(username, password, host, port, dbname) )
#con = None
#con = psycopg2.connect(database = dbname, user = username, password = password)

#plotly.tools.set_credentials_file(username='chocochun', api_key='VMvHUkFtulJ2hIt95b9R')
#cleandata = pickle.load(open("/Users/minchunzhou/Desktop/insight/cleandata.pickle", "rb"))
#rf = pickle.load(open("/Users/minchunzhou/Desktop/insight/model.pickle", "rb"))

mytestexample = pickle.load(open("/Users/minchunzhou/Desktop/insight/sample_RF_data.pickle", "rb"))
rf = pickle.load(open("/Users/minchunzhou/Desktop/insight/RF_Alldata.pickle", "rb"))

@app.route('/')
def cesareans_input():
    return render_template("index.html")

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/upload", methods=["POST"])
def upload():

    form = request.form
    
    print list(form.items())
    
    x = str(list(form.items())[0][1])
    
    mytestexample.width_inch = float(list(form.items())[0][1])
    mytestexample.height_inch = float(list(form.items())[3][1])
    mytestexample[list(form.items())[1][1]] = 1
    mytestexample.is_alive_auction = (str(list(form.items())[2][1]) == "yes")
    mytestexample['aspect_ratio'] = mytestexample.width_inch / mytestexample.height_inch
    mytestexample['area_in_inch'] = mytestexample.width_inch * mytestexample.height_inch
    
    print form.items()
    print mytestexample.width_inch
    print mytestexample.height_inch
    print mytestexample[list(form.items())[1][1]]
    print mytestexample.is_alive_auction
    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
    
    testimg = Img_feature(destination)    
    testimg.get_all_feature()   
    
    for i in testimg.result.columns:
        mytestexample[i] = testimg.result[i]
        
    for j in mytestexample.columns: 
        print mytestexample[j]
        
    predictions = rf.predict(mytestexample)

    pred_proba = rf.predict_proba(mytestexample)
    predict_ba  = pred_proba[0][1] * 100
    
    
    #print pred_proba
    
    return render_template('/output1_upload.html',
                           image_name=filename,
                           height = x,
                           proba = predict_ba,
                           domaincolor = testimg.DominantColor)
                           

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)
