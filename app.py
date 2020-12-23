import pandas as pd 
import numpy as np 
import pickle
from flask import Flask, request, render_template
import xgboost as xgb


app = Flask(__name__)

# load model from file
regressor = pickle.load(open("regressor.pickle.dat", "rb"))


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/data', methods=["GET","POST"])
def data():
    if request.method == "POST":
        # read the csv file 
        file = request.form['upload-file']
        data_file = pd.read_csv(file)
        pred_data = xgb.DMatrix(data = data_file)
        predictions = regressor.predict(pred_data)
        return render_template('data.html', data = str(list(predictions)))    

            # # data_file = pd.read_csv(request.files.get("csvfile"))
            # # # convert input into appropriate format acceptable by xgboost regressor model 
            # pred_data = xgb.DMatrix(data = data_file)
            # # # do the predictions
            # # predictions = regressor.predict(pred_data)
            # return render_template('data.html', data = str(list(data)))


if __name__=="__main__":
    app.run(debug = True)