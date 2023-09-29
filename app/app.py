from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from joblib import load
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def hello():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html')
    else:
        text = request.form['text']
        path = "predictions_pic.svg"
        model = load('model.joblib')
        np_arr = floats_string_to_np_arr(text)
        make_picture('df.pkl', model, np_arr, path)
       
        return render_template('index.html', path)

def make_picture(training_data_filename, model, new_inp_np_arr, output_file):
    data = pd.read_pickle(training_data_filename)
    data = data.drop('Unnamed: 0', axis = 1)
    years = data['YearsExperience']
    salary = data['Salary']
    x_new = np.array(list(range(12))).reshape(12,1)
    preds = model.predict(x_new)

    fig = px.scatter(x = years,
                     y = salary,
                     title = 'Years of Experience vs Salary',
                     labels = {'x': 'Years of Experience',
                               'y':'Salary ($USD)'})
    
    fig.add_trace(go.Scatter(x = x_new.reshape(12),
                             y = preds,
                             mode = 'lines',
                             name = 'Model'))
    
    new_preds = model.predict(new_inp_np_arr)
    
    
    fig.add_trace(go.Scatter(x = new_inp_np_arr.reshape(len(new_inp_np_arr)),
                             y = new_preds,
                             name = 'New Outputs',
                             mode = 'markers',
                             marker = dict(color = 'purple',
                                           size = 20,
                                           line = dict(color = 'purple', width = 2))))
    fig.write_image(output_file, width = 800, engine='kaleido')

    fig.show()

def floats_string_to_np_arr(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False
    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats), 1)