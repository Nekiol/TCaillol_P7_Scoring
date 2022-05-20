import flask
from flask import request, jsonify
import joblib
import pandas as pd
import shap
import json
import numpy as np

app = flask.Flask(__name__)
app.config["DEBUG"] = False

#Chargement du tableau et du modèle
df = pd.read_pickle("test_df.gz")
df.drop(columns=["index"], inplace=True)
df.set_index("SK_ID_CURR", inplace=True)
feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

model = joblib.load("pipeline_housing.joblib")

explainer = joblib.load("shap_explainer.joblib")


def make_prediction(client_id):
    return model.predict_proba([df[feats].loc[client_id]])[0, 1]
    
def explain(client_id):
    return explainer.shap_values(df[feats].loc[client_id].to_numpy().reshape(1, -1))[1][0][:]


@app.route('/', methods=['GET'])
def index():
    return {'message': 'Hello, stranger'}


@app.route('/score_min/', methods=['GET'])
def score_min():
    return {"score_min" : 0.55} 


@app.route('/predict', methods=["GET"])
def proba():
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        pred = make_prediction(client_id)
        return {"proba" : pred}
    else:
        return "Error"

@app.route('/feats/', methods=["GET"])
def feats_ret():
    return json.dumps(feats) 

@app.route('/importances', methods=["GET"])
def importances():
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        shap_vals = explain(client_id).tolist()
        return json.dumps(shap_vals)
    else:
        return "Error"    

@app.route('/bar', methods=["GET"])
def bar():
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        feat = str(request.args["feature"])
        
        dff = df[feat]
        retour = []
        retour.append(float(dff.loc[client_id]))
        retour.append(np.mean(dff))
        del dff      
     
        return json.dumps(retour)
    else:
        return "Error"     
@app.route('/boxplot', methods=["GET"])
def boxplot():
    if 'feature' in request.args:
        feat = str(request.args["feature"])
        
        dff = df[feat]
     
        return json.dumps(dff.tolist())
    else:
        return "Error"   
 
app.run()