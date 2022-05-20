# -*- coding: utf-8 -*-
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import joblib
import plotly.graph_objects as go
import pandas as pd
# import shap
import plotly.express as px
import numpy as np


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df = pd.read_pickle("test_df.gz")
df.drop(columns=["index"], inplace=True)
df.set_index("SK_ID_CURR", inplace=True)
feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

#Partie eventuellement à sortir dans l'API à part
model = joblib.load("pipeline_housing.joblib")

explainer = joblib.load("shap_explainer.joblib")

def make_prediction():
    return 

# A modifier
score_min = 0.2*100

app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                feats,
                'EXT_SOURCE_3',
                id='crossfilter-feature',
            )
        ],
        style={'width': '66%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Input(id="client_id", type="number", value=df.index.values[0]),
            html.Button(id="validation_bt", n_clicks=0, children="Valider")
        ],
        style={'width': '33%', "float" : "right", 'display': 'inline-block'})
    ]),
    html.Div(dcc.Graph(id='bar_mean', figure = {"layout" : {"height" : 800}}), 
        style={'width': '33%', 'display': 'inline-block', 'padding': '0 20', "vertical_align" : "middle"}),
    html.Div(dcc.Graph(id='boxplot', figure = {"layout" : {"height" : 800}}), 
        style={'width': '33%', 'display': 'inline-block', 'padding': '0 20', "vertical_align" : "middle"}),
    html.Div([
        dcc.Graph(id='score'),
        dcc.Graph(id='feature_imp')
        ], style={'display': 'inline-block', 'width': '33%', "float":"right"})
])
        

@app.callback(Output('score', 'figure'),
              Input('validation_bt', 'n_clicks'),
              State('client_id', 'value'))
def update_score(n_clicks, client_id):
    dff = df.loc[client_id]

    dff_feats = dff[feats].to_numpy().reshape(1,-1)
        
    val = model.predict_proba(dff_feats)[0, 1] * 100
    if val > score_min:
        accept = "Accepté"
        color = "darkgreen"
    else:
        accept = "Refusé"
        color = "darkred"
    fig1 = go.Figure()
    
    fig1.add_trace(go.Indicator(
        domain = {"x" : [0,1], "y" : [0,1]},
        
        title = {"text" : "Score", "font_size" : 40},
        value = val,
        number = {"font_size" : 50},
        
        mode = "gauge + number",
        
        gauge = {
            "shape" : "angular",
            "steps" : [
                {"range" : [0, score_min], "color" : "red"},
                {"range" : [score_min, 100], "color" : "green"}
                ],
            "bar" : {"color" : "black", "thickness" : 0.5},
            "axis" : {"range" : [None, 100]}
            }
        )
    )
    
    fig1.add_annotation(x=0.5, y=0.4, text=accept, font = dict(size = 30, color=color), showarrow = False)
    
    return fig1
      

@app.callback(Output('feature_imp', 'figure'),
              Input('validation_bt', 'n_clicks'),
              State('client_id', 'value'))
def update_fi(n_clicks, client_id):
    dff = df.loc[client_id]
    dff_feats = dff[feats].to_numpy().reshape(1,-1)
    
    shap_vals = explainer.shap_values(dff_feats)
    shap_vals = shap_vals[1][0][:] #1 pour le résultat positif au crédit, 0 parce que voilà, et toutes les features
    
    df_feats = pd.DataFrame(shap_vals, columns=["importances"])
    df_feats["feats"] = feats
    df_feats["abs"] = abs(df_feats["importances"])
    df_feats["Influence"]= np.where(df_feats["importances"]<0, "Negative", "Positive")
    df_feats.sort_values(by="abs", ascending=False, inplace=True)
    df_feats.drop(columns=["abs"], inplace=True)
    
    fig2 = px.bar(df_feats.iloc[:10],
           x= "importances",
           y = "feats", 
           color = "Influence",
           orientation="h",
           title = "Principales données influant sur le résultat")
    fig2.update_xaxes(title="Impact sur le résultat")
    fig2.update_yaxes(title="Variable étudiée")
    
    return fig2

@app.callback(Output('crossfilter-feature', 'value'),
              Input('feature_imp', 'clickData'))
def change_feat(clickdata):
    if clickdata == None:
        return "EXT_SOURCE_3"
    else:
        return clickdata["points"][0]["y"]
        
@app.callback(Output('bar_mean', 'figure'),
              Input('validation_bt', 'n_clicks'),
              Input("crossfilter-feature", "value"),
              State('client_id', 'value'))
def plot_bar(n_clicks, feature, client_id):
    dff = df[feature]
    
    fig3 = px.bar(
           x = ["client", "moyenne"],
           y = [dff.loc[client_id], np.mean(dff)],
           color = [dff.loc[client_id], np.mean(dff)],
           title = "Comparaison du client à la moyenne")
    fig3.update_xaxes(title="")
    fig3.update_yaxes(title="Valeur")
    
    return fig3
    
@app.callback(Output('boxplot', 'figure'),
              Input("crossfilter-feature", "value"))
def plot_box(feature):
    dff = df[feature]
    
    fig4 = px.box(dff, title = "Répartition de la variable dans la clientèle")
    fig4.update_xaxes(title="")
    fig4.update_yaxes(title="Valeur")
        
    return fig4

if __name__ == '__main__':
    app.run_server(debug=True)