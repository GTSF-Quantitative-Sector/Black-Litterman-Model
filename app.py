import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
import pandas_datareader as web
import yfinance as yf
from flask import Flask, request

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/frozen-caverns-62999', methods=['POST'])
def create_upside_downside():

    start = dt.datetime.strptime(request.json["start"], '%Y-%m-%d').date()
    end = dt.datetime.strptime(request.json["end"], '%Y-%m-%d').date()
    tckr_list = request.json["tckr_list"]
    per = request.json["per"]

    data = get_data(start, end, tckr_list, per)
    up, down = buckets(data)
    capture_calcs = capture(up, down)
    return capture_calcs.to_json()

def get_data(start, end, tckr_list, per):
    
    # Loop to grab the price data from yahoo finance and combine them into one pandas dataframe
    stock_data = pd.DataFrame()
    for tckr in tckr_list:
        data = yf.download(tckr, start, end)  # function to actually grab the data
        if per == 'D':
            data = data.pct_change().dropna()  # converts price data into yearly returns data
        elif per == 'M':
            data = data.iloc[((len(
                data.index) - 1) % 21)::21].pct_change().dropna()  # converts price data into Monthly returns data
        elif per == 'W':
            data = data.iloc[
                   ((len(data.index) - 1) % 5)::5].pct_change().dropna()  # converts price data into weekly returns data
        stock_data[tckr] = data['Close']  # appends to overall/output dataframe
    return stock_data

def buckets(data):
    avg = data.iloc[:, 0].mean() #grabs the average from the 1st column, which will always be the reference inputted by the user
    data['Direction'] = np.where(data.iloc[:, 0] >= avg, 1, 0) #look up np.where() for info on this function, np is numpy which is a common python library
    up_data = data.loc[data['Direction'] == 1].drop(['Direction'], axis = 1) #grabs up data
    down_data = data.loc[data['Direction'] == 0].drop(['Direction'], axis = 1) #grabs down data
    return up_data, down_data

def capture(up, down):
    cap = pd.DataFrame()

    for i in [up, down]:
        ref = i.iloc[:, 0] + 1
        lis = {}
        for j in i:
            stock = i[j] + 1
            lis[j] = (np.prod(stock)-1)/(np.prod(ref)-1) *100
        cap = cap.append(lis, ignore_index = True)
    cap.index = ['Up Capture', 'Down Capture']
    return cap

if __name__ == '__main__':
    app.run()
