import pandas as pd
import numpy as np
import pandas_datareader as web
import yfinance as yf
from flask import Flask, request, jsonify
from datetime import datetime as dt

app = Flask(__name__)
app.config["DEBUG"] = True

DEFAULT_TICKERS = ['BIL', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

@app.route('/api/v1/blmodel', methods=['POST'])
def create_black_litterman():

    p = request.json['p']
    q = request.json['q']
    weights = request.json['weights']

    new_q = []
    for el in q:
        if el != ['']:
            new_q.append(el)

    new_p = []
    for el in p:
        if el != ['', '', '', '', '', '', '', '', '', '']:
            new_p.append(el)

    new_weights = []
    for el in weights:
        if el != ['']:
            new_weights.append(el)

    print(new_p)
    print(new_q)
    print(new_weights)

    new_weights = np.transpose(np.array(new_weights))
    new_p = np.array(new_p)

    start = '2018-01-01'
    end = dt.now()
    returns = get_data(start, end, DEFAULT_TICKERS)
    weight_output, _ = bl(returns, new_weights, q=new_q, p=new_p)

    return jsonify(weight_output)

def bl(returns, weights, aversion=3, tau=.1, relative_confidence=1, q=None, p=None):

    if q is None or p is None:
        return list(weights.T[0]), None

    tau_omega = tau/relative_confidence # calculated or given
    
    #calculations
    cov = np.array(returns.cov())
    prior_returns = aversion * np.matmul(cov,weights)
    cov_tau = tau * cov
    mkt_var = float(np.matmul(weights.transpose(), np.matmul(cov, weights)))
    std = float(np.sqrt(np.matmul(np.matmul(weights.transpose(), cov), weights)))
    mkt_excess_return = aversion * mkt_var
    mkt_sharpe = mkt_excess_return/std
    omega = tau_omega * np.matmul(p, np.matmul(cov, p.transpose()))
    prior_prec_views = np.matmul(p, np.matmul(tau*cov, p.transpose()))
    post_returns = prior_returns + np.matmul(np.matmul(cov_tau, np.matmul(p.transpose(), np.linalg.inv(prior_prec_views + omega))),q - np.matmul(p, prior_returns))
    post_dist = np.add(cov, np.subtract(cov_tau, np.matmul(np.matmul(np.matmul(cov_tau, p.transpose()), np.linalg.inv(prior_prec_views + omega)), np.matmul(p, cov_tau))))
    unconstrained_w = np.matmul(post_returns.transpose(), np.linalg.inv(aversion*post_dist))

    #outputs for the assignment
    constrained_w = unconstrained_w/np.sum(unconstrained_w)
    er = float(np.matmul(post_returns.transpose(), constrained_w.transpose()))
    opt_std = float(np.sqrt(np.matmul(np.matmul(constrained_w, post_dist), constrained_w.transpose())))
    opt_sharpe = float(er/opt_std)
    final_w = pd.DataFrame(constrained_w, columns=returns.columns, index = [['Asset Class Weights']]).transpose()
    final_w = np.transpose(list(final_w.to_numpy().T[0]))
    final_stats = pd.DataFrame([er, opt_std, opt_sharpe], columns = [['Portfolio Statistics (Weekly)']], index = [['Expected Return', 'Standard Deviation', 'Sharpe Ratio']]) #Need to annualize
    return final_w, final_stats

def get_data(start, end=None, tckr_list=DEFAULT_TICKERS, per='W'):

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

    stock_data = stock_data.subtract(stock_data['BIL'], axis=0)
    stock_data = stock_data.drop('BIL', axis=1)

    return stock_data

if __name__ == '__main__':
    app.run()
