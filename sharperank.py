import copy
import datetime as dt
import numpy as np
import pandas as pd
import math
import sys
import scipy
import scipy.stats.stats
# QSTK Imports
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.qsdateutil as du

def get_symbols(s_list_index):
    dataobj = da.DataAccess("Yahoo")
    
    return dataobj.get_symbols_from_list(s_list_index)

def get_data(dt_start, dt_end, ls_symbols):
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt.timedelta(hours=16))
    ls_keys = ["open", "high", "low", "close", "volume", "actual_close"]
    dataobj = da.DataAccess('Yahoo')
    ldf_data = dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method="ffill")
        d_data[s_key] = d_data[s_key].fillna(method="bfill")
        d_data[s_key] = d_data[s_key].fillna(1.0)
    return d_data
def get_sharpe(df_prices,i_lookback):
    df_sharpe = np.NAN * copy.deepcopy(df_prices)
    for s_symbol in df_prices.columns:
        ts_price = df_prices[s_symbol]
        ts_mid = pd.rolling_mean(ts_price, i_lookback)
        ts_std = pd.rolling_std(ts_price, i_lookback)
        df_sharpe[s_symbol] = (ts_mid*math.sqrt(i_lookback))/ts_std 
    ldt_timestamps = df_sharpe.index
    for i in range(1, len(ldt_timestamps)):
	df_sharpe.ix[ldt_timestamps[i]]=scipy.stats.stats.rankdata(df_sharpe.ix[ldt_timestamps[i]]) 

    return df_sharpe 

def save_sharpe(df_sharpe, s_out_file_path):
    df_sharpe.to_csv(s_out_file_path, sep=",", header=True, index=True)


def get_bollingers(df_prices, i_lookback):
    df_bollingers = np.NAN * copy.deepcopy(df_prices)
    for s_symbol in df_prices.columns:
        ts_price = df_prices[s_symbol]
        ts_mid = pd.rolling_mean(ts_price, i_lookback)
        ts_std = pd.rolling_std(ts_price, i_lookback)
        df_bollingers[s_symbol] = (ts_price - ts_mid) / (ts_std) 
    return df_bollingers

def save_bollingers(df_bollingers, s_out_file_path):
    df_bollingers.to_csv(s_out_file_path, sep=",", header=True, index=True)

def find_bollinger_events(df_bollingers,trigger,market,switch,switch2):
    count = 0
    df_events = np.NAN * copy.deepcopy(df_bollingers)
    ldt_timestamps = df_bollingers.index
    for s_symbol in ls_symbols:
        for i in range(1, len(ldt_timestamps)):
            f_bollinger_today = df_bollingers[s_symbol].ix[ldt_timestamps[i]]
            f_bollinger_yest = df_bollingers[s_symbol].ix[ldt_timestamps[i - 1]]
            f_bollinger_index = df_bollingers[ls_symbols[-1]].ix[ldt_timestamps[i]]
            if f_bollinger_today*switch2 < trigger*switch2 and f_bollinger_yest*switch2 >= trigger*switch2 and (f_bollinger_index*switch)  >= (switch*market):
                df_events[s_symbol].ix[ldt_timestamps[i]] = 1
                count = count +1 
    return df_events, count

def find_sharpe_events(df_sharpe,trigger,market,switch,switch2,i_lookback):
    count = 0
    df_events = np.NAN * copy.deepcopy(df_sharpe)
    ldt_timestamps = df_sharpe.index
    for s_symbol in ls_symbols:
        for i in range(1, len(ldt_timestamps)):
            f_sharpe_today = df_sharpe[s_symbol].ix[ldt_timestamps[i]]
            f_sharpe_yest = df_sharpe[s_symbol].ix[ldt_timestamps[i - i_lookback ]]
            f_sharpe_index = df_sharpe[ls_symbols[-1]].ix[ldt_timestamps[i]]
            f_sharpe_index_yest = df_sharpe[ls_symbols[-1]].ix[ldt_timestamps[i-i_lookback]]
            if f_sharpe_today > (f_sharpe_yest + trigger*100) and f_sharpe_index*switch >= switch*(f_sharpe_index_yest + market*100):
            # if f_sharpe_today*switch2 < trigger*switch2 and f_sharpe_yest*switch2 >= trigger*switch2 and (f_sharpe_index*switch)  >= (switch*market):
                df_events[s_symbol].ix[ldt_timestamps[i]] = 1
                count = count +1
    return df_events, count

def find_momentum(df_prices, trigger, market, switch, switch2, i_lookback):
    ''' Finding the event dataframe '''
    df_momentum = np.NAN * copy.deepcopy(df_prices)
    count = 0
    df_events = np.NAN * copy.deepcopy(df_momentum)

    # Time stamps for the event range
    ldt_timestamps = df_momentum.index
    for s_sym in ls_symbols:
        for w in range(1, i_lookback):
            for i in range(1, len(ldt_timestamps)):
            # Calculating the returns for this timestamp
                if (i+w) < len(ldt_timestamps):
                    
                    f_symprice_beg = df_prices[s_sym].ix[ldt_timestamps[i]]
                    f_symprice_end = df_prices[s_sym].ix[ldt_timestamps[i + w]]
                    f_market_beg = df_prices[ls_symbols[-1]].ix[ldt_timestamps[i]]
                    f_market_end = df_prices[ls_symbols[-1]].ix[ldt_timestamps[i+w]]
            #f_marketprice_today = ts_market.ix[ldt_timestamps[i]]
            #f_marketprice_yest = ts_market.ix[ldt_timestamps[i - 1]]
            #f_symreturn_today = (f_symprice_today / f_symprice_yest) - 1
            #f_marketreturn_today = (f_marketprice_today / f_marketprice_yest) -1
 
            
            # Event is found if the symbol is down more then 3% while the
            # market is up more then 2%
            #if f_symreturn_today <= -0.03 and f_marketreturn_today >= 0.02:
            #       if (f_symprice_end - f_symprice_beg) < -(0.15/60)*w*f_symprice_beg:
                    if (f_symprice_end - f_symprice_beg) < -(trigger*switch)*f_symprice_beg and (f_market_end - f_market_beg) > (market*switch2)*f_market_beg :
                        df_momentum[s_sym].ix[ldt_timestamps[i]] = 1
                        count = count + 1
#                        print count, i, w, f_symprice_end, f_symprice_beg, trigger*switch, f_symprice_beg
    return df_momentum, count


def generate_order(ldt_dates, t, delta_t, s_symbol, i_num):
    l_buy_order = [ldt_dates[t], s_symbol, "Buy", i_num]  
    i = t + delta_t
    if t + delta_t >= len(ldt_dates):
        i = len(ldt_dates) - 1
    l_sell_order = [ldt_dates[i], s_symbol, "Sell", i_num]
    return l_buy_order, l_sell_order

def generate_orders(df_events, i_num, delta_t):
    t = 0
    ldt_dates = list(df_events.index)
    ls_symbols = list(df_events.columns)
    ls_orders = []
    for t in range(len(ldt_dates)):
        for s_symbol in ls_symbols:
            if df_events.ix[ldt_dates[t], s_symbol] == 1:
                l_buy_order, l_sell_order = generate_order(ldt_dates, t, delta_t, s_symbol, i_num)
                ls_orders.append(l_buy_order)
                ls_orders.append(l_sell_order)
    df_orders = pd.DataFrame(data=ls_orders, columns=["date", "sym", "type", "num"])
    # It is not possible to set "date" as index due duplicate keys
    df_orders = df_orders.sort(["date", "sym", "type"], ascending=[1, 1, 1])
    df_orders = df_orders.reset_index(drop=True)
    return df_orders

def save_orders(df_orders, s_out_file_path):
    na_dates = np.array([[dt_date.year, dt_date.month, dt_date.day] for dt_date in df_orders["date"]])
    df_dates = pd.DataFrame(data=na_dates, columns=["year", "month", "day"])
    del df_orders["date"]
    df_orders = df_dates.join(df_orders)
    df_orders.to_csv(s_out_file_path, sep=",", header=False, index=False)
    
if __name__ == '__main__':
#    print "start bollinger_events.py"

    s_list_index = "ase" 
    s_index = "FTSE.AT"
    s_lookback = sys.argv[1]
    s_delta_t = sys.argv[2]
    trigger= sys.argv[3] 
    market=sys.argv[4]
    switch=sys.argv[5]
    switch2=sys.argv[6]
    style=sys.argv[7]
    s_num = "100"
    s_start = "2012-09-01"
    s_end = "2015-02-04" 

    s_sharpe_file_path = "q4_sharpe" + ".csv"
    s_bollingers_file_path = "data\\q1_bollinger" + ".csv"
    s_events_file_path = "data\\q1_bollinger_events" + ".csv"
    s_events_img_path = "data\\q1_bollinger_events" + ".pdf"
    s_orders_file_path = "q4_orders" + ".csv"
    trigger=float(trigger) 
    market=float(market) 
    switch=float(switch)
    switch2=float(switch2)
    i_lookback = int(s_lookback)
    delta_t = int(s_delta_t)
    i_num = int(s_num)
    style=int(style)
    dt_start = dt.datetime.strptime(s_start, "%Y-%m-%d")
    dt_end = dt.datetime.strptime(s_end, "%Y-%m-%d")
    
    ls_symbols = get_symbols(s_list_index)
    ls_symbols.append(s_index)
    d_data = get_data(dt_start, dt_end, ls_symbols)
    print ls_symbols[-1]    
    
    #df_bollingers = get_bollingers(d_data["close"], i_lookback)
    #save_bollingers(df_bollingers, s_bollingers_file_path)
    #df_bollinger_events,count = find_bollinger_events(df_bollingers,trigger,market,switch,switch2)
    if (style==1):
    	df_momentum_events, count = find_momentum(d_data["actual_close"],trigger,market,switch,switch2,i_lookback)
    #df_orders = generate_orders(df_bollinger_events, i_num, delta_t)
    #sharpe = get_sharpe(d_data["actual_close"],i_lookback)
    #print sharpe["ALPHA.AT"]
    	df_orders = generate_orders(df_momentum_events, i_num, delta_t)
    	save_orders(df_orders, s_orders_file_path)
    	print count
    if (style==0):
   	print "sharpe rank"
        df_sharpe= get_sharpe(d_data["close"], i_lookback)
    	save_sharpe(df_sharpe, s_sharpe_file_path)
    	df_sharpe_events, count = find_sharpe_events(df_sharpe,trigger,market,switch,switch2,i_lookback)
    #df_orders = generate_orders(df_bollinger_events, i_num, delta_t)
    # sharpe = get_sharpe(d_data["actual_close"],i_lookback)
    #print sharpe["ALPHA.AT"]
    #for s_symbol in ls_symbols:
    #	print s_symbol    
    #   print sharpe[s_symbol]
    #	print sharpe[s_symbol].mean() 
    	df_orders = generate_orders(df_sharpe_events, i_num, delta_t)
    	save_orders(df_orders, s_orders_file_path)
    print count


#   print "end bollinger_events.py"  
