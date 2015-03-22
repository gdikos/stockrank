import datetime as dt
import numpy as np
import pandas as pd

# QSTK Imports
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.qsdateutil as du

def get_orders_list(s_file_path):
    l_columns = ["year", "month", "day", "sym", "type", "num"]
    df_orders_list = pd.read_csv(s_file_path, sep=',', header=None)
    df_orders_list = df_orders_list.dropna(axis=1, how='all')
    df_orders_list.columns = l_columns
    return df_orders_list

def get_orders(df_orders_list):
    na_orders_list = df_orders_list.values

    l_orders = []
    ld_daily_orders = None
    
    for order in na_orders_list:
        dt_date = dt.datetime(order[0], order[1], order[2], hour=16)
        d_order = {df_orders_list.columns[3]: order[3], \
                   df_orders_list.columns[4]: order[4], \
                   df_orders_list.columns[5]: int(order[5])}

        if l_orders != [] and dt_date == l_orders[-1][0]:
            l_orders[-1][1].append(d_order)
        else:
            ld_daily_orders = []
            ld_daily_orders.append(d_order)
            l_orders.append([dt_date, ld_daily_orders])
    
    na_orders = np.array(l_orders)
    df_orders = pd.DataFrame(na_orders[:, 1], index=na_orders[:, 0], columns=["ord"])
    df_orders = df_orders.sort()
    dt_start = df_orders.ix[0].name 
    dt_end = df_orders.ix[-1].name

    ls_symbols = list(set(df_orders_list["sym"]))
    ls_symbols.sort() # It is neccesary to sort due the use of set
    
    return df_orders, dt_start, dt_end, ls_symbols

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

def get_prices(dt_start, dt_end, ls_symbols, s_key="close"):
    # close = adjusted close
    # actual_close = actual close
    d_data = get_data(dt_start, dt_end, ls_symbols)
    return d_data[s_key]

def process_daily_orders(dt_date, df_orders, df_prices, df_num, df_val, df_res):
    op = 0
    daily_orders = list(df_orders.ix[dt_date, "ord"])
    for order in daily_orders:
        if order["type"] == "Buy":
            op = 1
        elif order["type"] == "Sell":
            op = -1
        df_num.ix[dt_date, order["sym"]] += op * order["num"]
        df_res.ix[dt_date, "cash"] += -op * order["num"] * df_prices.ix[dt_date, order["sym"]]

def update_port(dt_date, dt_last_orders_date, ls_symbols, df_num, df_res):
    for s_symbol in ls_symbols:
        df_num.ix[dt_date, s_symbol] = df_num.ix[dt_last_orders_date, s_symbol]
    df_res.ix[dt_date, "cash"] = df_res.ix[dt_last_orders_date, "cash"]

def value_port(dt_date, ls_symbols, df_prices, df_num, df_val, df_res):
    for s_symbol in ls_symbols:
        df_val.ix[dt_date, s_symbol] = df_num.ix[dt_date, s_symbol] * df_prices.ix[dt_date, s_symbol]
    df_res.ix[dt_date, "port"] = np.sum(df_val.ix[dt_date, :])
    df_res.ix[dt_date, "total"] = df_res.ix[dt_date, "port"] + df_res.ix[dt_date, "cash"]
    
def process_orders(df_orders, df_prices, cash):
    ldt_dates = list(df_prices.index)
    ls_symbols = list(df_prices.columns)
    df_num = pd.DataFrame(index=ldt_dates, columns=ls_symbols)
    df_val = pd.DataFrame(index=ldt_dates, columns=ls_symbols)
    df_res = pd.DataFrame(index=ldt_dates, columns=["port", "cash", "total"])
    
    df_num = df_num.fillna(0.0)
    df_val = df_val.fillna(0.0)
    df_res = df_res.fillna(0.0)
    df_res.ix[0, "cash"] = cash

    ldt_orders_dates = list(df_orders.index) 
    iter_orders_dates = iter(ldt_orders_dates)
    dt_orders_date = iter_orders_dates.next()
    dt_last_orders_date = dt_orders_date

    for dt_date in ldt_dates:
        update_port(dt_date, dt_last_orders_date, ls_symbols, df_num, df_res)
        
        if dt_date == dt_orders_date:
            process_daily_orders(dt_date, df_orders, df_prices, df_num, df_val, df_res)
            try:
                dt_last_orders_date = dt_orders_date
                dt_orders_date = iter_orders_dates.next()
            except StopIteration:
                pass
            
        value_port(dt_date, ls_symbols, df_prices, df_num, df_val, df_res)
    
    df_port = df_num.join(df_val, lsuffix="_num", rsuffix="_val").join(df_res)
    #df_port.to_csv("port.csv")
    return df_port

def save_values(df_port, s_out_file_path):
    ldt_dates = df_port.index
    na_dates = np.array([[dt_date.year, dt_date.month, dt_date.day] for dt_date in ldt_dates])
    na_total = np.array(df_port["total"])
    na_values = np.insert(arr=na_dates, obj=3, values=na_total, axis=1)
    df_values = pd.DataFrame(na_values, columns=["year", "month", "day", "total"])
    df_values.to_csv(s_out_file_path, sep=",", header=False, index=False)

if __name__ == '__main__':
    print "start market_sim.py"
    s_in_file_path = "q4_orders.csv"
    s_out_file_path = "q4_values.csv"
    s_cash = "100000"
    f_cash = float(s_cash)
    df_orders_list = get_orders_list(s_in_file_path)
    df_orders, dt_start, dt_end, ls_symbols = get_orders(df_orders_list)
    df_prices = get_prices(dt_start, dt_end, ls_symbols)
    df_port = process_orders(df_orders, df_prices, f_cash)
    save_values(df_port, s_out_file_path)
    print "end market_sim.py"
