import itertools
import math
import pylab
import datetime as dt
import numpy as np

# QSTK Imports
import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da


def get_data(ls_symbols, dt_start, dt_end):
    """
    -Input:
        ls_symbols: symbols.
        dt_start: start date.
        dt_end: end date.
    -Output:
        ldt_timestamps: timestamps between start and end dates (end included).
        nd_prices: close values to the given symbols in such a period of time.
    """
    # It is needed closing prices so the timestamp should be hours=16.
    dt_timeofday = dt.timedelta(hours=16)

    # Get a list of trading days between the start and the end.
    ldt_timestamps = du.getNYSEdays(dt_start, dt_end, dt_timeofday)

    # Creating an object of the dataaccess class with Yahoo as the source.
    c_dataobj = da.DataAccess('Yahoo')

    # Keys to be read from the data, it is good to read everything in one go.
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']

    # Reading the data, now d_data is a dictionary with the keys above.
    # Timestamps and symbols are the ones that were specified before.
    ldf_data = c_dataobj.get_data(ldt_timestamps, ls_symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method='ffill')
        d_data[s_key] = d_data[s_key].fillna(method='bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)
    
    print ls_symbols
    print ls_keys
    for s_key in ls_symbols:
        val = c_dataobj.getPathOfFile(s_key)
        print val
        val = c_dataobj.getPathOfCSVFile('ALPHA.AT')
        print val
    
    # Save into a csv files the "close values"    
    d_data['close'].to_csv("data.csv")

    # Getting the numpy ndarray of close prices.
    na_prices = d_data['close'].values
    return ldt_timestamps, na_prices

    
def run_simulation(ls_symbols, ldt_timestamps, na_prices, step=10):
    # It is neccesary normalize the prices in order to get cumulative return
    # na_cum_rets: numpy array of cumulative returns.
    # step: increments (1 = 1%)

    histo_sharpe=[]
    na_cum_rets = na_prices / na_prices[0, :]

    allocations = np.zeros(len(ls_symbols))
    
    inc = range(0, 101, step)
    inc = [i / 100.0 for i in inc] 

    error = 1e-6
    best_sharpe =-100 
    res_allocations = allocations # initialization
    res_vol = 0
    res_daily_ret = 0
    res_cum_ret = 0
    for allocations in itertools.product(inc, repeat=len(ls_symbols)):
        if abs(1.0 - sum(allocations)) <= error:
            f_vol, f_daily_ret, f_sharpe, f_cum_ret = run_trial(na_cum_rets, list(allocations))
            histo_sharpe.append(f_sharpe) 
            if best_sharpe < f_sharpe:
                best_sharpe = f_sharpe
                res_allocations = allocations
                res_vol = f_vol
                res_daily_ret = f_daily_ret
                res_cum_ret = f_cum_ret

    print "Optimal solution:"
    print "Symbols:", ls_symbols
    print "Allocations:", list(res_allocations)
    print "Sharpe Ratio:", best_sharpe
    print "Volatility (stdev of daily returns):", res_vol
    print "Average Daily Return:", res_daily_ret
    print "Cumulative Return:", res_cum_ret
    print "Optimal solution:"
    print "Symbols:", ls_symbols
    print "Allocations:", list(res_allocations)
    print "Sharpe Ratio:", best_sharpe
    print "Volatility (stdev of daily returns):", res_vol
    print "Average Daily Return:", res_daily_ret
    print "Cumulative Return:", res_cum_ret
    pylab.hist(histo_sharpe, bins=100, normed=True)
    pylab.xlabel('Sharpe')
    pylab.ylabel('frequency')
    pylab.title('Sharpe-ratio hist.')
    pylab.grid()
    pylab.savefig('sharpe_histo.png')
    pylab.show()
    
def run_trial(na_cum_rets, allocations):
    """
    Input:
        na_cum_rets: numpy array of cumulative returns.
        allocations: porcentages of each symbol (column) in the portafolio.
    Output:
        f_vol: standard deviation of daily returns of the total portfolio.
        f_daily_ret: average daily return of the total portfolio.
        f_sharpe: it is assumed the always there are 252 trading days in a
        year, and risk free rate equal to 0.
        f_cum_ret: Cumulative return of the total portfolio.
    """
    # It is assumed that the total investment amounts to 1 dollar.
    # na_daily_invs: numpy array that contents the daily portafolio investment,
    # in which each colum represents how much money is invested in that
    # symbol.
    na_daily_invs = allocations * na_cum_rets.copy()

    # Calculating the portafolio investment (sum colums by colum)
    na_daily_inv = na_daily_invs.sum(axis=1)
    
    # Calculating the daily return
    # It gets a copy of na_daily_inv becouse tsu.returnize0 does not return
    # anything, the function changes the argument.
    na_daily_ret = na_daily_inv.copy()
    tsu.returnize0(na_daily_ret)
    
    # It is equivalent to STDEV.P from excel
    f_vol = np.std(na_daily_ret)
    # print f_vol
    f_daily_ret = np.mean(na_daily_ret)
    # print f_daily_ret
    f_sharpe =  math.sqrt(252) * f_daily_ret / f_vol
    
    # The next instruction is correct because the inicial investment was 1 dollar
    f_cum_ret = na_daily_inv[-1]
    
    return f_vol, f_daily_ret, f_sharpe, f_cum_ret


def main():
    # List of symbols
   #ls_symbols = ['AMZN', 'EBAY', 'GS', 'C'] 
    ls_symbols = ['ALPHA.AT','ETE.AT','TPEIR.AT','MYTIL.AT','OPAP.AT']
    # ls_symbols = ["table"]
    # Start and End date of the charts
    dt_start = dt.datetime(2012, 9, 1)
    dt_end = dt.datetime(2014, 9, 1)

    ldt_timestamps, na_prices = get_data(ls_symbols, dt_start, dt_end)
    run_simulation(ls_symbols, ldt_timestamps, na_prices)


if __name__ == '__main__':
    main()
