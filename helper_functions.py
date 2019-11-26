import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def get_datetimes(df):
    return pd.to_datetime(df.columns.values[1:], format='%Y-%m')


def get_datetimes_v2(df, column=None):
    return pd.to_datetime(df[column], format='%Y-%m')


def melt_data(df):
    melted = pd.melt(df, id_vars=['RegionID', 'RegionName', 'City',
                                  'State', 'Metro', 'CountyName', 'SizeRank'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted.groupby('time').aggregate({'value': 'mean'})


def melt_data_v2(df):

    melted = pd.melt(df, id_vars=['RegionID', 'RegionName', 'City',
                                'State', 'Metro', 'CountyName', 'SizeRank'], var_name='time')
    melted['time'] = pd.to_datetime(melted['time'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted


def stationarity_check(ts):
    """
        Stationary check function that utilizes the Dickey-Fuller test on time series.
        This function also plots the rolling statistics of the input series

        Taken from flatiron course on Time Series Decomposition

        Parameters:
            ts {Pandas dataframe or series}
    """
        
    # Calculate rolling statistics
    rolmean = ts.rolling(window = 8, center = False).mean()
    rolstd = ts.rolling(window = 8, center = False).std()
    
    # Perform the Dickey Fuller Test
    dftest = adfuller(ts['#Passengers']) # change the passengers column as required 
    
    #Plot rolling statistics:
    fig = plt.figure(figsize=(12,6))
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Print Dickey-Fuller test results
    print ('Results of Dickey-Fuller Test:')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                                             'p-value',
                                             '#Lags Used',
                                             'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
    return None
