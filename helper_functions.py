import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace import sarimax
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from matplotlib.pylab import rcParams


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

def model_SARIMA(df=None, order=None, s_order=None, print_table=False, model_fit=False):
    ARIMA_MODEL = sm.tsa.statespace.SARIMAX(df,
                                    order=order,
                                    seasonal_order=s_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)

    output = ARIMA_MODEL.fit()
    if print_table:
        print(output.summary().tables[1])    
    return output

def diagnostics_plot(model=None, figsize=(15, 18)):
    model.plot_diagnostics(figsize=figsize)
    plt.show()
    return None

def one_step_ahead_forecast(df=None, start_date=None, end_date=None, arima_model=None,
                            plot_df=True, plot_interval=True, print_mse=False, figsize=(15,6)):
        
    """
        Parameters:
            df:
                Pandas dataframe with the date index and values to graph.
                This dataframe should only be the date indicies and the values associated with the respective indices
            start_date:
                The beginning date for our model to begin its prediction. Enter as a string
                Format:
                    YYYY  or YYYY-MM  or YYYY-MM-DD
            end_date:
                The end date for our prediction model.  Enter as a string
                Format:
                    'YYYY'  or 'YYYY-MM'  or 'YYYY-MM-DD'
            arima_model:
                The arima/sarima model to use for generating predictions 
                
            plot_df
                Boolean to plot the observed value of the dataframe passed into the function
                By default, the boolean is set to true
                
            plot_interval:
                Boolean to plot interval of confidence of our prediction.
                By default, the boolean is set to true
            
            figsize:
                Tuple of the width and height of the figure to be used in the
                
            print_mse: 
                Boolean to print the means square error of the prediction vs the oserved.
                By default, this boolean is set to false
            
        Return:
            Plots the dynamic prediction data for the model and dataframe passed into the function 
    """
    
    #type case the start and end date as a timestamp object
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    #Setup the confidence interval and the prediction of the model generated
    pred = arima_model.get_prediction(start=start, end=end)
    pred_conf = pred.conf_int()
    
    #Predicted vs real values with confidence intervals
    rcParams['figure.figsize'] = figsize

    #Plot observed values for orange county
    if plot_df:
        ax = df.plot(label='observed')
        
        #Plot predicted values
        pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.9)

    else:
        ax = pred.predicted_mean.plot(label='One-step ahead Forecast', alpha=.9)

#     #Plot predicted values
#     pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.9)

    #Plot the range for confidence intervals
    if plot_interval:
        ax.fill_between(pred_conf.index,
                        pred_conf.iloc[:, 0],
                        pred_conf.iloc[:, 1], alpha=.5)
#                         pred_conf.iloc[:, 1], color='g', alpha=.5)

    #Set axes labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean housing prices')
    plt.legend()

    #Plot the prediction and the forecast
    plt.show()
    
    # Get the Real and predicted values
    forecasted = pred.predicted_mean
    truth = df.value

    
    # Compute the mean square error
    if print_mse:
        mse = ((forecasted - truth) ** 2).mean()
        print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))


def dynamic_prediction(df=None, start_date=None, end_date=None, arima_model=None, 
                       plot_df=True, plot_interval=True, print_mse=False, figsize=(15,6)):
    """
        Parameters
            df:
                Pandas dataframe with the date index and values to graph
            start_date:
                The beginning date for our model to begin its prediction
            end_date:
                The end date for our prediction model
            arima_model:
                The arima/sarima model to use for generating predictions

            plot_df
                Boolean to plot the observed value of the dataframe passed into the function
                By default, the boolean is set to true

            plot_interval:
                Boolean to plot interval of confidence of our prediction.
                By default, the boolean is set to true

            figsize:
                Tuple of the width and height of the figure to be used in the
                
            print_mse: 
                Boolean to print the means square error of the prediction vs the oserved.
                By default, this boolean is set to false
            
        Return:
            Plots the dynamic prediction data for the model and dataframe passed into the function 
    """    
    # Setup the start and end date of the prediction
    pred_start_date = pd.to_datetime(start_date)
    pred_end_date = pd.to_datetime(end_date)

    #Create the dynamic prediction model and confidence intervals
    pred_dynamic = arima_model.get_prediction(start=pred_start_date, 
                                              end=pred_end_date, dynamic=True, full_results=True)
    pred_dynamic_conf = pred_dynamic.conf_int()
    
    # Get the Real and predicted values
    forecasted = pred_dynamic.predicted_mean
    truth = df.value

    # Compute the mean square error
    if print_mse:
        mse = ((forecasted - truth) ** 2).mean()
        print('The Mean Squared Error of our Dynamic forecasts is {}'.format(round(mse, 2)))
    
    # Plot the dynamic forecast with confidence intervals.
    if plot_df:
        ax = df.plot(label='observed', figsize=figsize)
        pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

    else:
        ax = pred_dynamic.predicted_mean.plot(label='Dynamic Forecast')
        
    if plot_interval:
        ax.fill_between(pred_dynamic_conf.index,
                        pred_dynamic_conf.iloc[:, 0],
                        pred_dynamic_conf.iloc[:, 1], color='g', alpha=.3)
        ax.fill_betweenx(ax.get_ylim(), pred_start_date, forecasted.index[-1], alpha=.1, zorder=-1)

    #Set labels for the axis
    ax.set_xlabel('Date')
    ax.set_ylabel('Housing Prices')

    #Display graphs
    plt.legend()
    plt.show()





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
