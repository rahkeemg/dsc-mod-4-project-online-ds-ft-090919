
# Mod 4 Project - Starter Notebook

In this project, we will take a look at a zillow dataset on the housing prices.


## Outline:

<ul>
    <li>Abstract</li>
    <li>Load and preprocess the data</li>
    <li>Filter for the desired area</li>
    <li>Find the best model to use for making predictions</li>
    <li>Graph predictions of our data</li>
    <li>Summary of the results</li>
</ul>

## Abstract 

The area that we will be looking at for this project is Central Florida.

We are going to create a time series model to find the best zip code to buy homes within this county of Florida.

![map of Orange County, Florda](images/Florida_by_Counties.png)

![Central Florida Counties](images/Central_Florida_Counties.png)


### Import libraries to run notebook


```
import warnings; warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import statsmodels.api as sm
import helper_functions as hf
import matplotlib.pyplot as plt
import itertools
from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace import sarimax
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

%matplotlib inline
%reload_ext autoreload
%autoreload 2
```


```
data_wide = pd.read_csv('./zillow_data.csv')
```


```
data_wide.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionID</th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
      <th>1996-04</th>
      <th>1996-05</th>
      <th>1996-06</th>
      <th>...</th>
      <th>2017-07</th>
      <th>2017-08</th>
      <th>2017-09</th>
      <th>2017-10</th>
      <th>2017-11</th>
      <th>2017-12</th>
      <th>2018-01</th>
      <th>2018-02</th>
      <th>2018-03</th>
      <th>2018-04</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>84654</td>
      <td>60657</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>Chicago</td>
      <td>Cook</td>
      <td>1</td>
      <td>334200.0</td>
      <td>335400.0</td>
      <td>336500.0</td>
      <td>...</td>
      <td>1005500</td>
      <td>1007500</td>
      <td>1007800</td>
      <td>1009600</td>
      <td>1013300</td>
      <td>1018700</td>
      <td>1024400</td>
      <td>1030700</td>
      <td>1033800</td>
      <td>1030600</td>
    </tr>
    <tr>
      <th>1</th>
      <td>90668</td>
      <td>75070</td>
      <td>McKinney</td>
      <td>TX</td>
      <td>Dallas-Fort Worth</td>
      <td>Collin</td>
      <td>2</td>
      <td>235700.0</td>
      <td>236900.0</td>
      <td>236700.0</td>
      <td>...</td>
      <td>308000</td>
      <td>310000</td>
      <td>312500</td>
      <td>314100</td>
      <td>315000</td>
      <td>316600</td>
      <td>318100</td>
      <td>319600</td>
      <td>321100</td>
      <td>321800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>91982</td>
      <td>77494</td>
      <td>Katy</td>
      <td>TX</td>
      <td>Houston</td>
      <td>Harris</td>
      <td>3</td>
      <td>210400.0</td>
      <td>212200.0</td>
      <td>212200.0</td>
      <td>...</td>
      <td>321000</td>
      <td>320600</td>
      <td>320200</td>
      <td>320400</td>
      <td>320800</td>
      <td>321200</td>
      <td>321200</td>
      <td>323000</td>
      <td>326900</td>
      <td>329900</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84616</td>
      <td>60614</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>Chicago</td>
      <td>Cook</td>
      <td>4</td>
      <td>498100.0</td>
      <td>500900.0</td>
      <td>503100.0</td>
      <td>...</td>
      <td>1289800</td>
      <td>1287700</td>
      <td>1287400</td>
      <td>1291500</td>
      <td>1296600</td>
      <td>1299000</td>
      <td>1302700</td>
      <td>1306400</td>
      <td>1308500</td>
      <td>1307000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>93144</td>
      <td>79936</td>
      <td>El Paso</td>
      <td>TX</td>
      <td>El Paso</td>
      <td>El Paso</td>
      <td>5</td>
      <td>77300.0</td>
      <td>77300.0</td>
      <td>77300.0</td>
      <td>...</td>
      <td>119100</td>
      <td>119400</td>
      <td>120000</td>
      <td>120300</td>
      <td>120300</td>
      <td>120300</td>
      <td>120300</td>
      <td>120500</td>
      <td>121000</td>
      <td>121500</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 272 columns</p>
</div>




```
data_long = hf.melt_data_v2(data_wide)
data_long['time'] = pd.to_datetime(data_long['time'], format='%Y-%m-%d')
data_long['RegionName'] = data_long['RegionName'].astype('str')

data_long.set_index(keys='time', inplace=True)
data_long.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionID</th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
      <th>value</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1996-04-01</th>
      <td>84654</td>
      <td>60657</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>Chicago</td>
      <td>Cook</td>
      <td>1</td>
      <td>334200.0</td>
    </tr>
    <tr>
      <th>1996-04-01</th>
      <td>90668</td>
      <td>75070</td>
      <td>McKinney</td>
      <td>TX</td>
      <td>Dallas-Fort Worth</td>
      <td>Collin</td>
      <td>2</td>
      <td>235700.0</td>
    </tr>
    <tr>
      <th>1996-04-01</th>
      <td>91982</td>
      <td>77494</td>
      <td>Katy</td>
      <td>TX</td>
      <td>Houston</td>
      <td>Harris</td>
      <td>3</td>
      <td>210400.0</td>
    </tr>
    <tr>
      <th>1996-04-01</th>
      <td>84616</td>
      <td>60614</td>
      <td>Chicago</td>
      <td>IL</td>
      <td>Chicago</td>
      <td>Cook</td>
      <td>4</td>
      <td>498100.0</td>
    </tr>
    <tr>
      <th>1996-04-01</th>
      <td>93144</td>
      <td>79936</td>
      <td>El Paso</td>
      <td>TX</td>
      <td>El Paso</td>
      <td>El Paso</td>
      <td>5</td>
      <td>77300.0</td>
    </tr>
  </tbody>
</table>
</div>



## Filter for the desired area within Florida

In this project, we will be looing at zip codes in Florida, specifically areas near Orlando


```
#separate out areas within the state of Florida
df_fl = data_long.loc[data_long.State=='FL']
df_fl.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 205265 entries, 1996-04-01 to 2018-04-01
    Data columns (total 8 columns):
    RegionID      205265 non-null int64
    RegionName    205265 non-null object
    City          205265 non-null object
    State         205265 non-null object
    Metro         198110 non-null object
    CountyName    205265 non-null object
    SizeRank      205265 non-null int64
    value         205265 non-null float64
    dtypes: float64(1), int64(2), object(5)
    memory usage: 14.1+ MB
    


```
df_fl.isna().sum()
```




    RegionID         0
    RegionName       0
    City             0
    State            0
    Metro         7155
    CountyName       0
    SizeRank         0
    value            0
    dtype: int64



Since the Metro area has a large amount of information that is missing, this column will not be used, although it is a good label of where homes are located related to nearest cities.  

We will use county inplace of metro area to identify the housing areas.

The benefit of using county as well is that there is not a great difference the unique counties and metro-areas within the state of Florida


```
print(f"Number of unique Metro areas in FL: {df_fl.Metro.nunique()}" + 
        f"\nNumber of unique counties in FL: {df_fl.CountyName.nunique()}" + 
     f"\nNumber of unique cities in FL: {df_fl.City.nunique()}")
```

    Number of unique Metro areas in FL: 28
    Number of unique counties in FL: 57
    Number of unique cities in FL: 401
    


```
df_fl.info()
df_fl.head()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 205265 entries, 1996-04-01 to 2018-04-01
    Data columns (total 8 columns):
    RegionID      205265 non-null int64
    RegionName    205265 non-null object
    City          205265 non-null object
    State         205265 non-null object
    Metro         198110 non-null object
    CountyName    205265 non-null object
    SizeRank      205265 non-null int64
    value         205265 non-null float64
    dtypes: float64(1), int64(2), object(5)
    memory usage: 14.1+ MB
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionID</th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
      <th>value</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1996-04-01</th>
      <td>71831</td>
      <td>32162</td>
      <td>The Villages</td>
      <td>FL</td>
      <td>The Villages</td>
      <td>Sumter</td>
      <td>12</td>
      <td>101000.0</td>
    </tr>
    <tr>
      <th>1996-04-01</th>
      <td>72463</td>
      <td>33160</td>
      <td>Sunny Isles Beach</td>
      <td>FL</td>
      <td>Miami-Fort Lauderdale</td>
      <td>Miami-Dade</td>
      <td>61</td>
      <td>337300.0</td>
    </tr>
    <tr>
      <th>1996-04-01</th>
      <td>72363</td>
      <td>33025</td>
      <td>Miramar</td>
      <td>FL</td>
      <td>Miami-Fort Lauderdale</td>
      <td>Broward</td>
      <td>83</td>
      <td>111600.0</td>
    </tr>
    <tr>
      <th>1996-04-01</th>
      <td>72578</td>
      <td>33411</td>
      <td>Royal Palm Beach</td>
      <td>FL</td>
      <td>Miami-Fort Lauderdale</td>
      <td>Palm Beach</td>
      <td>84</td>
      <td>126800.0</td>
    </tr>
    <tr>
      <th>1996-04-01</th>
      <td>72442</td>
      <td>33139</td>
      <td>Miami Beach</td>
      <td>FL</td>
      <td>Miami-Fort Lauderdale</td>
      <td>Miami-Dade</td>
      <td>85</td>
      <td>480200.0</td>
    </tr>
  </tbody>
</table>
</div>




```
from matplotlib import rc

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

rc('font', **font)

# NOTE: if you visualizations are too cluttered to read, try calling 'plt.gcf().autofmt_xdate()'!
```


```
df_fl.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RegionID</th>
      <th>RegionName</th>
      <th>City</th>
      <th>State</th>
      <th>Metro</th>
      <th>CountyName</th>
      <th>SizeRank</th>
      <th>value</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1996-04-01</th>
      <td>71831</td>
      <td>32162</td>
      <td>The Villages</td>
      <td>FL</td>
      <td>The Villages</td>
      <td>Sumter</td>
      <td>12</td>
      <td>101000.0</td>
    </tr>
    <tr>
      <th>1996-04-01</th>
      <td>72463</td>
      <td>33160</td>
      <td>Sunny Isles Beach</td>
      <td>FL</td>
      <td>Miami-Fort Lauderdale</td>
      <td>Miami-Dade</td>
      <td>61</td>
      <td>337300.0</td>
    </tr>
    <tr>
      <th>1996-04-01</th>
      <td>72363</td>
      <td>33025</td>
      <td>Miramar</td>
      <td>FL</td>
      <td>Miami-Fort Lauderdale</td>
      <td>Broward</td>
      <td>83</td>
      <td>111600.0</td>
    </tr>
    <tr>
      <th>1996-04-01</th>
      <td>72578</td>
      <td>33411</td>
      <td>Royal Palm Beach</td>
      <td>FL</td>
      <td>Miami-Fort Lauderdale</td>
      <td>Palm Beach</td>
      <td>84</td>
      <td>126800.0</td>
    </tr>
    <tr>
      <th>1996-04-01</th>
      <td>72442</td>
      <td>33139</td>
      <td>Miami Beach</td>
      <td>FL</td>
      <td>Miami-Fort Lauderdale</td>
      <td>Miami-Dade</td>
      <td>85</td>
      <td>480200.0</td>
    </tr>
  </tbody>
</table>
</div>




```
# Florida monthly means ovre the years
florida_monthly = df_fl.groupby(pd.Grouper(freq='MS'))
florida_monthly.value.mean().plot(figsize=(15,5))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21010424e10>




![png](mod_4_starter_notebook_files/mod_4_starter_notebook_15_1.png)



```
# Separate out the areas that are part of Central Florida
central_florida = ['Levy', 'Marion', 'Volusia', 'Citrus', 'Sumter',
                           'Lake', 'Seminole', 'Hernando', 'Orange', 'Brevard', 'Pasco',
                           'Pinellas', 'Hillsborough', 'Polk', ' Osceola', 'Indian River']
```


```
#Plot the means for each county in Central Florida
plt.figure(figsize=(18,7))

for county in central_florida:    
    county_df = df_fl.loc[(df_fl.CountyName==county), ['value']].resample('MS').mean()
    plt.plot(county_df, label=county)
plt.legend()
plt.show()

# df_fl_2011.loc[(df_fl_2011.CountyName=='Orange'), ['value']].resample('MS').mean()
```


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_17_0.png)


Overall, Central FLorida's counties resemble the entire state of Florida.  Each of these counties were affected by the housing market crash, but some were affected more than others.

Based on our graph below, there appears to be some seasonality that is occurring within the housing market for orange county as time passes by.

TO create our model, we will look at the average of Central Florida's mean prices by month. 


```
central_fl = df_fl.loc[(df_fl.CountyName.isin(central_florida)), ['value']].resample('MS').mean()
```


```
## Looking at difference by year
# Look at the distribution of the diffs and look at the one with the smallest standard deviation

plt.gcf().autofmt_xdate()

central_fl_diff = central_fl.diff(periods=1)
rcParams['figure.figsize'] = (12, 5)
plt.plot(central_fl_diff)

rcParams['figure.figsize'] = (12, 5)
plot_acf(central_fl['value'], title='Orange County Auto Correlation');
plot_pacf(central_fl['value'], title='Orange County Partial Auto Correlation');
```


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_21_0.png)



![png](mod_4_starter_notebook_files/mod_4_starter_notebook_21_1.png)



![png](mod_4_starter_notebook_files/mod_4_starter_notebook_21_2.png)


Based on what we see in our partial correlation plot, there is a high negative correlation somewhere between 220 - 245 lags.

This high negative appears at lag = 180 months


```
central_fl.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value</th>
    </tr>
    <tr>
      <th>time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1996-04-01</th>
      <td>87852.076677</td>
    </tr>
    <tr>
      <th>1996-05-01</th>
      <td>87926.198083</td>
    </tr>
    <tr>
      <th>1996-06-01</th>
      <td>87991.693291</td>
    </tr>
    <tr>
      <th>1996-07-01</th>
      <td>88042.492013</td>
    </tr>
    <tr>
      <th>1996-08-01</th>
      <td>88106.389776</td>
    </tr>
  </tbody>
</table>
</div>



# ARIMA Modeling

Before getting into the ARIMA modeling, combinations for the model needs to be created.  
Here, the parameters for all combinations of seasons are also added to our values for seasonal & non seasonal arima modeling.


```
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
# pdqs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# Generate combinations of SARIMA modeling with different seasonalities
pdqs = []    
for i in range(0,13):
    for x in pdq:
        pdqs.append((x[0], x[1], x[2], i))

pdqs
```




    [(0, 0, 0, 0),
     (0, 0, 1, 0),
     (0, 1, 0, 0),
     (0, 1, 1, 0),
     (1, 0, 0, 0),
     (1, 0, 1, 0),
     (1, 1, 0, 0),
     (1, 1, 1, 0),
     (0, 0, 0, 1),
     (0, 0, 1, 1),
     (0, 1, 0, 1),
     (0, 1, 1, 1),
     (1, 0, 0, 1),
     (1, 0, 1, 1),
     (1, 1, 0, 1),
     (1, 1, 1, 1),
     (0, 0, 0, 2),
     (0, 0, 1, 2),
     (0, 1, 0, 2),
     (0, 1, 1, 2),
     (1, 0, 0, 2),
     (1, 0, 1, 2),
     (1, 1, 0, 2),
     (1, 1, 1, 2),
     (0, 0, 0, 3),
     (0, 0, 1, 3),
     (0, 1, 0, 3),
     (0, 1, 1, 3),
     (1, 0, 0, 3),
     (1, 0, 1, 3),
     (1, 1, 0, 3),
     (1, 1, 1, 3),
     (0, 0, 0, 4),
     (0, 0, 1, 4),
     (0, 1, 0, 4),
     (0, 1, 1, 4),
     (1, 0, 0, 4),
     (1, 0, 1, 4),
     (1, 1, 0, 4),
     (1, 1, 1, 4),
     (0, 0, 0, 5),
     (0, 0, 1, 5),
     (0, 1, 0, 5),
     (0, 1, 1, 5),
     (1, 0, 0, 5),
     (1, 0, 1, 5),
     (1, 1, 0, 5),
     (1, 1, 1, 5),
     (0, 0, 0, 6),
     (0, 0, 1, 6),
     (0, 1, 0, 6),
     (0, 1, 1, 6),
     (1, 0, 0, 6),
     (1, 0, 1, 6),
     (1, 1, 0, 6),
     (1, 1, 1, 6),
     (0, 0, 0, 7),
     (0, 0, 1, 7),
     (0, 1, 0, 7),
     (0, 1, 1, 7),
     (1, 0, 0, 7),
     (1, 0, 1, 7),
     (1, 1, 0, 7),
     (1, 1, 1, 7),
     (0, 0, 0, 8),
     (0, 0, 1, 8),
     (0, 1, 0, 8),
     (0, 1, 1, 8),
     (1, 0, 0, 8),
     (1, 0, 1, 8),
     (1, 1, 0, 8),
     (1, 1, 1, 8),
     (0, 0, 0, 9),
     (0, 0, 1, 9),
     (0, 1, 0, 9),
     (0, 1, 1, 9),
     (1, 0, 0, 9),
     (1, 0, 1, 9),
     (1, 1, 0, 9),
     (1, 1, 1, 9),
     (0, 0, 0, 10),
     (0, 0, 1, 10),
     (0, 1, 0, 10),
     (0, 1, 1, 10),
     (1, 0, 0, 10),
     (1, 0, 1, 10),
     (1, 1, 0, 10),
     (1, 1, 1, 10),
     (0, 0, 0, 11),
     (0, 0, 1, 11),
     (0, 1, 0, 11),
     (0, 1, 1, 11),
     (1, 0, 0, 11),
     (1, 0, 1, 11),
     (1, 1, 0, 11),
     (1, 1, 1, 11),
     (0, 0, 0, 12),
     (0, 0, 1, 12),
     (0, 1, 0, 12),
     (0, 1, 1, 12),
     (1, 0, 0, 12),
     (1, 0, 1, 12),
     (1, 1, 0, 12),
     (1, 1, 1, 12)]




```
## Run multiple models with the different combination generated  for pdq & pdqs##
ans = []
for comb in pdq:
    for combs in pdqs:
        try:
            mod = hf.model_SARIMA(df=central_fl, order=comb, s_order=combs)
            ans.append([comb, combs, mod.aic, mod.bic])
#             print('ARIMA {} x {} : AIC Calculated ={}, BIC Calculated ={}'.format(comb, combs, mod.aic, mod.bic))
        except:
            continue
```

After running all of the possible combinations through the seasonal ARIMA model, the results of each combination was stored in a dataframe, so that we can easily search for the optimum model.


```
ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic', 'bic'])
ans_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pdq</th>
      <th>pdqs</th>
      <th>aic</th>
      <th>bic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(0, 0, 0)</td>
      <td>(0, 0, 0, 0)</td>
      <td>7090.036088</td>
      <td>7093.612037</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(0, 0, 0)</td>
      <td>(0, 0, 1, 0)</td>
      <td>6872.806440</td>
      <td>6879.950749</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(0, 0, 0)</td>
      <td>(1, 0, 0, 0)</td>
      <td>4713.930258</td>
      <td>4721.082156</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(0, 0, 0)</td>
      <td>(1, 0, 1, 0)</td>
      <td>4352.493231</td>
      <td>4363.209693</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(0, 0, 0)</td>
      <td>(0, 0, 0, 1)</td>
      <td>7090.036088</td>
      <td>7093.612037</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(0, 0, 0)</td>
      <td>(0, 0, 1, 1)</td>
      <td>6872.806440</td>
      <td>6879.950749</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(0, 0, 0)</td>
      <td>(0, 1, 0, 1)</td>
      <td>4714.914364</td>
      <td>4718.486518</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(0, 0, 0)</td>
      <td>(0, 1, 1, 1)</td>
      <td>4353.738444</td>
      <td>4360.875133</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(0, 0, 0)</td>
      <td>(1, 0, 0, 1)</td>
      <td>4713.930258</td>
      <td>4721.082156</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(0, 0, 0)</td>
      <td>(1, 0, 1, 1)</td>
      <td>4352.493231</td>
      <td>4363.209693</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(0, 0, 0)</td>
      <td>(1, 1, 0, 1)</td>
      <td>3638.499163</td>
      <td>3645.643471</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(0, 0, 0)</td>
      <td>(1, 1, 1, 1)</td>
      <td>3571.260462</td>
      <td>3581.965496</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(0, 0, 0)</td>
      <td>(0, 0, 0, 2)</td>
      <td>7090.036088</td>
      <td>7093.612037</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(0, 0, 0)</td>
      <td>(0, 0, 1, 2)</td>
      <td>6849.867842</td>
      <td>6857.004531</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(0, 0, 0)</td>
      <td>(0, 1, 0, 2)</td>
      <td>5059.334395</td>
      <td>5062.902739</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(0, 0, 0)</td>
      <td>(0, 1, 1, 2)</td>
      <td>4960.339395</td>
      <td>4967.460758</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(0, 0, 0)</td>
      <td>(1, 0, 0, 2)</td>
      <td>5060.874224</td>
      <td>5068.018532</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(0, 0, 0)</td>
      <td>(1, 0, 1, 2)</td>
      <td>4985.428542</td>
      <td>4996.133575</td>
    </tr>
    <tr>
      <th>18</th>
      <td>(0, 0, 0)</td>
      <td>(1, 1, 0, 2)</td>
      <td>4198.619304</td>
      <td>4205.748345</td>
    </tr>
    <tr>
      <th>19</th>
      <td>(0, 0, 0)</td>
      <td>(1, 1, 1, 2)</td>
      <td>4171.204263</td>
      <td>4181.886308</td>
    </tr>
    <tr>
      <th>20</th>
      <td>(0, 0, 0)</td>
      <td>(0, 0, 0, 3)</td>
      <td>7090.036088</td>
      <td>7093.612037</td>
    </tr>
    <tr>
      <th>21</th>
      <td>(0, 0, 0)</td>
      <td>(0, 0, 1, 3)</td>
      <td>32825.366887</td>
      <td>32832.495928</td>
    </tr>
    <tr>
      <th>22</th>
      <td>(0, 0, 0)</td>
      <td>(0, 1, 0, 3)</td>
      <td>5250.398927</td>
      <td>5253.963448</td>
    </tr>
    <tr>
      <th>23</th>
      <td>(0, 0, 0)</td>
      <td>(0, 1, 1, 3)</td>
      <td>5158.091234</td>
      <td>5165.197153</td>
    </tr>
    <tr>
      <th>24</th>
      <td>(0, 0, 0)</td>
      <td>(1, 0, 0, 3)</td>
      <td>5253.886184</td>
      <td>5261.022873</td>
    </tr>
    <tr>
      <th>25</th>
      <td>(0, 0, 0)</td>
      <td>(1, 0, 1, 3)</td>
      <td>5178.985791</td>
      <td>5189.679352</td>
    </tr>
    <tr>
      <th>26</th>
      <td>(0, 0, 0)</td>
      <td>(1, 1, 0, 3)</td>
      <td>4516.274975</td>
      <td>4523.388631</td>
    </tr>
    <tr>
      <th>27</th>
      <td>(0, 0, 0)</td>
      <td>(1, 1, 1, 3)</td>
      <td>4491.130146</td>
      <td>4501.789025</td>
    </tr>
    <tr>
      <th>28</th>
      <td>(0, 0, 0)</td>
      <td>(0, 0, 0, 4)</td>
      <td>7090.036088</td>
      <td>7093.612037</td>
    </tr>
    <tr>
      <th>29</th>
      <td>(0, 0, 0)</td>
      <td>(0, 0, 1, 4)</td>
      <td>40693.676630</td>
      <td>40700.797993</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>770</th>
      <td>(1, 1, 1)</td>
      <td>(0, 1, 0, 9)</td>
      <td>3613.586274</td>
      <td>3624.186443</td>
    </tr>
    <tr>
      <th>771</th>
      <td>(1, 1, 1)</td>
      <td>(0, 1, 1, 9)</td>
      <td>3380.857172</td>
      <td>3394.845845</td>
    </tr>
    <tr>
      <th>772</th>
      <td>(1, 1, 1)</td>
      <td>(1, 0, 0, 9)</td>
      <td>3469.243484</td>
      <td>3483.392821</td>
    </tr>
    <tr>
      <th>773</th>
      <td>(1, 1, 1)</td>
      <td>(1, 0, 1, 9)</td>
      <td>3458.449136</td>
      <td>3476.116084</td>
    </tr>
    <tr>
      <th>774</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 0, 9)</td>
      <td>3471.304956</td>
      <td>3485.309989</td>
    </tr>
    <tr>
      <th>775</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 1, 9)</td>
      <td>3368.600960</td>
      <td>3386.086801</td>
    </tr>
    <tr>
      <th>776</th>
      <td>(1, 1, 1)</td>
      <td>(0, 0, 0, 10)</td>
      <td>3571.260462</td>
      <td>3581.965496</td>
    </tr>
    <tr>
      <th>777</th>
      <td>(1, 1, 1)</td>
      <td>(0, 0, 1, 10)</td>
      <td>3433.582365</td>
      <td>3447.700081</td>
    </tr>
    <tr>
      <th>778</th>
      <td>(1, 1, 1)</td>
      <td>(0, 1, 0, 10)</td>
      <td>3540.191962</td>
      <td>3550.780250</td>
    </tr>
    <tr>
      <th>779</th>
      <td>(1, 1, 1)</td>
      <td>(0, 1, 1, 10)</td>
      <td>3412.143119</td>
      <td>3426.098870</td>
    </tr>
    <tr>
      <th>780</th>
      <td>(1, 1, 1)</td>
      <td>(1, 0, 0, 10)</td>
      <td>3455.931515</td>
      <td>3470.065073</td>
    </tr>
    <tr>
      <th>781</th>
      <td>(1, 1, 1)</td>
      <td>(1, 0, 1, 10)</td>
      <td>3435.460678</td>
      <td>3453.107824</td>
    </tr>
    <tr>
      <th>782</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 0, 10)</td>
      <td>3403.494599</td>
      <td>3417.466845</td>
    </tr>
    <tr>
      <th>783</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 1, 10)</td>
      <td>3334.859050</td>
      <td>3352.303738</td>
    </tr>
    <tr>
      <th>784</th>
      <td>(1, 1, 1)</td>
      <td>(0, 0, 0, 11)</td>
      <td>3571.260462</td>
      <td>3581.965496</td>
    </tr>
    <tr>
      <th>785</th>
      <td>(1, 1, 1)</td>
      <td>(0, 0, 1, 11)</td>
      <td>3416.558973</td>
      <td>3430.660784</td>
    </tr>
    <tr>
      <th>786</th>
      <td>(1, 1, 1)</td>
      <td>(0, 1, 0, 11)</td>
      <td>3525.786710</td>
      <td>3536.363068</td>
    </tr>
    <tr>
      <th>787</th>
      <td>(1, 1, 1)</td>
      <td>(0, 1, 1, 11)</td>
      <td>3377.425252</td>
      <td>3391.347807</td>
    </tr>
    <tr>
      <th>788</th>
      <td>(1, 1, 1)</td>
      <td>(1, 0, 0, 11)</td>
      <td>3443.294446</td>
      <td>3457.412162</td>
    </tr>
    <tr>
      <th>789</th>
      <td>(1, 1, 1)</td>
      <td>(1, 0, 1, 11)</td>
      <td>3418.310527</td>
      <td>3435.937792</td>
    </tr>
    <tr>
      <th>790</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 0, 11)</td>
      <td>3373.819459</td>
      <td>3387.758647</td>
    </tr>
    <tr>
      <th>791</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 1, 11)</td>
      <td>3370.722811</td>
      <td>3388.126005</td>
    </tr>
    <tr>
      <th>792</th>
      <td>(1, 1, 1)</td>
      <td>(0, 0, 0, 12)</td>
      <td>3571.260462</td>
      <td>3581.965496</td>
    </tr>
    <tr>
      <th>793</th>
      <td>(1, 1, 1)</td>
      <td>(0, 0, 1, 12)</td>
      <td>3420.715659</td>
      <td>3434.801503</td>
    </tr>
    <tr>
      <th>794</th>
      <td>(1, 1, 1)</td>
      <td>(0, 1, 0, 12)</td>
      <td>3589.373629</td>
      <td>3599.938012</td>
    </tr>
    <tr>
      <th>795</th>
      <td>(1, 1, 1)</td>
      <td>(0, 1, 1, 12)</td>
      <td>3431.947668</td>
      <td>3445.836750</td>
    </tr>
    <tr>
      <th>796</th>
      <td>(1, 1, 1)</td>
      <td>(1, 0, 0, 12)</td>
      <td>3431.185268</td>
      <td>3445.287079</td>
    </tr>
    <tr>
      <th>797</th>
      <td>(1, 1, 1)</td>
      <td>(1, 0, 1, 12)</td>
      <td>3415.343901</td>
      <td>3432.951205</td>
    </tr>
    <tr>
      <th>798</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 0, 12)</td>
      <td>3415.727959</td>
      <td>3429.633813</td>
    </tr>
    <tr>
      <th>799</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 1, 12)</td>
      <td>3393.598775</td>
      <td>3410.960128</td>
    </tr>
  </tbody>
</table>
<p>800 rows × 4 columns</p>
</div>



Here we see that the our best model has both the highest `aic`and `bic` scores.


```
ans_df.loc[ans_df['aic'].idxmin()]
```




    pdq         (1, 1, 1)
    pdqs    (1, 1, 1, 10)
    aic           3334.86
    bic            3352.3
    Name: 783, dtype: object




```
ans_df.loc[ans_df['bic'].idxmin()]
```




    pdq         (1, 1, 1)
    pdqs    (1, 1, 1, 10)
    aic           3334.86
    bic            3352.3
    Name: 783, dtype: object




```
ans_df.sort_values(by=['aic','bic'], ascending=True, inplace=True)
ans_df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pdq</th>
      <th>pdqs</th>
      <th>aic</th>
      <th>bic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>783</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 1, 10)</td>
      <td>3334.859050</td>
      <td>3352.303738</td>
    </tr>
    <tr>
      <th>775</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 1, 9)</td>
      <td>3368.600960</td>
      <td>3386.086801</td>
    </tr>
    <tr>
      <th>791</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 1, 11)</td>
      <td>3370.722811</td>
      <td>3388.126005</td>
    </tr>
    <tr>
      <th>790</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 0, 11)</td>
      <td>3373.819459</td>
      <td>3387.758647</td>
    </tr>
    <tr>
      <th>787</th>
      <td>(1, 1, 1)</td>
      <td>(0, 1, 1, 11)</td>
      <td>3377.425252</td>
      <td>3391.347807</td>
    </tr>
    <tr>
      <th>771</th>
      <td>(1, 1, 1)</td>
      <td>(0, 1, 1, 9)</td>
      <td>3380.857172</td>
      <td>3394.845845</td>
    </tr>
    <tr>
      <th>767</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 1, 8)</td>
      <td>3392.524334</td>
      <td>3410.050992</td>
    </tr>
    <tr>
      <th>799</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 1, 12)</td>
      <td>3393.598775</td>
      <td>3410.960128</td>
    </tr>
    <tr>
      <th>782</th>
      <td>(1, 1, 1)</td>
      <td>(1, 1, 0, 10)</td>
      <td>3403.494599</td>
      <td>3417.466845</td>
    </tr>
    <tr>
      <th>690</th>
      <td>(1, 1, 0)</td>
      <td>(1, 1, 0, 11)</td>
      <td>3407.564192</td>
      <td>3418.018583</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have the results from our SARIMA model, we will now take the results with best AIC and BIC and pass it into our Model to see how ti performed compared to the others

Fitting ARIMA Time Series Model 


```
### Get the results of our best parameters for our ARIMA model ###
order = ans_df.loc[ans_df['aic'].idxmin()]['pdq']
s_order = ans_df.loc[ans_df['aic'].idxmin()]['pdqs']
SARIMA_MODEL = hf.model_SARIMA(central_fl, order=order, s_order=s_order, print_table=True)
```

    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    ar.L1          0.9850      0.014     69.530      0.000       0.957       1.013
    ma.L1          0.4889      0.031     15.526      0.000       0.427       0.551
    ar.S.L10       0.0283      0.014      1.999      0.046       0.001       0.056
    ma.S.L10      -0.9909      0.359     -2.762      0.006      -1.694      -0.288
    sigma2      4.842e+04   1.53e+04      3.155      0.002    1.83e+04    7.85e+04
    ==============================================================================
    

Now, we are going to take a look where some of the residuals are deviating from the standard deviation and attempt to create a batter model with

### Plottting residuals 


```
hf.diagnostics_plot(model=SARIMA_MODEL)
```


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_39_0.png)


Based on the results of the plot diagnostics, we see that our data is not normally distributed. From here, we will continue to further look into our model and attempt to improve the results by removing outliers and the residuals that are causing issues within our model.

## Making predictions using our model parameters 


```
central_fl.idxmin()
```




    value   1996-04-01
    dtype: datetime64[ns]




```
hf.one_step_ahead_forecast(df=central_fl, start_date='2018', end_date='2021', arima_model=SARIMA_MODEL)
```


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_43_0.png)


### Dynamic forecasting of the data



```
hf.dynamic_prediction(df=central_fl, start_date='2018', end_date='2021', arima_model=SARIMA_MODEL)
```


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_45_0.png)


Based on the results of our dynamic prediction, the housign prices are forecast to steadily increase into 2025

## Findng the best zip code within Orange County

To find the best zip code within the given area, we wil use the following formula to calculate the return of investment:

$$\large R.O.I = \frac{(GFI - CoI)}{CoI}$$

- ROI = Return of Investment
- GFI = Gain from Investment
- CoI = Cost of Investment

Our Cost of Ivestment will be the average of 2017, since we do not have a complete dataset for 2018


To calculate GFI, we will take our cost of investment and subtract it from the average predicted means from 2018 to 2025.

We will then use the formula above to calculate the return of investment for each zip code observed


```
## Zipcodes of Orange County, Florida
zipcodes = df_fl.loc[(df_fl.CountyName.isin(central_florida))].RegionName.unique()
```


```
ROI_list = []
model_list = []

## Loop to get each zip code and calculate return on ivestment ##
for code in zipcodes:

    zip_df = df_fl.loc[(df_fl.RegionName==code), ['value']].resample('MS').mean()
    zip_model = hf.model_SARIMA(zip_df, order=order, s_order=s_order)

    pred = zip_model.get_prediction(start=pd.to_datetime('2018'), end=pd.to_datetime('2021'))

    ## Define the initial cost of investment as of 2017 ##
    cost_of_investment = zip_df['2017'].value.mean()

    ## Calculate gain from investmnt from 2018 up to 2021 ##
    gain_from_investment = pred.predicted_mean['2018':].mean()

    ## calculate Return of Investment for the observed zip code
    ROI = (gain_from_investment - cost_of_investment)/cost_of_investment
    ROI_list.append(ROI)
    model_list.append(zip_model)
```


```
df_results = pd.DataFrame(data=list(zip(zipcodes, ROI_list, model_list)), columns=['zip_code','ROI','model'])
```

# Step 6: Interpreting Results

Based on the results of our model, the top 5 zip codes to purchase a house from 2018-2025 are as follows:
    


```
df_results.sort_values(by='ROI', ascending=False, inplace=True)
df_results['county'] = df_results.zip_code.apply(
    lambda x: df_fl[df_fl.RegionName==x].CountyName.unique())
```


```
df_results.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>zip_code</th>
      <th>ROI</th>
      <th>model</th>
      <th>county</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>223</th>
      <td>34785</td>
      <td>0.821538</td>
      <td>&lt;statsmodels.tsa.statespace.sarimax.SARIMAXRes...</td>
      <td>[Sumter]</td>
    </tr>
    <tr>
      <th>79</th>
      <td>34652</td>
      <td>0.729564</td>
      <td>&lt;statsmodels.tsa.statespace.sarimax.SARIMAXRes...</td>
      <td>[Pasco]</td>
    </tr>
    <tr>
      <th>122</th>
      <td>34691</td>
      <td>0.642348</td>
      <td>&lt;statsmodels.tsa.statespace.sarimax.SARIMAXRes...</td>
      <td>[Pasco]</td>
    </tr>
    <tr>
      <th>239</th>
      <td>34488</td>
      <td>0.580343</td>
      <td>&lt;statsmodels.tsa.statespace.sarimax.SARIMAXRes...</td>
      <td>[Marion]</td>
    </tr>
    <tr>
      <th>212</th>
      <td>34690</td>
      <td>0.561948</td>
      <td>&lt;statsmodels.tsa.statespace.sarimax.SARIMAXRes...</td>
      <td>[Pasco]</td>
    </tr>
  </tbody>
</table>
</div>




```
## Loop and code to get the 

for index, row in df_results.head().iterrows():
    zip_df = df_fl.loc[(df_fl.RegionName==code), ['value']].resample('MS').mean()
    print(f'Zip code: {row["zip_code"]} \nCounty: {row["county"]}')
    hf.dynamic_prediction(df=zip_df, start_date='2018', end_date='2025', 
                               arima_model=row['model'], plot_interval=False, plot_df=False)

```

    Zip code: 34785 
    County: ['Sumter']
    


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_56_1.png)


    Zip code: 34652 
    County: ['Pasco']
    


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_56_3.png)


    Zip code: 34691 
    County: ['Pasco']
    


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_56_5.png)


    Zip code: 34488 
    County: ['Marion']
    


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_56_7.png)


    Zip code: 34690 
    County: ['Pasco']
    


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_56_9.png)



```
df_results.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>zip_code</th>
      <th>ROI</th>
      <th>model</th>
      <th>county</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>178</th>
      <td>32836</td>
      <td>0.051416</td>
      <td>&lt;statsmodels.tsa.statespace.sarimax.SARIMAXRes...</td>
      <td>[Orange]</td>
    </tr>
    <tr>
      <th>222</th>
      <td>33715</td>
      <td>0.040957</td>
      <td>&lt;statsmodels.tsa.statespace.sarimax.SARIMAXRes...</td>
      <td>[Pinellas]</td>
    </tr>
    <tr>
      <th>282</th>
      <td>34484</td>
      <td>0.014895</td>
      <td>&lt;statsmodels.tsa.statespace.sarimax.SARIMAXRes...</td>
      <td>[Sumter]</td>
    </tr>
    <tr>
      <th>95</th>
      <td>33626</td>
      <td>0.013987</td>
      <td>&lt;statsmodels.tsa.statespace.sarimax.SARIMAXRes...</td>
      <td>[Hillsborough]</td>
    </tr>
    <tr>
      <th>210</th>
      <td>33598</td>
      <td>-0.009668</td>
      <td>&lt;statsmodels.tsa.statespace.sarimax.SARIMAXRes...</td>
      <td>[Hillsborough]</td>
    </tr>
  </tbody>
</table>
</div>




```
for index, row in df_results.tail().iterrows():
    zip_df = df_fl.loc[(df_fl.RegionName==code), ['value']].resample('MS').mean()
    print(f'Zip code: {row["zip_code"]} \nCounty: {row["county"]}')
    hf.dynamic_prediction(df=zip_df, start_date='2018', end_date='2025', arima_model=row['model'],
                         plot_interval=False, plot_df=False)

```

    Zip code: 32836 
    County: ['Orange']
    


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_58_1.png)


    Zip code: 33715 
    County: ['Pinellas']
    


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_58_3.png)


    Zip code: 34484 
    County: ['Sumter']
    


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_58_5.png)


    Zip code: 33626 
    County: ['Hillsborough']
    


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_58_7.png)


    Zip code: 33598 
    County: ['Hillsborough']
    


![png](mod_4_starter_notebook_files/mod_4_starter_notebook_58_9.png)


## Conclusion

Based on the results of our modeling and testing, the best zip code to invest in is 34785 of Sumter County.
My recommmendation would be to purchase homes within the area code of 34785 of Sumter County.

## Future Works

In the future, I would like to implement a means to automatically optimize the models generated, to see if the best model maybe one that was not ranked as high as the top 5.
