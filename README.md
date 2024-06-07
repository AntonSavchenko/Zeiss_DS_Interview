# ZEISS Interview Take-home-task


```python
# Python environment:
# python      v. 3.10.10
# jupyter lab v. 4.1.5
# pandas      v. 2.2.1
# numpy       v. 1.26.4
# scipy       v. 1.12.0
# matplotlib  v. 3.8.2
# seaborn     v. 0.13.2

import pandas as pd
import numpy as np

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"figure.figsize":(12, 8)})
```

# 0. Import data and initial observations
Since the problem setup does not provide in-depth explanation of this data, some initial exploration is required


```python
# First, we load the dataframe and make sure the datetime has the correct dtype
temps = pd.read_csv('data//sample_temperature_data_for_coding_challenge.csv',
                    parse_dates=['datetime'])

# inspect the contents of the dataframe
temps.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>source_id</th>
      <th>datetime</th>
      <th>property_name</th>
      <th>temperature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MICDEV001</td>
      <td>2019-04-13 17:51:16+00:00</td>
      <td>heating_temperature</td>
      <td>33.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MICDEV001</td>
      <td>2019-04-13 17:51:16+00:00</td>
      <td>cooling_temperature</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MICDEV001</td>
      <td>2019-04-13 18:51:18+00:00</td>
      <td>heating_temperature</td>
      <td>34.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# To quickly check the type of data and information contained in the column, we count unique elements in each column
for c in temps.columns:
    print(f'Column "{c}":\t {temps[c].nunique()} unique value(s),\t {temps[c].isnull().sum()} missing value(s),',end='\t')
    # in case there aren't many distinct entries in the column, we can print them out along with their value counts
    if temps[c].nunique()<5:
        print(temps[c].value_counts(dropna=False).to_dict())
    else:
        print()

# The outcome indicates that source_id is always constant, and property_name indicates either a heating or cooling temperature. There are no missing values.
```

    Column "source_id":	 1 unique value(s),	 0 missing value(s),	{'MICDEV001': 1000}
    Column "datetime":	 716 unique value(s),	 0 missing value(s),	
    Column "property_name":	 2 unique value(s),	 0 missing value(s),	{'heating_temperature': 699, 'cooling_temperature': 301}
    Column "temperature":	 172 unique value(s),	 0 missing value(s),	
    


```python
# To better visualize the data, let's plot the temperature values over the datetime, separating the cooling and heating temperatures by color
ax = sns.scatterplot(data=temps, x='datetime', y='temperature',hue='property_name', hue_order=['cooling_temperature','heating_temperature'])
ax.tick_params(axis='x', labelrotation=45)
```


    
![png](README_files/README_5_0.png)
    


### Initial observations:
- The dataset consists of 1000 temperature measurements, spread over the period of roughly one year (Apr. 2019 - Feb. 2020), with clearly visible gaps in the range of days or even months
- There are no NULL values, and the temperature readings are all in range of [14.9,39.4] degrees
- The "cooling_temperature" readings clearly divided in two groups - one with temperatures around 15 degrees, and another mostly in the range of [19,34] degrees
- It is not yet clear if this is an indication of an error or some limitation of the data collection

### Next steps:
- Collect more information about clustering of cooling_temperature data
- Investigate possible temporal dependencies in the data
- Look for ways to detect abnormal behavior in the data

# 1. Closer data inspection


```python
# Looking at the distribution of temperature readings overall, it seems that both cooling and heating temperatures follow a multi-modal distribution
# The value of 15 degrees is most probably used as some "default" value for the cooling temperature
sns.histplot(data=temps, x='temperature', hue='property_name', bins=list(range(14,41)), multiple='dodge', shrink=0.9, hue_order=['cooling_temperature','heating_temperature'])
```




    <Axes: xlabel='temperature', ylabel='Count'>




    
![png](README_files/README_8_1.png)
    



```python
# With temporal data we can expect some oscillating patterns, that could be observed at different frequencies.
# To look at it more closely, we will introduce a few derivative features from the timestamps

temps['weekday'] = temps['datetime'].dt.weekday   # Monday = 0  ... Sunday = 6
temps['month'] = temps['datetime'].dt.month       # January = 1 ... December = 12
temps['hour'] = temps['datetime'].dt.hour         # 0 .. 23

fig, axes = plt.subplots(2,2)
# because some months don't have any data, we need to hard-code the bin counts for the histogram, otherwise we could set
# bins = temps[c].nunique()
for i,(c,bins) in enumerate(zip(['weekday','month','hour'],[7,12,24])):
    sns.histplot(data=temps, x=c, hue='property_name',multiple='stack', bins=bins, ax=axes[i%2,i//2], hue_order=['cooling_temperature','heating_temperature'])
```


    
![png](README_files/README_9_0.png)
    


### Observations:
- temporal data has distinct "worktime" pattern - more data is present for the wordays and working hours
- cooling temperature almost exclusively collected during "worktime"
- monthly data shows less regular behavior, but we can notice more cooling data present in summer months (esp. May, July), which can indicate correlation with the temperature of the environment


```python
# some timestamps appear multiple times in the dataset, and it would be interesting to see what kind of data is collected for the same dataset
timestamps = temps.groupby('datetime')['source_id'].count().rename('count')

# we can add this count as another feature to the initial dataset to determine possible correlations
temps2 = temps.join(timestamps, on='datetime')
```


```python
# From these observations we can conclude that
# - cooling_temperature reading almost always comes together with the heating_temperature reading
# - heating_temperature reading can appear on its own
# - but cooling_temperature reading extremely rarely appears on its own
# This could be either a sign of an anomaly, or simply a processing delay
temps2.groupby(['count','property_name'])[['source_id']].count()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>source_id</th>
    </tr>
    <tr>
      <th>count</th>
      <th>property_name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">1</th>
      <th>cooling_temperature</th>
      <td>17</td>
    </tr>
    <tr>
      <th>heating_temperature</th>
      <td>415</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>cooling_temperature</th>
      <td>284</td>
    </tr>
    <tr>
      <th>heating_temperature</th>
      <td>284</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let us look closer at the frequency of data timestamps. First, we will check the whole dataset to determine the time delay between consecutive measurements
timestamps.sort_index(inplace=True)
tdeltas = (timestamps.index[1:]-timestamps.index[0:-1]).to_series(name='overall time deltas')
tdeltas.describe()
```




    count                          715
    mean     0 days 09:35:15.132867132
    std      3 days 04:56:39.136836443
    min                0 days 00:03:47
    25%                0 days 01:00:01
    50%                0 days 01:00:02
    75%                0 days 01:00:02
    max               70 days 22:50:27
    Name: overall time deltas, dtype: object




```python
# Now, since we only have 17 "suspicious" single cooling_temperature readings, we can just iterate over them and determine the closest time stamp
single_cooling_meas = temps2.query('property_name=="cooling_temperature" and count==1').copy()
single_cooling_meas.sort_values('temperature',inplace=True)

# Of course, we need to remove these timestamps from the overall list first (given that there are only 716 of them, such a brute-force approach is acceptable)
other_timestamps = np.array(list(set(timestamps.index).difference(single_cooling_meas['datetime'])))
for _,row in single_cooling_meas.iterrows():
    print(row['temperature'],'\t',min(abs(row['datetime']-other_timestamps)))
```

    14.9 	 0 days 01:00:01
    15.0 	 0 days 01:00:01
    15.0 	 0 days 01:00:02
    15.0 	 0 days 01:00:02
    15.0 	 0 days 02:00:03
    15.1 	 0 days 01:00:01
    15.1 	 0 days 01:00:02
    19.5 	 0 days 01:00:01
    22.1 	 0 days 00:55:44
    25.3 	 0 days 00:48:02
    29.0 	 0 days 00:36:34
    30.3 	 0 days 00:19:21
    30.8 	 0 days 00:07:33
    31.1 	 0 days 00:24:11
    31.7 	 0 days 00:18:47
    32.4 	 0 days 00:12:34
    32.5 	 0 days 00:42:24
    

### Observations:
- Overall, the vast majority of the datapoints are collected hourly, less than 10% of points are collected at a higher frequency
- Disproving my initial hypothesis, there are no cases of processing delay, shortest period between consequtive measurements is over 200s
- 8 of the 17 single cooling_temperature measurements appear at the correct frequency, so it is the heating temperature measurement that's missing in these cases
- There is a very clear correlation between the temperature reading and the "regularity" of the timestamp - my hypothesis is that above ~20 degrees there might be a triggering condition that prompts a new value for the cooling

# 2. Distributions of values
So far we primarily looked at frequencies of values separately from the values themselves.
As the next step in investigating the underlying structure we can consider the distributions of temperature readings themselves - though we have seen the overall histograms, it could be useful to find more granular visual representations


```python
# A useful tool for visualizations of this kind are "violin plots" that extend the standard candlestick plots with more detailed density representation of data

# First, let us consider the hourly distribution of the temperature readings within the dataset
sns.violinplot(data=temps, x='hour', y='temperature', hue='property_name', split=True, inner='stick',
               cut=0, density_norm='count', hue_order=['cooling_temperature','heating_temperature'])
```




    <Axes: xlabel='hour', ylabel='temperature'>




    
![png](README_files/README_17_1.png)
    



```python
# Similarly, we can plot the information depending on the weekday
sns.violinplot(data=temps, x='weekday', y='temperature', hue='property_name', split=True, gap=.1, inner='box',
               cut=0, density_norm='count', hue_order=['cooling_temperature','heating_temperature'])
```




    <Axes: xlabel='weekday', ylabel='temperature'>




    
![png](README_files/README_18_1.png)
    



```python
# Lastly, looking at the monthly data we can evaluate potential relation to the temperature of the environment
sns.violinplot(data=temps, x='month', y='temperature', hue='property_name', split=True, gap=.1, inner='box',
               cut=0, density_norm='count', hue_order=['cooling_temperature','heating_temperature'])
```




    <Axes: xlabel='month', ylabel='temperature'>




    
![png](README_files/README_19_1.png)
    


### Observations:
- These visualizations confirm previous observations: There is a large discrepancy between the data collected during working hours on workdays, and the data collected outside of these time frames
- Monthly data mainly shows the difference in the volume of data, but there doesn't seem to be a big yearly trend. The low volume of cooling data for the winter months could be explained by lower environment temperature, yet could also be just correlated with overall reduced volume of measurements
- The biggest trends can be observed in the hourly data - beyond the separation between working hours and the rest, one can spot different phases appearing throught the workday - the second mode in the heating_temperature only seem present in the morning hours and could be related to some warm-up phases of the underlying process. It could also be the potential "anomaly", that either stems from the overheating in the preceding hours or absence of some sensor readings before the workday start.

# 3. Modeling
- We have determined that the most consistent patterns arise in the intra-day data, and it would be interesting to see if we could determine even more structure in these data.
- To me the most interesting aspect would be determining if there is a significant level of correlation between the datapoints that are collected at consecutive time points.
- If that is not the case, and each datapoint seems independent from the one before and the one after, then the best predictive model would simply approximate the sampled distribution of the point at specific hour of the day.
- However, if there is a high degree of correlation between the temperature readings, we can generate a predictive model that could either impute missing values throughout the day, or even predict values we would expect in the evening hours from the values we see in the morning hours.


```python
# To make it easier to extract data from each day, we will add more derivative features to the dataframe

temps['time'] = temps['datetime'].dt.time           # datetime portion for time of day
temps['dayofyear'] = temps['datetime'].dt.dayofyear # Given that the data spans less than a year, extracting ordinal day of the year is enough

# Given that the data is mostly collected on an hourly basis, I would expect to have just one measurement of each type (cooling/heating) at each hour of each day.
# Ideally I would need to inspect the cases with multiple measurements more closely, but given the time constraints we can simply average those measurements
temps.groupby(['dayofyear','hour','property_name'])['source_id'].count().value_counts()
```




    source_id
    1    876
    2     50
    3      5
    4      1
    5      1
    Name: count, dtype: int64




```python
# As we have stated already, observed behavior varies greatly within working hours and outside of working hours.
# Let's see if we can determine the cause of this discrepancy by plotting the daily trajectories

heating_working_hours = temps.query('weekday<5 and property_name=="heating_temperature"')

# Now let's average "duplicates" that appear more than once within one hour
heating_working_hours = heating_working_hours.groupby(['dayofyear','hour'])['temperature'].mean().reset_index()
```


```python
# With that, we can quickly check how the daily trajectories look like
sns.lineplot(data=heating_working_hours, x='hour', y='temperature',hue='dayofyear')
```




    <Axes: xlabel='hour', ylabel='temperature'>




    
![png](README_files/README_24_1.png)
    


### Observation: 
It appears that there is a very clear pattern:
- if the heating was on for the night, the heating temperature will be set to ~35 degrees from 7AM
- if there were no heating measurements before the workday began, then at some point throughout the day the process starts, probably from the room temperature, and brought up to ~35 degrees over the course of ~4 hours

### Possible modeling approach:
- given that the trajectories seem so straightforward, I don't see the need for complex machinery such as Gaussian Processes
- we can separate the data into two clusters - the "warmed up" days vs "cold start" days based on the temperature readings at 7am
- for the "cold start" cluster there might be a simple curve fitting, matching the first heating temperature reading of the day to the follow-up measurements
- some trajectories show temperatures falling below 32.5 degrees, which could indicate either an end of the batch cycle or possible faults (could be external termination of the process)


```python
# To quickly select the "cold start" curves, we can simply select the dayofyear values for which heating temperature was below 30 degrees:
cold_start_days = temps.query('property_name=="heating_temperature" and temperature<30')['dayofyear'].unique()

# Now we only select those days and heating_temperature measurements, and then create a new column indicating the time passed since the first measurement of the day
cold_start_df = temps[(temps['dayofyear'].isin(cold_start_days)) & (temps['property_name']=='heating_temperature')].copy()
cold_start_df['time_from_start'] = cold_start_df['datetime'].sub(cold_start_df.groupby(['dayofyear'])['datetime'].transform('first'))
```


```python
# Now we have the daily curves that start from the point of the first measurement - it is still not ideal, since the starting temperature values are different
sns.lineplot(data=cold_start_df, x='time_from_start', y='temperature',hue='dayofyear')
```




    <Axes: xlabel='time_from_start', ylabel='temperature'>




    
![png](README_files/README_27_1.png)
    


### Curve fitting
These curves have a recognizable shape - it looks like a proportional gain feedback. Let us assume that the heating temperature is used for a controlled warming of the underlying process from whichever initial state to the target temperature. And it is done in the way, that the rate of change of the temperature at any time point should be proportional to the discrepancy between the current measurement and the target measurement, i.e.

$\dot{x}(t) = p\cdot (target - x(t))$

If we transform the temperature measurements to the error coordinates $e(t) = target - x(t)$, then the solution of this ODE is an exponential decay function of the form

$e(t) = a \cdot \exp(-b\cdot t)$

For $a$ and $b$ being positive constants.

Let us find a quick way of determining these parameters and check if it would fit the observed data.


```python
# First, let us select the trajectoried that we will use for curve fitting - I would propose taking those that start from a very low heating_temperature measurement:
selected_days = temps.query('property_name=="heating_temperature" and temperature<22.7')['dayofyear'].unique()

# we can now extract these days' data, converting the timedelta variables to seconds
fit_data = cold_start_df[cold_start_df['dayofyear'].isin(selected_days)][['time_from_start','temperature','dayofyear']]
fit_data['time_from_start'] = fit_data['time_from_start'].dt.total_seconds()

# one of the trajectories seems slightly off, which might indicate an outlier
sns.lineplot(data=fit_data, x='time_from_start', y='temperature',hue='dayofyear')

# Let's for now remove that day's trajectory to improve the fit
fit_data = fit_data[fit_data['dayofyear']!=191]
```


    
![png](README_files/README_29_0.png)
    



```python
# Next step is to estimate the value for the target temperature. We can, of course, just pick a value based on the visual inspection of the data,
# but we could also derive it from the dataset.

# Let's assume that the target temperature is a median of maximal temperature readings achieved by each day's trajectory
target = cold_start_df.groupby(['dayofyear'])['temperature'].max().median()
print(f'Target temperature is estimated as {target:0.2f} degrees.')
```

    Target temperature is estimated as 34.50 degrees.
    


```python
# Now we can utilize the curve_fit tool from scipy library
from scipy.optimize import curve_fit

(a,b),_ = curve_fit(lambda t,a,b: a*np.exp(-b*t),  fit_data['time_from_start'],target - fit_data['temperature'], p0 = (12,0.0001))

print(f'Fit exponential curve has the form: e(t) = {a:0.2f}*exp(-{b:0.5f}*t)')
```

    Fit exponential curve has the form: e(t) = 12.15*exp(-0.00018*t)
    


```python
# As the last step we can go back to all the data selected in "cold_start_days" and see if they fit the model.
# Since each trajectory's initial temperature reading is different, we would assume that it starts along the fitted trajectory, and see if it deviates later on.
# To calculate the starting point we can invert the fitted function as follows:
# target - heating_temperature[init] = a*exp(-b*t[init])    ===>    t[init] = - log((target - heating_temperature[init])/a)/b

def get_init_offset(temp,target=target, a=a,b=b):
    return -np.log((target - temp)/a)/b

cold_start_df['fit_time'] = cold_start_df['time_from_start'].dt.total_seconds() + cold_start_df.groupby(['dayofyear'])['temperature'].transform('first').apply(get_init_offset)
```


```python
# Now each day's trajectory is placed on the fitted line and we can check if they stay relatively close to it during the course of the day
sns.lineplot(data=cold_start_df, x='fit_time', y='temperature',hue='dayofyear',palette=sns.color_palette("light:b", as_cmap=True))

# Let's add the actual fitted model on top as a comparison
xmodel = np.linspace(-100,52000,1000)
ymodel = target - a*np.exp(-b*xmodel)
plt.plot(xmodel,ymodel,color='#ff7f0e',lw=4,ls='--')
```




    [<matplotlib.lines.Line2D at 0x12be3847fd0>]




    
![png](README_files/README_33_1.png)
    


### Final observations:
- The daily trajectories seem to closely follow the fitted curve in general
- Whether the increasing variance on the tail end of the trajectories is an indication of inaccurate model fit or physical effects of the underlying process is still unclear
- We haven't yet checked if there is a relation between these values and the cooling_temperatures, which could also explain the variance

