# Databricks notebook source
import json
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from prophet.plot import plot_plotly, plot_components_plotly, add_changepoints_to_plot
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

# COMMAND ----------

OSFcast=spark.read.parquet('/mnt/AFcovid19/AFAnalytics/MLPipeline/Fcast/')



# COMMAND ----------

# OSFcast=OSFcast[["TimePeriod","Available"]]

# COMMAND ----------

# OSFcast = OSFcast.withColumnRenamed("Available","SeatsAvailable")
           


# COMMAND ----------

# OSFcast.write.mode('overwrite').parquet('/mnt/AFcovid19/AFAnalytics/MLPipeline/Fcast/')

# COMMAND ----------

from pyspark.sql.functions import year
from pyspark.sql.functions import to_date
from pyspark.sql import SparkSession	
from pyspark.sql.functions import *

# COMMAND ----------

#Convert to Pandas
pdf=OSFcast.toPandas()

# COMMAND ----------

pdf.head()

# COMMAND ----------

 #feature engineering
pdf['TimePeriod'] = pd.to_datetime(pdf['TimePeriod'])

# COMMAND ----------

#Selecet columns that matters
dfa=pdf[['TimePeriod','SeatsAvailable']]

# COMMAND ----------

#Prophet feature engineering
dfa= dfa.rename(columns={'TimePeriod': 'ds','SeatsAvailable': 'y'})

# COMMAND ----------

dfa

# COMMAND ----------

dfa=dfa.groupby(['ds'],as_index=False)['y'].sum()

# COMMAND ----------

dfa

# COMMAND ----------

# x=dfa.groupby(['ds']).size().reset_index(name='count')



# COMMAND ----------

#You cant forecast where a category has less than 2 rows so filter them out
# y=x[x['count'] <2]

# COMMAND ----------

# y

# COMMAND ----------

# fil_os = y.OSName.unique()

# COMMAND ----------

# fil_os

# COMMAND ----------

# os=['Primary','Control','Duty']



# COMMAND ----------


# df2=dfa[~dfa['OSName'].isin(fil_os)]
# df2=df2[df2['OSName'].isin(os)]

# COMMAND ----------

df2=dfa.copy()

# COMMAND ----------

#Prophet Model training in loop. It also saves all forecast results



fcsta_all = pd.DataFrame()
plots_a={}
components_a={}

    
ma = Prophet(changepoint_prior_scale=0.09)
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
for i, month in enumerate(months):
    df2[month] = (df2['ds'].dt.month == i + 1).values.astype('float')
    ma.add_regressor(month)

ma.fit(df2 )
future_a= ma.make_future_dataframe(periods=365)
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
for i, month in enumerate(months):
       future_a[month] = (future_a['ds'].dt.month == i + 1).values.astype('float')

forecast_a= ma.predict(future_a)
fcsta_all = pd.concat([fcsta_all, forecast_a], ignore_index=True)
fig=plot_plotly(ma, forecast_a)
fig1=plot_components_plotly(ma,forecast_a)

# COMMAND ----------

fig.update_layout(showlegend=True,title='Trend and Forecast of Seats Availability',autosize=False,
width=1700,
height=600,template='seaborn')
labels_to_show_in_legend = ["Predicted", "Actual"]

for trace in fig['data']: 
    if (not trace['name'] in labels_to_show_in_legend):
        trace['showlegend'] = False

fig.show()

fig1.update_layout(title='Trend and Forecast of Seats Availability',autosize=False,
width=1700,
height=800,template ='plotly_dark')
fig1.show()

# COMMAND ----------

import plotly.graph_objs as go
fig=go.Figure()



    
dff= fcsta_all.copy()
dfy= df2.copy()

fig.add_trace(go.Scatter(
name = 'predicted trend',
mode = 'lines',
x = list(dff['ds']),
y = list(dff['yhat']),
marker=dict(
    color='red',
    line=dict(width=3))))

fig.add_trace(go.Scatter(
name = 'upper band',
mode = 'lines',
x = list(dff['ds']),
y = list(dff['yhat_upper']),
line= dict(color='#57b88f'),
fill = 'tonexty'))

fig.add_trace(go.Scatter(
name= 'lower band',
mode = 'lines',
x = list(dff['ds']),
y = list(dff['yhat_lower']),
line= dict(color='#1705ff')))

fig.add_trace(go.Scatter(
name = 'Actual',
mode = 'markers',
x = list(dfy['ds']),
y = list(dfy['y']),
marker=dict(
  color='pink',
  line=dict(width=2))))


    

fig.update_layout(title='Trend and Forecast of Seats Availability',showlegend=True,template='plotly_dark')
fig.show()


# COMMAND ----------


