import os

from sklearn import metrics
from math import sqrt
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

from pyapnea.oscar.oscar_loader import load_session
from pyapnea.oscar.oscar_getter import event_data_to_dataframe
from pyapnea.oscar.oscar_constants import *

data_path = '../data/'
filename = '63c6e928.001'
value_channel = ChannelID.CPAP_FlowRate.value
oscar_session_data = load_session(os.path.join(data_path, filename))
df = event_data_to_dataframe(oscar_session_data, value_channel)
label = [item[5] for item in CHANNELS if item[1].value == value_channel][0]

# test NA values
print(df.isna().sum())

# fig1 = px.line(df, x="time_absolute", y=label)
# fig1.show()
# fig2 = px.box(df,y=label)
# fig2.show()

# predicting naive

df[label + '_t-1'] = df[label].shift(1)
df_naive = df[[label, label + '_t-1']][1:]

true = df_naive[label]
prediction = df_naive[label + '_t-1']
error = sqrt(metrics.mean_squared_error(true, prediction))
print('RMSE for Naive Method 1: ', error)

df[label + '_rm'] = df[label].rolling(3).mean().shift(1)
df_naive = df[[label, label + '_rm']].dropna()
true = df_naive[label]
prediction = df_naive[label + '_rm']
error = sqrt(metrics.mean_squared_error(true, prediction))
print('RMSE for Naive Method 2: ', error)

split = len(df) - int(0.2 * len(df))
train, test = df[label][0:split], df[label][split:]
test_idx = df['time_absolute'][split:]

# ARIMA
model = SARIMAX(train.values, order=(5, 1, 0))
model_fit = model.fit()

# do not understand predict (start=len(test) ??)
predictions = model_fit.predict(len(test))
test_ = pd.DataFrame(test)
test_['predictions'] = predictions[0:len(test)]
test_['time_absolute'] = test_idx

fig1 = px.line(df, x="time_absolute", y=label)
fig1.add_scatter(x=test_["time_absolute"], y=test_['predictions'], mode='lines')
fig1.show()

error = sqrt(metrics.mean_squared_error(test.values, predictions[0:len(test)]))
print('Test RMSE for SARIMAX: ', error)
