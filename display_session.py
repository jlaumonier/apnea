import os.path
from os import listdir, path
from os.path import isfile, join

import pandas as pd
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.express as px

from pyapnea.oscar.oscar_loader import load_session
from pyapnea.oscar.oscar_getter import event_data_to_dataframe
from pyapnea.oscar.oscar_constants import *

app = Dash(__name__)

#data_path = 'data/'
# TODO Rethink of date source
data_path = '/home/julien/OSCAR/Profiles/Julien/ResMed_23221085377/Events'
list_files = [{'label': f, 'value': f} for f in listdir(data_path) if isfile(join(data_path, f))]

list_channel_options = []

# read all files
global_df = pd.DataFrame(columns=['Col1'])
for f in list_files:
    oscar_session_data = load_session(os.path.join(data_path, f['value']))
    list_existing_channel_options = [channel.code for channel in oscar_session_data.data.channels]
    if ChannelID.CPAP_Obstructive.value in list_existing_channel_options:
        df = event_data_to_dataframe(oscar_session_data, ChannelID.CPAP_Obstructive.value)
        if global_df.empty:
            global_df = df
        else:
            global_df = pd.concat([global_df, df])

# change UTC to own local time - TODO should be in PyApnea
global_df['time_absolute'] = global_df['time_absolute'].dt.tz_localize('UTC')
global_df['local_time'] = global_df['time_absolute'].dt.tz_convert('America/Montreal')
global_df['hour'] = global_df['local_time'].dt.hour

fig_freq_hour = px.histogram(global_df, x="hour")

app.layout = html.Div(children=[
    html.H1(children='Oscar Apnea data visualization'),
    dash_table.DataTable(global_df.to_dict(orient='records'), [{"name": i, "id": i} for i in global_df.columns]),
    dcc.Graph(figure=fig_freq_hour),
    dcc.Dropdown(options=list_files, value=list_files[0]['value'], id='list_files'),
    dcc.Dropdown(options=list_channel_options, value=ChannelID.CPAP_Pressure.value, id='list_channel'),
    dcc.Graph(id='graph-data')
])




@app.callback(
    Output(component_id='list_channel', component_property='options'),
    Input(component_id='list_files', component_property='value')
)
def update_output(value):
    oscar_session_data = load_session(os.path.join(data_path, value))
    list_existing_channel_options = [channel.code for channel in oscar_session_data.data.channels]
    list_channel_options = [{'label': channel[5], 'value': channel[1].value} for channel in CHANNELS if
                            channel[1].value in list_existing_channel_options]
    return list_channel_options


@app.callback(
    Output(component_id='graph-data', component_property='figure'),
    Input(component_id='list_channel', component_property='value'),
    Input(component_id='list_files', component_property='value')
)
def update_output(value_channel, value_file):
    oscar_session_data = load_session(os.path.join(data_path, value_file))
    label = [item[5] for item in CHANNELS if item[1].value == value_channel][0]
    df = event_data_to_dataframe(oscar_session_data, value_channel)
    if df is not None:
        fig = px.line(df, x="time_absolute", y=label)
        if label+'2' in df.columns.to_list():
            fig.add_scatter(df, x=df["time_absolute"], y=[label+'2'], mode='lines')
        return fig
    else:
        print('No data')


if __name__ == '__main__':
    app.run_server(debug=True)
