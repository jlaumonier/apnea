from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px

from oscar_tools.oscar_loader import load_session, event_data_to_dataframe
from oscar_tools.schema import *

app = Dash(__name__)

oscar_session_data = load_session('data/63c6e928.001')
list_existing_channel_options = [channel.code for channel in oscar_session_data.data.channels]
list_channel_options = [{'label': channel[5], 'value': channel[1].value} for channel in CHANNELS if
                        channel[1].value in list_existing_channel_options]

app.layout = html.Div(children=[
    html.H1(children='Oscar Apnea data visualization'),
    dcc.Dropdown(options=list_channel_options, value=ChannelID.CPAP_Pressure.value, id='list_channel'),
    dcc.Graph(id='graph-data')
])


@app.callback(
    Output(component_id='graph-data', component_property='figure'),
    Input(component_id='list_channel', component_property='value')
)
def update_output(value):
    label = [item[5] for item in CHANNELS if item[1].value == value][0]
    df = event_data_to_dataframe(oscar_session_data, value)
    if df is not None:
        fig = px.line(df, x="time_absolute", y=label)
        return fig
    else:
        print('No data')


if __name__ == '__main__':
    app.run_server(debug=True)
