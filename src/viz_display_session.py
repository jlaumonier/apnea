import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from tqdm import tqdm

from src.data.datasets.processed_dataset import ProcessedDataset
from src.data.datasets.raw_oscar_dataset import RawOscarDataset
from src.data.datasets.slpdb_dataset import SLPDB_Dataset
from src.data.utils import get_annotations_ends

app = Dash(__name__)

#processed_dataset = ProcessedDataset(data_path='../test/data/processing/windowed', output_type='dataframe')
processed_dataset = SLPDB_Dataset(data_path='../data/raw-slpdb/physionet.org/files/slpdb/1.0.0', output_type='dataframe', limits=slice(1, 2, None))
#processed_dataset = RawOscarDataset(data_path='../test/data/raw/', output_type='dataframe')
index_dataset = {}

list_df_annot = []
with tqdm(total=len(processed_dataset), position=0, leave=False, colour='red', ncols=80) as pbar:
    for idx_df, df in enumerate(processed_dataset):
        index_dataset[str(df.index[0])] = idx_df
        df_annotation = get_annotations_ends(df)
        df_annotation = df_annotation[['Event', 'ApneaEvent']]
        list_df_annot.append(df_annotation)
        pbar.update(1)

df_annotations = pd.concat(list_df_annot)
df_annotations.reset_index(inplace=True)
list_index_keys = list(index_dataset.keys())
list_index_keys.sort()

app.layout = html.Div(children=[
    html.H1(children='Oscar Apnea data visualization'),
    # dcc.Input(id="session_id", value=0, type="number", min=0, max=len(processed_dataset)),
    dcc.Dropdown(options=list_index_keys, value=list_index_keys[0], id='list_index'),
    dash_table.DataTable(df_annotations.to_dict(orient='records'),
                         [{"name": i, "id": i} for i in df_annotations.columns]),
    #dash_table.DataTable(id='df'),
    dcc.Graph(id='graph-data')
])


def df_to_fig(df: pd.DataFrame):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df.index, y=df['FlowRate'], name="FlowRate", mode='lines'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['ApneaEvent'], name="ApneaEvent", mode='lines'),
        secondary_y=True,
    )
    fig.update_yaxes(range=[-1, 1], secondary_y=True)
    return fig


@app.callback(
    Output(component_id='graph-data', component_property='figure'),
    #Output(component_id='df', component_property='data'),
    #Output(component_id='df', component_property='columns'),
    Input(component_id='list_index', component_property='value')
)
def update_output(index):
    session_id = index_dataset[index]
    df = processed_dataset[session_id]
    fig = df_to_fig(df)
    df.reset_index(inplace=True)
    # data = df.to_dict(orient='records')
    # cols = [{"name": i, "id": i} for i in df.columns]

    return fig #, data, cols


if __name__ == '__main__':
    app.run_server(debug=True)
