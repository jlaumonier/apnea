import pandas as pd
import plotly.express as px
import datetime

from oscar_tools.oscar_loader import load_session, get_channel_from_code
from oscar_tools.schema import *

oscar_session_data = load_session('data/63c6e928.001')

channel = get_channel_from_code(oscar_session_data, ChannelID.CPAP_FlowRate)
# Testing plot. Here is the Flow Rata
gain = channel.events[0].gain
if channel.events[0].t8 == 0:
    channel.events[0].time = range(0, channel.events[0].evcount*int(channel.events[0].rate), int(channel.events[0].rate))
df = pd.DataFrame(data={'time': channel.events[0].time,
                        'data': channel.events[0].data})
df['data_gain'] = df['data'] * gain
df['time_absolute'] = df['time'] + channel.events[0].ts1
df['time_absolute'] = pd.to_datetime(df['time_absolute'], unit='ms')
fig = px.line(df, x="time_absolute", y="data_gain")
fig.show()
