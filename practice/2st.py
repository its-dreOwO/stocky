import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

airline_data =  pd.read_csv(f'C:/Users/Acer/Desktop/study/ADy201m/practice/airline_data.csv', encoding = "ISO-8859-1", dtype={'Div1Airport': str, 'Div1TailNum': str, 'Div2Airport': str, 'Div2TailNum': str})
data = airline_data.sample(n=500, random_state=42)

line_data = data.groupby('Month')['ArrDelay'].mean().reset_index()

fig=go.Figure()
fig.add_trace(go.Scatter(x=line_data['Month'], y=line_data['ArrDelay'], mode='lines', marker=dict(color='green')))
fig.update_layout(title='Month vs Average Flight Delay Time', xaxis_title='Month', yaxis_title='ArrDelay')
fig.show()
