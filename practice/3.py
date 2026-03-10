import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Read the airline data into pandas dataframe
airline_data =  pd.read_csv(f'C:/Users/Acer/Desktop/study/ADy201m/practice/airline_data.csv', encoding = "ISO-8859-1", dtype={'Div1Airport': str, 'Div1TailNum': str, 'Div2Airport': str, 'Div2TailNum': str})
data = airline_data.sample(n=500, random_state=42)
bar_data = data.groupby(['DestState'])['Flights'].sum().reset_index()

fig = px.bar( x=bar_data['DestState'], y=bar_data['Flights'], title='Total numbers of flights to the destination state splilt by reporting air')
fig.update_layout(xaxis_title='DestState', yaxis_title='Flights')

fig.show()