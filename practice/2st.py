import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Read the airline data into pandas dataframe
airline_data =  pd.read_csv(f'C:/Users/Acer/Desktop/study/ADy201m/practice/airline_data.csv', encoding = "ISO-8859-1", dtype={'Div1Airport': str, 'Div1TailNum': str, 'Div2Airport': str, 'Div2TailNum': str})
data = airline_data.sample(n=500, random_state=42)

# # Group the data by Month and compute average over arrival delay time.
line_data = data.groupby('Month')['ArrDelay'].mean().reset_index()
# fig= go.Figure()
# fig

# fig.add_trace(go.Scatter(x=Month, y=ArrDelay, mode='lines', marker=dict(color='green')))
# fig.update_layout(title='Month vs Average Flight Delay Time', xaxis_title='Month', yaxis_title='ArrDelay')
# fig.show()

fig=go.Figure()
##Next we will create a line plot by using the add_trace function and use the go.scatter() function within it
# In go.Scatter we define the x-axis data,y-axis data and define the mode as lines with color of the marker as green
fig.add_trace(go.Scatter(x=line_data['Month'], y=line_data['ArrDelay'], mode='lines', marker=dict(color='green')))
# Create line plot here
## Here we update these values under function attributes such as title,xaxis_title and yaxis_title
fig.update_layout(title='Month vs Average Flight Delay Time', xaxis_title='Month', yaxis_title='ArrDelay')
fig.show()