
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st
import datetime
from datetime import date, timedelta
from VolatilityAnalyzer import VolatilityAnalyzer


@st.cache_data
def load_data(stock, from_date=None, to_date=None):
    """
    Load historical data for a given cryptocurrency pair.

    Parameters:
    - stock (str): The cryptocurrency pair ('BTC-USDT', 'ETH-USDC', 'ETH-BTC').

    Returns:
    - DataFrame: A pandas DataFrame containing historical data with selected columns.

    Notes: 
    - Data for different pairs can be downloaded from 
    Example:
    >>> btc_data = load_data('BTC-USDT')
    >>> print(btc_data.head())
    """

    try:
         # Load data based on the selected cryptocurrency pair
        print(f"Loading {stock} Data")

        if stock == 'TOY':
            data = {
            'Date': pd.date_range(start='2022-01-01', periods=10, freq='D'),
            'Close': [10, 9, 10,9, 11, 11, 11, 9, 10, 10]
            }
            data = pd.DataFrame(data)
            return data

       
        data = pd.read_csv(f"data/Binance_{stock.replace('-', '')}_d.csv")
        
        # Reverse the order of data
        data = data[::-1]

        # Convert 'Date' column to datetime and set it as index
        data['Date'] = data['Date'].astype('datetime64[s]')
        s = pd.to_datetime(data['Date'], unit='s')
        data.index = s

        # Select relevant columns
        selected_columns = ['Open', 'High', 'Low', 'Close']
        data = data[selected_columns]

        if from_date is not None: 
            print(f'Before from_date filtering: {data.shape} rows')        
            data = data[from_date:]
            print(f'After from_date filtering @ {from_date}, leaves {data.shape} rows')

        if to_date is not None:
            print(f'Before to_date filtering: {data.shape} rows')  
            data = data[:to_date]
            print(f'After to_date filtering @ {to_date}, leaves {data.shape} rows')
        
        
        print(data.head())
        return data
    except Exception as ex:
        print(ex)
        return None

def visualize(selected_pool, data, lookback, selected_price, vol_name):
    """
    Visualizes trend predictions based on historical data.

    Parameters:
    - selected_pool (str): The cryptocurrency pair (e.g., 'BTC-USDT').
    - data (DataFrame): Historical data with trend predictions.
    - lookback (int): Number of historical values considered for trend prediction.
    - selected_price (str): The price attribute for visualization (e.g., 'Close').

    Returns:
    - None: Displays an interactive plot.

    Example:
    >>> visualize(selected_pool='BTC-USDT', data=df, lookback=30, selected_price='Close')
    """
    # Create a reference line for the original prices
    print(f"My Hover data is {[vol_name]}")
    reference_line = go.Scatter(x=data.index,
                                y=pd.Series(data[selected_price]),
                                mode="lines",
                                line=go.scatter.Line(color="gray"),
                                showlegend=True, name=f"{selected_price} Price")
    
    # Create an interactive scatter plot for trend visualization
    layout = go.Layout(title=f"{vol_name.capitalize()} Volatility of {selected_pool} based on historic {lookback} values")
    #fig = go.Figure()
    
    # Create an interactive scatter plot for trend visualization
    # fig = (px.scatter(data, x=data.index, y=data[selected_price], 
    #             hover_data=[vol_name], size=data[vol_name],               
    #             title=f"{vol_name.capitalize()} Volatility of {selected_pool} based on historic {lookback} values")
    # .update_layout(title_font_size=24)
    # .update_xaxes(showgrid=True)
    # .update_traces(
    #     line=dict(dash="dot", width=4),
    #     selector=dict(type="scatter", mode="lines"))
    # )
    # Add scatter trace
    scatter_trace = go.Scatter(x=data.index, 
                           y=data[selected_price], 
                           mode='markers', 
                           marker=dict(symbol='circle', color='blue', size=abs(data[vol_name])*10),
                           hoverinfo='text',
                           text=[f"{vol_name}: {vol}" for vol in data[vol_name]],
                           name=f"{vol_name} Volatility",
                          )

    # Add the reference line to the plot
    #fig.add_trace(reference_line, row=1, col=1)
    #fig.add_trace(ma_line, row=1, col=1)
    
    # Update layout
    # fig.update_layout(
    #     xaxis=dict(title='Date'),
    #     yaxis=dict(title=selected_price),
    #     legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    # )

    #fig.add_trace(reference_line)
    #fig.add_trace(scatter_trace)

    # Update layout
    # fig.update_layout(
    #     title=f"{vol_name.capitalize()} Volatility of {selected_pool} based on historic {lookback} values",
    #     title_font_size=24,
    #     xaxis=dict(title='Date'),
    #     yaxis=dict(title=selected_price),
    #     showlegend=True
    # )

    fig = go.Figure(data=[reference_line, scatter_trace], layout=layout)

    # Show the interactive plot
    return fig


if __name__ == '__main__':
    START = None
    END = None
    min_lookback_period = 7
    va = VolatilityAnalyzer()

    st.title("Volatility Prediction APP")
    pools = ("BTC-USDT", "ETH-USDC", "ETH-BTC", "ETH-USDT")
    pool_start_date=(datetime.date(2017, 8, 17), datetime.date(2018, 12, 15), datetime.date(2017, 7, 14), datetime.date(2017, 8, 17))
    prices = ("Close", "Open", "High", "Low")
    #strategy = ("MA", "EMA")

    #selected_pool = 0
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        selected_pool = st.selectbox("Select Pair", pools, index=0)
        #symbol = st.selectbox('Choose stock symbol', options=['AAPL', 'MSFT', 'GOOG', 'AMZN'], index=1)
    with c2:
        if selected_pool is None:
            idx = 0
        else:
            idx = pools.index(selected_pool)  # Use pools.index to get the index of selected_pool

        START = st.date_input("Start-Date", min_value=pool_start_date[idx], value=pool_start_date[idx])
    with c3:
        if selected_pool is None:
            idx = 0
        else:
            idx = pools.index(selected_pool)  # Use pools.index to get the index of selected_pool

        END = st.date_input("End-Date", min_value=pool_start_date[idx]+timedelta(days=min_lookback_period), max_value=datetime.date(2024, 2, 29), value=datetime.date(2024, 2, 29))


    st.markdown('---')

    st.sidebar.subheader('Settings')
    st.sidebar.caption('Adjust charts settings and then press apply')

    with st.sidebar.form('settings_form'):        
        #strategy = st.selectbox("Select Strategy", strategy, index=0)
        
        selected_price = st.selectbox("Select Price", prices, index=0)

        lookback = st.slider("Lookback days (Long Term)", min_value=7, max_value=365, value=15)

        #threshold = st.slider("Threshold", min_value=0.0, max_value=0.3, value=0.05)

        show_data = st.checkbox('Show data table', False)
        st.form_submit_button('Apply')
    
    # Load historical data
    df = load_data(selected_pool, from_date=START, to_date=END)
    

    # # Perform trend prediction
    if df is not None:
        df_result_stat = va.get_statistical_volatility(data=df, lookback=lookback, selected_price=selected_price)
        
        df_result_spec = va.get_spectral_volatility(data=df, lookback=lookback, selected_price=selected_price)

        df = pd.merge(df_result_spec, df_result_stat, on=['Date', 'Open','Close', 'Low', 'High'], how='inner')
        print(df.tail())
        #df_result = predict_trend(df, strategy=strategy, lookback=lookback, selected_price=selected_price, threshold=threshold)
        #print(df_result.head())
        fig = visualize(selected_pool=selected_pool, data = df, lookback=lookback, vol_name="Statistical", selected_price=selected_price )
        st.plotly_chart(fig)

        fig1 = visualize(selected_pool=selected_pool, data = df, lookback=lookback, vol_name="Spectral", selected_price=selected_price )
        st.plotly_chart(fig1)


    else:
        print('Data not available for the selected pool')
    
    if show_data:
        st.markdown('---')
        st.dataframe(df.tail(50))