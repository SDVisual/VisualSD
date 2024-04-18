

import datetime
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import cufflinks as cf
import matplotlib.pyplot as plt



StockInfo = {}
StockInfo_df = pd.DataFrame()

APP_NAME = "Visual Stock Data"

# Set page configuration
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="auto",)

color_code = "#0ECCEC"
header_html = f'<h1 style="color:{color_code};">{APP_NAME} </h1>'
st.markdown(header_html, unsafe_allow_html=True)

# # Display the image with the caption
# st.image('Logo.png')
# st.write("")
#
# # Define sidebar elements
# st.sidebar.image("Side.png", use_column_width=True)

# # Display title with blue color using Markdown
# st.markdown(f"<h1 style='color:blue;'>{APP_NAME}</h1>", unsafe_allow_html=True)

# Input box for user to enter symbol
new_symbol = st.text_input("Add Stock Symbol to Symbols List (e.g., AAPL)").strip().upper()


# Retrieve the last valid symbol entered by the user, default to 'AAPL' if none
DEFAULT_SYMBOL = st.session_state.valid_tickers[-1] if st.session_state.valid_tickers else 'AAPL'

# Retrieve the selected ticker index and symbol from the session state
selected_ticker_index = st.session_state.selected_ticker_index
selected_symbol = st.session_state.valid_tickers[selected_ticker_index]

# # Display the selected ticker index and symbol
# st.write("Selected ticker index:", selected_ticker_index)
# st.write("Selected symbol:", selected_symbol)

# Check if the entered symbol is empty or consists only of whitespace characters
if not new_symbol or new_symbol.isspace():
    new_symbol = DEFAULT_SYMBOL
# else:
#     if new_symbol in st.session_state.valid_tickers:
#         st.warning(f"'{new_symbol}' is already in Symbols List - Clear Text")


# Check if the entered symbol is valid
historical_data = yf.Ticker(new_symbol).history(period='1d')
income_statement = yf.Ticker(new_symbol).income_stmt

if new_symbol != DEFAULT_SYMBOL and historical_data.empty or income_statement.empty:
    st.error("Invalid symbol. Please enter Only Stocks symbols.")


else:
    if new_symbol not in st.session_state.valid_tickers:
        st.session_state.valid_tickers.append(new_symbol)
        st.text(f"Symbol Added to Symbols List - {new_symbol} ")
        # Update selected ticker index to the newly added symbol
        st.session_state.selected_ticker_index = len(st.session_state.valid_tickers) - 1



# Select box to choose ticker
ticker = st.sidebar.selectbox('Symbols List - Select Box', st.session_state.valid_tickers,
                              index=selected_ticker_index)

# Update session state with the newly selected symbol index
st.session_state.selected_ticker_index = st.session_state.valid_tickers.index(ticker)


# Display a message box in the sidebar
st.sidebar.info("- For the best experience, maximize your screen.")
st.sidebar.info("- Close side bar for better visualization.")
st.sidebar.info("- Recommended dark mode in setting menu.")
st.sidebar.markdown("&copy;Dan Oren. All rights reserved.", unsafe_allow_html=True)


# # Sidebar date inputs
# start_date = st.sidebar.date_input('Start date - Historical Prices', datetime.datetime(2021, 1, 1))
# end_date = st.sidebar.date_input('End date', datetime.datetime.now().date())


StockInfo = yf.Ticker(ticker).info

balance_sheetYear = yf.Ticker(ticker).balance_sheet
balance_sheetQuarterly = yf.Ticker(ticker).quarterly_balance_sheet

# Default to annual income statement
balance_sheet = balance_sheetYear

symbol = StockInfo["shortName"]
color_code = "#0ECCEC"  # Color for the symbol

st.write(f'<span style="font-size:30px;">Balance Sheet - </span>'
         f'<span style="color:{color_code}; font-size:30px;">{symbol}</span>', unsafe_allow_html=True)

# # Combine st.write() with HTML-styled header
# st.write(f'<span style="color:white; font-size:30px;">Balance Sheet - </span>'
#          f'<span style="color:{color_code}; font-size:30px;">{symbol}</span>', unsafe_allow_html=True)

st.write("")

# Checkbox layout
col1, col2, col3, col4 = st.columns(4)

# Checkbox to select between annual and quarterly
is_quarterly = col1.checkbox("Quarterly Balance Sheet", value=False)

# Checkbox to toggle display of extended balance sheet
is_extended = col1.checkbox("Extended Balance Sheet", value=False)

# # Checkbox to select between annual and quarterly
# is_quarterly = st.checkbox("Show Quarterly Balance Sheet", value=False)

# Update income statement based on the checkbox value
if is_quarterly:
    balance_sheet = balance_sheetQuarterly

# Define desired order for the first section
desired_order_first = [
    'Total Assets', 'Current Assets', 'Total Non Current Assets',
    'Cash Cash Equivalents And Short Term Investments',
    'Total Liabilities Net Minority Interest', 'Current Liabilities',
    'Total Non Current Liabilities Net Minority Interest', 'Total Equity Gross Minority Interest',
    'Total Capitalization', 'Common Stock Equity', 'Net Tangible Assets',
    'Working Capital', 'Invested Capital', 'Tangible Book Value',
    'Total Debt', 'Net Debt', 'Share Issued', 'Ordinary Shares Number',
    'Treasury Shares Number'
]

desired_order = [
    'Total Assets', 'Current Assets', 'Cash Cash Equivalents And Short Term Investments',
    'Cash And Cash Equivalents', 'Cash Financial',
    'Cash Equivalents', 'Other Short Term Investments', 'Receivables', 'Accounts Receivable', 'Other Receivables',
    'Inventory', 'Other Current Assets', 'Total Non Current Assets', 'Net PPE', 'Gross PPE',
    'Land And Improvements', 'Machinery Furniture Equipment', 'Leases', 'Accumulated Depreciation',
    'Investments And Advances',
    'Investmentin Financial Assets', 'Available For Sale Securities', 'Other Investments',
    'Non Current Deferred Assets', 'Non Current Deferred Taxes Assets',
    'Other Non Current Assets', 'Total Liabilities Net Minority Interest', 'Current Liabilities',
    'Payables And Accrued Expenses', 'Payables',
    'Accounts Payable', 'Current Debt And Capital Lease Obligation', 'Current Debt', 'Commercial Paper',
    'Other Current Borrowings', 'Current Deferred Liabilities',
    'Current Deferred Revenue', 'Other Current Liabilities', 'Total Non Current Liabilities Net Minority Interest',
    'Long Term Debt And Capital Lease Obligation',
    'Long Term Debt', 'Tradeand Other Payables Non Current', 'Other Non Current Liabilities',
    'Total Equity Gross Minority Interest', 'Stockholders Equity',
    'Capital Stock', 'Common Stock', 'Retained Earnings', 'Gains Losses Not Affecting Retained Earnings',
    'Other Equity Adjustments',
    'Total Capitalization', 'Common Stock Equity', 'Net Tangible Assets', 'Working Capital', 'Invested Capital',
    'Tangible Book Value',
    'Total Debt', 'Net Debt', 'Share Issued', 'Ordinary Shares Number', 'Treasury Shares Number'
]




st.write("<span style='font-size: 16px;'>* values in millions $</span>", unsafe_allow_html=True)



if is_extended:
    balance_sheet = balance_sheet.reindex(desired_order, fill_value='0')
else:
    balance_sheet = balance_sheet.reindex(desired_order_first, fill_value='0')

# balance_sheet = balance_sheet.reindex(desired_order_first, fill_value='0')
balance_sheet = balance_sheet.drop('Properties', errors='ignore')

# Convert values to millions
balance_sheet = balance_sheet.astype(float) / 1_000_000  # Divide by 1 million

# Convert column headers to datetime
balance_sheet.columns = pd.to_datetime(balance_sheet.columns)

# Sort the columns in ascending order of dates
balance_sheet = balance_sheet.sort_index(axis=1)

# Format the column headers to remove the timestamp
balance_sheet.columns = [col.strftime('%d/%m/%Y') for col in balance_sheet.columns]

# Drop rows where all cells are '0' or empty spaces
balance_sheet = balance_sheet[(balance_sheet != '0')].dropna()

# st.write(balance_sheet)

# Apply the formatting function to the balance sheet DataFrame
balance_sheet = balance_sheet.applymap(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)

# st.subheader(f"Balance Sheet for {StockInfo['shortName']} (In M$)")

# Apply styling to the balance sheet DataFrame
styled_balance_sheet = balance_sheet.style.set_table_styles([
    {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('border', '2px solid blue')]},
    {'selector': 'th, td', 'props': [('text-align', 'center'), ('border', '1px solid blue')]},
    {'selector': 'th', 'props': [('text-align', 'left')]}
])


st.write(balance_sheet)
st.write("")
st.write("")


# Define the color code for the "Chart Zone" text
color_code_chart_zone = "white"  # Example color code

# Display the styled header using st.write() with HTML
st.write(
    f'<span style="font-size:30px;">Chart Zone</span>',
    unsafe_allow_html=True
)
st.write(f'* All charts are interactive by clicking legend elements')
st.write(f'* values in millions $')


col1, col2 = st.columns([0.8, 0.2])  # Adjust the width ratio of col1 and col2 as needed



with col1:


    
    # Transpose the balance sheet DataFrame
    transposed_balance_sheet = balance_sheet.transpose()

    # Select relevant columns from the transposed balance sheet DataFrame
    selected_columns = ['Current Assets', 'Total Non Current Assets', 'Current Liabilities',
                        'Total Non Current Liabilities Net Minority Interest',
                        'Total Equity Gross Minority Interest']
    selected_data = transposed_balance_sheet[selected_columns]
    colors1 = ['#0080ff', '#5483b3', '#7da0ca', '#c1e8ff']  # Add more colors if needed
    # Define colors for each column
    colors = {'Current Assets': 'blue',
              'Total Non Current Assets': '#0080ff',
              'Current Liabilities': 'red',
              'Total Non Current Liabilities Net Minority Interest': 'lightcoral',
              'Total Equity Gross Minority Interest': '#5483b3'}

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces to the figure as bars
    for column in selected_data.columns:
        fig.add_trace(
            go.Bar(x=selected_data.index, y=selected_data[column], name=column, marker_color=colors[column]))

    # Update layout if needed
    fig.update_layout(
        title='Balance Sheet Elements Break Down',
        title_x=0.3,
        xaxis_title='',
        yaxis_title='Amount (M$)'
    )

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})




    # Define a function to convert strings with commas to float

    def str_to_float(value):
        try:
            return float(value.replace(',', ''))
        except ValueError:
            return np.nan


    # Apply the conversion function to all values in the DataFrame
    balance_sheet_numeric = balance_sheet.applymap(str_to_float)

    # Calculate percentage change for each metric between consecutive periods
    percentage_change_balance = balance_sheet_numeric.pct_change(axis=1) * 100

    # Convert the first column to numeric, coercing errors to NaN
    percentage_change_balance.iloc[:, 0] = pd.to_numeric(percentage_change_balance.iloc[:, 0], errors='coerce')

    # Replace NaN values with 0
    percentage_change_balance.iloc[:, 0] = percentage_change_balance.iloc[:, 0].fillna(0)


    st.write("")
    st.subheader(f"Assets & Liabilities")

# ******************************************   CHARTS 'Total Assets', 'Current Assets', 'Total Non Current Assets', ' Cash  & Cash Equivalents And Short Term Investments' ***************************************************

col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])



with col1:

    data_percentage_change_balance = percentage_change_balance.loc['Total Assets'].transpose()


    fig = go.Figure()

    # Add bar trace for total assets
    fig.add_trace(
        go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Total Assets'], name='Total Assets',
               marker_color="#0ECCEC"))

    # Add line trace for growth rate
    fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                             mode='lines+markers', name='Growth Rate', line=dict(color='red'), yaxis='y2'))

    # Add text annotations for growth rate values above the linear points
    for i, value in enumerate(data_percentage_change_balance.values):
        fig.add_annotation(
            x=data_percentage_change_balance.index[i],  # x-coordinate for annotation
            y=data_percentage_change_balance.values[i] + 0.7,  # Shift the text 0.05 above the point
            text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
            showarrow=False,  # whether to show arrow or not
            font=dict(size=15, color='black'),  # font properties for the annotation text
            bgcolor='yellow',
            yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
            align='left',  # alignment of the text
            xanchor='left',  # anchor point along x-axis for alignment

        )

    # Update layout
    fig.update_layout(title='Total Assets',
                      title_x=0.35,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount (M$)',
                      yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(x=0.4, y=1.15, xanchor='center', yanchor='top',
                                  orientation='h'))  # Set legend to horizontal orientation

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})






with col2:
    data_percentage_change_balance = percentage_change_balance.loc['Current Assets'].transpose()

    # Create a figure
    fig = go.Figure()

    # Add bar trace for total assets
    fig.add_trace(
        go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Current Assets'], name='Current Assets',
               marker_color="#0ECCEC"))

    # Add line trace for growth rate
    fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                             mode='lines+markers', name='Growth Rate', line=dict(color='red'), yaxis='y2'))

    # Add text annotations for growth rate values above the linear points
    for i, value in enumerate(data_percentage_change_balance.values):
        fig.add_annotation(x=data_percentage_change_balance.index[i],  # x-coordinate for annotation
                           y=data_percentage_change_balance.values[i] + 0.7,  # Shift the text 0.05 above the point
                           text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
                           showarrow=False,  # whether to show arrow or not
                           font=dict(color='black', size=15),  # color of the annotation text
                           bgcolor='yellow',
                           yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                           align='left',  # alignment of the text
                           xanchor='left')  # anchor point along x-axis for alignment

    # Update layout
    fig.update_layout(title='Current Assets',
                      title_x=0.4,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount',
                      yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(x=0.5, y=1.15, xanchor='center', yanchor='top',
                                  orientation='h'))  # Set legend to horizontal orientation

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col3:
    data_percentage_change_balance = percentage_change_balance.loc['Total Non Current Assets'].transpose()

    # Create a figure
    fig = go.Figure()

    # Add bar trace for total assets
    fig.add_trace(
        go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Total Non Current Assets'],
               name='Total Non Current Assets',
               marker_color="#0ECCEC"))

    # Add line trace for growth rate
    fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                             mode='lines+markers', name='Growth Rate', line=dict(color='red'), yaxis='y2'))

    # Add text annotations for growth rate values above the linear points
    for i, value in enumerate(data_percentage_change_balance.values):
        fig.add_annotation(x=data_percentage_change_balance.index[i],  # x-coordinate for annotation
                           y=data_percentage_change_balance.values[i] + 0.7,  # Shift the text 0.05 above the point
                           text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
                           showarrow=False,  # whether to show arrow or not
                           font=dict(color='black', size=15),  # color of the annotation text
                           bgcolor='yellow',
                           yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                           align='left',  # alignment of the text
                           xanchor='left')  # anchor point along x-axis for alignment

    # Update layout
    fig.update_layout(title='Total Non Current Assets',
                      title_x=0.35,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount (M$)',
                      yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(x=0.5, y=1.15, xanchor='center', yanchor='top',
                                  orientation='h'))  # Set legend to horizontal orientation

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ******************************************   CHARTS 'Total Liabilities Net Minority Interest', 'Current Liabilities' & MORE *************************************************


st.write("")

st.subheader(f"Cash & Debt ")
with col1:


    data_percentage_change_balance = percentage_change_balance.loc[
        'Total Liabilities Net Minority Interest'].transpose()
    # Create a figure
    fig = go.Figure()

    # Add bar trace for total assets
    fig.add_trace(
        go.Bar(x=data_percentage_change_balance.index,
               y=balance_sheet.loc['Total Liabilities Net Minority Interest'],
               name='Total Liabilities Net Minority Interest ',
               marker_color='red'))

    # Add line trace for growth rate
    fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                             mode='lines+markers', name='Growth Rate', line=dict(color='blue'), yaxis='y2'))

    # Add text annotations for growth rate values above the linear points
    for i, value in enumerate(data_percentage_change_balance.values):
        fig.add_annotation(x=data_percentage_change_balance.index[i],  # x-coordinate for annotation
                           y=data_percentage_change_balance.values[i] + 0.7,  # Shift the text 0.05 above the point
                           text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
                           showarrow=False,  # whether to show arrow or not
                           font=dict(color='black', size=15),  # color of the annotation text
                           bgcolor='yellow',
                           yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                           align='left',  # alignment of the text
                           xanchor='left')  # anchor point along x-axis for alignment

    # Update layout
    fig.update_layout(title='Total Liabilities Net Minority Interest',
                      title_x=0.25,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount (M$)',
                      yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(x=0.42, y=1.15, xanchor='center', yanchor='top',
                                  orientation='h'))  # Set legend to horizontal orientation

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col2:

    data_percentage_change_balance = percentage_change_balance.loc['Current Liabilities'].transpose()
    # Create a figure
    fig = go.Figure()

    # Add bar trace for total assets
    fig.add_trace(
        go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Current Liabilities'],
               name='Current Liabilities ',
               marker_color='red'))

    # Add line trace for growth rate
    fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                             mode='lines+markers', name='Growth Rate', line=dict(color='blue'), yaxis='y2'))

    # Add text annotations for growth rate values above the linear points
    for i, value in enumerate(data_percentage_change_balance.values):
        fig.add_annotation(x=data_percentage_change_balance.index[i],  # x-coordinate for annotation
                           y=data_percentage_change_balance.values[i] + 0.7,  # Shift the text 0.05 above the point
                           text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
                           showarrow=False,  # whether to show arrow or not
                           font=dict(color='black', size=15),  # color of the annotation text
                           bgcolor='yellow',
                           yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                           align='left',  # alignment of the text
                           xanchor='left')  # anchor point along x-axis for alignment

    # Update layout
    fig.update_layout(title='Current Liabilities',
                      title_x=0.35,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount (M$)',
                      yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(x=0.45, y=1.15, xanchor='center', yanchor='top',
                                  orientation='h'))  # Set legend to horizontal orientation

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col3:

    data_percentage_change_balance = percentage_change_balance.loc[
        'Total Non Current Liabilities Net Minority Interest'].transpose()
    # Create a figure
    fig = go.Figure()

    # Add bar trace for total assets
    fig.add_trace(
        go.Bar(x=data_percentage_change_balance.index,
               y=balance_sheet.loc['Total Non Current Liabilities Net Minority Interest'],
               name='Total Non Current Liabilities Net Minority Interest ',
               marker_color='red'))

    # Add line trace for growth rate
    fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                             mode='lines+markers', name='Growth Rate', line=dict(color='blue'), yaxis='y2'))

    # Add text annotations for growth rate values above the linear points
    for i, value in enumerate(data_percentage_change_balance.values):
        fig.add_annotation(x=data_percentage_change_balance.index[i],  # x-coordinate for annotation
                           y=data_percentage_change_balance.values[i] + 0.7,  # Shift the text 0.05 above the point
                           text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
                           showarrow=False,  # whether to show arrow or not
                           font=dict(color='black', size=15),  # color of the annotation text
                           bgcolor='yellow',
                           yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                           align='left',  # alignment of the text
                           xanchor='left')  # anchor point along x-axis for alignment

    # Update layout
    fig.update_layout(title='Total Non Current Liabilities',
                      title_x=0.35,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount (M$)',
                      yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(x=0.45, y=1.15, xanchor='center', yanchor='top',
                                  orientation='h'))  # Set legend to horizontal orientation

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})



col1, col2, col3 = st.columns([0.3, 0.3, 0.4])


with col1:
    data_percentage_change_balance = percentage_change_balance.loc[
        'Cash Cash Equivalents And Short Term Investments'].transpose()

    # Create a figure
    fig = go.Figure()

    # Add bar trace for total assets
    fig.add_trace(
        go.Bar(x=data_percentage_change_balance.index,
               y=balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'],
               name='Cash Cash Equivalents And Short Term Investments',
               marker_color='green'))

    # Add line trace for growth rate
    fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                             mode='lines+markers', name='Growth Rate', line=dict(color='red'), yaxis='y2'))

    # Add text annotations for growth rate values above the linear points
    for i, value in enumerate(data_percentage_change_balance.values):
        fig.add_annotation(x=data_percentage_change_balance.index[i],  # x-coordinate for annotation
                           y=data_percentage_change_balance.values[i] + 0.7,  # Shift the text 0.05 above the point
                           text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
                           showarrow=False,  # whether to show arrow or not
                           font=dict(color='black', size=15),  # color of the annotation text
                           bgcolor='yellow',
                           yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                           align='left',  # alignment of the text
                           xanchor='left')  # anchor point along x-axis for alignment

    # Update layout
    fig.update_layout(title='Cash & Cash Equivalents And Short Term Investments',
                      title_x=0.25,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount (M$)',
                      yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(x=0.5, y=1.15, xanchor='center', yanchor='top',
                                  orientation='h'))  # Set legend to horizontal orientation

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})





with col2:

    data_percentage_change_balance = percentage_change_balance.loc['Total Debt'].transpose()
    # Create a figure
    fig = go.Figure()

    # Add bar trace for total assets
    fig.add_trace(
        go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Total Debt'], name='Total Debt',
               marker_color='red'))

    # Add line trace for growth rate
    fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                             mode='lines+markers', name='Growth Rate', line=dict(color='blue'), yaxis='y2'))

    # Add text annotations for growth rate values above the linear points
    for i, value in enumerate(data_percentage_change_balance.values):
        fig.add_annotation(x=data_percentage_change_balance.index[i],  # x-coordinate for annotation
                           y=data_percentage_change_balance.values[i] + 0.7,  # Shift the text 0.05 above the point
                           text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
                           showarrow=False,  # whether to show arrow or not
                           font=dict(color='black', size=15),  # color of the annotation text
                           bgcolor='yellow',
                           yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                           align='left',  # alignment of the text
                           xanchor='left')  # anchor point along x-axis for alignment

    # Update layout
    fig.update_layout(title='Total Debt',
                      title_x=0.35,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount (M$)',
                      yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(x=0.35, y=1.15, xanchor='center', yanchor='top',
                                  orientation='h'))  # Set legend to horizontal orientation

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
