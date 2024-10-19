
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
    initial_sidebar_state="auto" ,)

color_code = "#0ECCEC"
header_html = f'<h2 style="color:{color_code};">{APP_NAME} </h2>'
st.markdown(header_html, unsafe_allow_html=True)

col1, col2 = st.columns([0.35, 0.65])

with col1:
    
    # Input box for user to enter symbol
    new_symbol = st.text_input("Add symbol to Symbols List (e.g., AAPL)", placeholder="Search Stocks Symbols").strip().upper()


# Retrieve the last valid symbol entered by the user, default to 'AAPL' if none
DEFAULT_SYMBOL = st.session_state.valid_tickers[-1] if st.session_state.valid_tickers else 'AAPL'

# Retrieve the selected ticker index and symbol from the session state
selected_ticker_index = st.session_state.selected_ticker_index
selected_symbol = st.session_state.valid_tickers[selected_ticker_index]


# Check if the entered symbol is empty or consists only of whitespace characters
if not new_symbol or new_symbol.isspace():
    new_symbol = DEFAULT_SYMBOL


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

st.sidebar.info("- Design Your Own Charts.")
st.sidebar.info("- Compare Balance Sheets Between Your Symbols.")
st.sidebar.info("- Easy Download Data Tables.")
st.sidebar.info("- For the best experience, maximize your screen.")
# st.sidebar.info("- Recommended dark mode in setting menu.")
st.sidebar.info("- This app version is less suitable for stocks in the finance industry")
st.sidebar.markdown("&copy;VisualSD. All rights reserved.", unsafe_allow_html=True)



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

balance_sheet_Design = balance_sheet.reindex(desired_order, fill_value='0')


if is_extended:
    balance_sheet = balance_sheet.reindex(desired_order, fill_value='0')
else:
    balance_sheet = balance_sheet.reindex(desired_order_first, fill_value='0')


balance_sheet = balance_sheet.drop('Properties', errors='ignore')

# Convert values to millions
balance_sheet = balance_sheet.astype(float) / 1_000_000  # Divide by 1 million

# Convert column headers to datetime
balance_sheet.columns = pd.to_datetime(balance_sheet.columns)

# Sort the columns in ascending order of dates
balance_sheet = balance_sheet.sort_index(axis=1)

# Format the column headers to remove the timestamp
balance_sheet.columns = [col.strftime('%d/%m/%Y') for col in balance_sheet.columns]

# Show only the latest 4 dates
balance_sheet = balance_sheet.iloc[:, -4:]
# Replace "None" with 0
balance_sheet = balance_sheet.fillna(0)


# Apply the formatting function to the balance sheet DataFrame
balance_sheet = balance_sheet.applymap(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)

# st.subheader(f"Balance Sheet for {StockInfo['shortName']} (In M$)")

# Apply styling to the balance sheet DataFrame
styled_balance_sheet = balance_sheet.style.set_table_styles([
    {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('border', '2px solid blue')]},
    {'selector': 'th, td', 'props': [('text-align', 'center'), ('border', '1px solid blue')]},
    {'selector': 'th', 'props': [('text-align', 'left')]}
])

st.dataframe(styled_balance_sheet)

st.write("")
st.write("")


####################################### COMPARE Income Elements #################################


col1, col2 = st.columns([0.7, 0.3])


with col1:
    # Initialize session state variables if they don't exist
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = None
    if 'comparison_table_visible' not in st.session_state:
        st.session_state.comparison_table_visible = False
    if 'valid_tickers' not in st.session_state:
        st.session_state.valid_tickers = []

    # Define the balance sheet elements to compare
    balance_sheet_elements = [
        ('Choose All', 'all', None),  # Option to choose all elements
        ('Total Assets', 'Total Assets', 'Billions'),
        ('Current Assets', 'Current Assets', 'Billions'),
        ('Cash Cash Equivalents And Short Term Investments', 'Cash Cash Equivalents And Short Term Investments',
         'Billions'),
        ('Cash And Cash Equivalents', 'Cash And Cash Equivalents', 'Billions'),
        ('Cash Financial', 'Cash Financial', 'Billions'),
        ('Cash Equivalents', 'Cash Equivalents', 'Billions'),
        ('Other Short Term Investments', 'Other Short Term Investments', 'Billions'),
        ('Receivables', 'Receivables', 'Billions'),
        ('Accounts Receivable', 'Accounts Receivable', 'Billions'),
        ('Other Receivables', 'Other Receivables', 'Billions'),
        ('Inventory', 'Inventory', 'Billions'),
        ('Other Current Assets', 'Other Current Assets', 'Billions'),
        ('Total Non Current Assets', 'Total Non Current Assets', 'Billions'),
        ('Net PPE', 'Net PPE', 'Billions'),
        ('Gross PPE', 'Gross PPE', 'Billions'),
        ('Land And Improvements', 'Land And Improvements', 'Billions'),
        ('Machinery Furniture Equipment', 'Machinery Furniture Equipment', 'Billions'),
        ('Leases', 'Leases', 'Billions'),
        ('Accumulated Depreciation', 'Accumulated Depreciation', 'Billions'),
        ('Investments And Advances', 'Investments And Advances', 'Billions'),
        ('Investmentin Financial Assets', 'Investmentin Financial Assets', 'Billions'),
        ('Available For Sale Securities', 'Available For Sale Securities', 'Billions'),
        ('Other Investments', 'Other Investments', 'Billions'),
        ('Non Current Deferred Assets', 'Non Current Deferred Assets', 'Billions'),
        ('Non Current Deferred Taxes Assets', 'Non Current Deferred Taxes Assets', 'Billions'),
        ('Other Non Current Assets', 'Other Non Current Assets', 'Billions'),
        ('Total Liabilities Net Minority Interest', 'Total Liabilities Net Minority Interest', 'Billions'),
        ('Current Liabilities', 'Current Liabilities', 'Billions'),
        ('Payables And Accrued Expenses', 'Payables And Accrued Expenses', 'Billions'),
        ('Payables', 'Payables', 'Billions'),
        ('Accounts Payable', 'Accounts Payable', 'Billions'),
        ('Current Debt And Capital Lease Obligation', 'Current Debt And Capital Lease Obligation', 'Billions'),
        ('Current Debt', 'Current Debt', 'Billions'),
        ('Commercial Paper', 'Commercial Paper', 'Billions'),
        ('Other Current Borrowings', 'Other Current Borrowings', 'Billions'),
        ('Current Deferred Liabilities', 'Current Deferred Liabilities', 'Billions'),
        ('Current Deferred Revenue', 'Current Deferred Revenue', 'Billions'),
        ('Other Current Liabilities', 'Other Current Liabilities', 'Billions'),
        ('Total Non Current Liabilities Net Minority Interest', 'Total Non Current Liabilities Net Minority Interest',
         'Billions'),
        ('Long Term Debt And Capital Lease Obligation', 'Long Term Debt And Capital Lease Obligation', 'Billions'),
        ('Long Term Debt', 'Long Term Debt', 'Billions'),
        ('Tradeand Other Payables Non Current', 'Tradeand Other Payables Non Current', 'Billions'),
        ('Other Non Current Liabilities', 'Other Non Current Liabilities', 'Billions'),
        ('Total Equity Gross Minority Interest', 'Total Equity Gross Minority Interest', 'Billions'),
        ('Stockholders Equity', 'Stockholders Equity', 'Billions'),
        ('Capital Stock', 'Capital Stock', 'Billions'),
        ('Common Stock', 'Common Stock', 'Billions'),
        ('Retained Earnings', 'Retained Earnings', 'Billions'),
        ('Gains Losses Not Affecting Retained Earnings', 'Gains Losses Not Affecting Retained Earnings', 'Billions'),
        ('Other Equity Adjustments', 'Other Equity Adjustments', 'Billions'),
        ('Total Capitalization', 'Total Capitalization', 'Billions'),
        ('Common Stock Equity', 'Common Stock Equity', 'Billions'),
        ('Net Tangible Assets', 'Net Tangible Assets', 'Billions'),
        ('Working Capital', 'Working Capital', 'Billions'),
        ('Invested Capital', 'Invested Capital', 'Billions'),
        ('Tangible Book Value', 'Tangible Book Value', 'Billions'),
        ('Total Debt', 'Total Debt', 'Billions'),
        ('Net Debt', 'Net Debt', 'Billions'),
        ('Share Issued', 'Share Issued', 'Billions'),
        ('Ordinary Shares Number', 'Ordinary Shares Number', 'Billions'),
        ('Treasury Shares Number', 'Treasury Shares Number', 'Billions')
    ]


    # Define function to display comparison table
    def display_comparison_table(selected_symbols, selected_elements):
        # Check if there are enough symbols for comparison
        if len(selected_symbols) < 2:
            st.warning("Not enough symbols for comparison. Please add more symbols.")
            return

        # Initialize a list to store DataFrames
        data_frames = []

        # Loop through selected tickers and fetch the stock information
        for ticker in selected_symbols:
            stock = yf.Ticker(ticker)
            balance_sheet_compare = stock.balance_sheet


            # Get the last available date
            last_date = balance_sheet_compare.columns[0]

            # Collect the elements for the current ticker
            data = {'Ticker': ticker, 'Date': last_date.strftime('%Y-%m-%d')}
            for elem in balance_sheet_elements:
                if elem[1] == 'all':
                    continue
                value = balance_sheet_compare.at[elem[1], last_date] if elem[1] in balance_sheet_compare.index else None

                # Check if value is not None and not NaN, otherwise set it to 0
                if value is not None and not (isinstance(value, float) and value != value):
                    if elem[2] == 'Millions':
                        value = f"{value / 1_000_000:.2f}M"
                    elif elem[2] == 'Billions':
                        value = f"{value / 1_000_000_000:.2f}B"
                    elif elem[2] == '2 decimals':
                        value = f"{value:.2f}"
                    elif elem[2] == 'percentage':
                        value = f"{value * 100:.2f}%"
                else:
                    value = "0"

                data[elem[0]] = value

            # Append the DataFrame to the list
            data_frames.append(pd.DataFrame([data]))

        # Concatenate all DataFrames in the list
        comparison_df = pd.concat(data_frames, ignore_index=True)

        # Set 'Ticker' as the index of the DataFrame
        comparison_df.set_index('Ticker', inplace=True)

        # Display the comparison table
        st.write("Comparison Table Of Annual Balance Sheet Elements:")
        st.write("")
        if 'Choose All' in selected_elements:
            selected_elements = [elem[0] for elem in balance_sheet_elements if elem[0] != 'Choose All']
        st.dataframe(comparison_df[['Date'] + selected_elements])



    # Check if the visibility flag is set to True and the user clicks the button
    if st.button("Compare Annual Balance Sheets Between Your Symbols"):
        if 'comparison_table_visible' not in st.session_state:
            st.session_state.comparison_table_visible = True

        st.session_state.comparison_table_visible = not st.session_state.comparison_table_visible

        # Check if there are enough symbols for comparison
        if len(st.session_state.valid_tickers) < 2:
            st.warning("Not enough symbols to compare. Please add symbols to your list.")
            st.session_state.comparison_table_visible = False

    # Check if ticker has changed
    if st.session_state.current_ticker != ticker:
        st.session_state.current_ticker = ticker
        st.session_state.comparison_table_visible = False

    # Check if the visibility flag is set to True and the user switches symbols in the list
    if st.session_state.get('comparison_table_visible', False):

        # Create a dropdown list with multiple selection for choosing symbols to compare
        all_symbols_option = 'All Symbols'
        selected_symbols = st.multiselect("Select symbols to compare:",
                                          [all_symbols_option] + st.session_state.valid_tickers)

        # If "All Symbols" is selected, use all available symbols
        if all_symbols_option in selected_symbols:
            selected_symbols = st.session_state.valid_tickers

        # Create a dropdown list with multiple selection for choosing elements to compare
        selected_elements = st.multiselect("Select elements to compare:",
                                           [elem[0] for elem in balance_sheet_elements])

        # Check if the user has selected at least one symbol and one element
        if st.button("Let's compare"):
            if selected_symbols and selected_elements:
                display_comparison_table(selected_symbols, selected_elements)
            else:
                st.warning("Please select at least TWO symbols and ONE element to compare.")
    else:
        # Turn off the visibility flag if the user switches symbols in the list
        st.session_state.comparison_table_visible = False



st.write("")
st.write("")
st.write("")

































# Define the color code for the "Chart Zone" text
color_code_chart_zone = "white"  # Example color code

# Display the styled header using st.write() with HTML
st.write(
    f'<span style="font-size:30px;">Chart Zone</span>',
    unsafe_allow_html=True
)
st.write(f'*All charts are interactive by clicking legend elements')
st.write(f'*values in millions $')


col1, col2 = st.columns([0.6, 0.4])  # Adjust the width ratio of col1 and col2 as needed



with col1:

    # Transpose the balance sheet DataFrame
    transposed_balance_sheet = balance_sheet.transpose()

    # Select relevant columns from the transposed balance sheet DataFrame
    selected_columns = ['Current Assets', 'Current Liabilities', 'Total Non Current Assets',
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
        title='',
        title_x=0.3,
        xaxis_title='',
        yaxis_title='Amount (M$)',
        height=500,  # Set a fixed height
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.45, font=dict(size=15)),
        # Center the legend
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


    # st.write("")
    st.subheader(f"Assets & Liabilities")
    st.write("")
# ******************************************   CHARTS 'Total Assets', 'Current Assets', 'Total Non Current Assets', ' Cash  & Cash Equivalents And Short Term Investments' ***************************************************

col1, col2, col3, col4 = st.columns([0.30, 0.30, 0.30, 0.1])



with col1:

    data_percentage_change_balance = percentage_change_balance.loc['Total Assets'].transpose()


    fig = go.Figure()

    # Add bar trace for total assets
    fig.add_trace(
        go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Total Assets'], name='Total Assets',
               marker_color="#0ECCEC"))

    # Add line trace for growth rate
    fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                             mode='lines+markers', name='Growth Trend Line', line=dict(color='red'), yaxis='y2'))

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
    fig.update_layout(title='',
                      title_x=0.30,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount (M$)',
                      yaxis2=dict(title='% Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45, font=dict(size=15)),  # Adjust legend position
                      )


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
                             mode='lines+markers', name='Growth Trend Line', line=dict(color='red'), yaxis='y2'))

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
    fig.update_layout(title='',
                      title_x=0.4,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount',
                      yaxis2=dict(title='% Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45, font=dict(size=15)))  # Adjust legend position

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col3:
    data_percentage_change_balance = percentage_change_balance.loc['Total Non Current Assets'].transpose()

    # Create a figure
    fig = go.Figure()

    # Add bar trace for total assets
    fig.add_trace(
        go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Total Non Current Assets'],
               name='Non Current Assets',
               marker_color="#0ECCEC"))

    # Add line trace for growth rate
    fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                             mode='lines+markers', name='Growth Trend Line', line=dict(color='red'), yaxis='y2'))

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
    fig.update_layout(title='',
                      title_x=0.35,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount (M$)',
                      yaxis2=dict(title='% Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45, font=dict(size=15)))  # Adjust legend position

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
                             mode='lines+markers', name='Growth Trend Line', line=dict(color='blue'), yaxis='y2'))

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
    fig.update_layout(title='',
                      title_x=0.25,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount (M$)',
                      yaxis2=dict(title='% Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45, font=dict(size=15)))  # Adjust legend position

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
                             mode='lines+markers', name='Growth Trend Line', line=dict(color='blue'), yaxis='y2'))

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
    fig.update_layout(title='',
                      title_x=0.35,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount (M$)',
                      yaxis2=dict(title='% Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45, font=dict(size=15)))  # Adjust legend position

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
               name='Non Current Liabilities Net Minority Interest',
               marker_color='red'))

    # Add line trace for growth rate
    fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                             mode='lines+markers', name='Growth Trend Line', line=dict(color='blue'), yaxis='y2'))

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
    fig.update_layout(title='',
                      title_x=0.35,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount (M$)',
                      yaxis2=dict(title='% Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45, font=dict(size=15)))  # Adjust legend position

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})



col1, col2, col3 = st.columns([0.35, 0.35, 0.3])


with col1:
    data_percentage_change_balance = percentage_change_balance.loc[
        'Cash Cash Equivalents And Short Term Investments'].transpose()

    # Create a figure
    fig = go.Figure()

    # Add bar trace for total assets
    fig.add_trace(
        go.Bar(x=data_percentage_change_balance.index,
               y=balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'],
               name='C&C Equivalents And Short Term Investments',
               marker_color='green'))

    # Add line trace for growth rate
    fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                             mode='lines+markers', name='Growth Trend Line', line=dict(color='red'), yaxis='y2'))

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
    fig.update_layout(title='',
                      title_x=0.20,  # Set the title position to the center
                      xaxis_title='',
                      yaxis_title='Amount (M$)',
                      yaxis2=dict(title='% Growth', overlaying='y', side='right', showgrid=False),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.40, font=dict(size=15)))  # Set legend to horizontal orientation

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})





with col2:

    if 'Total Debt' in balance_sheet.transpose():  # Checking if 'Total Debt' is present in the columns
        data_percentage_change_balance = percentage_change_balance.loc['Total Debt'].transpose()

        # Create a figure
        fig = go.Figure()

        # Add bar trace for total assets
        fig.add_trace(
            go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Total Debt'], name='Total Debt',
                   marker_color='red'))

        # Add line trace for growth rate
        fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                                 mode='lines+markers', name='Growth Trend Line', line=dict(color='blue'), yaxis='y2'))

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
        fig.update_layout(title='',
                          title_x=0.35,  # Set the title position to the center
                          xaxis_title='',
                          yaxis_title='Amount (M$)',
                          yaxis2=dict(title='% Growth', overlaying='y', side='right', showgrid=False),
                          legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45, font=dict(size=15)))  # Set legend to horizontal orientation


        # Display the chart without the menu
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.write("")
        st.write("")



#####################################################################################################





col1, col2 = st.columns([0.6, 0.4])

with col1:

    st.write("")
    # Custom subheader with color
    color_code = "#0ECCEC"

    st.markdown(f"<h2>Chart Zone - <span style='color: {color_code};'>Design Your Own Chart</span></h2>",
                unsafe_allow_html=True)

    st.write("Design a Dynamic Visual Chart & Compare Balance Sheet Key Elements")
    st.write("")

    # Assuming the 'Balance Sheet' DataFrame is provided as per the initial input
    # elements = [
    #     'Total Assets', 'Current Assets', 'Total Non Current Assets',
    #     'Cash Cash Equivalents And Short Term Investments',
    #     'Total Liabilities Net Minority Interest', 'Current Liabilities',
    #     'Total Non Current Liabilities Net Minority Interest', 'Total Equity Gross Minority Interest',
    #     'Total Capitalization', 'Common Stock Equity', 'Net Tangible Assets',
    #     'Working Capital', 'Invested Capital', 'Tangible Book Value',
    #     'Total Debt', 'Net Debt', 'Share Issued', 'Ordinary Shares Number',
    #     'Treasury Shares Number'
    # ]

    elements = [
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

    balance_sheet_Design = balance_sheet_Design.drop('Properties', errors='ignore')

    # Convert values to millions
    balance_sheet_Design = balance_sheet_Design.astype(float) / 1_000_000  # Divide by 1 million

    # Convert column headers to datetime
    balance_sheet_Design.columns = pd.to_datetime(balance_sheet_Design.columns)

    # Sort the columns in ascending order of dates
    balance_sheet_Design = balance_sheet_Design.sort_index(axis=1)

    # Format the column headers to remove the timestamp
    balance_sheet_Design.columns = [col.strftime('%d/%m/%Y') for col in balance_sheet_Design.columns]

    # Show only the latest 4 dates
    balance_sheet_Design = balance_sheet_Design.iloc[:, -4:]
    # Replace "None" with 0
    balance_sheet_Design = balance_sheet_Design.fillna(0)


    # Apply the formatting function to the balance sheet DataFrame
    balance_sheet_Design = balance_sheet_Design.applymap(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
    # st.write(balance_sheet_Design)



    # Transposing the income_statement DataFrame to have dates as rows and elements as columns
    data_chart = balance_sheet_Design.loc[elements].transpose()

    # Convert values to float after removing commas
    data_chart = data_chart.replace({',': ''}, regex=True).astype(float)

    # Convert data index to datetime objects if they are not already datetime objects
    data_chart.index = pd.to_datetime(data_chart.index)


    # Option to choose chart type
    chart_type = st.radio('Select Chart Type', ('Single Axis', 'Dual Axis'))

    chart_style = st.selectbox('Select Chart Style', ('Bar Chart', 'Line Chart'))

    if chart_type == 'Single Axis':
        # Dropdown for selecting one axis
        single_axis = st.selectbox('Select Metric', options=[None] + elements, index=0)

        go_button_single = st.button('Make Chart', key='go_single')

        if go_button_single:
            if single_axis and single_axis in data_chart.columns:
                fig = go.Figure()
                if chart_style == 'Bar Chart':
                    fig.add_trace(go.Bar(
                        x=data_chart.index,
                        y=data_chart[single_axis].astype(float),
                        name=single_axis,
                        text=[f"${'{:,.0f}'.format(val)}" for val in data_chart[single_axis].astype(float)],
                        textposition='auto',
                        insidetextanchor='start',
                        marker=dict(color='blue', line=dict(width=2, color='black')),
                        insidetextfont=dict(size=15),
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=data_chart.index,
                        y=data_chart[single_axis].astype(float),
                        mode='lines+markers',
                        name=single_axis,
                        text=[f"${'{:,.0f}'.format(val)}" for val in data_chart[single_axis].astype(float)],
                        textposition='top center',
                        marker=dict(color='blue', size=10, line=dict(width=2, color='black')),
                        line=dict(width=2, color='blue'),
                    ))

                # Calculate percentage growth and add linear line
                growth = data_chart[single_axis].pct_change() * 100
                growth.iloc[0] = 0  # Set the first value to 0%
                fig.add_trace(go.Scatter(
                    x=data_chart.index,
                    y=growth,
                    mode='lines+markers',
                    name=f'{single_axis} Growth Trend',
                    yaxis='y2',
                    marker=dict(color='red', size=8, line=dict(width=2, color='black')),
                    line=dict(width=2, color='red'),
                ))

                # Add annotations for percentage growth
                for i in range(len(data_chart)):
                    growth_value = growth.iloc[i]
                    fig.add_annotation(
                        x=data_chart.index[i],
                        y=growth.iloc[i],
                        text=f"{growth_value:.2f}%",
                        showarrow=False,  # don't show an arrow
                        yref='y2',
                        xanchor='right',  # anchor point for x-coordinate
                        yanchor='middle',  # anchor point for y-coordinate
                        align='left',  # alignment of the annotation text
                        font=dict(color="black", size=15),  # Change the font color to black
                        bgcolor='yellow',
                        xshift=5,  # horizontal shift of the annotation
                        yshift=20,  # vertical shift of the annotation

                    )

                fig.update_layout(
                    xaxis=dict(tickvals=data_chart.index, ticktext=data_chart.index.strftime('%d/%m/%Y')),
                    yaxis=dict(title='Amount (M$)'),
                    yaxis2=dict(title='% Growth', overlaying='y', side='right', showgrid=False, position=1),
                    width=800,
                    height=500,
                    # title_text=f'{single_axis} Over Time',
                    title_text="",
                    title_x=0.5,
                    title_y=0.98,
                    title_xanchor='center',
                    title_yanchor='top',
                    font=dict(size=15),
                    legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.45,
                                font=dict(size=15)),
                )

                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})



            else:
                st.write("Please select a valid element for the metric.")

    else:
        # Dropdown select boxes and "Go" button for dual axis
        col1, col2, col3 = st.columns(3)

        with col1:
            x_axis = st.selectbox('X-Axis Metric', options=[None] + elements, index=0)

        y_axis_options = [col for col in elements if col != x_axis]

        with col2:
            y_axis = st.selectbox('Y-Axis Metric', options=[None] + y_axis_options, index=0)

        go_button_dual = st.button('Make Chart', key='go_dual')

        # Plot the selected elements
        if go_button_dual:
            if x_axis and y_axis and x_axis in data_chart.columns and y_axis in data_chart.columns:
                fig = go.Figure()
                if chart_style == 'Bar Chart':
                    fig.add_trace(go.Bar(
                        x=data_chart.index,
                        y=data_chart[x_axis].astype(float),
                        name=x_axis,
                        text=[f"${'{:,.0f}'.format(val)}" for val in data_chart[x_axis].astype(float)],
                        textposition='auto',
                        insidetextanchor='start',
                        marker=dict(color='blue', line=dict(width=2, color='black')),
                        insidetextfont=dict(size=15),
                    ))
                    fig.add_trace(go.Bar(
                        x=data_chart.index,
                        y=data_chart[y_axis].astype(float),
                        name=y_axis,
                        text=[f"${'{:,.0f}'.format(val)}" for val in data_chart[y_axis].astype(float)],
                        textposition='auto',
                        insidetextanchor='start',
                        marker=dict(color='red', line=dict(width=2, color='black')),
                        insidetextfont=dict(size=15),
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=data_chart.index,
                        y=data_chart[x_axis].astype(float),
                        mode='lines+markers',
                        name=x_axis,
                        text=[f"${'{:,.0f}'.format(val)}" for val in data_chart[x_axis].astype(float)],
                        textposition='top center',
                        marker=dict(color='blue', size=10, line=dict(width=2, color='black')),
                        line=dict(width=2, color='blue'),
                    ))
                    fig.add_trace(go.Scatter(
                        x=data_chart.index,
                        y=data_chart[y_axis].astype(float),
                        mode='lines+markers',
                        name=y_axis,
                        text=[f"${'{:,.0f}'.format(val)}" for val in data_chart[y_axis].astype(float)],
                        textposition='top center',
                        marker=dict(color='red', size=10, line=dict(width=2, color='black')),
                        line=dict(width=2, color='red'),
                    ))

                fig.update_layout(
                    xaxis=dict(tickvals=data_chart.index, ticktext=data_chart.index.strftime('%d/%m/%Y')),
                    yaxis=dict(title='Amount (M$)'),
                    width=800,
                    height=500,
                    # title_text=f'{x_axis} and {y_axis} Over Time',
                    title_text="",
                    title_x=0.5,
                    title_y=0.98,
                    title_xanchor='center',
                    title_yanchor='top',
                    font=dict(size=15),
                    legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.45,
                                font=dict(size=15)),
                )

                # Calculate percentage growth for x_axis
                growth_x = ((data_chart[x_axis] - data_chart[x_axis].shift(1)) / data_chart[x_axis].shift(1)) * 100
                growth_x.iloc[0] = 0  # Set the first value to 0%

                # Add percentage growth annotation for x_axis
                for i in range(0, len(data_chart)):
                    fig.add_annotation(x=data_chart.index[i], y=data_chart[x_axis].iloc[i],
                                       text=f"{growth_x.iloc[i]:.2f}%",
                                       showarrow=False,
                                       font=dict(color="black", size=15),
                                       xshift=-30,
                                       yshift=20,  # Adjusted y-shift to position above the element
                                       bgcolor="yellow",
                                       bordercolor="black",
                                       borderwidth=1,
                                       opacity=0.8)

                # Calculate percentage growth for y_axis
                growth_y = ((data_chart[y_axis] - data_chart[y_axis].shift(1)) / data_chart[y_axis].shift(1)) * 100
                growth_y.iloc[0] = 0  # Set the first value to 0%

                # Add percentage growth annotation for y_axis
                for i in range(0, len(data_chart)):
                    fig.add_annotation(x=data_chart.index[i], y=data_chart[y_axis].iloc[i],
                                       text=f"{growth_y.iloc[i]:.2f}%",
                                       showarrow=False,
                                       font=dict(color="black", size=15),
                                       xshift=25,
                                       yshift=20,  # Adjusted y-shift to position above the element
                                       bgcolor="yellow",
                                       bordercolor="black",
                                       borderwidth=1,
                                       opacity=0.8)

                # Add a label to indicate that the annotation represents percentage change
                fig.add_annotation(
                    x=0.5,
                    y=-0.2,
                    xref='paper',
                    yref='paper',
                    # text='Percentage change (%)',
                    text=f'Percentage change (%){" QoQ" if is_quarterly else " YoY"}',  # Corrected parentheses
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="yellow",
                    bordercolor="black",
                    borderwidth=1,
                    opacity=0.8
                )

                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.write("Please select valid elements for both X-axis and Y-axis.")
