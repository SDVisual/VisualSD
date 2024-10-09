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
header_html = f'<h2 style="color:{color_code};">{APP_NAME} </h2>'
st.markdown(header_html, unsafe_allow_html=True)


col1, col2 = st.columns([0.35, 0.65])

with col1:

    # Input box for user to enter symbol
    new_symbol = st.text_input("Add symbol to Symbols List (e.g., AAPL)", placeholder="Search Stocks").strip().upper()


# Retrieve the last valid symbol entered by the user, default to 'AAPL' if none
DEFAULT_SYMBOL = st.session_state.valid_tickers[-1] if st.session_state.valid_tickers else 'AAPL'

# Retrieve the selected ticker index and symbol from the session state
selected_ticker_index = st.session_state.selected_ticker_index
selected_symbol = st.session_state.valid_tickers[selected_ticker_index]



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

st.sidebar.info("- Design Your Own Charts.")
st.sidebar.info("- Compare Free Cash Flow Between Your Symbols.")
st.sidebar.info("- Easy Download Data Tables.")
st.sidebar.info("- For the best experience, maximize your screen.")
st.sidebar.info("- This app version is less suitable for stocks in the finance industry")
st.sidebar.markdown("&copy;VisualSD. All rights reserved.", unsafe_allow_html=True)




# ***************************************     Cash Flow   ************************************************************************

StockInfo = yf.Ticker(ticker).info
symbol = StockInfo["shortName"]
color_code = "#0ECCEC"  # Color for the symbol

st.write(f'<span style="font-size:30px;">Cash Flow - </span>'
         f'<span style="color:{color_code}; font-size:30px;">{symbol}</span>', unsafe_allow_html=True)

# # Combine st.write() with HTML-styled header
# st.write(f'<span style="color:white; font-size:30px;">Cash Flow - </span>'
#          f'<span style="color:{color_code}; font-size:30px;">{symbol}</span>', unsafe_allow_html=True)

balance_sheetYear = yf.Ticker(ticker).balance_sheet
balance_sheetQuarterly = yf.Ticker(ticker).quarterly_balance_sheet
income_statementYear = yf.Ticker(ticker).income_stmt
income_statementQuarterly = yf.Ticker(ticker).quarterly_income_stmt

# st.write(balance_sheet.transpose()['Ordinary Shares Number'])
cash_flowYear = yf.Ticker(ticker).cash_flow
cash_flowQuarterly = yf.Ticker(ticker).quarterly_cashflow

# Default to annual income statement
cash_flow = cash_flowYear
balance_sheet = balance_sheetYear
income_statement = income_statementYear

# Checkbox to select between annual and quarterly
is_quarterly = st.checkbox("Quarterly Cash Flow", value=False)

# Checkbox to toggle display of extended balance sheet
is_extended = st.checkbox("Extended Cash Flow Statement", value=False)

# Update income statement based on the checkbox value
if is_quarterly:
    cash_flow = cash_flowQuarterly
    balance_sheet = balance_sheetQuarterly
    income_statement = income_statementQuarterly

desired_order_first = [
    'Free Cash Flow', 'Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'End Cash Position',
    'Changes In Cash',
    'Income Tax Paid Supplemental Data', 'Interest Paid Supplemental Data', 'Capital Expenditure',
    'Issuance Of Debt', 'Repayment Of Debt',
    'Repurchase Of Capital Stock'
]

desired_order = [
    'Free Cash Flow', 'Operating Cash Flow', 'Cash Flow From Continuing Operating Activities',
    'Net Income From Continuing Operations', 'Operating Gains Losses', 'Gain Loss On Sale Of PPE',
    'Gain Loss On Investment Securities', 'Earnings Losses From Equity Investments',
    'Depreciation Amortization Depletion', 'Depreciation And Amortization', 'Depreciation',
    'Deferred Tax', 'Deferred Income Tax', 'Asset Impairment Charge', 'Stock Based Compensation',
    'Other Non Cash Items',
    'Change In Working Capital', 'Change In Receivables', 'Changes In Account Receivables', 'Change In Inventory',
    'Change In Prepaid Assets',
    'Change In Payables And Accrued Expense', 'Change In Payable', 'Change In Account Payable',
    'Change In Accrued Expense', 'Investing Cash Flow',
    'Cash Flow From Continuing Investing Activities', 'Net PPE Purchase And Sale', 'Purchase Of PPE',
    'Net Business Purchase And Sale',
    'Purchase Of Business', 'Sale Of Business', 'Net Investment Purchase And Sale', 'Purchase Of Investment',
    'Sale Of Investment', 'Net Other Investing Changes',
    'Financing Cash Flow', 'Cash Flow From Continuing Financing Activities', 'Net Issuance Payments Of Debt',
    'Net Long Term Debt Issuance', 'Long Term Debt Issuance',
    'Long Term Debt Payments', 'Net Short Term Debt Issuance', 'Short Term Debt Issuance',
    'Net Common Stock Issuance', 'Common Stock Payments',
    'Proceeds From Stock Option Exercised', 'Net Other Financing Charges', 'End Cash Position', 'Changes In Cash',
    'Beginning Cash Position',
    'Income Tax Paid Supplemental Data', 'Interest Paid Supplemental Data', 'Capital Expenditure',
    'Issuance Of Debt', 'Repayment Of Debt',
    'Repurchase Of Capital Stock'
]


cash_flow_Design = cash_flow.reindex(desired_order, fill_value='0')


if is_extended:
    cash_flow = cash_flow.reindex(desired_order, fill_value='0')
else:
    cash_flow = cash_flow.reindex(desired_order_first, fill_value='0')



def str_to_float(value):
    try:
        return float(value.replace(',', ''))  # Assuming the numbers are formatted with commas
    except ValueError:
        return value  # Return the original value if conversion fails





# Convert values to millions
cash_flow = cash_flow.astype(float) / 1_000_000

# Fill empty cells with '0'
cash_flow.fillna('0', inplace=True)

# Convert column headers to datetime
cash_flow.columns = pd.to_datetime(cash_flow.columns)

# Format the column headers to remove the timestamp
cash_flow.columns = cash_flow.columns.strftime('%Y-%m-%d')

# Sort the columns in ascending order of dates
cash_flow = cash_flow.sort_index(axis=1)

# Apply formatting to the DataFrame
cash_flow_formatted = cash_flow.applymap(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)

cash_flow = cash_flow_formatted

# Apply the conversion function to all values in the DataFrame
cash_flow_numeric = cash_flow.applymap(str_to_float)


# Show only the latest 4 dates
cash_flow = cash_flow.iloc[:, -4:]


# Calculate percentage change for each metric between consecutive periods
percentage_change_cash_flow = cash_flow_numeric.pct_change(axis=1) * 100

# Convert the first column to numeric, coercing errors to NaN
percentage_change_cash_flow.iloc[:, 0] = pd.to_numeric(percentage_change_cash_flow.iloc[:, 0], errors='coerce')

# Replace NaN values with 0
percentage_change_cash_flow.iloc[:, 0] = percentage_change_cash_flow.iloc[:, 0].fillna(0)

percentage_change_cash_flow = percentage_change_cash_flow.iloc[:, -4:]

# Fill the first column of percentage_change_cash_flow DataFrame with 0
percentage_change_cash_flow.iloc[:, 0] = 0

# st.write(percentage_change_cash_flow)


st.write("<span style='font-size: 16px;'>* All values in millions $</span>", unsafe_allow_html=True)
# Apply styling to the cash flow DataFrame
styled_cash_flow = cash_flow.style.set_table_styles([
    {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('border', '2px solid blue')]},
    {'selector': 'th, td', 'props': [('text-align', 'center'), ('border', '1px solid blue')]},
    {'selector': 'th', 'props': [('text-align', 'center')]},  # Center align headers
    {'selector': 'th:first-child, td:first-child', 'props': [('text-align', 'left')]}  # Align first column to left
])




st.write(cash_flow)
st.write("")
st.write("")


col1, col2 = st.columns([0.7, 0.3])


with col1:
    # Initialize session state variables if they don't exist
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = None
    if 'comparison_table_visible' not in st.session_state:
        st.session_state.comparison_table_visible = False
    if 'valid_tickers' not in st.session_state:
        st.session_state.valid_tickers = []

    # Define the Cash Flow elements to compare

    cash_flow_elements = [
        ('Choose All', 'all', 'Billions'),  # Option to choose all elements
        ('Free Cash Flow', 'Free Cash Flow', 'Billions'),
        ('Operating Cash Flow', 'Operating Cash Flow', 'Billions'),
        ('Cash Flow From Continuing Operating Activities', 'Cash Flow From Continuing Operating Activities', 'Billions'),
        ('Net Income From Continuing Operations', 'Net Income From Continuing Operations', 'Billions'),
        ('Operating Gains Losses', 'Operating Gains Losses', 'Billions'),
        ('Gain Loss On Sale Of PPE', 'Gain Loss On Sale Of PPE', 'Billions'),
        ('Gain Loss On Investment Securities', 'Gain Loss On Investment Securities', 'Billions'),
        ('Earnings Losses From Equity Investments', 'Earnings Losses From Equity Investments', 'Billions'),
        ('Depreciation Amortization Depletion', 'Depreciation Amortization Depletion', 'Billions'),
        ('Depreciation And Amortization', 'Depreciation And Amortization', 'Billions'),
        ('Depreciation', 'Depreciation', 'Billions'),
        ('Deferred Tax', 'Deferred Tax', 'Billions'),
        ('Deferred Income Tax', 'Deferred Income Tax', 'Billions'),
        ('Asset Impairment Charge', 'Asset Impairment Charge', 'Billions'),
        ('Stock Based Compensation', 'Stock Based Compensation', 'Billions'),
        ('Other Non Cash Items', 'Other Non Cash Items', 'Billions'),
        ('Change In Working Capital', 'Change In Working Capital', 'Billions'),
        ('Change In Receivables', 'Change In Receivables', 'Billions'),
        ('Changes In Account Receivables', 'Changes In Account Receivables', 'Billions'),
        ('Change In Inventory', 'Change In Inventory', 'Billions'),
        ('Change In Prepaid Assets', 'Change In Prepaid Assets', 'Billions'),
        ('Change In Payables And Accrued Expense', 'Change In Payables And Accrued Expense', 'Billions'),
        ('Change In Payable', 'Change In Payable', 'Billions'),
        ('Change In Account Payable', 'Change In Account Payable', 'Billions'),
        ('Change In Accrued Expense', 'Change In Accrued Expense', 'Billions'),
        ('Investing Cash Flow', 'Investing Cash Flow', 'Billions'),
        ('Cash Flow From Continuing Investing Activities', 'Cash Flow From Continuing Investing Activities', 'Billions'),
        ('Net PPE Purchase And Sale', 'Net PPE Purchase And Sale', 'Billions'),
        ('Purchase Of PPE', 'Purchase Of PPE', 'Billions'),
        ('Net Business Purchase And Sale', 'Net Business Purchase And Sale', 'Billions'),
        ('Purchase Of Business', 'Purchase Of Business', 'Billions'),
        ('Sale Of Business', 'Sale Of Business', 'Billions'),
        ('Net Investment Purchase And Sale', 'Net Investment Purchase And Sale', 'Billions'),
        ('Purchase Of Investment', 'Purchase Of Investment', 'Billions'),
        ('Sale Of Investment', 'Sale Of Investment', 'Billions'),
        ('Net Other Investing Changes', 'Net Other Investing Changes', 'Billions'),
        ('Financing Cash Flow', 'Financing Cash Flow', 'Billions'),
        ('Cash Flow From Continuing Financing Activities', 'Cash Flow From Continuing Financing Activities', 'Billions'),
        ('Net Issuance Payments Of Debt', 'Net Issuance Payments Of Debt', 'Billions'),
        ('Net Long Term Debt Issuance', 'Net Long Term Debt Issuance', 'Billions'),
        ('Long Term Debt Issuance', 'Long Term Debt Issuance', 'Billions'),
        ('Long Term Debt Payments', 'Long Term Debt Payments', 'Billions'),
        ('Net Short Term Debt Issuance', 'Net Short Term Debt Issuance', 'Billions'),
        ('Short Term Debt Issuance', 'Short Term Debt Issuance', 'Billions'),
        ('Net Common Stock Issuance', 'Net Common Stock Issuance', 'Billions'),
        ('Common Stock Payments', 'Common Stock Payments', 'Billions'),
        ('Proceeds From Stock Option Exercised', 'Proceeds From Stock Option Exercised', 'Billions'),
        ('Net Other Financing Charges', 'Net Other Financing Charges', 'Billions'),
        ('End Cash Position', 'End Cash Position', 'Billions'),
        ('Changes In Cash', 'Changes In Cash', 'Billions'),
        ('Beginning Cash Position', 'Beginning Cash Position', 'Billions'),
        ('Income Tax Paid Supplemental Data', 'Income Tax Paid Supplemental Data', 'Billions'),
        ('Interest Paid Supplemental Data', 'Interest Paid Supplemental Data', 'Billions'),
        ('Capital Expenditure', 'Capital Expenditure', 'Billions'),
        ('Issuance Of Debt', 'Issuance Of Debt', 'Billions'),
        ('Repayment Of Debt', 'Repayment Of Debt', 'Billions'),
        ('Repurchase Of Capital Stock', 'Repurchase Of Capital Stock', 'Billions')
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
            cash_flow_compare = stock.cash_flow


            # Get the last available date
            last_date = cash_flow_compare.columns[0]

            # Collect the elements for the current ticker
            data = {'Ticker': ticker, 'Date': last_date.strftime('%Y-%m-%d')}
            for elem in cash_flow_elements:
                if elem[1] == 'all':
                    continue
                value = cash_flow_compare.at[elem[1], last_date] if elem[1] in cash_flow_compare.index else None

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
        st.write("Comparison Table Of Annual Cash Flow Elements:")
        st.write("")
        if 'Choose All' in selected_elements:
            selected_elements = [elem[0] for elem in balance_sheet_elements if elem[0] != 'Choose All']
        st.dataframe(comparison_df[['Date'] + selected_elements])



    # Check if the visibility flag is set to True and the user clicks the button
    if st.button("Compare Annual Free Cash Flow Between Your Symbols"):
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
                                           [elem[0] for elem in cash_flow_elements])

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
color_code_chart_zone = "light white"  # Example color code

# Display the styled header using st.write() with HTML
st.write(
    f'<span style="font-size:30px;">Chart Zone</span>', unsafe_allow_html=True)

st.write(f'*All charts are interactive by clicking legend elements')

st.write("")


col1, col2 = st.columns([0.65, 0.35])

with col1:

    # Transpose the balance sheet DataFrame
    transposed_cash_flow_sheet = cash_flow.transpose()

    # Select relevant columns from the transposed balance sheet DataFrame
    selected_columns = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Free Cash Flow']

    selected_data = transposed_cash_flow_sheet[selected_columns]


    # Remove commas from the values and then convert to float
    selected_data = selected_data.replace(',', '', regex=True).astype(float)
    # Convert selected data to numeric type
    selected_data = selected_data.astype(float)

    # Define colors for each column based on their values
    colors = {}

    colors['Operating Cash Flow'] = ['blue' for _ in selected_data['Operating Cash Flow']]
    colors['Investing Cash Flow'] = ['purple' for _ in selected_data['Investing Cash Flow']]
    colors['Financing Cash Flow'] = ['yellow' for _ in selected_data['Financing Cash Flow']]
    colors['Free Cash Flow'] = ['white' for _ in selected_data['Free Cash Flow']]

    # # Additional conditions for specific columns
    # colors['Operating Cash Flow'] = ['#006400' if value > 0 else 'red' if value < 0 else 'white' for value in
    #                                  selected_data['Operating Cash Flow']]
    # colors['Investing Cash Flow'] = ['green' if value > 0 else '#ff6347' if value < 0 else 'white' for value in
    #                                  selected_data['Investing Cash Flow']]
    # colors['Financing Cash Flow'] = ['#3cb371' if value > 0 else '#f08080' if value < 0 else 'white' for value in
    #                                  selected_data['Financing Cash Flow']]
    # colors['Free Cash Flow'] = ['lightgreen' if value > 0 else '#ffa07a' if value < 0 else 'white' for value in
    #                             selected_data['Free Cash Flow']]

    # Convert column headers to datetime
    cash_flow.columns = pd.to_datetime(cash_flow.columns)

    # Create a Plotly figure Chart ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Free Cash Flow']
    fig = go.Figure()

    # Add traces to the figure as bars
    for column in selected_data.columns:
        fig.add_trace(
            go.Bar(x=cash_flow.columns, y=selected_data[column], name=column, marker_color=colors[column]))

        # Update layout to set the x-axis tick values
        fig.update_layout(xaxis=dict(tickmode='array',  # Set tick values manually
                                     tickvals=cash_flow.columns,
                                     # Use the dates in cash_flow columns as tick values
                                     tickformat='%m-%y'))  # Format the tick labels as 'YYYY-MM'

    # Update layout if needed
    fig.update_layout(
        title='',
        title_x=0.3,
        xaxis_title='',
        yaxis_title='Amount (M$)',
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.45, font=dict(size=15)),
    )

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

col1, col2, col3, col4 = st.columns([0.3, 0.3, 0.3, 0.1])

with col1:

    #   ***********************  FcF Chart & % Growth ************************

    # Extract the percentage growth for Free Cash Flow
    free_cash_flow_growth = percentage_change_cash_flow.loc['Free Cash Flow']

    # Create a Plotly figure Free Cash Flow
    fig = go.Figure()

    # Add bar trace for Free Cash Flow
    fig.add_trace(
        go.Bar(x=percentage_change_cash_flow.columns, y=cash_flow.loc['Free Cash Flow'], name='Free Cash Flow',
               marker_color='green'))

    # Add line trace for percentage growth
    fig.add_trace(
        go.Scatter(x=percentage_change_cash_flow.columns, y=free_cash_flow_growth, name='Growth Trend Line',
                   mode='lines+markers', line=dict(color='red'), yaxis='y2'))

    # Add text annotations for growth values above the bars
    for i, value in enumerate(free_cash_flow_growth):
        fig.add_annotation(x=percentage_change_cash_flow.columns[i],  # x-coordinate for annotation
                           y=value + 1,  # Shift the text 1 above the point
                           text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
                           showarrow=False,  # whether to show arrow or not
                           font=dict(color='black', size=15),  # color of the annotation text
                           bgcolor='yellow',
                           yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                           align='left',  # alignment of the text
                           xanchor='left')  # anchor point along x-axis for alignment

    # Update layout
    fig.update_layout(
        title='',
        title_x=0.25,
        xaxis=dict(
            title='',
            tickmode='array',  # Set tick values manually
            tickvals=percentage_change_cash_flow.columns,  # Use the dates from the DataFrame
            tickformat='%m/%y'  # Set the tick format to display as MM/YY
        ),
        yaxis_title='Amount (M$)',
        yaxis2=dict(title='Percentage Growth (%)', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45, font=dict(size=15)),  # Adjust legend position
    )

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Apply str_to_float function only to columns containing string values
    ordinary_shares_number = balance_sheet.transpose()['Ordinary Shares Number'].apply(
        lambda x: str_to_float(x) if isinstance(x, str) else x)


    # Apply str_to_float function to 'Free Cash Flow' column
    free_cash_flow = cash_flow.transpose()['Free Cash Flow'].apply(str_to_float)

    # Perform division operation after applying conversion function to both columns
    free_cash_flow_per_share = (free_cash_flow * 1000000) / ordinary_shares_number

    free_cash_flow_per_share.dropna(inplace=True)




    # Convert Series to DataFrame with a single column
    free_cash_flow_per_share_df = pd.DataFrame(free_cash_flow_per_share, columns=['Free Cash Flow Per Share'])

    # Calculate percentage change for each metric between consecutive periods
    percentage_change_cash_flow_per_share = free_cash_flow_per_share_df[
                                                'Free Cash Flow Per Share'].pct_change() * 100

    # Fill missing or None values with 0
    free_cash_flow_per_share_df['Free Cash Flow Per Share'] = free_cash_flow_per_share_df[
        'Free Cash Flow Per Share'].fillna(0)

    # Fill None values with 0
    percentage_change_cash_flow_per_share = percentage_change_cash_flow_per_share.fillna(0)

    Capital_Expenditure_percentage_change = percentage_change_cash_flow.loc['Capital Expenditure']

    # Transpose the cash_flow DataFrame
    cash_flow_transposed = cash_flow.transpose()

    # Access the 'Capital Expenditure' column after transposing
    Capital_Expenditure = cash_flow_transposed['Capital Expenditure'].str.replace(',', '').str.replace('-',
                                                                                                       '').astype(
        float)
    Capital_Expenditure = Capital_Expenditure.abs()  # Convert values to positive

    # Create a Plotly figure for Capital Expenditure ********************************************

    # Create a Plotly figure for Capital Expenditure
    fig = go.Figure()

    # Add bar trace for Capital Expenditure
    fig.add_trace(
        go.Bar(
            x=Capital_Expenditure.index,
            y=Capital_Expenditure.values,
            name='Capital Expenditure',
            marker_color='blue'
        )
    )

    # Add line trace for percentage growth
    fig.add_trace(
        go.Scatter(
            x=Capital_Expenditure.index,
            y=Capital_Expenditure_percentage_change,
            name='Growth Trend Line',
            mode='lines+markers',
            line=dict(color='red'),
            yaxis='y2'  # Assign the line to the secondary y-axis
        )
    )

    # Add text annotations for growth values above the bars
    for i, value in enumerate(Capital_Expenditure_percentage_change):
        fig.add_annotation(
            x=Capital_Expenditure.index[i],  # x-coordinate for annotation
            y=value + 1,  # Shift the text 1 above the point
            text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
            showarrow=False,  # whether to show arrow or not
            font=dict(color='black', size=15),  # color of the annotation text
            bgcolor='yellow',
            yref='y2',  # reference point on the y-axis (in this case, it's the secondary y-axis)
            align='left',  # alignment of the text
            xanchor='left'  # anchor point along x-axis for alignment
        )

    # Update layout
    fig.update_layout(
        title='',
        title_x=0.25,
        xaxis=dict(
            title='',
            tickmode='array',  # Set tick values manually
            tickvals=Capital_Expenditure.index,  # Use the dates from the DataFrame
            tickformat='%m/%Y'  # Set the tick format to display as MM/YYYY
        ),
        yaxis=dict(title='Amount (M$)'),
        yaxis2=dict(title='Percentage Growth (%)', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45, font=dict(size=15)),  # Adjust legend position
    )

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col2:

    # Create a Plotly figure Free Cash Flow Per Share
    fig = go.Figure()

    # Add bar trace for Free Cash Flow
    fig.add_trace(
        go.Bar(x=free_cash_flow_per_share_df.index,
               y=free_cash_flow_per_share_df['Free Cash Flow Per Share'],
               name='Free Cash Flow Per Share',
               marker_color='green'))

    # Add line trace for percentage growth
    fig.add_trace(
        go.Scatter(x=free_cash_flow_per_share_df.index,
                   y=percentage_change_cash_flow_per_share,
                   name='Growth Trend Line',
                   mode='lines+markers',
                   line=dict(color='red'),
                   yaxis='y2'))

    # Add text annotations for growth values above the bars
    for i, value in enumerate(percentage_change_cash_flow_per_share):
        fig.add_annotation(x=free_cash_flow_per_share_df.index[i],  # x-coordinate for annotation
                           y=value + 1,  # Shift the text 1 above the point
                           text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
                           showarrow=False,  # whether to show arrow or not
                           font=dict(color='black', size=15),  # color of the annotation text
                           bgcolor='yellow',
                           yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                           align='left',  # alignment of the text
                           xanchor='left')  # anchor point along x-axis for alignment

    # Update layout
    fig.update_layout(
        title='',
        title_x=0.3,
        xaxis=dict(
            title='',
            tickmode='array',  # Set tick values manually
            tickvals=free_cash_flow_per_share_df.index,  # Use the dates from the DataFrame
            tickformat='%m/%Y'  # Set the tick format to display as MM/YYYY
        ),
        yaxis_title='$ Per Share',
        yaxis2=dict(title='Percentage Growth (%)', overlaying='y', side='right', showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45, font=dict(size=15)),  # Adjust legend position
    )

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Extract rows for 'End Cash Position' and 'Changes In Cash' from the DataFrame
    cash_position = cash_flow.loc['End Cash Position']
    changes_in_cash = cash_flow.loc['Changes In Cash']

    # Create a Plotly figure for End Cash Position and Changes In Cash
    fig = go.Figure()

    # Add bar trace for End Cash Position
    fig.add_trace(
        go.Bar(
            x=cash_position.index,
            y=cash_position.values,
            name='End Cash Position',
            marker_color='green'
        )
    )

    # Add bar trace for Changes In Cash
    fig.add_trace(
        go.Bar(
            x=changes_in_cash.index,
            y=changes_in_cash.values,
            name='Changes In Cash',
            marker_color='blue'
        )
    )

    # Update layout
    fig.update_layout(
        title='',
        title_x=0.28,
        xaxis=dict(title='', tickformat='%m/%Y'),  # Set the tick format to display as MM/YYYY
        yaxis=dict(title='Amount (M$)'),
        barmode='group',  # Group bars
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45, font=dict(size=15)),  # Adjust legend position
    )

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col3:


    # Convert values to millions
    income_statement = income_statement.astype(float) / 1_000_000


    # Transpose the cash_flow DataFrame to convert the row into a column
    cash_flow_transposed = cash_flow.transpose()

    # Transpose the income_statement DataFrame to convert the row into a column
    income_statement_transposed = income_statement.transpose()

    # Convert 'Total Revenue' column to numeric type with error handling
    income_statement_transposed['Total Revenue'] = pd.to_numeric(income_statement_transposed['Total Revenue'],
                                                                 errors='coerce')

    # Remove commas from the 'Free Cash Flow' column and convert to numeric type
    cash_flow_transposed['Free Cash Flow'] = cash_flow_transposed['Free Cash Flow'].str.replace(',', '')
    cash_flow_transposed['Free Cash Flow'] = pd.to_numeric(cash_flow_transposed['Free Cash Flow'], errors='coerce')

    # Calculate free cash flow margin as a percentage

    free_cash_flow_margin_percentage = (cash_flow_transposed['Free Cash Flow'] / income_statement_transposed[
        'Total Revenue']) * 100

    free_cash_flow_margin_percentage.dropna(inplace=True)

    # Create a Plotly figure for the bar chart
    fig = go.Figure()

    # Define colors based on the values (red for negative, green for non-negative)
    colors = ['red' if val < 0 else 'green' for val in free_cash_flow_margin_percentage.values]

    # Add bar trace for free cash flow margin
    fig.add_trace(
        go.Bar(
            x=free_cash_flow_margin_percentage.index.strftime('%Y-%m-%d'),  # Convert datetime index to strings
            y=free_cash_flow_margin_percentage.values,
            marker_color=colors,
            name='Free Cash Flow Margin',
            text=[f'{val:.2f}%' for val in free_cash_flow_margin_percentage.values],  # Text for each bar
            textposition='auto',
            # Position of the text (auto places the text inside the bars if there's enough space, otherwise outside)
            textfont=dict(size=15),  # Set font size to 14
        )
    )

    # Update layout
    fig.update_layout(
        title=f"FCF Margin {'QoQ' if is_quarterly else 'YoY'}",
        title_x=0.35,
        title_y=0.85,  # Adjust the title position downward by changing this value
        xaxis=dict(title='', tickmode='array',
                   tickvals=free_cash_flow_margin_percentage.index.strftime('%Y-%m-%d'),
                   ticktext=free_cash_flow_margin_percentage.index.strftime('%m/%Y')),
        # Set custom tick values and text
        yaxis=dict(title='Percentage (%)'),
    )

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Transpose the cash_flow DataFrame
    transposed_cash_flow = cash_flow.transpose()

    # Extract data for the desired columns
    Issuance_Of_Debt = transposed_cash_flow['Issuance Of Debt']
    Repayment_Of_Debt = transposed_cash_flow['Repayment Of Debt']

    Repayment_Of_Debt = cash_flow_transposed['Repayment Of Debt'].str.replace(',', '').str.replace('-', '').astype(
        float)
    Repayment_Of_Debt = Repayment_Of_Debt.abs()  # Convert values to positive

    # Create a Plotly figure
    fig = go.Figure()

    # Add bar trace for Issuance Of Debt
    fig.add_trace(
        go.Bar(x=Issuance_Of_Debt.index, y=Issuance_Of_Debt.values, name='Issuance Of Debt', marker_color='green'))

    # Add bar trace for Repayment Of Debt
    fig.add_trace(
        go.Bar(x=Repayment_Of_Debt.index, y=Repayment_Of_Debt.values, name='Repayment Of Debt', marker_color='red'))

    # Update layout
    fig.update_layout(
        title='',
        xaxis=dict(title='', tickformat='%m/%Y'),  # Set the tick format to display as MM/YYYY
        title_x=0.3,
        yaxis=dict(title='Amount (M$)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45, font=dict(size=15)),  # Adjust legend position
    )

    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})







#######################################     Chart Zone - Design Your Own Chart             ##############################################################


col1, col2 = st.columns([0.6, 0.4])

with col1:

    st.write("")
    # Custom subheader with color
    color_code = "#0ECCEC"

    st.markdown(f"<h2>Chart Zone - <span style='color: {color_code};'>Design Your Own Chart</span></h2>",
                unsafe_allow_html=True)

    st.write("Design a Dynamic Visual Chart & Compare Cash Flow Key Elements")
    st.write("")

    # Assuming the 'Balance Sheet' DataFrame is provided as per the initial input

    elements = [
        'Free Cash Flow', 'Operating Cash Flow', 'Cash Flow From Continuing Operating Activities',
        'Net Income From Continuing Operations', 'Operating Gains Losses', 'Gain Loss On Sale Of PPE',
        'Gain Loss On Investment Securities', 'Earnings Losses From Equity Investments',
        'Depreciation Amortization Depletion', 'Depreciation And Amortization', 'Depreciation',
        'Deferred Tax', 'Deferred Income Tax', 'Asset Impairment Charge', 'Stock Based Compensation',
        'Other Non Cash Items',
        'Change In Working Capital', 'Change In Receivables', 'Changes In Account Receivables', 'Change In Inventory',
        'Change In Prepaid Assets',
        'Change In Payables And Accrued Expense', 'Change In Payable', 'Change In Account Payable',
        'Change In Accrued Expense', 'Investing Cash Flow',
        'Cash Flow From Continuing Investing Activities', 'Net PPE Purchase And Sale', 'Purchase Of PPE',
        'Net Business Purchase And Sale',
        'Purchase Of Business', 'Sale Of Business', 'Net Investment Purchase And Sale', 'Purchase Of Investment',
        'Sale Of Investment', 'Net Other Investing Changes',
        'Financing Cash Flow', 'Cash Flow From Continuing Financing Activities', 'Net Issuance Payments Of Debt',
        'Net Long Term Debt Issuance', 'Long Term Debt Issuance',
        'Long Term Debt Payments', 'Net Short Term Debt Issuance', 'Short Term Debt Issuance',
        'Net Common Stock Issuance', 'Common Stock Payments',
        'Proceeds From Stock Option Exercised', 'Net Other Financing Charges', 'End Cash Position', 'Changes In Cash',
        'Beginning Cash Position',
        'Income Tax Paid Supplemental Data', 'Interest Paid Supplemental Data', 'Capital Expenditure',
        'Issuance Of Debt', 'Repayment Of Debt',
        'Repurchase Of Capital Stock']




    cash_flow = cash_flow_Design.drop('Properties', errors='ignore')


    def str_to_float(value):
        try:
            return float(value.replace(',', ''))  # Assuming the numbers are formatted with commas
        except ValueError:
            return value  # Return the original value if conversion fails


    # Convert values to millions
    cash_flow_Design = cash_flow_Design.astype(float) / 1_000_000

    # Fill empty cells with '0'
    cash_flow_Design.fillna('0', inplace=True)

    # Convert column headers to datetime
    cash_flow_Design.columns = pd.to_datetime(cash_flow_Design.columns)

    # Format the column headers to remove the timestamp
    cash_flow_Design.columns = cash_flow_Design.columns.strftime('%Y-%m-%d')

    # Sort the columns in ascending order of dates
    cash_flow_Design = cash_flow_Design.sort_index(axis=1)

    # Apply formatting to the DataFrame
    cash_flow_Design = cash_flow_Design.applymap(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)



    # Show only the latest 4 dates
    cash_flow_Design = cash_flow_Design.iloc[:, -4:]

    # st.write(cash_flow_Design)


    # Transposing the income_statement DataFrame to have dates as rows and elements as columns
    data_chart = cash_flow_Design.loc[elements].transpose()

    # Convert values to float after removing commas
    data_chart = data_chart.replace({',': ''}, regex=True).astype(float)

    # Convert data index to datetime objects if they are not already datetime objects
    data_chart.index = pd.to_datetime(data_chart.index)

    # # Display the transposed income statement dataframe
    # st.dataframe(data_chart)
    # st.write("")

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
