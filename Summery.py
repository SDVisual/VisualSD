import datetime
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import cufflinks as cf
import matplotlib.pyplot as plt


StockInfo = {}
StockInfo_df = pd.DataFrame()

APP_NAME = "Stock Data Visualization"

# Set page configuration
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="collapsed",

)



# Display title with blue color using Markdown
st.markdown(f"<h1 style='color:blue;'>{APP_NAME}</h1>", unsafe_allow_html=True)



# Initialize session state for selected ticker index and valid tickers
if 'selected_ticker_index' not in st.session_state:
    st.session_state.selected_ticker_index = 0

if 'valid_tickers' not in st.session_state:
    st.session_state.valid_tickers = []

# Retrieve the last valid symbol entered by the user, default to 'AAPL' if none
DEFAULT_SYMBOL = st.session_state.valid_tickers[-1] if st.session_state.valid_tickers else 'AAPL'



# Input box for user to enter symbol
new_symbol = st.text_input("Add Symbol to Select Box (e.g., AAPL)").strip().upper()

# Check if the entered symbol is empty or consists only of whitespace characters
if not new_symbol or new_symbol.isspace():
    new_symbol = DEFAULT_SYMBOL

# Check if the entered symbol is valid
historical_data = yf.Ticker(new_symbol).history(period='1d')

if new_symbol != DEFAULT_SYMBOL and historical_data.empty:
    st.error("Invalid symbol. Please enter a valid symbol.")

else:
    # Add valid symbol to session state if it's not already present
    if new_symbol not in st.session_state.valid_tickers:
        st.session_state.valid_tickers.append(new_symbol)
        st.text(f"Symbol Added to Select Box - {new_symbol} ")

        # Update selected ticker index to the newly added symbol
        st.session_state.selected_ticker_index = len(st.session_state.valid_tickers) - 1


# Retrieve the index of the selected ticker symbol from the session state
selected_ticker_index = st.session_state.selected_ticker_index

# Select box to choose ticker
ticker = st.sidebar.selectbox('Symbols List - Select Box', st.session_state.valid_tickers,
                              index=selected_ticker_index)

# Update session state with the newly selected symbol index
st.session_state.selected_ticker_index = st.session_state.valid_tickers.index(ticker)

# Sidebar date inputs
start_date = st.sidebar.date_input('Start date - Historical Prices', datetime.datetime(2021, 1, 1))
end_date = st.sidebar.date_input('End date', datetime.datetime.now().date())


# if menu_option == "Company Summary":

df_ticker = yf.download(ticker, start=start_date, end=end_date).reset_index()

StockInfo = yf.Ticker(ticker).info

# Check if "companyOfficers" exists before dropping
if "companyOfficers" in StockInfo:
    del StockInfo["companyOfficers"]


def replace_empty_nested_dicts(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) and not value:
                data[key] = 0
            else:
                replace_empty_nested_dicts(value)
    return data


StockInfo = replace_empty_nested_dicts(StockInfo)

StockInfo_df = pd.DataFrame.from_dict(StockInfo, orient='index').reset_index()

# Convert all columns to strings
StockInfo_df = StockInfo_df.astype(str)
# st.write(StockInfo_df)

# Define the desired order of columns
desired_order_StockInfo = [
    "address1", "city", "state", "zip", "country", "phone", "website",
    "industry", "industryKey", "industryDisp", "sector", "sectorKey", "sectorDisp",
    "longBusinessSummary", "fullTimeEmployees", "auditRisk", "boardRisk", "compensationRisk",
    "shareHolderRightsRisk", "overallRisk", "governanceEpochDate", "compensationAsOfEpochDate",
    "maxAge", "priceHint", "previousClose", "open", "dayLow", "dayHigh",
    "regularMarketPreviousClose", "regularMarketOpen", "regularMarketDayLow", "regularMarketDayHigh",
    "dividendRate", "dividendYield", "exDividendDate", "payoutRatio", "fiveYearAvgDividendYield",
    "beta", "trailingPE", "forwardPE", "volume", "regularMarketVolume", "averageVolume",
    "averageVolume10days", "averageDailyVolume10Day", "bidSize", "askSize", "marketCap",
    "fiftyTwoWeekLow", "fiftyTwoWeekHigh", "priceToSalesTrailing12Months", "fiftyDayAverage",
    "twoHundredDayAverage", "trailingAnnualDividendRate", "trailingAnnualDividendYield", "currency",
    "enterpriseValue", "profitMargins", "floatShares", "sharesOutstanding", "sharesShort",
    "sharesShortPriorMonth", "sharesShortPreviousMonthDate", "dateShortInterest", "sharesPercentSharesOut",
    "heldPercentInsiders", "heldPercentInstitutions", "shortRatio", "shortPercentOfFloat",
    "impliedSharesOutstanding",
    "bookValue", "priceToBook", "lastFiscalYearEnd", "nextFiscalYearEnd", "mostRecentQuarter",
    "earningsQuarterlyGrowth", "netIncomeToCommon", "trailingEps", "forwardEps", "pegRatio",
    "lastSplitFactor", "lastSplitDate", "enterpriseToRevenue", "enterpriseToEbitda", "52WeekChange",
    "SandP52WeekChange", "lastDividendValue", "lastDividendDate", "exchange", "quoteType",
    "symbol", "underlyingSymbol", "shortName", "longName", "firstTradeDateEpochUtc", "timeZoneFullName",
    "timeZoneShortName", "uuid", "messageBoardId", "gmtOffSetMilliseconds", "currentPrice",
    "targetHighPrice", "targetLowPrice", "targetMeanPrice", "targetMedianPrice", "recommendationMean",
    "recommendationKey", "numberOfAnalystOpinions", "totalCash", "totalCashPerShare", "ebitda",
    "totalDebt", "quickRatio", "currentRatio", "totalRevenue", "debtToEquity", "revenuePerShare",
    "returnOnAssets", "returnOnEquity", "freeCashflow", "operatingCashflow", "earningsGrowth",
    "revenueGrowth", "grossMargins", "ebitdaMargins", "operatingMargins", "financialCurrency",
    "trailingPegRatio"
]

# Define the desired order of labels for Stock Info
label_mapping = {
    'Last Price': 'currentPrice',
    'Previous Close': 'previousClose',
    'Open': 'regularMarketOpen',
    'Day High': 'dayHigh',
    'Day Low': 'dayLow',
    '52 Week Low': 'fiftyTwoWeekLow',
    '52 Week High': 'fiftyTwoWeekHigh',
    '50d Average Price': 'fiftyDayAverage',
    '200d Average Price': 'twoHundredDayAverage',
    'Volume': 'volume',
    'Avg. Volume (10d)': 'averageVolume10days',
    'Shares Outstanding (In M$)': 'sharesOutstanding',
    'Market Cap (In B$)': 'marketCap',
    'Company EV': 'enterpriseValue',
    'PE Ratio (TTM)': 'trailingPE',
    'Price to Sales (TTM)': 'priceToSalesTrailing12Months',
    'Beta (5Y Monthly)': 'beta',
    'Dividend Yield': 'dividendYield',
    'Dividend': 'dividendRate',
    'Short % Of Float': 'shortPercentOfFloat',
    'Shares Short': 'sharesShort',
    '1YTarget Est': 'targetMeanPrice',
}

pairs = [
    ('Last Price', 'Market Cap (In B$)'),
    ('Previous Close', 'Company EV'),
    ('Open', 'PE Ratio (TTM)'),
    ('Day High', 'Price to Sales (TTM)'),
    ('Day Low', 'Beta (5Y Monthly)'),
    ('52 Week Low', 'Dividend Yield'),
    ('52 Week High', 'Dividend'),
    ('50d Average Price', '1YTarget Est'),
    ('200d Average Price', 'Short % Of Float'),
    ('Volume', 'Shares Short'),
    ('Avg. Volume (10d)', ''),
    ('Shares Outstanding (In M$)', '')
]

# col1, col2 = st.columns([0.5, 0.5])  # Adjust the width ratio of col1 and col2 as needed
col1, col2, col3 = st.columns([0.45, 0.45, 0.05])  # Adjust the width ratio of col1 and col2 as needed

with col1:

    # Display stock summary with reduced spacing
   
    st.subheader(f'Stock Summary - {StockInfo["shortName"]}')
    st.write(f"<h1 style='color:blue; font-size: smaller; margin-bottom: 0px;'>{StockInfo['sector']}</h1>", unsafe_allow_html=True)
    st.write(f"<h1 style='color:blue; font-size: smaller; margin-bottom: 0px;'>{StockInfo['industry']}</h1>", unsafe_allow_html=True)



    
    # st.subheader(f'Stock Summary - {StockInfo["shortName"]}')
    # st.markdown(f"<h1 style='color:blue; font-size: smaller; margin-bottom: 1px;'>{StockInfo['sector']}</h1>", unsafe_allow_html=True)
    # st.markdown(f"<h1 style='color:blue; font-size: smaller; margin-bottom: 1px;'>{StockInfo['industry']}</h1>", unsafe_allow_html=True)
    
    # # Display the industry description
    # st.write(StockInfo['industryDisp'])

    # st.markdown(f"<h1 style='color:blue; font-size: smaller;'>{StockInfo['sectorDisp']}</h1>", unsafe_allow_html=True)


    st.markdown("## \n\n\n")  # Add an empty line

    # Iterate through pairs and display labels with values or "N/A"
    for label1, label2 in pairs:
        key1 = label_mapping.get(label1)
        key2 = label_mapping.get(label2) if label2 else None
        value1 = StockInfo.get(key1, 'N/A') if key1 else 'N/A'
        value2 = StockInfo.get(key2, 'N/A') if key2 else 'N/A'

        # Format labels with HTML for bold
        formatted_label1 = f"{label1}" if label1 else ''
        formatted_label2 = f"{label2}" if label2 else ''

        # Divide values by billions or millions based on labels
        if key1 and label1 in ['Market Cap (In B$)', 'Company EV']:
            value1 = int(float(
                value1) / 1_000_000_000) if value1 != 'N/A' else 'N/A'  # Divide by billions and convert to integer
        elif key1 == 'Avg. Volume (10d)' or key1 == 'Volume':
            value1 = int(float(value1)) if value1 != 'N/A' else 'N/A'  # Convert to integer
            if label1 == 'Avg. Volume (10d)' or label1 == 'Volume':
                value1 = f"{int(value1):,}" if value1 != 'N/A' else 'N/A'  # Format without decimal places

        if key2 and label2 in ['Market Cap (In B$)', 'Company EV']:
            value2 = int(float(
                value2) / 1_000_000_000) if value2 != 'N/A' else 'N/A'  # Divide by billions and convert to integer
        elif key2 == 'Avg. Volume (10d)' or key2 == 'Volume':
            value2 = int(float(value2)) if value2 != 'N/A' else 'N/A'  # Convert to integer
            if label2 == 'Avg. Volume (10d)' or label2 == 'Volume':
                value2 = f"{int(value2):,}" if value2 != 'N/A' else 'N/A'  # Format without decimal places

        if label1 == 'Short % Of Float':
            value1 = float(value1) * 100 if value1 != 'N/A' else 'N/A'  # Multiply by 100 to convert to percentage
            formatted_value1 = f"{value1:.2%}" if value1 != 'N/A' else 'N/A'
        else:
            formatted_value1 = f"{value1}" if value1 != 'N/A' else 'N/A'

        if label2 == 'Short % Of Float':
            value2 = float(value2) * 100 if value2 != 'N/A' else 'N/A'  # Multiply by 100 to convert to percentage
            formatted_value2 = f"{value2:.2%}" if value2 != 'N/A' else ''
        else:
            formatted_value2 = f"{value2}" if value2 != 'N/A' else ''

        # Format Dividend Yield and Dividend
        if label1 == 'Dividend Yield':
            formatted_value1 = f"{value1:.2%}" if value1 != 'N/A' else 'N/A'
        elif label1 == 'Dividend':
            formatted_value1 = f"{value1:,.2f}" if value1 != 'N/A' else 'N/A'
        else:
            formatted_value1 = f"{value1:,.2f}" if value1 != 'N/A' else 'N/A'

        if label2 == 'Dividend Yield':
            formatted_value2 = f"{value2:.2%}" if value2 != 'N/A' else ''
        elif label2 == 'Dividend':
            formatted_value2 = f"{value2:,.2f}" if value2 != 'N/A' else ''
        else:
            formatted_value2 = f"{value2:,.2f}" if value2 != 'N/A' else ''

            # Calculate the difference and percentage change if label1 or label2 is "Last Price"
            if label1 == 'Last Price':
                value1 = float(value1)
                previous_close = float(StockInfo.get('previousClose', 0))
                last_price_difference = value1 - previous_close
                percentage_change = (last_price_difference / previous_close) * 100 if previous_close != 0 else 0
                change_color = "green" if last_price_difference >= 0 else "red"
                formatted_value1 += f" ({last_price_difference:+.2f},{percentage_change:+.2f}%)"

        # Combine label and value, ensuring the total length is up to 40 characters
        label1_value1 = f"{formatted_label1}: {formatted_value1}"
        label2_value2 = f"{formatted_label2}: {formatted_value2}" if formatted_label2 else ''

        # Display pairs in the same line without the "|" string
        st.text(f"{label1_value1:<40} {label2_value2}")

    st.markdown("## \n\n\n")
    st.subheader(f'Company Summery')

    st.markdown("## \n\n\n\n\n")  # Add an empty line
    st.write(StockInfo['longBusinessSummary'])
    st.markdown("## \n\n")  # Add an empty line
    st.write("Company Website:", StockInfo['website'])

# Column 2: *****************************      Candlestick chart         ***************************
with col2:
    # Centered subheader with dates
    st.markdown("<h3 style='text-align: center;'>Stock Chart For Dates Entered: {} to {}</h3>".format(
        start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")), unsafe_allow_html=True)

    if df_ticker.empty:
        st.warning(f"No data found for {ticker} in the selected date range.")
    else:
        candlestick_chart = go.Figure(data=[go.Candlestick(x=df_ticker['Date'],
                                                           open=df_ticker['Open'],
                                                           high=df_ticker['High'],
                                                           low=df_ticker['Low'],
                                                           close=df_ticker['Close'])])

        candlestick_chart.update_layout(xaxis_rangeslider_visible=False,
                                        xaxis=dict(type='date', range=[start_date, end_date]),
                                        height=600)

        # Hide Plotly toolbar and directly display the chart

        st.plotly_chart(candlestick_chart, use_container_width=True, config={'displayModeBar': False})

        if st.checkbox('Show Stock Price History Data'):
            st.subheader('Stock History')
            st.write(df_ticker)



