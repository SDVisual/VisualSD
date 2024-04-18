# import datetime
from datetime import datetime, timedelta
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

# Display the image with the caption
# st.image('Logo.png')


# # Define sidebar elements
# st.sidebar.image("Side.png", use_column_width=True)

# # Display title with blue color using Markdown
# st.markdown(f"<h1 style='color:blue;'>{APP_NAME}</h1>", unsafe_allow_html=True)


# Initialize session state for selected ticker index and valid tickers
if 'selected_ticker_index' not in st.session_state:
    st.session_state.selected_ticker_index = 0

if 'valid_tickers' not in st.session_state:
    st.session_state.valid_tickers = []

# Retrieve the last valid symbol entered by the user, default to 'AAPL' if none
DEFAULT_SYMBOL = st.session_state.valid_tickers[-1] if st.session_state.valid_tickers else 'AAPL'



# Input box for user to enter symbol
new_symbol = st.text_input("Add Stock Symbol to Symbols List (e.g., AAPL)").strip().upper()



# Check if the entered symbol is empty or consists only of whitespace characters
if not new_symbol or new_symbol.isspace():
    new_symbol = DEFAULT_SYMBOL



# Check if the entered symbol is valid
historical_data = yf.Ticker(new_symbol).history(period='1d')
income_statement = yf.Ticker(new_symbol).income_stmt

if new_symbol != DEFAULT_SYMBOL and historical_data.empty or income_statement.empty:
    st.error("Invalid symbol. Please enter Only Stocks symbols.")

else:
    # Add valid symbol to session state if it's not already present
    if new_symbol not in st.session_state.valid_tickers:
        st.session_state.valid_tickers.append(new_symbol)
        st.text(f"Symbol Added to Symbols List - {new_symbol} ")

        # Update selected ticker index to the newly added symbol
        st.session_state.selected_ticker_index = len(st.session_state.valid_tickers) - 1



# Retrieve the index of the selected ticker symbol from the session state
selected_ticker_index = st.session_state.selected_ticker_index

# Select box to choose ticker
ticker = st.sidebar.selectbox('Symbols List - Select Box', st.session_state.valid_tickers,
                              index=selected_ticker_index)

# Update session state with the newly selected symbol index
st.session_state.selected_ticker_index = st.session_state.valid_tickers.index(ticker)

# # Sidebar date inputs
# start_date = st.sidebar.date_input('Start date - Historical Prices', datetime(2000, 1, 1))
# end_date = st.sidebar.date_input('End date', datetime.now().date())


# Display a message box in the sidebar
st.sidebar.info("- For the best experience, maximize your screen.")
st.sidebar.info("- Close side bar for better visualization.")
st.sidebar.info("- recommend dark mode in setting menu.")
st.sidebar.info("- רונן אתה אלוף")


# df_ticker = yf.download(ticker, start=start_date, end=end_date).reset_index()

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
    'Avg.Volume (10d)': 'averageVolume10days',
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
    'Forward PE': 'forwardPE'
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
    ('200d Average Price', 'Forward PE'),
    ('Volume', 'Shares Short'),
    ('Avg.Volume (10d)', 'Short % Of Float'),
    ('Shares Outstanding (In M$)', '')
]


col1, col2, col3, col4 = st.columns([0.3, 0.03, 0.40, 0.10])  # Adjust the width ratio of col1 and col2 as needed




with col1:

    # Define the color code
    color_code = "#0ECCEC"


    font_size = "30px"  # You can adjust the font size as needed

    # Render subheader with customized font size and color
    st.markdown(f'<h2 style="color:{color_code}; font-size:{font_size}">{StockInfo["shortName"]}</h2>', unsafe_allow_html=True)

    # Write the sector and industry with custom styling
    st.write(
        f"<h1 style='font-size: larger; margin-bottom: 5px; display: inline;'>Sector - {StockInfo['sector']}</h1>"
        f"<h1 style='font-size: larger; margin-bottom: 5px; display: inline;'>Industry - {StockInfo['industry']}</h1>",
        unsafe_allow_html=True
    )



    st.write("")
    st.write("")
    st.write("")

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
        st.text(f"{label1_value1:<45} {label2_value2}")

    st.write("")
    st.write("")




    st.subheader(f'Company Summery')



    st.write(StockInfo['longBusinessSummary'])
    st.write("Full Time Employees:", str(StockInfo['fullTimeEmployees']))
    st.write("Company Website:", StockInfo['website'])

    st.write("")
    st.subheader("Recommendation")
    st.write("Number Of Analyst Opinions: ",
             "<span style='font-size: 16px;'>" + str(StockInfo['numberOfAnalystOpinions']) + "</span>",
             unsafe_allow_html=True)
    st.write("Recommendation Key: ",
             "<span style='font-size: 16px;'>" + StockInfo['recommendationKey'].upper() + "</span>",
             unsafe_allow_html=True)
    st.write("Target Low Price: ", "<span style='font-size: 16px;'>" + str(StockInfo['targetLowPrice']) + "</span>",
             unsafe_allow_html=True)
    st.write("Target Mean Price: ", "<span style='font-size: 16px;'>" + str(StockInfo['targetMeanPrice']) + "</span>",
             unsafe_allow_html=True)
    st.write("Target High Price: ", "<span style='font-size: 16px;'>" + str(StockInfo['targetHighPrice']) + "</span>",
             unsafe_allow_html=True)


    # st.write("Number Of Analyst Opinions:", StockInfo['numberOfAnalystOpinions'])
    # st.write("Recommendation Key:", StockInfo['recommendationKey'].upper())
    # st.write("Target Low Price:", StockInfo['targetLowPrice'])
    # st.write("Target Mean Price:", StockInfo['targetMeanPrice'])
    # st.write("Target High Price:", StockInfo['targetHighPrice'])
    # st.write("")
    # st.write("")


    # st.write(StockInfo)

with col2:
    st.write("")
    st.write("")

    # Fetch data based on the selected time period or default to '1Y'
    selected_time_period = st.session_state.get('selected_time_period', '1Y')
    df_ticker = yf.download(ticker, period='max').reset_index()
    end_date = datetime.now()
    # Buttons for selecting different time periods
    time_periods = ['7D', '2M', '6M', 'YTD', '1Y', '5Y', 'MAX']
    for _ in range(6):
        st.write("")

    # Display buttons in a single row
    button_container = st.container()

    # # # Initialize selected_time_period to '1Y' as default
    # selected_time_period = '1Y'

    with button_container:
        button_spacing = 7  # Adjust spacing between buttons
        st.write('<style>div.row-widget.stHorizontal {flex-wrap: nowrap;}</style>', unsafe_allow_html=True)

        for period in time_periods:
            if st.button(period):
                selected_time_period = period
                st.session_state.selected_time_period = period

    # Calculate start date based on selected time period

    if selected_time_period == '7D':
        start_date = end_date - timedelta(days=7)
    elif selected_time_period == '2M':
        start_date = end_date - timedelta(days=60)
    elif selected_time_period == '6M':
        start_date = end_date - timedelta(days=180)
    elif selected_time_period == 'YTD':
        start_date = datetime(end_date.year, 1, 1)
    elif selected_time_period == '1Y':
        start_date = end_date - timedelta(days=365)
    elif selected_time_period == '5Y':
        start_date = end_date - timedelta(days=5 * 365)
    else:  # 'MAX'
        start_date = df_ticker['Date'].min()  # Get the earliest date from the dataframe

    # Fetch data for the selected time period again
    df_ticker = yf.download(ticker, start=start_date, end=end_date).reset_index()

with col3:
    st.write("")
    st.write("")
    if df_ticker.empty:
        st.warning(f"No data found for {ticker} in the selected date range.")
    else:
        # Calculate additional information
        max_price = df_ticker['High'].max()
        min_price = df_ticker['Low'].min()
        range_low_to_high = ((max_price - min_price) / min_price) * 100

        initial_close = df_ticker.iloc[0]['Close']  # Closing price for the oldest date
        final_close = df_ticker.iloc[-1]['Close']  # Closing price for the earliest date
        yield_percentage = (((final_close / initial_close) - 1) * 100)

        # Determine color based on yield
        yield_color = 'red' if yield_percentage < 0 else 'green'



        candlestick_chart = go.Figure()

        # Add candlestick trace
        candlestick_chart.add_trace(go.Candlestick(x=df_ticker['Date'],
                                                    open=df_ticker['Open'],
                                                    high=df_ticker['High'],
                                                    low=df_ticker['Low'],
                                                    close=df_ticker['Close'],
                                                    name='Candlestick'))

        # Add volume bars in light blue
        candlestick_chart.add_trace(go.Bar(x=df_ticker['Date'],
                                            y=df_ticker['Volume'],
                                            yaxis='y2',
                                            name='Shares Volume',
                                            marker_color='rgba(52, 152, 219, 0.3)'))

        # Set the title of the chart with both main and additional information
        candlestick_chart.update_layout(
            title_text="<span style='text-align: center;'>Stock Chart For Dates: {} to {}</span><br>"
                       "<span style='font-size: 18px;'>Low Price: {:.2f} | High Price: {:.2f} | Range Low To High: {:.2f}%</span><br>"
                       "<span style='font-size: 18px;'>                                                  Return for the period: <span style='color:{};'>{:.2f}%</span></span>".format(
                start_date.strftime("%d-%m-%Y"), end_date.strftime("%d-%m-%Y"),
                min_price, max_price, range_low_to_high, yield_color, yield_percentage),
            title_x=0.2,  # Center the title
            title_font_size=25,  # Increase font size
            title_y=0.95,  # Adjust title vertical position
            title_yanchor='top',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)  # Adjust legend position
        )

        candlestick_chart.update_layout(xaxis_rangeslider_visible=False,
                                        xaxis=dict(type='date', range=[start_date, end_date]),
                                        yaxis=dict(title='Price', showgrid=True),  # Remove gridlines from y-axis
                                        yaxis2=dict(title='Volume',
                                                    overlaying='y',
                                                    side='right',  # Move to the right side
                                                    position=1,  # Move outside the plot area
                                                    showgrid=False),  # Remove gridlines from y2-axis
                                        height=600)

        # Hide Plotly toolbar and directly display the chart
        st.plotly_chart(candlestick_chart, use_container_width=True, config={'displayModeBar': False})

    if st.checkbox('Show Stock Price History Data'):
        st.subheader('Stock History - {}'.format(selected_time_period))
        sorted_df = df_ticker.sort_values(by='Date', ascending=False)
        st.write(sorted_df)

#









