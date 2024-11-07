
from datetime import datetime, timedelta
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import cufflinks as cf
import matplotlib.pyplot as plt

APP_NAME = "Visual Stock Data"

# Set page configuration
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="auto")

col1, col2 = st.columns([0.7, 0.3])

StockInfo = {}
StockInfo_df = pd.DataFrame()

color_code = "#0ECCEC"
header_html = f'<h2 style="color:{color_code};">{APP_NAME} </h2>'
st.markdown(header_html, unsafe_allow_html=True)

# Initialize session state for selected ticker index and valid tickers
if 'selected_ticker_index' not in st.session_state:
    st.session_state.selected_ticker_index = 0

if 'valid_tickers' not in st.session_state:
    st.session_state.valid_tickers = []

# Retrieve the last valid symbol entered by the user, default to 'AAPL' if none
DEFAULT_SYMBOL = st.session_state.valid_tickers[-1] if st.session_state.valid_tickers else 'AAPL'


col1, col2 = st.columns([0.35, 0.65])

with col1:

    # st.write('<hr style="height:4px;border:none;color:#0ECCEC;background-color:#0ECCEC;">', unsafe_allow_html=True)

    # Input box for user to enter symbol
    new_symbol = st.text_input("Add symbol to Symbols List (Example given, AAPL)",
                               placeholder="Add Stock Symbol").strip().upper()

    # Check if the entered symbol is empty or consists only of whitespace characters
    if not new_symbol or new_symbol.isspace():
        new_symbol = DEFAULT_SYMBOL

    # Check if the entered symbol is valid
    historical_data = yf.Ticker(new_symbol).history(period='1d')
    income_statement = yf.Ticker(new_symbol).income_stmt

    if new_symbol != DEFAULT_SYMBOL and historical_data.empty or income_statement.empty:

        st.error("Invalid symbol. Please enter only Stocks symbols.")

    else:
        # Add valid symbol to session state if it's not already present
        if new_symbol not in st.session_state.valid_tickers:
            st.session_state.valid_tickers.append(new_symbol)
            st.text(f"{new_symbol} - Added to Symbols List")

            # Update selected ticker index to the newly added symbol
            st.session_state.selected_ticker_index = len(st.session_state.valid_tickers) - 1

    # Retrieve the index of the selected ticker symbol from the session state
    selected_ticker_index = st.session_state.selected_ticker_index


    # st.write("")

col1, col2, col3, col4 = st.columns([0.35, 0.35, 0.15, 0.15])


    ############################   SP 500 SELECT SECTORS ######################################

with col1:

    # Initialize session state if it doesn't already exist
    if 'valid_tickers' not in st.session_state:
        st.session_state.valid_tickers = []
    if 'selected_sector' not in st.session_state:
        st.session_state.selected_sector = ""
    if 'selected_sub_industry' not in st.session_state:
        st.session_state.selected_sub_industry = ""
    if 'selected_ticker_index' not in st.session_state:
        st.session_state.selected_ticker_index = 0

    # Load the data from the webpage
    sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500data = pd.read_html(sp500url)[0]

    # Extract relevant columns
    data = sp500data[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry',
                      'Headquarters Location', 'Date added', 'Founded']]



    # Dropdown for GICS Sector
    gics_sector = st.selectbox('Or Select S&P 500 Sectors & Companies', ["Select a sector"] + sorted(data['GICS Sector'].unique()))

    # # Display a message if "Financials" is selected
    # if gics_sector == "Financials":
    #     st.warning("Note: This App Version is less suitable for stocks in the Finance Industry")

    # Filter data based on selected GICS Sector
    filtered_data = data[data['GICS Sector'] == gics_sector]

# Display sub-industry dropdown in col2 only if a sector is selected
with col2:


    # Display sub-industry dropdown only if a sector is selected
    if gics_sector != "Select a sector":

        # Optional dropdown for GICS Sub-Industry
        gics_sub_industry = st.selectbox('Optional', ["Select a sub-industry"] + sorted(
            filtered_data['GICS Sub-Industry'].unique()))
        filtered_data = filtered_data[filtered_data['GICS Sub-Industry'] == gics_sub_industry] if gics_sub_industry != "Select a sub-industry" else filtered_data
    else:
        gics_sub_industry = "Select a sub-industry"

    # Filter out symbols containing a dot (.) in their names
    filtered_data = filtered_data[~filtered_data['Symbol'].str.contains(r'\.')]

    # Update the session state with the selected sector's and sub-industry's tickers only if a new sector or sub-industry is selected
    if gics_sector != "Select a sector" and (
            gics_sector != st.session_state.selected_sector or gics_sub_industry != st.session_state.selected_sub_industry):
        st.session_state.valid_tickers = filtered_data['Symbol'].tolist()
        st.session_state.selected_sector = gics_sector
        st.session_state.selected_sub_industry = gics_sub_industry
        # Reset the selected ticker index if the valid_tickers list is updated
        st.session_state.selected_ticker_index = 0


col1, col2 = st.columns([0.7, 0.3])
with col1:

    # Display the filtered data

    # Check if filtered_data is not empty
    if filtered_data is not None and not filtered_data.empty:
        show_info = st.checkbox('Show selected Sector & companies information')

        if show_info:
            st.write('Filtered Sector Companies Info:')
            st.dataframe(filtered_data)


    # Select box to choose ticker from the sidebar
    ticker = st.sidebar.selectbox('Symbols List - Select Box', st.session_state.valid_tickers,
                                  index=st.session_state.selected_ticker_index)

    # Update session state with the newly selected symbol index
    if ticker in st.session_state.valid_tickers:
        st.session_state.selected_ticker_index = st.session_state.valid_tickers.index(ticker)




    if st.sidebar.button('Clear Symbols List'):
        st.session_state.valid_tickers = []
        st.session_state.selected_ticker_index = 0
        st.sidebar.success("List cleared. Make New Symbols List !")





# Display a message box in the sidebar
st.sidebar.info("- For the best experience, maximize your screen.")
st.sidebar.info("- Compare Ratios/Profitability Between Your Symbols.")
st.sidebar.info("- Easy Download Data Tables.")
# st.sidebar.info("- Recommended dark mode in setting menu.")
st.sidebar.info("- This App Version is Less suitable for stocks in the Finance Industry")
st.sidebar.markdown("&copy;VisualSD. All rights reserved.", unsafe_allow_html=True)



col1, col2 = st.columns([0.7, 0.3])

with col1:


    st.write('<hr style="height:4px;border:none;color:#0ECCEC;background-color:#0ECCEC;">', unsafe_allow_html=True)
    StockInfo = yf.Ticker(ticker).info
    rec_summery = yf.Ticker(ticker).recommendations_summary

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
        'S.Outstanding (B)': 'sharesOutstanding',
        'Market Cap (In B$)': 'marketCap',
        'Company EV': 'enterpriseValue',
        'PE Ratio(TTM)': 'trailingPE',
        'Price to Sales(TTM)': 'priceToSalesTrailing12Months',
        'Beta (5Y Monthly)': 'beta',
        'Dividend Yield': 'dividendYield',
        'Dividend': 'dividendRate',
        'Short % Of Float': 'shortPercentOfFloat',
        'Shares Short': 'sharesShort',
        '1YTarget Est': 'targetMeanPrice',
        'Gross Margins(TTM)': 'grossMargins',
        'Operating Margins(TTM)': 'operatingMargins',
        'Revenue(TTM)': 'totalRevenue',
        'ROE(TTM)': 'returnOnEquity',
        'ROA(TTM)': 'returnOnAssets',
        'Debt To Equity': 'debtToEquity',
        'Diluted EPS(TTM)': 'trailingEps',
        'Profit Margins(TTM)': 'profitMargins',
        'Forward PE': 'forwardPE',
        'Ebitda Margins(TTM)': 'ebitdaMargins',
        'Ebitda(TTM)': 'ebitda',
        'EV/Ebitda': 'enterpriseToEbitda'
    }



col1, col2, col3, col4 = st.columns([0.3, 0.03, 0.3, 0.01])  # Adjust the width ratio of col1 and col2 as needed

with col1:

    # Define the color code
    color_code = "#0ECCEC"

    font_size = "25px"  # You can adjust the font size as needed

    # Render subheader with customized font size and color
    st.markdown(f'<h2 style="color:{color_code}; font-size:{font_size}">{StockInfo["shortName"]}</h2>',
                unsafe_allow_html=True)

    # Write the sector and industry with custom styling
    st.write(
        f"<h1 style='font-size: larger; margin-bottom: 5px; display: inline;'>Exchange - {StockInfo['exchange']}</h1>",
        f"<h1 style='font-size: larger; margin-bottom: 5px; display: inline;'>Sector - {StockInfo['sector']}</h1>",
        f"<h1 style='font-size: larger; margin-bottom: 5px; display: inline;'>Industry - {StockInfo['industry']}</h1>",


        unsafe_allow_html=True
    )

    st.write("")
    st.write("")


col1, col2, col3, col4 = st.columns([0.5, 0.2, 0.09, 0.01])  # Adjust the width ratio of col1 and col2 as needed

with col2:

    st.write("")
    st.write("")

    # Fetch data based on the selected time period or default to '1Y'
    selected_time_period = st.session_state.get('selected_time_period', '3M')
    df_ticker = yf.download(ticker, period='max').reset_index()
    end_date = datetime.now()
    # Buttons for selecting different time periods
    time_periods = ['7D', '3M', '6M', 'YTD', '1Y', '5Y', 'MAX']
    for _ in range(1):
        st.write("")

    # Display buttons in a single row
    button_container = st.container()



    with button_container:
        button_spacing = 1  # Adjust spacing between buttons
        st.write('<style>div.row-widget.stHorizontal {flex-wrap: nowrap;}</style>', unsafe_allow_html=True)

        for period in time_periods:
            if st.button(period):
                selected_time_period = period
                st.session_state.selected_time_period = period

    # Calculate start date based on selected time period


    if selected_time_period == '7D':
        start_date = end_date - timedelta(days=7)
    elif selected_time_period == '3M':
        start_date = end_date - timedelta(days=90)
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

    if selected_time_period == '1D':
        # Fetch data for the selected time period again
        df_ticker = yf.download(ticker, start=start_date, end=end_date).reset_index()
    else:
        # Fetch data for the selected time period again
        df_ticker = yf.download(ticker, start=start_date, end=end_date).reset_index()



# with col1:


#     # Check if the DataFrame is empty
#     if df_ticker.empty:
#         st.warning(f"No data found for {ticker} in the selected date range.")
#     else:
#         # Filter the DataFrame to exclude non-trading days
#         df_ticker = df_ticker[df_ticker['Volume'] > 0]

#         # Calculate additional information
#         max_price = df_ticker['High'].max()
#         min_price = df_ticker['Low'].min()
#         range_low_to_high = ((max_price - min_price) / min_price) * 100

#         initial_close = df_ticker.iloc[0]['Close']  # Closing price for the oldest date
#         final_close = df_ticker.iloc[-1]['Close']  # Closing price for the latest date
#         yield_percentage = (((final_close / initial_close) - 1) * 100)

#         # Determine color based on yield
#         yield_color = 'red' if yield_percentage < 0 else 'green'

       
#         candlestick_chart = go.Figure()

#         # Add candlestick trace
#         candlestick_chart.add_trace(go.Candlestick(
#             x=df_ticker['Date'],
#             open=df_ticker['Open'],
#             high=df_ticker['High'],
#             low=df_ticker['Low'],
#             close=df_ticker['Close'],
#             name='Candlestick'
#         ))

#         # Add volume bars in light blue
#         candlestick_chart.add_trace(go.Bar(
#             x=df_ticker['Date'],
#             y=df_ticker['Volume'],
#             yaxis='y2',
#             name='Shares Volume',
#             marker_color='rgba(52, 152, 219, 0.3)'
#         ))

#         # Add line trace for close prices
#         candlestick_chart.add_trace(go.Scatter(
#             x=df_ticker['Date'],
#             y=df_ticker['Close'],
#             mode='lines',
#             name='Close Price',
#             line=dict(color='lightblue', width=2),
#             showlegend=True
#         ))

#         # Set the title of the chart with both main and additional information
#         candlestick_chart.update_layout(
#             title_text="<span style='text-align: center;'>                           {} Chart </span><br>"
#                        "<span style='font-size: 18px;'>Low: {:.2f} | High: {:.2f} | Range: {:.2f}%</span><br>"
#                        "<span style='font-size: 18px;'>                 Return for the period: <span style='color:{};'>{:.2f}%</span></span>".format(
#                 selected_time_period, min_price, max_price, range_low_to_high, yield_color, yield_percentage),
#             title_x=0.25,  # Center the title
#             title_font_size=22,  # Increase font size
#             title_y=0.95,  # Adjust title vertical position
#             title_yanchor='top',
#             legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)  # Adjust legend position
#         )


#         candlestick_chart.update_layout(
#             xaxis_rangeslider_visible=False,
#             xaxis=dict(type='date',  # Set type to 'date'
#                        range=[start_date, end_date],
#                        rangebreaks=[dict(bounds=["sat", "mon"])],  # Adjust this based on your non-trading days
#                        ),
#             yaxis=dict(title='Price', showgrid=True),
#             yaxis2=dict(
#                 title='',
#                 overlaying='y',
#                 side='right',  # Move to the right side
#                 position=1,  # Move outside the plot area
#                 showgrid=False  # Remove gridlines from y2-axis
#             ),
#             height=500
#         )

#         # Hide Plotly toolbar and directly display the chart
#         st.plotly_chart(candlestick_chart, use_container_width=True, config={'displayModeBar': False})


with col1:
    # Check if the DataFrame is empty
    if df_ticker.empty:
        st.warning(f"No data found for {ticker} in the selected date range.")
    else:
        # Filter the DataFrame to exclude non-trading days
        df_ticker = df_ticker[df_ticker['Volume'] > 0]

        # Calculate additional information
        max_price = df_ticker['High'].max()
        min_price = df_ticker['Low'].min()
        range_low_to_high = ((max_price - min_price) / min_price) * 100

        initial_close = float(df_ticker.iloc[0]['Close'])  # Ensure single numeric value
        final_close = float(df_ticker.iloc[-1]['Close'])   # Ensure single numeric value
        yield_percentage = (final_close / initial_close - 1) * 100
        yield_color = 'red' if yield_percentage < 0 else 'green'


        # Initialize candlestick chart
        candlestick_chart = go.Figure()

        # Add candlestick trace
        candlestick_chart.add_trace(go.Candlestick(
            x=df_ticker['Date'],
            open=df_ticker['Open'],
            high=df_ticker['High'],
            low=df_ticker['Low'],
            close=df_ticker['Close'],
            name='Candlestick'
        ))

        # Add volume bars
        candlestick_chart.add_trace(go.Bar(
            x=df_ticker['Date'],
            y=df_ticker['Volume'],
            yaxis='y2',
            name='Shares Volume',
            marker_color='rgba(52, 152, 219, 0.3)'
        ))

        # Add line for close prices
        candlestick_chart.add_trace(go.Scatter(
            x=df_ticker['Date'],
            y=df_ticker['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='lightblue', width=2),
            showlegend=True
        ))

        # Set chart title and formatting
        candlestick_chart.update_layout(
            title_text=(
                f"<span style='text-align: center;'>{ticker} Chart </span><br>"
                f"<span style='font-size: 18px;'>Low: {min_price:.2f} | High: {max_price:.2f} | "
                f"Range: {range_low_to_high:.2f}%</span><br>"
                f"<span style='font-size: 18px;'>Return: <span style='color:{yield_color};'>"
                f"{yield_percentage:.2f}%</span></span>"
            ),
            title_x=0.5,  # Center title horizontally
            title_font_size=22,
            title_y=0.95,
            title_yanchor='top',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )

        # Configure layout for axes and rangebreaks
        candlestick_chart.update_layout(
            xaxis_rangeslider_visible=False,
            xaxis=dict(
                type='date',
                range=[start_date, end_date],
                rangebreaks=[{"bounds": ["sat", "mon"]}]
            ),
            yaxis=dict(title='Price', showgrid=True),
            yaxis2=dict(
                title='',
                overlaying='y',
                side='right',
                position=1,
                showgrid=False
            ),
            height=500
        )

        # Display the chart in Streamlit
        st.plotly_chart(candlestick_chart, use_container_width=True, config={'displayModeBar': False})






col1, col2 = st.columns([0.8, 0.2])

with col1:


    # Initialize session state for checkbox and previous ticker
    if 'show_stock_data' not in st.session_state:
        st.session_state.show_stock_data = False

    if 'previous_ticker' not in st.session_state:
        st.session_state.previous_ticker = ''


    # Check if the ticker has changed
    ticker_changed = st.session_state.previous_ticker != ticker

    # Update the previous ticker to the current one
    st.session_state.previous_ticker = ticker

    # Reset checkbox if ticker changes
    if ticker_changed:
        st.session_state.show_stock_data = False

    st.write('Chart dates : {} to {}'.format(start_date.strftime("%d-%m-%Y"), end_date.strftime("%d-%m-%Y")))
    st.write("")

    # Generate a unique key for the checkbox based on the ticker
    checkbox_key = f"show_stock_data_{ticker}"

    # Checkbox for showing stock price history data
    show_stock_data = st.checkbox('Show Stock Price History Data', value=st.session_state.show_stock_data,
                                  key=checkbox_key)

    # Display the stock data only if the checkbox is checked
    if show_stock_data:
        st.subheader('Stock History - {}'.format(selected_time_period))
        # Assuming df_ticker is defined earlier in your code as the dataframe containing the stock data
        sorted_df = df_ticker.sort_values(by='Date', ascending=False)
        st.write(sorted_df)




pairs = [
    ('Last Price', 'Market Cap (In B$)'),
    ('Volume', 'Avg.Volume (10d)'),
    ('Previous Close', '52 Week Low' ),
    ('Open', '52 Week High'),
    ('Day High', 'Beta (5Y Monthly)'),
    ('Day Low', '50d Average Price'),
    ('Dividend', '200d Average Price'),
    ('Dividend Yield', 'S.Outstanding (B)'),
    ('1YTarget Est', 'Short % Of Float'),


    ('PE Ratio(TTM)', 'Diluted EPS(TTM)'),
    ('Forward PE', 'Ebitda(TTM)'),
    ('EV/Ebitda', 'Revenue(TTM)'),
    ('Price to Sales(TTM)', 'Gross Margins(TTM)'),
    ('ROA(TTM)', 'Operating Margins(TTM)'),
    ('ROE(TTM)', 'Profit Margins(TTM)'),

]

# 'Ebitda Margins(TTM)': 'ebitdaMargins',
# : 'ebitda'

col1, col2 = st.columns([0.4, 0.3])
with col1:
    st.write('<hr style="height:4px;border:none;color:#0ECCEC;background-color:#0ECCEC;">', unsafe_allow_html=True)


col1, col2 = st.columns([0.8, 0.2])

with col1:

    # Get the current date
    current_date = datetime.now().strftime("%d-%m-%Y")

    # Display the subheader
    st.subheader(f'**Trading Information**')

    # Display the date in a smaller font
    st.markdown(f"<p style='font-size:12px;'>As of Date: {current_date} NasdaqGS</p>", unsafe_allow_html=True)


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
        if key1 and label1 in ['Market Cap (In B$)', 'Company EV', 'Revenue(TTM)', 'Ebitda(TTM)']:
            value1 = float(value1) / 1_000_000_000 if value1 != 'N/A' else 'N/A'  # Divide by billions and convert to integer
        elif key1 == 'Avg. Volume (10d)' or key1 == 'Volume':
            value1 = int(float(value1)) if value1 != 'N/A' else 'N/A'  # Convert to integer
            if label1 == 'Avg. Volume (10d)' or label1 == 'Volume':
                value1 = f"{int(value1):,}" if value1 != 'N/A' else 'N/A'  # Format without decimal places

        if key2 and label2 in ['Market Cap (In B$)', 'Revenue(TTM)', 'S.Outstanding (B)', 'Ebitda(TTM)']:
            value2 = float(value2) / 1_000_000_000 if value2 != 'N/A' else 'N/A'  # Divide by billions and convert to integer
        elif key2 == 'Avg. Volume (10d)' or key2 == 'Volume':
            value2 = int(float(value2)) if value2 != 'N/A' else 'N/A'  # Convert to integer
            if label2 == 'Avg. Volume (10d)' or label2 == 'Volume':
                value2 = f"{int(value2):,}" if value2 != 'N/A' else 'N/A'  # Format without decimal places

        if label1 == 'Short % Of Float':
            value1 = float(
                value1) * 100 if value1 != 'N/A' else 'N/A'  # Multiply by 100 to convert to percentage
            formatted_value1 = f"{value1:.2%}" if value1 != 'N/A' else 'N/A'
        else:
            formatted_value1 = f"{value1}" if value1 != 'N/A' else 'N/A'

        if label2 == 'Short % Of Float':
            value2 = float(
                value2) * 100 if value2 != 'N/A' else 'N/A'  # Multiply by 100 to convert to percentage
            formatted_value2 = f"{value2:.2%}" if value2 != 'N/A' else ''
        else:
            formatted_value2 = f"{value2}" if value2 != 'N/A' else ''

        # Format Dividend Yield and Dividend
        if label1 == 'Dividend Yield':
            formatted_value1 = f"{value1:.2%}" if value1 != 'N/A' else 'N/A'
        elif label1 == 'Dividend':
            formatted_value1 = f"{value1:,.2f}" if value1 != 'N/A' else 'N/A'
        elif label1 == 'Volume':
            formatted_value1 = f"{int(value1):,}" if value1 != 'N/A' else 'N/A'

        else:
            formatted_value1 = f"{value1:,.2f}" if value1 != 'N/A' else 'N/A'

        if label2 == 'Dividend Yield':
            formatted_value2 = f"{value2:.2%}" if value2 != 'N/A' else ''

        if label1 == 'PE Ratio(TTM)':
            st.subheader(f'**Ratios/Profitability**')

        if label1 == 'ROA(TTM)':
            formatted_value1 = f"{value1:.2%}" if value1 != 'N/A' else 'N/A'
        if label1 == 'ROE(TTM)':
            formatted_value1 = f"{value1:,.2%}" if value1 != 'N/A' else 'N/A'



        if label2 == 'Operating Margins(TTM)':
            formatted_value2 = f"{value2:.2%}" if value2 != 'N/A' else ''
        elif label2 == 'Gross Margins(TTM)':
            formatted_value2 = f"{value2:.2%}" if value2 != 'N/A' else ''
        elif label2 == 'Profit Margins(TTM)':
            formatted_value2 = f"{value2:.2%}" if value2 != 'N/A' else ''
        elif label2 == 'Ebitda(TTM)':
            formatted_value2 = f"{value2:,.2f}B" if value2 != 'N/A' else ''

        elif label2 == 'Market Cap (In B$)':
            if value2 == 'N/A':
                formatted_value2 = ''
            elif value2 < 1:
                formatted_value2 = f"{value2:,.3f}B"
            else:
                formatted_value2 = f"{value2:,.2f}B"

        elif label2 == 'Revenue(TTM)':
            formatted_value2 = f"{value2:,.2f}B" if value2 != 'N/A' else ''
        elif label2 == 'Avg.Volume (10d)':
            formatted_value2 = f"{value2:,.0f}" if value2 != 'N/A' else ''
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



st.write("")
st.write("")

col1, col2 = st.columns([0.7, 0.3])


with col1:

    # Initialize session state variables if they don't exist
    if 'current_ticker' not in st.session_state:
        st.session_state.current_ticker = None

    # Define the elements to compare
    elements = [
        ('Choose All', 'all', None),  # Option to choose all elements
        ('Revenue(TTM)', 'totalRevenue', 'Billions'),
        ('PE Ratio(TTM)', 'trailingPE', '2 decimals'),
        ('Forward PE', 'forwardPE', '2 decimals'),
        ('Price to Sales(TTM)', 'priceToSalesTrailing12Months', '2 decimals'),
        ('Diluted EPS(TTM)', 'trailingEps', '2 decimals'),
        ('Gross Margins(TTM)', 'grossMargins', 'percentage'),
        ('Operating Margins(TTM)', 'operatingMargins', 'percentage'),
        ('Profit Margins(TTM)', 'profitMargins', 'percentage'),
        ('ROA(TTM)', 'returnOnAssets', 'percentage'),
        ('ROE(TTM)', 'returnOnEquity', 'percentage'),
        ('Beta (5Y Monthly)', 'beta', '2 decimals'),
        ('Quarterly Earnings Growth(YoY)', 'earningsGrowth', 'percentage'),
        ('Dividend Yield', 'dividendYield', 'percentage'),
        ('Current Ratio', 'currentRatio', '2 decimals'),
        ('Quick Ratio', 'quickRatio', '2 decimals')
    ]


    # Define function to display comparison table
    def display_comparison_table(selected_symbols, selected_elements):
        # Check if there are enough symbols for comparison
        if len(selected_symbols) < 2:
            st.warning("Not enough symbols for comparison. Please add more symbols.")
            return

        # Initialize an empty DataFrame to store comparison data
        comparison_df = pd.DataFrame(columns=[elem[0] for elem in elements])

        # Loop through selected tickers and fetch the stock information
        for ticker in selected_symbols:
            stock_info = yf.Ticker(ticker).info

            # Collect the elements for the current ticker
            data = {}
            for elem in elements:
                value = stock_info.get(elem[1], None)
                if value is not None:
                    if elem[2] == 'Billions':
                        value = f"{value / 1_000_000_000:.2f}B"
                    elif elem[2] == '2 decimals':
                        value = f"{value:.2f}"
                    elif elem[2] == 'percentage':
                        value = f"{value * 100:.2f}%"
                data[elem[0]] = value

            data['Ticker'] = ticker

            # Append the data to the comparison DataFrame using pd.concat
            comparison_df = pd.concat([comparison_df, pd.DataFrame([data])], ignore_index=True)

        # Set 'Ticker' as the index of the DataFrame
        comparison_df.set_index('Ticker', inplace=True)

        # Display the comparison table
        st.write("Comparison Table Of Ratios/Profitability")
        if 'Choose All' in selected_elements:
            selected_elements = [elem[0] for elem in elements if elem[0] != 'Choose All']
        st.dataframe(comparison_df[selected_elements])


    # Check if the visibility flag is set to True and the user clicks the button
    if st.button("Compare Ratios/Profitability Between Your Symbols"):
        if 'comparison_table_visible' not in st.session_state:
            st.session_state.comparison_table_visible = True

        st.session_state.comparison_table_visible = not st.session_state.comparison_table_visible

        # Check if there are enough symbols for comparison
        if len(st.session_state.valid_tickers) < 2:
            st.warning("Not enough symbols to compare. Please add symbols to your list.")
            st.session_state.comparison_table_visible = False

    # If ticker changes, reset all visibility states
    if st.session_state.current_ticker != ticker:
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
        selected_elements = st.multiselect("Select elements to compare:", [elem[0] for elem in elements])

        # Check if the user has selected at least one symbol and one element
        if st.button("Let's compare"):
            if selected_symbols and selected_elements:
                display_comparison_table(selected_symbols, selected_elements)
            else:
                st.warning("Please select at least TWO symbols and ONE element to compare.")
    else:
        # Turn off the visibility flag if the user switches symbols in the list
        st.session_state.comparison_table_visible = False




col1, col2 = st.columns([0.4, 0.3])  # Adjust the width ratio of col1 and col2 as needed

with col1:

    st.write('<hr style="height:4px;border:none;color:#0ECCEC;background-color:#0ECCEC;">', unsafe_allow_html=True)

    st.subheader(f'Company Summery')

    st.write(StockInfo['longBusinessSummary'])
    if 'fullTimeEmployees' in StockInfo:
        st.write("Full Time Employees:", str(StockInfo['fullTimeEmployees']))


    webfiling = "https://www.sec.gov/edgar/search"
    st.write("Company Website:", StockInfo['website'])
    st.write("Search Company Filing (8-k/10-k):", webfiling)


    st.write('<hr style="height:4px;border:none;color:#0ECCEC;background-color:#0ECCEC;">', unsafe_allow_html=True)

    st.subheader("Analyst Recommendation")

    calendar = yf.Ticker(ticker).calendar
    # Extracting the earnings dates
    calendar_dates = calendar["Earnings Date"]

    # Formatting the dates as strings
    calendar_dates_str = [date.strftime("%d-%m-%Y") for date in calendar_dates]

    # If there's only one earnings date, display it as a single date
    if len(calendar_dates_str) == 1:
        st.write("Next Earnings Date:", calendar_dates_str[0])
    else:
        # If there are multiple earnings dates, display them as a range
        st.write("Next Earnings Date:", " to ".join(calendar_dates_str))

    if pd.isna(StockInfo.get('recommendationKey')) or StockInfo.get(
            'recommendationKey') == "none":  # Check if recommendationKey is empty or equals "NONE"
        st.write("No Analyst Recommendation For Current Company")
    else:
        st.write("Number Of Analyst Opinions: ",
                 "<span style='font-size: 16px;'>" + str(StockInfo['numberOfAnalystOpinions']) + "</span>",
                 unsafe_allow_html=True)
        st.write("Recommendation Key: ",
                 "<span style='font-size: 16px;'>" + StockInfo['recommendationKey'].upper() + "</span>",
                 unsafe_allow_html=True)
        st.write("Target Low Price: ",
                 "<span style='font-size: 16px;'>" + str(StockInfo['targetLowPrice']) + "</span>",
                 unsafe_allow_html=True)
        st.write("Target Mean Price: ",
                 "<span style='font-size: 16px;'>" + str(StockInfo['targetMeanPrice']) + "</span>",
                 unsafe_allow_html=True)
        st.write("Target High Price: ",
                 "<span style='font-size: 16px;'>" + str(StockInfo['targetHighPrice']) + "</span>",
                 unsafe_allow_html=True)

        st.write('<hr style="height:4px;border:none;color:#0ECCEC;background-color:#0ECCEC;">',
                 unsafe_allow_html=True)

########################### Dividends #################################################################

        # Initialize session state for visibility
        if 'dividend_visibility' not in st.session_state:
            st.session_state.dividend_visibility = {
                "history": False,
                "chart": False
            }
        if 'visibility' not in st.session_state:
            st.session_state.visibility = {
                "institutional": False,
                "major": False,
                "insider": False,
                "transactions": False,
                "roster": False
            }
        if 'current_ticker' not in st.session_state:
            st.session_state.current_ticker = None


        # If ticker changes, reset all visibility states
        if st.session_state.current_ticker != ticker:
            st.session_state.dividend_visibility = {
                "history": False,
                "chart": False
            }
            st.session_state.visibility = {
                "institutional": False,
                "major": False,
                "insider": False,
                "transactions": False,
                "roster": False
            }
            st.session_state.current_ticker = ticker

        ########################### Dividends #################################################################



        calendar_data = yf.Ticker(ticker).calendar

        dividend_date = calendar_data.get('Dividend Date', None)
        ex_dividend_date = calendar_data.get('Ex-Dividend Date', None)

        output = ""
        if dividend_date is not None:
            output += f"Dividend Date: {dividend_date}  ||"

        if ex_dividend_date is not None:
            output += f"   Ex-Dividend Date: {ex_dividend_date}"

        st.write(output)


        # Get dividends data
        StockDiv = yf.Ticker(ticker).dividends

        # Check if dividends data is not empty
        if not StockDiv.empty:
            # Button to toggle dividends history visibility
            if st.button("Show Dividends History"):
                st.session_state.dividend_visibility["history"] = not st.session_state.dividend_visibility[
                    "history"]

            # Show dividends history if visibility is True
            if st.session_state.dividend_visibility["history"]:
                dividends_df = pd.DataFrame({'Dividends': StockDiv})
                dividends_df['Dividends Growth (%)'] = StockDiv.pct_change() * 100
                dividends_df = dividends_df.sort_index(ascending=False)
                st.write(dividends_df)

            # Check if the DataFrame has more than one row before showing the chart
            if len(StockDiv) > 1:
                # Button to toggle dividends history chart visibility
                if st.button("Show Dividends History Chart"):
                    st.session_state.dividend_visibility["chart"] = not st.session_state.dividend_visibility[
                        "chart"]

                # Show dividends history chart if visibility is True
                if st.session_state.dividend_visibility["chart"]:
                    st.line_chart(StockDiv)
            else:
                st.write("*Chart not available due to insufficient data*")
        else:
            st.write("*No Dividends History For Current Company*")

        st.write('<hr style="height:4px;border:none;color:#0ECCEC;background-color:#0ECCEC;">',
                 unsafe_allow_html=True)

        ########################### Holders #################################################################

        # Fetch data with exception handling
        try:
            StockInsider = yf.Ticker(ticker).insider_transactions
            StockInstitutional = yf.Ticker(ticker).institutional_holders
            StockMajor = yf.Ticker(ticker).major_holders
            StockInsiderPurchases = yf.Ticker(ticker).insider_purchases
            StockInsider_roster_holders = yf.Ticker(ticker).insider_roster_holders
            StockFund = yf.Ticker(ticker).mutualfund_holders
        except Exception as e:
            st.write("*Currently Stock Holders Not Available*")
            StockInsider = None
            StockInstitutional = None
            StockMajor = None
            StockInsiderPurchases = None
            StockInsider_roster_holders = None
            StockFund = None

        # Button to toggle major holders data visibility
        if not all(df is None for df in [StockMajor, StockInstitutional, StockFund]) and st.button("Major Holders"):
            st.session_state.visibility["major"] = not st.session_state.visibility["major"]

        # Display major holders data if visibility is True
        if st.session_state.visibility["major"]:
            if StockMajor is not None:
                st.write("**- Major Holders -**")
                st.write(StockMajor)
            if StockInstitutional is not None:
                st.write("**- Top Institutional Holders -**")
                st.write(StockInstitutional)
            if StockFund is not None:
                st.write("**- Top Mutual Fund Holders -**")
                st.write(StockFund)

        # Button to toggle insider holders data visibility
        if not all(df is None for df in
                   [StockInsiderPurchases, StockInsider, StockInsider_roster_holders]) and st.button(
                "Insider Holders & Transactions"):
            st.session_state.visibility["insider"] = not st.session_state.visibility["insider"]

        # Display insider holders data if visibility is True
        if st.session_state.visibility["insider"]:
            if StockInsiderPurchases is not None:
                st.write("**- Insider Purchases Last 6 Months -**")
                st.write(StockInsiderPurchases)
            if StockInsider is not None:
                st.write("**- Insider Transactions Reported - Last Two Years -**")
                st.write(StockInsider)
            if StockInsider_roster_holders is not None:
                st.write("**- Insider Roster -**")
                st.write(
                    "*- Insider roster data is derived solely from the last 24 months of Form 3 & Form 4 SEC filings.*")
                st.write(StockInsider_roster_holders)

        st.write('<hr style="height:4px;border:none;color:#0ECCEC;background-color:#0ECCEC;">', unsafe_allow_html=True)

