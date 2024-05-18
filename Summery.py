
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


def show_disclaimer():
    # st.title("Disclaimer")
    color_code = "#0ECCEC"
    header_html = f'<h2 style="color:{color_code};">DISCLAIMER</h2>'
    st.markdown(header_html, unsafe_allow_html=True)

    with st.form(key="disclaimer_form"):

        # # Set the width of the form frame to match the column width
        # st.markdown("<style>div[data-testid='stForm'] div{max-width:600px}</style>", unsafe_allow_html=True)
        disclaimer_content = (
            "- This Web Application (first Beta Version for Desktop Computers) Aims to enhance the accessibility and comprehension of financial data by providing visual representations of various financial metrics, including stock summaries, income statements, balance sheets and cash flow statements. It is designed to facilitate the understanding of new companies by presenting data visually, allowing users to interpret information beyond mere numerical values."
            "\n\n- The information presented in this application is for informational purposes only and should not be considered a substitute for professional financial consultation. Users are encouraged to conduct their own research and consult with qualified financial advisors before making any investment decisions."
            "\n\n- The creators of this application do not guarantee the accuracy, completeness, or reliability of the information retrieved from Financial Data APIs."
            "\n\n- By continuing to use this application, you agree that you have read and understood this disclaimer, and you acknowledge that the creators of this application are not liable for any investment decisions made based on the information presented."
        )


        # Place the disclaimer content within column col1 with width 0.6
        col1, col2 = st.columns([0.6, 0.4])

        with col1:
            st.write(disclaimer_content)  # Display the disclaimer content

            agree_disclaimer = st.checkbox("I have read and agree to the disclaimer")
            continue_button = st.form_submit_button(label="Click for Continue")
            # Apply custom CSS style to adjust the form frame width

    return agree_disclaimer, continue_button


# Check if the disclaimer has already been accepted
if "disclaimer_accepted" not in st.session_state:
    agree_disclaimer, continue_button = show_disclaimer()

    # Wait until the user checks the box and presses the button to continue
    if agree_disclaimer and continue_button:
        # Set session state to indicate that the disclaimer has been accepted
        st.session_state["disclaimer_accepted"] = True

        # Clear the entire layout and proceed with loading the app
        st.empty()

else:


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

    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        # Input box for user to enter symbol
        new_symbol = st.text_input("Add symbol to Symbols List (e.g., AAPL)",
                                   placeholder="Search Stocks Symbols").strip().upper()

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


    # Display a message box in the sidebar
    st.sidebar.info("- For the best experience, maximize your screen.")
    st.sidebar.info("- Close side bar for better visualization.")
    # st.sidebar.info("- Recommended dark mode in setting menu.")
    st.sidebar.info("- This app version is less suitable for stocks in the finance industry")

    st.sidebar.markdown("&copy;VisualSD. All rights reserved.", unsafe_allow_html=True)

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
        'S.Outstanding': 'sharesOutstanding',
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
            f"<h1 style='font-size: larger; margin-bottom: 5px; display: inline;'>Sector - {StockInfo['sector']}</h1>"
            f"<h1 style='font-size: larger; margin-bottom: 5px; display: inline;'>Industry - {StockInfo['industry']}</h1>",
            unsafe_allow_html=True
        )

        st.write("")
        st.write("")
        st.write("")

    col1, col2, col3, col4 = st.columns([0.6, 0.2, 0.09, 0.01])  # Adjust the width ratio of col1 and col2 as needed

    with col2:

        st.write("")
        st.write("")

        # Fetch data based on the selected time period or default to '1Y'
        selected_time_period = st.session_state.get('selected_time_period', '1Y')
        df_ticker = yf.download(ticker, period='max').reset_index()
        end_date = datetime.now()
        # Buttons for selecting different time periods
        time_periods = ['1D', '7D', '3M', '6M', 'YTD', '1Y', '5Y', 'MAX']
        for _ in range(1):
            st.write("")

        # Display buttons in a single row
        button_container = st.container()

        # # # Initialize selected_time_period to '1Y' as default
        # selected_time_period = '1Y'

        with button_container:
            button_spacing = 1  # Adjust spacing between buttons
            st.write('<style>div.row-widget.stHorizontal {flex-wrap: nowrap;}</style>', unsafe_allow_html=True)

            for period in time_periods:
                if st.button(period):
                    selected_time_period = period
                    st.session_state.selected_time_period = period

        # Calculate start date based on selected time period

        if selected_time_period == '1D':
            start_date = end_date - timedelta(days=2)
        elif selected_time_period == '7D':
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

    with col1:


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
                title_text="<span style='text-align: center;'>        Chart Dates: {} to {}</span><br>"
                           "<span style='font-size: 18px;'>       Low: {:.2f} | High: {:.2f} | Range Low To High: {:.2f}%</span><br>"
                           "<span style='font-size: 18px;'>                             Return for the period: <span style='color:{};'>{:.2f}%</span></span>".format(
                    start_date.strftime("%d-%m-%Y"), end_date.strftime("%d-%m-%Y"),
                    min_price, max_price, range_low_to_high, yield_color, yield_percentage),
                title_x=0.15,  # Center the title
                title_font_size=22,  # Increase font size
                title_y=0.95,  # Adjust title vertical position
                title_yanchor='top',
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
                # Adjust legend position
            )

            candlestick_chart.update_layout(xaxis_rangeslider_visible=False,
                                            xaxis=dict(type='date', range=[start_date, end_date]),
                                            yaxis=dict(title='Price', showgrid=True),
                                            # Remove gridlines from y-axis
                                            yaxis2=dict(title='',
                                                        overlaying='y',
                                                        side='right',  # Move to the right side
                                                        position=1,  # Move outside the plot area
                                                        showgrid=False),  # Remove gridlines from y2-axis
                                            height=500)

            # Hide Plotly toolbar and directly display the chart
            st.plotly_chart(candlestick_chart, use_container_width=True, config={'displayModeBar': False})

    col1, col2 = st.columns([0.8, 0.2])

    with col1:
        if st.checkbox('Show Stock Price History Data', value=False):
            st.subheader('Stock History - {}'.format(selected_time_period))
            sorted_df = df_ticker.sort_values(by='Date', ascending=False)
            st.write(sorted_df)

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
        ('S.Outstanding', '')
    ]



    with col1:
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
                value1 = float(value1) / 1_000_000_000 if value1 != 'N/A' else 'N/A'  # Divide by billions and convert to integer
            elif key1 == 'Avg. Volume (10d)' or key1 == 'Volume':
                value1 = int(float(value1)) if value1 != 'N/A' else 'N/A'  # Convert to integer
                if label1 == 'Avg. Volume (10d)' or label1 == 'Volume':
                    value1 = f"{int(value1):,}" if value1 != 'N/A' else 'N/A'  # Format without decimal places

            if key2 and label2 in ['Market Cap (In B$)', 'Company EV']:
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




    col1, col2 = st.columns([0.3, 0.3])  # Adjust the width ratio of col1 and col2 as needed

    with col1:

        st.write('<hr style="height:4px;border:none;color:#0ECCEC;background-color:#0ECCEC;">', unsafe_allow_html=True)

        st.subheader(f'Company Summery')

        st.write(StockInfo['longBusinessSummary'])
        if 'fullTimeEmployees' in StockInfo:
            st.write("Full Time Employees:", str(StockInfo['fullTimeEmployees']))

        st.write("Company Website:", StockInfo['website'])
        # st.write("****************************************************************************************************")
        # # Adjust the size of the line using CSS
        # st.write('<hr style="height:4px;border:none;color:#333;background-color:#333;">', unsafe_allow_html=True)
        # # Adjust the size and color of the line using CSS
        # st.write('<hr style="height:5px;border:none;color:blue;background-color:blue;">', unsafe_allow_html=True)
        # # Adjust the size and color of the line using CSS

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


          


