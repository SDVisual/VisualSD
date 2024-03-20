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

st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialize session state
if 'valid_tickers' not in st.session_state:
    st.session_state.valid_tickers = []

# Default symbol to show
DEFAULT_SYMBOL = ''



# Display title with blue color using Markdown

st.markdown(f"<h1 style='color:blue;'>{APP_NAME}</h1>", unsafe_allow_html=True)

# Input box for user to enter symbol
new_symbol = st.text_input("Add Symbol to Select Box (e.g., AAPL)", DEFAULT_SYMBOL).strip().upper()


st.markdown("## \n\n\n")  # Add an empty line




# Check if the entered symbol is empty or consists only of whitespace characters
if not new_symbol or new_symbol.isspace():
    new_symbol = DEFAULT_SYMBOL




# Check if the entered symbol is valid
historical_data = yf.Ticker(new_symbol).history(period='1d')

if new_symbol != DEFAULT_SYMBOL and historical_data.empty:
    st.error("Invalid symbol. Please enter a valid symbol.")

else:
    # Add valid symbol to session state

    if new_symbol not in st.session_state.valid_tickers:
        st.session_state.valid_tickers.append(new_symbol)
        st.text(f"Symbol Added to Select Box - {new_symbol} ")




# Automatically select the last valid symbol entered by the user
index = st.session_state.valid_tickers.index(new_symbol)
# st.text(index)

# Update dropdown options with all ticker symbols
if st.session_state.valid_tickers:
    # Select box to choose ticker
    ticker = st.sidebar.selectbox('Symbols List - Select Box', st.session_state.valid_tickers,
                                  index=index)


# Sidebar date inputs
start_date = st.sidebar.date_input('Start date - Historical Prices', datetime.datetime(2021, 1, 1))
end_date = st.sidebar.date_input('End date', datetime.datetime.now().date())




# Add a menu to the sidebar
menu_option = st.sidebar.radio("Menu", ["Company Summary", "Income Statements", "Balance Sheet", "Cash Flow"])



if menu_option == "Company Summary":


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
        "heldPercentInsiders", "heldPercentInstitutions", "shortRatio", "shortPercentOfFloat", "impliedSharesOutstanding",
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

        st.subheader(f'Stock Summary for {StockInfo["shortName"]}')
        st.markdown(f'<span style="font-size: larger;">Sector - {StockInfo["sector"]}</span><br>'
                    f'<span style="font-size: smaller;">{StockInfo["industryDisp"]}</span>',
                    unsafe_allow_html=True)

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





    st.markdown("## \n\n\n")  # Add an empty line
    st.markdown("## \n\n\n")  # Add an empty line
    st.markdown("## \n\n\n")  # Add an empty line


    # # Write DataFrame to Excel file
    # excel_filename = rf"C:\Users\danny\Desktop\WEB APP\{ticker}.xlsx"
    #
    # # excel_filename = "C:/Users/danny/OneDrive/Desktop/python/YTest/stock_info.xlsx"  # Adjust the path and filename as needed
    # StockInfo_df.to_excel(excel_filename, index=False)
    # st.success(f"Stock information has been written to {excel_filename}")

# ****************************************************************************************************************************

elif menu_option == "Income Statements":


    # st.text(ticker)
    StockInfo = yf.Ticker(ticker).info
    income_statementYear = yf.Ticker(ticker).income_stmt
    IncomeStatementQuarterly = yf.Ticker(ticker).quarterly_income_stmt

    # Default to annual income statement
    income_statement = income_statementYear

    # Checkbox to select between annual and quarterly
    is_quarterly = st.checkbox("Show Quarterly Income Statement", value=False)

    # Checkbox to toggle display of extended balance sheet
    is_extended = st.checkbox("Show extended Income Statment", value=False)

    # Update income statement based on the checkbox value
    if is_quarterly:
        income_statement = IncomeStatementQuarterly


    # Define desired order for the first section
    desired_order_first = [
        'Total Revenue', 'Cost Of Revenue', 'Gross Profit', 'Operating Expense',
        'Selling General And Administration', 'Research And Development', 'Operating Income',
        'Net Non Operating Interest Income Expense', 'Other Income Expense', 'Other Non Operating Income Expenses', 'Pretax Income', 'Tax Provision', 'Net Income Common Stockholders',
        'Net Income', 'EBIT', 'EBITDA', 'Basic EPS', 'Diluted EPS'
    ]

    desired_order = [
        'Total Revenue', 'Operating Revenue', 'Cost Of Revenue', 'Gross Profit', 'Operating Expense',
        'Selling General And Administration', 'Research And Development', 'Operating Income',
        'Net Non Operating Interest Income Expense', 'Interest Income Non Operating', 'Interest Expense Non Operating',
        'Other Income Expense', 'Special Income Charges', 'Restructuring And Mergern Acquisition',
        'Other Non Operating Income Expenses', 'Pretax Income', 'Tax Provision', 'Net Income Common Stockholders',
        'Net Income', 'Net Income Including Noncontrolling Interests', 'Net Income Continuous Operations',
        'Diluted NI Availto Com Stockholders', 'Basic EPS', 'Diluted EPS', 'Basic Average Shares', 'Diluted Average Shares',
        'Total Operating Income As Reported', 'Total Expenses', 'Net Income From Continuing And Discontinued Operation',
        'Normalized Income', 'Interest Income', 'Interest Expense', 'Net Interest Income', 'EBIT', 'EBITDA',
        'Reconciled Cost Of Revenue', 'Reconciled Depreciation', 'Net Income From Continuing Operation Net Minority Interest',
        'Net Income Including Noncontrolling Interests', 'Total Unusual Items Excluding Goodwill', 'Total Unusual Items', 'Normalized EBITDA', 'Tax Rate For Calcs',
        'Tax Effect Of Unusual Items'
    ]

    if is_extended:
        income_statement = income_statement.reindex(desired_order, fill_value='0')
    else:
        income_statement = income_statement.reindex(desired_order_first, fill_value='0')




    # Convert column headers to datetime
    income_statement.columns = pd.to_datetime(income_statement.columns)

    # Sort the columns in ascending order of dates
    income_statement = income_statement.sort_index(axis=1)

    # Format the column headers to remove the timestamp
    income_statement.columns = [col.strftime('%d/%m/%Y') for col in income_statement.columns]

    # Check where 'Total Revenue' is NaN in each column
    nan_mask = income_statement.loc['Total Revenue'].isna()

    # Drop columns where 'Total Revenue' is NaN for all values
    income_statement = income_statement.loc[:, ~nan_mask]



    # % CHANGE DF *********************************

    # Convert the DataFrame values to numeric type, ignoring errors

    income_statement_numeric = income_statement.apply(pd.to_numeric, errors='coerce')


    # Calculate the percentage of revenue for each item in the income statement
    revenue_percentage_df = income_statement_numeric.div(income_statement_numeric.loc['Total Revenue']) * 100


    exclude_rows = ['Basic EPS', 'Diluted EPS', 'Tax Rate For Calcs']
    income_statement = income_statement.apply(
        lambda row: row.map(lambda x: f"{x / 1:.2f}" if isinstance(x, (
         int, float)) and row.name in exclude_rows else f"{x / 1e6:,.0f}" if isinstance(x, (int, float)) else x),
        axis=1
    )



    st.subheader(f"Income Statement for  {StockInfo['shortName']} (In M$)")

    styled_income_statement = income_statement.style.set_table_styles([
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('border', '2px solid blue')]},
        {'selector': 'th, td', 'props': [('text-align', 'center'), ('border', '1px solid blue')]},
        {'selector': 'th', 'props': [('text-align', 'left')]}
    ])

    # Convert the styled DataFrame to HTML
    styled_income_statement_html = styled_income_statement.render()

    # Use st.markdown to add a vertical scroll bar without expanding
    st.markdown(
        f'<div style="max-height: 400px; overflow-y: auto;">{styled_income_statement_html}</div>',
        unsafe_allow_html=True
    )

    st.markdown("## \n\n\n")  # Add an empty line
    st.markdown("## \n\n\n")  # Add an empty line

    st.write(f'* All charts are interactive by clicking legend elements')
    st.markdown("## \n\n\n")  # Add an empty line



    col1, col2 = st.columns([0.6, 0.4])  # Adjust the width ratio of col1 and col2 as needed

    data = revenue_percentage_df.loc[['Cost Of Revenue', 'Gross Profit', 'Selling General And Administration',
                                      'Research And Development', 'Operating Expense', 'Operating Income', 'Net Income']].transpose()

    # Define a dictionary to map full names to shorter abbreviations
    name_mapping = {
        'Selling General And Administration': 'SG&A',
        'Research And Development': 'R&D'
        # Add more mappings as needed
    }

    # Plot bar chart Visual BreakDown Income Statment (As%)***************************************************

    with col1:
        if not data.empty:
            fig = go.Figure()

            # Define colors for each quarter
            colors = ['blue', 'red', 'yellow', 'purple']  # Add more colors if needed

            # Create traces for each quarter or year based on checkbox value
            for i, date in enumerate(data.index):
                if is_quarterly:
                    label = f'Q{pd.to_datetime(date).quarter} {pd.to_datetime(date).year}'
                else:
                    label = str(pd.to_datetime(date).year)

                fig.add_trace(go.Bar(
                    x=[name_mapping.get(col, col) for col in data.columns],  # Use the mapped names as x-axis labels
                    y=data.loc[date],
                    name=label,  # Use the formatted date as the legend label
                    text=[f"{value:.2f}%" for value in data.loc[date]],
                    textposition='auto',
                    insidetextanchor='start',
                    marker=dict(color=colors[i % len(colors)], line=dict(width=2, color='black')),
                    insidetextfont=dict(size=15),
                ))

            # Update layout
            fig.update_layout(
                barmode='group',
                xaxis=dict(title=''),  # Set x-axis title
                yaxis=dict(title='%  of  Total  Revenue'),
                width=800,
                height=500,
                title_text=f'Visual BreakDown Income Statment Margins By {"Quarters" if is_quarterly else "Years"} (As%)',
                # Update title
                title_x=0.5,
                title_y=0.98,
                title_xanchor='center',
                title_yanchor='top',
                font=dict(size=15),
                legend=dict(orientation="h", yanchor="bottom", y=1.07, xanchor="center", x=0.5),
                # Center the legend
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.write("Income statement is empty.")







    col1, col2, col3 = st.columns([0.3, 0.3, 0.4])  # Adjust the width ratio of col1 and col2 as needed

    data = income_statement.loc[['Net Income', 'Total Revenue', 'Operating Income']].transpose()

    # Convert data index to datetime objects if they are not already datetime objects
    data.index = pd.to_datetime(data.index)


# Plot bar chart Company Revenues YoY ***************************************************


    with col1:
        if not income_statement.empty:
            fig = go.Figure()

            # Define colors for each trace
            colors = ['blue', 'yellow', 'green']  # Add more colors if needed

            # Create traces outside the loop
            for i, metric in enumerate(['Total Revenue', 'Operating Income', 'Net Income']):
                # Check if the metric is 'Net Income' and if so, set the color to red for negative values
                bar_colors = ['red' if metric == 'Net Income' and float(value.replace(',', '')) < 0 else colors[i] for
                              value in data[metric]]

                fig.add_trace(go.Bar(
                    x=data.index.strftime('%Y-%m-%d' if is_quarterly else '%Y'),
                    # Adjust date format based on checkbox value
                    y=[float(value.replace(',', '')) for value in data[metric]],
                    name=metric,
                    text=[f"${'{:,.0f}'.format(float(value.replace(',', '')))}" for value in data[metric]],
                    textposition='auto',
                    insidetextanchor='start',
                    marker=dict(color=bar_colors, line=dict(width=2, color='black')),
                    insidetextfont=dict(size=15) if metric == 'Total Revenue' or metric == 'Operating Income' else dict(
                        size=15),
                ))

            # Update layout
            fig.update_layout(
                barmode='group',
                xaxis=dict(tickvals=data.index if is_quarterly else data.index.year,
                           ticktext=data.index.strftime('%Y-%m-%d' if is_quarterly else '%Y')),
                # Adjust ticktext based on checkbox value
                yaxis=dict(title='Amount (M$)'),
                width=300,
                height=500,
                title_text=f'Company Revenues YoY' + (" (By Quarters)" if is_quarterly else " (By Years)"),
                # Update title based on checkbox value
                title_x=0.5,
                title_y=0.98,
                title_xanchor='center',
                title_yanchor='top',
                font=dict(size=12),
                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45),  # Center the legend
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.write("Income statement is empty.")


    # Plot bar chart Company Expenses YoY **********************************
    with col2:
        if not income_statement.empty:
            data_expenses = income_statement.loc[
                ['Operating Expense', 'Selling General And Administration', 'Research And Development']].transpose()

            # Convert data_expenses index to datetime objects if they are not already datetime objects
            data_expenses.index = pd.to_datetime(data_expenses.index)

            fig_expenses = go.Figure()

            # Define colors for each trace
            colors_expenses = ['red', 'blue', 'lightblue']

            # Create traces outside the loop
            for i, metric in enumerate(
                    ['Operating Expense', 'Selling General And Administration', 'Research And Development']):
                legend_label = 'SG&A' if metric == 'Selling General And Administration' else (
                    'R&D' if metric == 'Research And Development' else metric)
                fig_expenses.add_trace(go.Bar(
                    x=data_expenses.index.strftime('%Y-%m-%d' if is_quarterly else '%Y'),
                    # Adjust date format based on checkbox value
                    y=[float(value.replace(',', '')) for value in data_expenses[metric]],
                    name=legend_label,
                    text=[f"${'{:,.0f}'.format(float(value.replace(',', '')))}" for value in data_expenses[metric]],
                    textposition='auto',
                    insidetextanchor='start',
                    marker=dict(color=colors_expenses[i], line=dict(width=2, color='black')),
                    insidetextfont=dict(size=15),  # Use the same size as in the first chart
                ))

            # Update layout to match the first chart
            fig_expenses.update_layout(
                barmode='group',
                xaxis=dict(tickvals=data_expenses.index if is_quarterly else data_expenses.index.year,
                           ticktext=data_expenses.index.strftime('%Y-%m-%d' if is_quarterly else '%Y')),
                # Adjust ticktext based on checkbox value
                yaxis=dict(title='Amount (M$)'),
                width=300,  # Use the same width as in the first chart
                height=500,  # Use the same height as in the first chart
                title_text=f'Company Expenses YoY' + (" (By Quarters)" if is_quarterly else " (By Years)"),
                # Update title based on checkbox value
                title_x=0.5,
                title_y=0.98,
                title_xanchor='center',
                title_yanchor='top',
                font=dict(size=12),  # Use the same font size as in the first chart
                legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45),  # Center the legend
            )

            st.plotly_chart(fig_expenses, use_container_width=True, config={'displayModeBar': False})
        else:
            st.write("Income statement is empty.")





# Plot bar Revenue Growth ***************************************************************************************


    col1, col2 = st.columns([0.7, 0.3])  # Adjust the width ratio of col1 and col2 as needed

    percentage_change_df = income_statement_numeric

    # Calculate percentage change/Growth for each metric between consecutive periods
    percentage_change_df = percentage_change_df.pct_change(axis=1) * 100

    # Convert the first column to numeric, coercing errors to NaN
    percentage_change_df.iloc[:, 0] = pd.to_numeric(percentage_change_df.iloc[:, 0], errors='coerce')

    # Replace NaN values with 0
    percentage_change_df.iloc[:, 0] = percentage_change_df.iloc[:, 0].fillna(0)

    data_percentage_change_df = percentage_change_df.loc[
        ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']].transpose()



    # Now you can proceed with the rest of your code
    with col1:


        # Add title in the middle with smaller font size
        st.markdown("<h2 style='text-align: center; color: blue'>Income Statement Growth Rate</h2>", unsafe_allow_html=True)

        # Use Streamlit's columns layout manager to display charts side by side
        col1, col2, col3, col4 = st.columns(4)

        # Define colors for each metric
        line_colors = ['blue', 'red', 'green', 'orange']

        # Create a new figure for the line chart
        line_fig = go.Figure()

        # Define the list of metrics
        metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']


        # Iterate over each metric and create a chart
        for metric, col in zip(metrics, [col1, col2, col3, col4]):
            # Create a new chart for the current metric
            with col:
                if not data_percentage_change_df.empty:
                    # Create a new figure for the current metric
                    fig = go.Figure()

                    # Add trace for the current metric
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(data_percentage_change_df.index).strftime(
                            '%Y-%m-%d' if is_quarterly else '%Y'),
                        y=data_percentage_change_df[metric],
                        mode='lines',
                        name=metric,
                        line=dict(color=line_colors[metrics.index(metric)]),  # Assign color from the list
                    ))

                    # Add annotations for each point on the line
                    for index, row in data_percentage_change_df.iterrows():
                        index_dt = pd.to_datetime(index)  # Convert index to datetime object
                        fig.add_annotation(
                            x=index_dt.strftime('%Y-%m-%d' if is_quarterly else '%Y'),
                            # Use specific x-value for each data point
                            y=row[metric],  # y-coordinate of the annotation
                            text=f"{row[metric]:.2f}%",  # text to display (format as needed)
                            showarrow=False,  # don't show an arrow
                            font=dict(color=line_colors[metrics.index(metric)]),  # color of the annotation text
                            xanchor='right',  # anchor point for x-coordinate
                            yanchor='middle',  # anchor point for y-coordinate
                            align='left',  # alignment of the annotation text
                            xshift=5,  # horizontal shift of the annotation
                            yshift=0,  # vertical shift of the annotation
                        )

                    # Update layout for the current chart
                    fig.update_layout(
                        title=f'{metric} {"QoQ" if is_quarterly else "YoY"}',
                        title_x=0.3,  # Set the title's horizontal position to the center
                        xaxis=dict(title=''),
                        yaxis=dict(title='% Growth'),
                        width=800,  # Adjust the width of each chart as needed
                        height=400,  # Adjust the height of each chart as needed
                        font=dict(size=12),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=1),
                    )

                    # Plot the current chart
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.write("Data is empty for metric:", metric)


# Basic EPS and Diluted EPS data for all years *****************************************************

    col1, col2, col3 = st.columns([0.2, 0.2, 0.5])

    with col1:
        # Extract Basic EPS and Diluted EPS data for all years
        basic_eps = income_statement.loc['Basic EPS'].astype(float)
        diluted_eps = income_statement.loc['Diluted EPS'].astype(float)

        # Create a bar chart
        fig = go.Figure()

        # Add Basic EPS bars
        basic_eps_index = basic_eps.index.astype(str)
        basic_eps_x = basic_eps_index.map(
            lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if is_quarterly else pd.to_datetime(x).strftime('%Y'))
        fig.add_trace(go.Bar(
            x=basic_eps_x,
            y=basic_eps.values,
            name='Basic EPS',
            text=["{:,.2f}$".format(value) for value in basic_eps.values],  # Add $ and format to two decimal places
            textposition='auto',
        ))

        # Add Diluted EPS bars
        diluted_eps_index = diluted_eps.index.astype(str)
        diluted_eps_x = diluted_eps_index.map(
            lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if is_quarterly else pd.to_datetime(x).strftime('%Y'))
        fig.add_trace(go.Bar(
            x=diluted_eps_x,
            y=diluted_eps.values,
            name='Diluted EPS',
            text=["{:,.2f}$".format(value) for value in diluted_eps.values],  # Add $ and format to two decimal places
            textposition='auto',
        ))

        # Update layout
        fig.update_layout(
            title='Basic EPS vs Diluted EPS',
            title_x=0.35,
            xaxis=dict(title='' if is_quarterly else ''),
            yaxis=dict(title='EPS Value'),
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            font=dict(
                size=14  # Adjust font size of values on bars
            )
        )

        # Display the chart without the menu
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})




    with col2:
        # Extract EBITDA and Net Income data for all years without converting to float
        ebit = income_statement.loc['EBIT']
        ebitda = income_statement.loc['EBITDA']
        net_income = income_statement.loc['Net Income']

        # Create a bar chart
        fig = go.Figure()


        # Add EBITDA bars
        ebitda_x = ebitda.index.astype(str)
        ebitda_x = ebitda_x.map(
            lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if is_quarterly else pd.to_datetime(x).strftime('%Y'))
        fig.add_trace(go.Bar(
            x=ebitda_x,
            y=ebitda.values,
            name='EBITDA',
            text=ebitda.values,
            textposition='auto',

        ))

        # Add EBIT bars
        ebit_x = ebit.index.astype(str)
        ebit_x = ebit_x.map(
            lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if is_quarterly else pd.to_datetime(x).strftime('%Y'))
        fig.add_trace(go.Bar(
            x=ebit_x,
            y=ebit.values,
            name='EBIT',
            text=ebit.values,
            textposition='auto',
            # marker=dict(color='rgba(0, 0, 255, 0.5)')  # Medium blue color
        ))

        # Add Net Income bars
        net_income_x = net_income.index.astype(str)
        net_income_x = net_income_x.map(
            lambda x: pd.to_datetime(x).strftime('%Y-%m-%d') if is_quarterly else pd.to_datetime(x).strftime('%Y'))


        fig.add_trace(go.Bar(
            x=net_income_x,
            y=net_income.values,
            name='Net Income',
            text=net_income.values,
            textposition='auto',
            marker=dict(color='green')  # Medium blue color
        ))

        # Update layout
        fig.update_layout(
            title='EBIT VS EBITDA vs Net Income (M$)',
            title_x=0.35,
            xaxis=dict(title='' if is_quarterly else ''),
            yaxis=dict(title='Value'),
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            font=dict(
                size=14  # Adjust font size of values on bars
            )
        )
        # Display the chart without the menu
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    #     # Extract 'Net Income' and 'EBITDA' data
    # net_income_str = income_statement.loc['Net Income']
    # ebitda_str = income_statement.loc['EBITDA']
    #
    # # Remove commas and convert to numeric type
    # net_income_numeric = net_income_str.str.replace(',', '').astype(float)
    # ebitda_numeric = ebitda_str.str.replace(',', '').astype(float)
    #
    # # Calculate the percentage of Net Income to EBITDA
    # percentage_net_income_to_ebitda = (net_income_numeric / ebitda_numeric) * 100
    #
    # # Create a DataFrame with year and percentage results
    #
    # result_df = pd.DataFrame({'Net Income as % of EBITDA': percentage_net_income_to_ebitda})
    #
    # result_df.reset_index(inplace=True)
    # result_df['Date'] = pd.to_datetime(result_df['index']).dt.date
    # result_df.drop(columns=['index'], inplace=True)
    #
    # # Display the DataFrame
    # st.write("Year and Net Income as % of EBITDA:")
    # st.dataframe(result_df[['Net Income as % of EBITDA', 'Date']].set_index('Date'))

elif menu_option == "Balance Sheet":

    StockInfo = yf.Ticker(ticker).info

    balance_sheetYear = yf.Ticker(ticker).balance_sheet
    balance_sheetQuarterly = yf.Ticker(ticker).quarterly_balance_sheet

    # Default to annual income statement
    balance_sheet = balance_sheetYear



    # Checkbox layout
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # Checkbox to select between annual and quarterly
    is_quarterly = col1.checkbox("Show Quarterly Balance Sheet", value=False)

    # Checkbox to toggle display of extended balance sheet
    is_extended = col1.checkbox("Show extended Balance Sheet", value=False)

    # # Checkbox to select between annual and quarterly
    # is_quarterly = st.checkbox("Show Quarterly Balance Sheet", value=False)

    # Update income statement based on the checkbox value
    if is_quarterly:
        balance_sheet = balance_sheetQuarterly

    # Define desired order for the first section
    desired_order_first = [
        'Total Assets', 'Current Assets', 'Total Non Current Assets', 'Cash Cash Equivalents And Short Term Investments',
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

    # # Checkbox to toggle display of extended balance sheet
    # is_extended = st.checkbox("Show extended Balance Sheet", value=False)

    st.subheader(f"Balance Sheet for {StockInfo['shortName']} (In M$)")

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

    # Convert the styled DataFrame to HTML
    styled_balance_sheet_html = styled_balance_sheet.render()

    # Use st.markdown to add a vertical scrollbar without expanding
    st.markdown(
        f'<div style="max-height: 400px; overflow-y: auto;">{styled_balance_sheet_html}</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## \n\n\n")  # Add an empty line
        st.markdown("## \n\n\n")  # Add an empty line
        st.write(f'* All charts are interactive by clicking legend elements')
        st.markdown("## \n\n\n")  # Add an empty line
        # Transpose the balance sheet DataFrame
        transposed_balance_sheet = balance_sheet.transpose()

        # Select relevant columns from the transposed balance sheet DataFrame
        selected_columns = ['Current Assets', 'Total Non Current Assets', 'Current Liabilities',
                            'Total Non Current Liabilities Net Minority Interest',
                            'Total Equity Gross Minority Interest']
        selected_data = transposed_balance_sheet[selected_columns]

        # Define colors for each column
        colors = {'Current Assets': 'green',
                  'Total Non Current Assets': 'lightgreen',
                  'Current Liabilities': 'red',
                  'Total Non Current Liabilities Net Minority Interest': 'lightcoral',
                  'Total Equity Gross Minority Interest': 'blue'}

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

        # st.write(balance_sheet)



        # Assuming balance_sheet is your DataFrame containing the balance sheet data



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

        # Display the resulting DataFrame
        # st.write(percentage_change_balance)


# ******************************************   CHARTS 'Total Assets', 'Current Assets', 'Total Non Current Assets', ' Cash  & Cash Equivalents And Short Term Investments' ***************************************************

    col1, col2, col3, col4, col5 = st.columns([0.15, 0.15, 0.15, 0.15, 0.4])

    with col1:

        data_percentage_change_balance = percentage_change_balance.loc['Total Assets'].transpose()
        # Create a figure
        fig = go.Figure()

        # Add bar trace for total assets
        fig.add_trace(
            go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Total Assets'], name='Total Assets',
                   marker_color='blue'))

        # Add line trace for growth rate
        fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                                 mode='lines+markers', name='Growth Rate', line=dict(color='red'), yaxis='y2'))

        # Add text annotations for growth rate values above the linear points
        for i, value in enumerate(data_percentage_change_balance.values):
            fig.add_annotation(x=data_percentage_change_balance.index[i],  # x-coordinate for annotation
                               y=data_percentage_change_balance.values[i] + 0.7,  # Shift the text 0.05 above the point
                               text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
                               showarrow=False,  # whether to show arrow or not
                               font=dict(color='yellow', size=15),  # color of the annotation text
                               yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                               align='left',  # alignment of the text
                               xanchor='left')  # anchor point along x-axis for alignment

        # Update layout
        fig.update_layout(title='Total Assets',
                          title_x=0.25,  # Set the title position to the center
                          xaxis_title='',
                          yaxis_title='Amount',
                          yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                          legend=dict(x=0.35, y=1.15, xanchor='center', yanchor='top', orientation='h'))  # Set legend to horizontal orientation


        # Display the chart without the menu
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col2:
        data_percentage_change_balance = percentage_change_balance.loc['Current Assets'].transpose()

        # Create a figure
        fig = go.Figure()

        # Add bar trace for total assets
        fig.add_trace(
            go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Current Assets'], name='Current Assets',
                   marker_color='blue'))

        # Add line trace for growth rate
        fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                                 mode='lines+markers', name='Growth Rate', line=dict(color='red'), yaxis='y2'))

        # Add text annotations for growth rate values above the linear points
        for i, value in enumerate(data_percentage_change_balance.values):
            fig.add_annotation(x=data_percentage_change_balance.index[i],  # x-coordinate for annotation
                               y=data_percentage_change_balance.values[i] + 0.7,  # Shift the text 0.05 above the point
                               text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
                               showarrow=False,  # whether to show arrow or not
                               font=dict(color='yellow', size=15),  # color of the annotation text
                               yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                               align='left',  # alignment of the text
                               xanchor='left')  # anchor point along x-axis for alignment

        # Update layout
        fig.update_layout(title='Current Assets',
                          title_x=0.25,  # Set the title position to the center
                          xaxis_title='',
                          yaxis_title='Amount',
                          yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                          legend=dict(x=0.35, y=1.15, xanchor='center', yanchor='top',
                                      orientation='h'))  # Set legend to horizontal orientation

        # Display the chart without the menu
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col3:
        data_percentage_change_balance = percentage_change_balance.loc['Total Non Current Assets'].transpose()

        # Create a figure
        fig = go.Figure()

        # Add bar trace for total assets
        fig.add_trace(
            go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Total Non Current Assets'], name='Total Non Current Assets',
                   marker_color='blue'))

        # Add line trace for growth rate
        fig.add_trace(go.Scatter(x=data_percentage_change_balance.index, y=data_percentage_change_balance.values,
                                 mode='lines+markers', name='Growth Rate', line=dict(color='red'), yaxis='y2'))

        # Add text annotations for growth rate values above the linear points
        for i, value in enumerate(data_percentage_change_balance.values):
            fig.add_annotation(x=data_percentage_change_balance.index[i],  # x-coordinate for annotation
                               y=data_percentage_change_balance.values[i] + 0.7,  # Shift the text 0.05 above the point
                               text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
                               showarrow=False,  # whether to show arrow or not
                               font=dict(color='yellow', size=15),  # color of the annotation text
                               yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                               align='left',  # alignment of the text
                               xanchor='left')  # anchor point along x-axis for alignment

        # Update layout
        fig.update_layout(title='Total Non Current Assets',
                          title_x=0.25,  # Set the title position to the center
                          xaxis_title='',
                          yaxis_title='Amount',
                          yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                          legend=dict(x=0.5, y=1.15, xanchor='center', yanchor='top',
                                      orientation='h'))  # Set legend to horizontal orientation

        # Display the chart without the menu
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col4:
        data_percentage_change_balance = percentage_change_balance.loc['Cash Cash Equivalents And Short Term Investments'].transpose()

        # Create a figure
        fig = go.Figure()

        # Add bar trace for total assets
        fig.add_trace(
            go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'],
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
                               font=dict(color='yellow', size=15),  # color of the annotation text
                               yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                               align='left',  # alignment of the text
                               xanchor='left')  # anchor point along x-axis for alignment

        # Update layout
        fig.update_layout(title='Cash & Cash Equivalents And Short Term Investments',
                          title_x=0,  # Set the title position to the center
                          xaxis_title='',
                          yaxis_title='Amount',
                          yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                          legend=dict(x=0.5, y=1.2, xanchor='center', yanchor='top',
                                      orientation='h'))  # Set legend to horizontal orientation

        # Display the chart without the menu
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ******************************************   CHARTS 'Total Liabilities Net Minority Interest', 'Current Liabilities' & MORE *************************************************

    with col1:

            data_percentage_change_balance = percentage_change_balance.loc['Total Liabilities Net Minority Interest'].transpose()
            # Create a figure
            fig = go.Figure()

            # Add bar trace for total assets
            fig.add_trace(
                go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Total Liabilities Net Minority Interest'], name='Total Liabilities Net Minority Interest ',
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
                                   font=dict(color='yellow', size=15),  # color of the annotation text
                                   yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                                   align='left',  # alignment of the text
                                   xanchor='left')  # anchor point along x-axis for alignment

            # Update layout
            fig.update_layout(title='Total Liabilities Net Minority Interest',
                              title_x=0.15,  # Set the title position to the center
                              xaxis_title='',
                              yaxis_title='Amount',
                              yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                              legend=dict(x=0.35, y=1.15, xanchor='center', yanchor='top', orientation='h'))  # Set legend to horizontal orientation

            # Display the chart without the menu
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


    with col2:

            data_percentage_change_balance = percentage_change_balance.loc['Current Liabilities'].transpose()
            # Create a figure
            fig = go.Figure()

            # Add bar trace for total assets
            fig.add_trace(
                go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Current Liabilities'], name='Current Liabilities ',
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
                                   font=dict(color='yellow', size=15),  # color of the annotation text
                                   yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                                   align='left',  # alignment of the text
                                   xanchor='left')  # anchor point along x-axis for alignment

            # Update layout
            fig.update_layout(title='Current Liabilities',
                              title_x=0.25,  # Set the title position to the center
                              xaxis_title='',
                              yaxis_title='Amount',
                              yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                              legend=dict(x=0.35, y=1.15, xanchor='center', yanchor='top', orientation='h'))  # Set legend to horizontal orientation

            # Display the chart without the menu
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


    with col3:

            data_percentage_change_balance = percentage_change_balance.loc['Total Non Current Liabilities Net Minority Interest'].transpose()
            # Create a figure
            fig = go.Figure()

            # Add bar trace for total assets
            fig.add_trace(
                go.Bar(x=data_percentage_change_balance.index, y=balance_sheet.loc['Total Non Current Liabilities Net Minority Interest'], name='Total Non Current Liabilities Net Minority Interest ',
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
                                   font=dict(color='yellow', size=15),  # color of the annotation text
                                   yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                                   align='left',  # alignment of the text
                                   xanchor='left')  # anchor point along x-axis for alignment

            # Update layout
            fig.update_layout(title='Total Non Current Liabilities',
                              title_x=0.25,  # Set the title position to the center
                              xaxis_title='',
                              yaxis_title='Amount',
                              yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                              legend=dict(x=0.35, y=1.15, xanchor='center', yanchor='top', orientation='h'))  # Set legend to horizontal orientation

            # Display the chart without the menu
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


    with col4:

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
                                   font=dict(color='yellow', size=15),  # color of the annotation text
                                   yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                                   align='left',  # alignment of the text
                                   xanchor='left')  # anchor point along x-axis for alignment

            # Update layout
            fig.update_layout(title='Total Debt',
                              title_x=0.25,  # Set the title position to the center
                              xaxis_title='',
                              yaxis_title='Amount',
                              yaxis2=dict(title='Percentage Growth', overlaying='y', side='right', showgrid=False),
                              legend=dict(x=0.35, y=1.15, xanchor='center', yanchor='top', orientation='h'))  # Set legend to horizontal orientation

            # Display the chart without the menu
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


# ***************************************     Cash Flow   *************************************************************************************************

elif menu_option == "Cash Flow":


    StockInfo = yf.Ticker(ticker).info

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
    is_quarterly = st.checkbox("Show Quarterly Cash Flow", value=False)

    # Checkbox to toggle display of extended balance sheet
    is_extended = st.checkbox("Show extended Cash Flow Statement", value=False)


    # Update income statement based on the checkbox value
    if is_quarterly:
        cash_flow = cash_flowQuarterly
        balance_sheet = balance_sheetQuarterly
        income_statement = income_statementQuarterly

    desired_order_first = [
        'Free Cash Flow', 'Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'End Cash Position', 'Changes In Cash',
        'Income Tax Paid Supplemental Data', 'Interest Paid Supplemental Data', 'Capital Expenditure',
        'Issuance Of Debt', 'Repayment Of Debt',
        'Repurchase Of Capital Stock'
    ]

    desired_order = [
        'Free Cash Flow', 'Operating Cash Flow', 'Cash Flow From Continuing Operating Activities', 'Net Income From Continuing Operations', 'Operating Gains Losses', 'Gain Loss On Sale Of PPE',
        'Gain Loss On Investment Securities', 'Earnings Losses From Equity Investments', 'Depreciation Amortization Depletion', 'Depreciation And Amortization', 'Depreciation',
        'Deferred Tax', 'Deferred Income Tax', 'Asset Impairment Charge', 'Stock Based Compensation', 'Other Non Cash Items',
        'Change In Working Capital', 'Change In Receivables', 'Changes In Account Receivables', 'Change In Inventory', 'Change In Prepaid Assets',
        'Change In Payables And Accrued Expense', 'Change In Payable', 'Change In Account Payable', 'Change In Accrued Expense', 'Investing Cash Flow',
        'Cash Flow From Continuing Investing Activities', 'Net PPE Purchase And Sale', 'Purchase Of PPE', 'Net Business Purchase And Sale',
        'Purchase Of Business', 'Sale Of Business', 'Net Investment Purchase And Sale', 'Purchase Of Investment', 'Sale Of Investment', 'Net Other Investing Changes',
        'Financing Cash Flow', 'Cash Flow From Continuing Financing Activities', 'Net Issuance Payments Of Debt', 'Net Long Term Debt Issuance', 'Long Term Debt Issuance',
        'Long Term Debt Payments', 'Net Short Term Debt Issuance', 'Short Term Debt Issuance', 'Net Common Stock Issuance', 'Common Stock Payments',
        'Proceeds From Stock Option Exercised', 'Net Other Financing Charges', 'End Cash Position', 'Changes In Cash', 'Beginning Cash Position',
        'Income Tax Paid Supplemental Data', 'Interest Paid Supplemental Data', 'Capital Expenditure', 'Issuance Of Debt', 'Repayment Of Debt',
        'Repurchase Of Capital Stock'
    ]



    if is_extended:
        cash_flow = cash_flow.reindex(desired_order, fill_value='0')
    else:
        cash_flow = cash_flow.reindex(desired_order_first, fill_value='0')


    def str_to_float(value):
        try:
            return float(value.replace(',', ''))  # Assuming the numbers are formatted with commas
        except ValueError:
            return value  # Return the original value if conversion fails



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

    # Calculate percentage change for each metric between consecutive periods
    percentage_change_cash_flow = cash_flow_numeric.pct_change(axis=1) * 100

    # Convert the first column to numeric, coercing errors to NaN
    percentage_change_cash_flow.iloc[:, 0] = pd.to_numeric(percentage_change_cash_flow.iloc[:, 0], errors='coerce')

    # Replace NaN values with 0
    percentage_change_cash_flow.iloc[:, 0] = percentage_change_cash_flow.iloc[:, 0].fillna(0)
    # st.write(percentage_change_cash_flow)



    st.subheader(f"Cash Flow for {StockInfo['shortName']} (In M$)")

    # Apply styling to the cash flow DataFrame
    styled_cash_flow = cash_flow.style.set_table_styles([
        {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('border', '2px solid blue')]},
        {'selector': 'th, td', 'props': [('text-align', 'center'), ('border', '1px solid blue')]},
        {'selector': 'th', 'props': [('text-align', 'center')]},  # Center align headers
        {'selector': 'th:first-child, td:first-child', 'props': [('text-align', 'left')]}  # Align first column to left
    ])

    # Convert the styled DataFrame to HTML
    styled_cash_flow_html = styled_cash_flow.render()

    # Use st.markdown to add a vertical scrollbar without expanding
    st.markdown(
        f'<div style="max-height: 400px; overflow-y: auto;">{styled_cash_flow_html}</div>',
        unsafe_allow_html=True
    )

    st.markdown("## \n\n\n")  # Add an empty line
    st.markdown("## \n\n\n")  # Add an empty line
    st.write(f'* All charts are interactive by clicking legend elements')
    col1, col2 = st.columns(2)

    with col1:

        # Transpose the balance sheet DataFrame
        transposed_cash_flow_sheet = cash_flow.transpose()

        # Select relevant columns from the transposed balance sheet DataFrame
        selected_columns = ['Operating Cash Flow', 'Investing Cash Flow', 'Financing Cash Flow', 'Free Cash Flow']

        selected_data = transposed_cash_flow_sheet[selected_columns]

        # Define colors for each column
        colors = {'Operating Cash Flow': 'blue',
                  'Investing Cash Flow': 'lightgreen',
                  'Financing Cash Flow': 'red',
                  'Free Cash Flow': 'green'}



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
            title='Cash Flow Elements Break Down',
            title_x=0.3,
            xaxis_title='',
            yaxis_title='Amount ($)'
        )

        # Display the chart without the menu
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    col1, col2, col3, col4, col5, col6 = st.columns(6)

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
            go.Scatter(x=percentage_change_cash_flow.columns, y=free_cash_flow_growth, name='Percentage Growth',
                       mode='lines+markers', line=dict(color='red'), yaxis='y2'))

        # Add text annotations for growth values above the bars
        for i, value in enumerate(free_cash_flow_growth):
            fig.add_annotation(x=percentage_change_cash_flow.columns[i],  # x-coordinate for annotation
                               y=value + 1,  # Shift the text 1 above the point
                               text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
                               showarrow=False,  # whether to show arrow or not
                               font=dict(color='yellow', size=15),  # color and size of the annotation text
                               yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                               align='left',  # alignment of the text
                               xanchor='left')  # anchor point along x-axis for alignment

        # Update layout
        fig.update_layout(
            title='Company Free Cash Flow',
            title_x=0.25,
            xaxis=dict(
                title='',
                tickmode='array',  # Set tick values manually
                tickvals=percentage_change_cash_flow.columns,  # Use the dates from the DataFrame
                tickformat='%m/%y'  # Set the tick format to display as MM/YY
            ),
            yaxis_title='Amount ($)',
            yaxis2=dict(title='Percentage Growth (%)', overlaying='y', side='right', showgrid=False),
            legend=dict(x=0.40, y=1.15, xanchor='center', yanchor='top', orientation='h')
        )

        # Display the chart without the menu
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})




        # Apply str_to_float function only to columns containing string values
        ordinary_shares_number = balance_sheet.transpose()['Ordinary Shares Number'].apply(
            lambda x: str_to_float(x) if isinstance(x, str) else x)

        # # Print Ordinary Shares Number after conversion
        # st.write("Ordinary Shares Number (After Conversion):")
        # st.write(ordinary_shares_number)

        # Apply str_to_float function to 'Free Cash Flow' column
        free_cash_flow = cash_flow.transpose()['Free Cash Flow'].apply(str_to_float)

        # Perform division operation after applying conversion function to both columns
        free_cash_flow_per_share = free_cash_flow / ordinary_shares_number

        # # Print Free Cash Flow per Share
        # st.write("Free Cash Flow per Share (After Division):")
        # st.write(free_cash_flow_per_share)



        # Convert Series to DataFrame with a single column
        free_cash_flow_per_share_df = pd.DataFrame(free_cash_flow_per_share, columns=['Free Cash Flow Per Share'])

        # Calculate percentage change for each metric between consecutive periods
        percentage_change_cash_flow_per_share = free_cash_flow_per_share_df['Free Cash Flow Per Share'].pct_change() * 100


        # Fill missing or None values with 0
        free_cash_flow_per_share_df['Free Cash Flow Per Share'] = free_cash_flow_per_share_df['Free Cash Flow Per Share'].fillna(0)

        # Fill None values with 0
        percentage_change_cash_flow_per_share = percentage_change_cash_flow_per_share.fillna(0)

        Capital_Expenditure_percentage_change = percentage_change_cash_flow.loc['Capital Expenditure']

        # Transpose the cash_flow DataFrame
        cash_flow_transposed = cash_flow.transpose()

        # Access the 'Capital Expenditure' column after transposing
        Capital_Expenditure = cash_flow_transposed['Capital Expenditure'].str.replace(',', '').str.replace('-', '').astype(float)
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
                name='Percentage Growth',
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
                font=dict(color='yellow', size=15),  # color and size of the annotation text
                yref='y2',  # reference point on the y-axis (in this case, it's the secondary y-axis)
                align='left',  # alignment of the text
                xanchor='left'  # anchor point along x-axis for alignment
            )

        # Update layout
        fig.update_layout(
            title='Company Capital Expenditure',
            title_x=0.25,
            xaxis=dict(
                title='',
                tickmode='array',  # Set tick values manually
                tickvals=Capital_Expenditure.index,  # Use the dates from the DataFrame
                tickformat='%m/%Y'  # Set the tick format to display as MM/YYYY
            ),
            yaxis=dict(title='Amount ($)'),
            yaxis2=dict(title='Percentage Growth (%)', overlaying='y', side='right', showgrid=False),
            legend=dict(x=0.40, y=1.15, xanchor='center', yanchor='top', orientation='h')
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
                       name='Percentage Growth',
                       mode='lines+markers',
                       line=dict(color='red'),
                       yaxis='y2'))

        # Add text annotations for growth values above the bars
        for i, value in enumerate(percentage_change_cash_flow_per_share):
            fig.add_annotation(x=free_cash_flow_per_share_df.index[i],  # x-coordinate for annotation
                               y=value + 1,  # Shift the text 1 above the point
                               text=f"{value:.2f}%",  # text to be displayed (formatted to two decimal places)
                               showarrow=False,  # whether to show arrow or not
                               font=dict(color='yellow', size=15),  # color and size of the annotation text
                               yref='y2',  # reference point on the y-axis (in this case, it's the y2 axis)
                               align='left',  # alignment of the text
                               xanchor='left')  # anchor point along x-axis for alignment

        # Update layout
        fig.update_layout(
            title='Free Cash Flow Per Share',
            title_x=0.25,
            xaxis=dict(
                title='',
                tickmode='array',  # Set tick values manually
                tickvals=free_cash_flow_per_share_df.index,  # Use the dates from the DataFrame
                tickformat='%m/%Y'  # Set the tick format to display as MM/YYYY
            ),
            yaxis_title='Amount ($)',
            yaxis2=dict(title='Percentage Growth (%)', overlaying='y', side='right', showgrid=False),
            legend=dict(x=0.45, y=1.18, xanchor='center', yanchor='top', orientation='h')
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
            title='End Cash Position and Changes In Cash',
            title_x=0.20,
            xaxis=dict(title='', tickformat='%m/%Y'),  # Set the tick format to display as MM/YYYY
            yaxis=dict(title='Amount ($)'),
            barmode='group',  # Group bars
            legend=dict(x=0.45, y=1.16, xanchor='center', yanchor='top', orientation='h')
        )

        # Display the chart without the menu
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col3:

        # st.write(cash_flow)
        # st.write(income_statement)

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
                textposition='auto',  # Position of the text (auto places the text inside the bars if there's enough space, otherwise outside)
                textfont=dict(size=15, color='yellow'),  # Set font size to 14
            )
        )

        # Update layout
        fig.update_layout(
            title='Free Cash Flow Margin',
            title_x=0.35,
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

        Repayment_Of_Debt = cash_flow_transposed['Repayment Of Debt'].str.replace(',', '').str.replace('-', '').astype(float)
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
            title='Issuance and Repayment of Debt',
            xaxis=dict(title='', tickformat='%m/%Y'),  # Set the tick format to display as MM/YYYY
            title_x=0.25,
            yaxis=dict(title='Amount ($)'),
            legend=dict(x=0.45, y=1.16, xanchor='center', yanchor='top', orientation='h')
        )

        # Display the chart without the menu
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


