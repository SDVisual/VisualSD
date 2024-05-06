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


# Input box for user to enter symbol
new_symbol = st.text_input("Add symbol to Symbols List (e.g., AAPL)", placeholder="Search Stocks").strip().upper()

# st.write("")
# st.write("")

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


# # Sidebar date inputs
# start_date = st.sidebar.date_input('Start date - Historical Prices', datetime.datetime(2021, 1, 1))
# end_date = st.sidebar.date_input('End date', datetime.datetime.now().date())

# Display a message box in the sidebar
st.sidebar.info("- For the best experience, maximize your screen.")
st.sidebar.info("- Close side bar for better visualization.")
st.sidebar.info("- Recommended dark mode in setting menu.")
st.sidebar.info("- This app version is less suitable for stocks in the finance industry")

st.sidebar.markdown("&copy;VisualSD by Dan Oren. All rights reserved.", unsafe_allow_html=True)

StockInfo = yf.Ticker(ticker).info
income_statementYear = yf.Ticker(ticker).income_stmt
IncomeStatementQuarterly = yf.Ticker(ticker).quarterly_income_stmt

# Default to annual income statement
income_statement = income_statementYear




symbol = StockInfo["shortName"]
color_code = "#0ECCEC"  # Color for the symbol

# Combine st.write() with HTML-styled header
st.write(f'<span style="font-size:30px;">Income Statement - </span>'
         f'<span style="color:{color_code}; font-size:30px;">{symbol}</span>', unsafe_allow_html=True)



st.write("")

# Checkbox to select between annual and quarterly
is_quarterly = st.checkbox("Quarterly Income Statement", value=False)

# Checkbox to toggle display of extended balance sheet
is_extended = st.checkbox("Extended Income Statment", value=False)

# Update income statement based on the checkbox value
if is_quarterly:
    income_statement = IncomeStatementQuarterly

# Define desired order for the first section
desired_order_first = [
    'Total Revenue', 'Cost Of Revenue', 'Gross Profit', 'Operating Expense',
    'Selling General And Administration', 'Research And Development', 'Operating Income',
    'Net Non Operating Interest Income Expense', 'Other Income Expense', 'Other Non Operating Income Expenses',
    'Pretax Income', 'Tax Provision', 'Net Income Common Stockholders',
    'Net Income', 'EBIT', 'EBITDA', 'Basic EPS', 'Diluted EPS'
]

desired_order = [
    'Total Revenue', 'Operating Revenue', 'Cost Of Revenue', 'Gross Profit', 'Operating Expense',
    'Selling General And Administration', 'Research And Development', 'Operating Income',
    'Net Non Operating Interest Income Expense', 'Interest Income Non Operating', 'Interest Expense Non Operating',
    'Other Income Expense', 'Special Income Charges', 'Restructuring And Mergern Acquisition',
    'Other Non Operating Income Expenses', 'Pretax Income', 'Tax Provision', 'Net Income Common Stockholders',
    'Net Income', 'Net Income Including Noncontrolling Interests', 'Net Income Continuous Operations',
    'Diluted NI Availto Com Stockholders', 'Basic EPS', 'Diluted EPS', 'Basic Average Shares',
    'Diluted Average Shares',
    'Total Operating Income As Reported', 'Total Expenses', 'Net Income From Continuing And Discontinued Operation',
    'Normalized Income', 'Interest Income', 'Interest Expense', 'Net Interest Income', 'EBIT', 'EBITDA',
    'Reconciled Cost Of Revenue', 'Reconciled Depreciation',
    'Net Income From Continuing Operation Net Minority Interest',
    'Net Income Including Noncontrolling Interests', 'Total Unusual Items Excluding Goodwill',
    'Total Unusual Items', 'Normalized EBITDA', 'Tax Rate For Calcs',
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



st.write(f'*Values in millions $')



styled_income_statement = income_statement.style.set_table_styles([
    {'selector': 'table', 'props': [('border-collapse', 'collapse'), ('border', '2px solid blue')]},
    {'selector': 'th, td', 'props': [('text-align', 'center'), ('border', '1px solid blue'), ('font-size', '30px')]},
    {'selector': 'th', 'props': [('text-align', 'left')]},
])



st.dataframe(styled_income_statement)

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
st.write("")

col1, col2 = st.columns([0.6, 0.4])  # Adjust the width ratio of col1 and col2 as needed

data = revenue_percentage_df.loc[['Cost Of Revenue', 'Gross Profit', 'Selling General And Administration',
                                  'Research And Development', 'Operating Expense', 'Operating Income',
                                  'Net Income']].transpose()

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
        colors = ['#0080ff', '#5483b3', '#7da0ca', '#c1e8ff']  # Add more colors if needed

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

                
                insidetextanchor='start',
                marker=dict(color=colors[i % len(colors)], line=dict(width=1, color='black')),
                insidetextfont=dict(size=20),
            ))

        # Update layout
        fig.update_layout(
            barmode='group',
            xaxis=dict(title=''),  # Set x-axis title
            yaxis=dict(title='%  of  Total  Revenue'),
            height=400,
            title_text=f'Income Statment Margins By {"Quarters" if is_quarterly else "Years"} ',
            # Update title
            title_x=0.5,
            title_y=0.98,
            title_xanchor='center',
            title_yanchor='top',
            font=dict(size=18),
            legend=dict(orientation="h", yanchor="bottom", y=1.07, xanchor="center", x=0.45, font=dict(size=15)),
            # Center the legend
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.write("Income statement is empty.")





st.subheader(f"Revenues & Expenses ")





st.write("")
st.write("")
col1, col2, col3 = st.columns([0.35, 0.35, 0.4])  # Adjust the width ratio of col1 and col2 as needed

data = income_statement.loc[['Net Income', 'Total Revenue', 'Operating Income']].transpose()

# Convert data index to datetime objects if they are not already datetime objects
data.index = pd.to_datetime(data.index)

# Plot bar chart Company Revenues YoY ***************************************************

with col1:
    if not income_statement.empty:
        fig = go.Figure()

        # Define colors for each trace
        colors = ['blue', '#0080ff', '#c1e8ff']  # Add more colors if needed

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
            title_text=f'',
            # title_text=f'Company Revenues' + (" QoQ" if is_quarterly else " YoY"),
            # Update title based on checkbox value
            title_x=0.5,
            title_y=0.98,
            title_xanchor='center',
            title_yanchor='top',
            font=dict(size=15),
            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45, font=dict(size=15)),  # Center the legend
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
        colors_expenses = ['red', '#cd5c5c', '#fa8072']

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
            # title_text=f'Company Expenses' + (" QoQ" if is_quarterly else " YoY"),
            title_text=f'',
            # Update title based on checkbox value
            title_x=0.5,
            title_y=0.98,
            title_xanchor='center',
            title_yanchor='top',
            font=dict(size=15),  # Use the same font size as in the first chart
            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45, font=dict(size=15)),  # Center the legend
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

# st.write(income_statement)





data = income_statement.loc[['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']].transpose()
# Convert string values to numeric
data = pd.DataFrame(data).apply(lambda x: x.str.replace(',', '').astype(float))


# Now you can proceed with the rest of your code
with col1:
    # # Add title in the middle with smaller font size
    # st.markdown("<h2 style='text-align: left; color: white'>Company Growth Trend</h2>", unsafe_allow_html=True)
    st.subheader(f"Growth Trend {'QoQ' if is_quarterly else 'YoY'}")


    st.write("")

    # Use Streamlit's columns layout manager to display charts side by side
    col1, col2 = st.columns(2)

    # Define colors for each metric
    line_colors = ['red', 'red']

    # Create a new figure for the line chart
    line_fig = go.Figure()

    # Define the list of metrics
    metrics = ['Total Revenue', 'Gross Profit']

    # Iterate over each metric and create a chart
    for metric, col in zip(metrics, [col1, col2]):

        # Regular CAGR calculation
        start_value = data[metric].iloc[0]
        end_value = data[metric].iloc[-1]
        num_periods = len(data)
        Ncagr = False

        if start_value == 0:
            # Handle zero starting value
            cagr = 0  # Or set a specific value (e.g., "Not Applicable")

        if start_value < 0 and end_value > 0:
            # Scenario: Negative Starting Value and Positive Ending Value (N, P)
            dif = abs(start_value) + end_value
            cagr = ((dif / abs(start_value)) ** (1 / num_periods)) - 1
        elif start_value < 0 and end_value < 0:
            # Scenario: Negative Starting Value and Negative Ending Value (N, N)
            if abs(end_value) < abs(start_value):
                # Lost less money
                cagr = ((end_value / start_value) ** (1 / num_periods)) + 1
            else:
                # Lost more money
                cagr = -((end_value / start_value) ** (1 / num_periods)) - 1
        elif start_value > 0 and end_value < 0:

            Ncagr = True
            st.write(Ncagr)

        else:
            # Regular CAGR calculation (P, P or N, N with same sign)
            cagr = ((end_value / start_value) ** (1 / num_periods)) - 1

        # Create a new chart for the current metric
        with col:
            if not data_percentage_change_df.empty:
                # Create a new figure for the current metric
                fig = go.Figure()

                # Add trace for the line chart
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(data_percentage_change_df.index).strftime('%Y-%m-%d' if is_quarterly else '%YY'),
                    y=data_percentage_change_df[metric],
                    mode='lines',
                    name="Growth Trend Line",
                    yaxis='y2',  # Assign the line to the secondary y-axis
                    line=dict(color=line_colors[metrics.index(metric)], width=3)  # Assign color from the list
                ))

                # Add annotations for each point on the line
                for index, row in data_percentage_change_df.iterrows():
                    index_dt = pd.to_datetime(index)  # Convert index to datetime object
                    fig.add_annotation(
                        x=index_dt.strftime('%Y-%m-%d' if is_quarterly else '%YY'),
                        y=row[metric],  # y-coordinate of the annotation
                        text=f"{row[metric]:.2f}%",  # text to display (format as needed)
                        showarrow=False,  # don't show an arrow
                        yref='y2',
                        xanchor='right',  # anchor point for x-coordinate
                        yanchor='middle',  # anchor point for y-coordinate
                        align='left',  # alignment of the annotation text
                        bgcolor='yellow',
                        xshift=5,  # horizontal shift of the annotation
                        yshift=0,  # vertical shift of the annotation
                    )

                # Add trace for the bar chart representing metric values
                fig.add_trace(go.Bar(
                    x=pd.to_datetime(data_percentage_change_df.index).strftime('%Y-%m-%d' if is_quarterly else '%YY'),
                    y=data[metric],
                    name=f"{metric}",
                    marker_color='#0080ff',  # Blue bars with transparency
                    opacity=1,  # Set the opacity for bars
                    yaxis='y',  # Use the primary y-axis for bars

                ))

                if Ncagr == True:
                    title = f"{metric} {'QoQ' if is_quarterly else 'YoY'}<br>"
                    if not is_quarterly:
                        title += f"{''if not is_quarterly else 'Yearly CAGR:'}<span style=\"color: red;\">No CAGR (End value negative)</span>"

                else:
                    color = 'red' if cagr < 0 else 'green'  # Change to 'red' if CAGR is negative, else 'green'
                    title = title = f"{metric} {'QoQ' if is_quarterly else 'YoY'}<br>" + (
                        '' if is_quarterly else f"{'    ' if is_quarterly else 'Yearly CAGR'}: <span style=\"color: {color};\">{cagr:.2%}</span>")

                # Update layout for the current chart
                fig.update_layout(
                    title=title,
                    title_x=0.30,  # Set the title's horizontal position to the center
                    xaxis=dict(title=''),
                    yaxis=dict(title=f"{metric} (M$)"),
                    yaxis2=dict(title='% Growth', overlaying='y', side='right', showgrid=False),  # Add secondary y-axis
                    width=800,  # Adjust the width of each chart as needed
                    height=400,  # Adjust the height of each chart as needed
                    font=dict(size=15, color='black'),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5, font=dict(size=15))  # Adjust legend position
                )

                # Plot the current chart
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            else:
                st.write("Data is empty for metric:", metric)



    # Define colors for each metric
    line_colors = ['red', 'red']

    # Create a new figure for the line chart
    line_fig = go.Figure()

    # Define the list of metrics
    metrics = ['Operating Income', 'Net Income']



    # Iterate over each metric and create a chart
    for metric, col in zip(metrics, [col1, col2]):

        Ncagr = False
        # Regular CAGR calculation
        start_value = data[metric].iloc[0]
        end_value = data[metric].iloc[-1]
        num_periods = len(data)

        if start_value == 0:
            # Handle zero starting value
            cagr = 0  # Or set a specific value (e.g., "Not Applicable")

        if start_value < 0 and end_value > 0:
            # Scenario: Negative Starting Value and Positive Ending Value (N, P)
            dif = abs(start_value) + end_value
            cagr = ((dif / abs(start_value)) ** (1 / num_periods)) - 1
        elif start_value < 0 and end_value < 0:
            # Scenario: Negative Starting Value and Negative Ending Value (N, N)
            if abs(end_value) < abs(start_value):
                # Lost less money
                cagr = ((end_value / start_value) ** (1 / num_periods)) + 1
            else:
                # Lost more money
                cagr = -((end_value / start_value) ** (1 / num_periods)) - 1
        elif start_value > 0 and end_value < 0:

            Ncagr = True
            st.write(Ncagr)

        else:
            # Regular CAGR calculation (P, P or N, N with same sign)
            cagr = ((end_value / start_value) ** (1 / num_periods)) - 1

        # Create a new chart for the current metric
        with col:
            if not data_percentage_change_df.empty:
                # Create a new figure for the current metric
                fig = go.Figure()

                # Add trace for the line chart
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(data_percentage_change_df.index).strftime('%Y-%m-%d' if is_quarterly else '%YY'),
                    y=data_percentage_change_df[metric],
                    mode='lines',
                    name="Growth Trend Line",
                    yaxis='y2',  # Assign the line to the secondary y-axis
                    line=dict(color=line_colors[metrics.index(metric)], width=3)  # Assign color from the list
                ))

                # Add annotations for each point on the line
                for index, row in data_percentage_change_df.iterrows():
                    index_dt = pd.to_datetime(index)  # Convert index to datetime object
                    fig.add_annotation(
                        x=index_dt.strftime('%Y-%m-%d' if is_quarterly else '%YY'),
                        y=row[metric],  # y-coordinate of the annotation
                        text=f"{row[metric]:.2f}%",  # text to display (format as needed)
                        showarrow=False,  # don't show an arrow
                        yref='y2',
                        xanchor='right',  # anchor point for x-coordinate
                        yanchor='middle',  # anchor point for y-coordinate
                        align='left',  # alignment of the annotation text
                        bgcolor='yellow',
                        xshift=5,  # horizontal shift of the annotation
                        yshift=0,  # vertical shift of the annotation
                    )

                # Add trace for the bar chart representing metric values
                fig.add_trace(go.Bar(
                    x=pd.to_datetime(data_percentage_change_df.index).strftime('%Y-%m-%d' if is_quarterly else '%YY'),
                    y=data[metric],
                    name=f"{metric}",
                    marker_color='#0080ff',  # Blue bars with transparency
                    opacity=1,  # Set the opacity for bars
                    yaxis='y',  # Use the primary y-axis for bars

                ))

                if Ncagr == True:
                    title = f"{metric} {'QoQ' if is_quarterly else 'YoY'}<br>"
                    if not is_quarterly:
                        title += f"{'' if not is_quarterly else 'Yearly CAGR:'} <span style=\"color: red;\">No CAGR (End value negative)</span>"

                else:
                    color = 'red' if cagr < 0 else 'green'  # Change to 'red' if CAGR is negative, else 'green'
                    title = title = f"{metric} {'QoQ' if is_quarterly else 'YoY'}<br>" + (
                        '' if is_quarterly else f"{'    ' if is_quarterly else 'Yearly CAGR'}: <span style=\"color: {color};\">{cagr:.2%}</span>")

                fig.update_layout(
                    title=title,
                    title_x=0.35,  # Set the title's horizontal position to the center
                    xaxis=dict(title=''),
                    yaxis=dict(title=f"{metric} (M$)"),
                    yaxis2=dict(title='% Growth', overlaying='y', side='right', showgrid=False),  # Add secondary y-axis
                    width=800,  # Adjust the width of each chart as needed
                    height=400,  # Adjust the height of each chart as needed
                    font=dict(size=15, color='black'),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5, font=dict(size=15))
                    # Adjust legend position
                )

                # Plot the current chart
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            else:
                st.write("Data is empty for metric:", metric)







# Basic EPS and Diluted EPS data for all years *****************************************************
st.write("")
st.subheader(f"Profitability")
col1, col2, col3 = st.columns([0.35, 0.35, 0.3])

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
        marker=dict(color='blue')
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
        marker=dict(color='#0080ff')  # Medium blue color
    ))

    # Update layout
    fig.update_layout(
        #
        title_text=f'',
        title_x=0.38,
        xaxis=dict(title='' if is_quarterly else ''),
        yaxis=dict(title='EPS Value'),
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.45,
            font=dict(size=15),
        ),
        font=dict(
            size=15  # Adjust font size of values on bars
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
        marker=dict(color='blue')

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
        marker=dict(color='#0080ff')  # Medium blue color
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
        marker=dict(color='#c1e8ff')  # Medium blue color
    ))

    # Update layout
    fig.update_layout(
        # title='EBIT VS EBITDA vs Net Income (M$)',
        title_text=f'',
        title_x=0.35,
        xaxis=dict(title='' if is_quarterly else ''),
        yaxis=dict(title='Value'),
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.45,
            font=dict(size=15),
        ),
        font=dict(
            size=18  # Adjust font size of values on bars
        )
    )
    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
