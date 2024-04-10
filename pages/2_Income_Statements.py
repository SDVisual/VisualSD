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

# Display the image with the caption
st.image('Logo.png')
st.write("")

# Define sidebar elements
st.sidebar.image("Side.png", use_column_width=True)

# # Display title with blue color using Markdown
# st.markdown(f"<h1 style='color:blue;'>{APP_NAME}</h1>", unsafe_allow_html=True)


# Input box for user to enter symbol
new_symbol = st.text_input("Add Stock Symbol to Symbols List (e.g., AAPL)").strip().upper()

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
else:
    if new_symbol in st.session_state.valid_tickers:
        st.warning(f"'{new_symbol}' is already in Symbols List - Clear Text")


# Check if the entered symbol is valid
historical_data = yf.Ticker(new_symbol).history(period='1d')

if new_symbol != selected_symbol and historical_data.empty:
    st.error("Invalid symbol. Please enter a valid symbol.")
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

# # Add a menu to the sidebar
# menu_option = st.sidebar.radio("Menu", ["Company Summary", "Income Statements", "Balance Sheet", "Cash Flow"])

StockInfo = yf.Ticker(ticker).info
income_statementYear = yf.Ticker(ticker).income_stmt
IncomeStatementQuarterly = yf.Ticker(ticker).quarterly_income_stmt

# Default to annual income statement
income_statement = income_statementYear

# color_code = "#0ECCEC"
# font_size = "25px"  # You can adjust the font size as needed
#
# # Render subheader with customized font size and color
# st.markdown(f'<h2 style="color:{color_code}; font-size:{font_size}">{StockInfo["shortName"]}</h2>', unsafe_allow_html=True)


symbol = StockInfo["shortName"]
color_code = "#0ECCEC"  # Color for the symbol

# Combine st.write() with HTML-styled header
st.write(f'<span style="color:white; font-size:30px;">Income Statement - </span>'
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



st.write(f'*values in millions $')



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

col1, col2 = st.columns([0.5, 0.5])  # Adjust the width ratio of col1 and col2 as needed

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
            legend=dict(orientation="h", yanchor="bottom", y=1.07, xanchor="center", x=0.45),
            # Center the legend
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.write("Income statement is empty.")


st.subheader(f"revenues & expenses ")
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
            font=dict(size=15),
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
            font=dict(size=15),  # Use the same font size as in the first chart
            legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.45),  # Center the legend
        )

        st.plotly_chart(fig_expenses, use_container_width=True, config={'displayModeBar': False})
    else:
        st.write("Income statement is empty.")





# Plot bar Revenue Growth ***************************************************************************************

col1, col2 = st.columns([0.6, 0.4])  # Adjust the width ratio of col1 and col2 as needed

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
    # # Add title in the middle with smaller font size
    # st.markdown("<h2 style='text-align: left; color: white'>Company Growth Trend</h2>", unsafe_allow_html=True)
    st.subheader(f"Growth Trend")

    st.write("")
    # Use Streamlit's columns layout manager to display charts side by side
    col1, col2 = st.columns(2)

    # Define colors for each metric
    line_colors = ['blue', 'red', 'green', 'orange']

    # Create a new figure for the line chart
    line_fig = go.Figure()

    # Define the list of metrics
    metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']

    # Iterate over each metric and create a chart
    for metric, col in zip(metrics, [col1, col2]):
        # Create a new chart for the current metric
        with col:
            if not data_percentage_change_df.empty:
                # Create a new figure for the current metric
                fig = go.Figure()

                # Add trace for the current metric
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(data_percentage_change_df.index).strftime(
                        '%Y-%m-%d' if is_quarterly else '%Y-%m'),
                    y=data_percentage_change_df[metric],
                    mode='lines',
                    name=metric,
                    line=dict(color=line_colors[metrics.index(metric)]),  # Assign color from the list
                ))

                # Add annotations for each point on the line
                for index, row in data_percentage_change_df.iterrows():
                    index_dt = pd.to_datetime(index)  # Convert index to datetime object
                    fig.add_annotation(
                        x=index_dt.strftime('%Y-%m-%d' if is_quarterly else '%Y-%m'),
                        # Use specific x-value for each data point
                        y=row[metric],  # y-coordinate of the annotation
                        text=f"{row[metric]:.2f}%",  # text to display (format as needed)
                        showarrow=False,  # don't show an arrow
                        font=dict(color='white'),  # color of the annotation text
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
                    font=dict(size=15),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=1),
                )

                # Plot the current chart
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.write("Data is empty for metric:", metric)




    # Define colors for each metric
    line_colors = ['green', 'orange']

    # Create a new figure for the line chart
    line_fig = go.Figure()

    # Define the list of metrics
    metrics = ['Operating Income', 'Net Income']

    # Iterate over each metric and create a chart
    for metric, col in zip(metrics, [col1, col2]):
        # Create a new chart for the current metric
        with col:
            if not data_percentage_change_df.empty:
                # Create a new figure for the current metric
                fig = go.Figure()

                # Add trace for the current metric
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(data_percentage_change_df.index).strftime(
                        '%Y-%m-%d' if is_quarterly else '%Y-%m'),
                    y=data_percentage_change_df[metric],
                    mode='lines',
                    name=metric,
                    line=dict(color=line_colors[metrics.index(metric)]),  # Assign color from the list
                ))

                # Add annotations for each point on the line
                for index, row in data_percentage_change_df.iterrows():
                    index_dt = pd.to_datetime(index)  # Convert index to datetime object
                    fig.add_annotation(
                        x=index_dt.strftime('%Y-%m-%d' if is_quarterly else '%Y-%m'),
                        # Use specific x-value for each data point
                        y=row[metric],  # y-coordinate of the annotation
                        text=f"{row[metric]:.2f}%",  # text to display (format as needed)
                        showarrow=False,  # don't show an arrow
                        font=dict(color='white'),  # color of the annotation text
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
                    font=dict(size=15),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=1),
                )

                # Plot the current chart
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            else:
                st.write("Data is empty for metric:", metric)




# Basic EPS and Diluted EPS data for all years *****************************************************
st.subheader(f"profitability")
col1, col2, col3 = st.columns([0.3, 0.3, 0.4])

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
        title_x=0.4,
        xaxis=dict(title='' if is_quarterly else ''),
        yaxis=dict(title='EPS Value'),
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.45
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
            x=0.45
        ),
        font=dict(
            size=15  # Adjust font size of values on bars
        )
    )
    # Display the chart without the menu
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

