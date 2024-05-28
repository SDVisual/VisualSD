import streamlit as st

APP_NAME = "Visual Stock Data"

# Set page configuration
st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="auto"
)

def show_disclaimer():
    color_code = "#0ECCEC"
    header_html = f'<h2 style="color:{color_code};">DISCLAIMER</h2>'
    st.markdown(header_html, unsafe_allow_html=True)

    disclaimer_content = (
        "- This Web Application (first Beta Version for Desktop Computers) aims to enhance the accessibility and comprehension of financial data by providing visual representations of various financial metrics, including stock summaries, income statements, balance sheets and cash flow statements. It is designed to facilitate the understanding of new companies by presenting data visually, allowing users to interpret information beyond mere numerical values."
        "\n\n- The information presented in this application is for informational purposes only and should not be considered a substitute for professional financial consultation. Users are encouraged to conduct their own research and consult with qualified financial advisors before making any investment decisions."
        "\n\n- The creators of this application do not guarantee the accuracy, completeness, or reliability of the information retrieved from Financial Data APIs."
        "\n\n- By continuing to use this application, you agree that you have read and understood this disclaimer, and you acknowledge that the creators of this application are not liable for any investment decisions made based on the information presented."
    )

    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.write(disclaimer_content)  # Display the disclaimer content

# Display the disclaimer
show_disclaimer()
