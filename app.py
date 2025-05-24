import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
import PyPDF2

# Load the .env file for API key
load_dotenv()

# Groq API configuration
groq_api = os.getenv('groq_api')
client = Groq(api_key=groq_api)

# Function to extract the text from the uploaded PDF
def extract_text_from_pdf(uploaded_file):
    # Reading the uploaded file directly from memory
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

# Function to analyze portfolio using Groq API
def analyze_portfolio_from_text(portfolio_text):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages = [
                {
                    "role": "user",
                    "content": f"Portfolio: {portfolio_text}. Please provide recommendations and analysis of the stock portfolio. List each stock recommendation on a separate line with details. Identify which stocks should be kept, which should be sold, and provide suggestions for new stocks to buy (that are not already in the portfolio). Present the results in a table format, showing the stocks to keep, sell, and buy in separate columns. Strictly write any information or statement only once. Tell quantity of how many stocks to buy and sell."
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        # Accessing the content directly from the 'message' object
        choices = completion.choices[0].message.content
        return choices
    
    except Exception as e:
        return f"Error during API request: {str(e)}"

# Function to parse and display the result in a table format
def display_recommendations(recommendations):
    
    # Extracting tables and recommendations
    keep_start = recommendations.find("| **Keep**")
    sell_start = recommendations.find("| **Sell**")
    buy_start = recommendations.find("| **Buy**")
    
    keep_section = recommendations[keep_start + len("| **Keep** |"):sell_start].strip().split(",")
    sell_section = recommendations[sell_start + len("| **Sell** |"):buy_start].strip().split(",")
    buy_section = recommendations[buy_start + len("| **Buy** |"):].strip().split(",")
    
    # Display Full Recommendation
    st.write("### Full Recommendation")
    st.markdown(recommendations)

def main():
    st.title("Stock Portfolio Analyzer")
    
    st.markdown("""
    The system will analyze your stock portfolio and provide recommendations.
    Upload a PDF file or enter the portfolio manually below.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your portfolio PDF", type="pdf")

    # Text input area as an alternative
    manual_input = st.text_area("Or enter your portfolio details manually:")

    portfolio_text = ""

    if uploaded_file is not None:
        # Priority to PDF if both are provided
        portfolio_text = extract_text_from_pdf(uploaded_file)
        st.success("PDF uploaded and text extracted.")
    elif manual_input.strip():
        portfolio_text = manual_input
        st.success("Manual input received.")
    else:
        st.info("Please upload a PDF file or enter the portfolio manually.")

    if portfolio_text:
        st.write("Processing portfolio... Please wait.")
        recommendations = analyze_portfolio_from_text(portfolio_text)
        
        if recommendations:
            display_recommendations(recommendations)
        else:
            st.error("Sorry, there was an error processing the portfolio.")

    st.markdown("""
    **Note:** You can use either the upload or manual entry method.
    """)

if __name__ == "__main__":
    main()
