import PyPDF2
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_api = os.getenv('groq_api')
client = Groq(api_key=groq_api)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text()

    return text


def analyze_portfolio_from_pdf(pdf_path):
    portfolio = extract_text_from_pdf(pdf_path)
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"Portfolio: {portfolio}. Provide recommendations and analysis of the stock portfolio with stock information. Give response for every stock in different lines and provide some stock to buy that are not in the portfolio. Give the result in the format that it shows the results in a table form with which one to keep, which one to sell and the recommendations to buy in separate columns."}],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    # Accessing the content directly from the 'message' object
    choices = completion.choices[0].message.content
    return choices


pdf_path = "Testing/Portfolio.pdf"
recommendations = analyze_portfolio_from_pdf(pdf_path)
print(recommendations)
