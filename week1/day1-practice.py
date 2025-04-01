import requests
from bs4 import BeautifulSoup

class WebScraper:
    def __init__(self):
        pass

    def scrape_text(self, url, element_type, element_attribute=None, attribute_value=None):
        """
        Scrapes text from a website.

        Args:
            url (str): The URL of the website to scrape.
            element_type (str): The HTML element type to target (e.g., "p", "h1", "div").
            element_attribute (str, optional): The attribute to filter by (e.g., "class", "id"). Defaults to None.
            attribute_value (str, optional): The value of the attribute to filter by. Defaults to None.

        Returns:
            list: A list of strings containing the extracted text.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            soup = BeautifulSoup(response.content, "html.parser")

            if element_attribute and attribute_value:
                elements = soup.find_all(element_type, {element_attribute: attribute_value})
            else:
                elements = soup.find_all(element_type)

            text_list = [element.get_text(strip=True) for element in elements]
            return text_list

        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL: {e}")
            return []
        except Exception as e:
            print(f"An error occured: {e}")
            return []
        
import openai
import os
# from IPython.display import display, Markdown
from dotenv import load_dotenv
load_dotenv()

class TextSummarizer:
    def __init__(self, openai_api_key):
        openai.api_key = openai_api_key

    def summarize_text(self, text):
        """Summarizes text using the ChatGPT API."""
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates company"},
                    {"role": "user", "content": f"Summarize the following text: {text}"},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error summarizing text: {e}")
            return ""
        
# # Example usage:
openai_api_key = os.getenv('OPENAI_API_KEY') #replace with your api key.
# # url = "https://www.example.com" #replace with your target website.

from taipy import Gui

page = """
<|{url}|input|label=Enter Page url|action_keys="Enter"|>


<|Summarize|button|on_action=summary|>

<|{summary_display}|text|>
"""

# Example usage:
# openai_api_key = os.getenv('OPENAI_API_KEY') #replace with your api key.
url = "" #replace with your target website.
summary_display = ""

def summary(state):
    # state.summary_display = "Processing..."

    scraper = WebScraper()
    summarizer = TextSummarizer(openai_api_key)

    scraped_text = scraper.scrape_text(state.url, "p")
    summarized = summarizer.summarize_text(scraped_text)

    if "Error" in scraped_text:
        state.summary_display = "Error: Unable to scrape the text."
        return
    elif "Error" in summarized:
        state.summary_display = "Error: Unable to summarize the text."

    state.summary_display = summarized
    return

gui = Gui(page)
gui.run(port=8080, title="Web Scraper and Summarizer", width=800, height=600, use_reloader=True)