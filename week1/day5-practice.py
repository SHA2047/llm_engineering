import requests
from bs4 import BeautifulSoup


# def scrape_text_and_links(url, element_type, element_attribute=None, attribute_value=None):
#     """
#     Scrapes text and hyperlinks from a website.

#     Args:
#         url (str): The URL of the website to scrape.
#         element_type (str): The HTML element type to target (e.g., "p", "a", "div").
#         element_attribute (str, optional): The attribute to filter by (e.g., "class", "id"). Defaults to None.
#         attribute_value (str, optional): The value of the attribute to filter by. Defaults to None.

#     Returns:
#         list: A list of dictionaries, where each dictionary contains the text and href (if available) of an element.
#     """
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.content, "html.parser")

#         if element_attribute and attribute_value:
#             elements = soup.find_all(element_type, {element_attribute: attribute_value})
#         else:
#             elements = soup.find_all(element_type)

#         results = []
#         for element in elements:
#             text = element.get_text(strip=True)
#             href = element.get("href")  # Get the href attribute, if it exists
#             results.append({"text": text, "href": href})

#         return results

#     except requests.exceptions.RequestException as e:
#         print(f"Error fetching URL: {e}")
#         return []
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return []

# Example usage:
url = "https://edwarddonner.com/" #replace with the url you wish to scrape
data = scrape_text_and_links(url, "a") #example scraping anchor tags
for item in data:
    print(f"Text: {item['text']}, Link: {item['href']}")

data = scrape_text_and_links(url, "p") #example scraping paragraph tags.
for item in data:
    print(f"Text: {item['text']}, Link: {item['href']}")
        
# import openai # type: ignore
# import os
# # from IPython.display import display, Markdown
# from dotenv import load_dotenv
# load_dotenv()

# class TextSummarizer:
#     def __init__(self, openai_api_key):
#         openai.api_key = openai_api_key

#     def summarize_text(self, text):
#         """Summarizes text using the ChatGPT API."""
#         try:
#             response = openai.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant that Creates a Company Brochure using the provided scraped text from its website"},
#                     {"role": "user", "content": f"Summarize the following text: {text}"},
#                 ],
#             )
#             return response.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"Error summarizing text: {e}")
#             return ""
        
# # # Example usage:
# openai_api_key = os.getenv('OPENAI_API_KEY') #replace with your api key.
# # # url = "https://www.example.com" #replace with your target website.

# from taipy import Gui

# page = """
# <|{url}|input|label=Enter Page url|action_keys="Enter"|>


# <|Summarize|button|on_action=summary|>

# <|{summary_display}|text|>
# """

# # Example usage:
# # openai_api_key = os.getenv('OPENAI_API_KEY') #replace with your api key.
# url = "" #replace with your target website.
# summary_display = ""

# def summary(state):
#     # state.summary_display = "Processing..."

#     scraper = WebScraper()
#     summarizer = TextSummarizer(openai_api_key)

#     scraped_text = scraper.scrape_text(state.url, "a")
#     summarized = summarizer.summarize_text(scraped_text)

#     if "Error" in scraped_text:
#         state.summary_display = "Error: Unable to scrape the text."
#         return
#     elif "Error" in summarized:
#         state.summary_display = "Error: Unable to summarize the text."

#     state.summary_display = summarized
#     return

# gui = Gui(page)
# gui.run(port=8080, title="Web Scraper and Summarizer", width=800, height=600, use_reloader=True)