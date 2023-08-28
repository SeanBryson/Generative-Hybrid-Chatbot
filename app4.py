import openai
import gradio as gr
import requests
import json

# Read eBay App ID from a file
with open('D:\Python_Workspace\ebay_key.txt', 'r') as file:
    EBAY_APP_ID = file.readline().strip()

# eBay Finding API endpoint for searching items
EBAY_ENDPOINT = f"https://svcs.ebay.com/services/search/FindingService/v1?OPERATION-NAME=findItemsByKeywords&SERVICE-VERSION=1.0.0&SECURITY-APPNAME={EBAY_APP_ID}&RESPONSE-DATA-FORMAT=JSON&REST-PAYLOAD&keywords={{}}&paginationInput.entriesPerPage=1"

openai.organization = "org-MYzdbZoWTu1PpVARD32T829L"
with open('D:\Python_Workspace\Building a Chatbot.txt', 'r') as file:
    openai.api_key = file.readline().strip()

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]

def chatbot(input):
    if "product" in input or "item" in input or "buy" in input:
        search_term = input.replace("product", "").replace("item", "").replace("buy", "").strip()
        response = requests.get(EBAY_ENDPOINT.format(search_term))
        data = response.json()

        # Parse the JSON response
        if data['findItemsByKeywordsResponse'][0]['searchResult'][0]['@count'] != '0':
            item_title = data['findItemsByKeywordsResponse'][0]['searchResult'][0]['item'][0]['title'][0]
            item_url = data['findItemsByKeywordsResponse'][0]['searchResult'][0]['item'][0]['viewItemURL'][0]
            reply = f"I found a product for you: '{item_title}'. [View on eBay]({item_url})"
        else:
            reply = "Sorry, I couldn't find any products matching your query."
    else:
        if input:
            messages.append({"role": "user", "content": input})
            chat = openai.Completion.create(
                model="text-davinci-002",  # or "gpt-3.5-turbo"
                prompt=input,
                max_tokens=150
            )
            reply = chat.choices[0].text.strip()
            messages.append({"role": "assistant", "content": reply})
            return reply

    return reply

inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=True)
