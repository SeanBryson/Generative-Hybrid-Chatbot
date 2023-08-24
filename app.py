import openai
import gradio as gr
import requests
BEST_BUY_API_KEY = "YOUR_BEST_BUY_API_KEY"
BEST_BUY_ENDPOINT = "https://api.bestbuy.com/v1/products(search={})?format=json&apiKey={}"
openai.organization = "org-MYzdbZoWTu1PpVARD32T829L"
file = open('D:\Python_Workspace\ChatbotV2\Building a Chatbot.txt')
openai.api_key = file.readline()
#print(openai.Model.list())

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]



def chatbot(input):
    if "product" in input or "buy" in input:  # Simple keyword detection for product-related queries
        search_term = input.split(" ")[-1]  # Assuming the last word is the product name for simplicity
        response = requests.get(BEST_BUY_ENDPOINT.format(search_term, BEST_BUY_API_KEY))
        data = response.json()
        if data["total"] > 0:
            product = data["products"][0]  # Taking the first product for simplicity
            reply = f"I found a product for you: {product['name']} priced at ${product['salePrice']}."
        else:
            reply = "Sorry, I couldn't find any products matching your query."
    else:
        # Your existing OpenAI chatbot code
        ...

    return reply


inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=True)