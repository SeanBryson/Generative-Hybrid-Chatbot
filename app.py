import openai
import gradio as gr
import requests
import xml.etree.ElementTree as ET

ARXIV_ENDPOINT = "http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results=1"
openai.organization = "org-MYzdbZoWTu1PpVARD32T829L"
file = open('D:\Python_Workspace\ChatbotV2\Building a Chatbot.txt')
openai.api_key = file.readline()
#print(openai.Model.list())

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]

def chatbot(input):
    if "paper" in input or "research" in input or "arxiv" in input:
        search_term = input.replace("paper", "").replace("research", "").replace("arxiv", "").strip()
        response = requests.get(ARXIV_ENDPOINT.format(search_term))
        data = response.text

        # Parse the XML response using ElementTree
        root = ET.fromstring(data)
        entry = root.find('{http://www.w3.org/2005/Atom}entry')
        if entry is not None:
            title = entry.find('{http://www.w3.org/2005/Atom}title').text
            reply = f"I found a paper for you: {title}."
        else:
            reply = "Sorry, I couldn't find any papers matching your query."
    else:
        if input:
            messages.append({"role": "user", "content": input})
            chat = openai.ChatCompletion.create(
                model="gpt-4", messages=messages
            )
            reply = chat.choices[0].message.content
            messages.append({"role": "assistant", "content": reply})
            return reply

    return reply


inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=True)