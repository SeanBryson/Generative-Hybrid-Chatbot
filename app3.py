# iTunes - Music Version 
# only works with prompts like "find me a song"
import openai
import gradio as gr
import requests
import json

ITUNES_ENDPOINT = "https://itunes.apple.com/search?term={}&limit=1&entity=song"
openai.organization = "org-MYzdbZoWTu1PpVARD32T829L"
file = open('D:\Python_Workspace\ChatbotV2\Building a Chatbot.txt')
openai.api_key = file.readline()
print(openai.api_key)

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]

def chatbot(input):
    if "song" in input or "track" in input or "music" in input:
        search_term = input.replace("song", "").replace("track", "").replace("music", "").strip()
        response = requests.get(ITUNES_ENDPOINT.format(search_term))
        data = response.json()

        # Parse the JSON response
        if data['resultCount'] > 0:
            track_name = data['results'][0]['trackName']
            artist_name = data['results'][0]['artistName']
            reply = f"I found a song for you: '{track_name}' by {artist_name}."
        else:
            reply = "Sorry, I couldn't find any songs matching your query."
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
