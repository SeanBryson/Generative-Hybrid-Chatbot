# iTunes - Music Version 
import openai
import gradio as gr
import requests
import json

ITUNES_ENDPOINT = "https://itunes.apple.com/search?term={}&limit=5&entity=song"
openai.organization = "org-MYzdbZoWTu1PpVARD32T829L"
file = open('D:\Python_Workspace\Building a Chatbot.txt')
openai.api_key = file.readline()

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]

def get_song_intent(input):
    messages.append({"role": "user", "content": input})
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
    intent = chat.choices[0].message.content
    messages.append({"role": "assistant", "content": intent})
    return intent

def chatbot(input):
    if "song" in input or "track" in input or "music" in input:
        search_term = get_song_intent(input)
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
        reply = get_song_intent(input)

    return reply

inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=True)
