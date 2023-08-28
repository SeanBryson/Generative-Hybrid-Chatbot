# iTunes - Music Version 
import openai
import gradio as gr
import requests
import json

ITUNES_ENDPOINT = "https://itunes.apple.com/search?term={}&limit={}&entity={}"
openai.organization = "org-MYzdbZoWTu1PpVARD32T829L"
file = open('D:\Python_Workspace\Building a Chatbot.txt')
openai.api_key = file.readline()

messages = [
    {"role": "system", "content": "You are a music suggestion bot with API access to iTunes. Please return at least one song"},
]

def get_song_intent(input):
    messages.append({"role": "user", "content": input})
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
    refined_intent = chat.choices[0].message.content
    messages.append({"role": "assistant", "content": refined_intent})
    return refined_intent

def extract_artist_name(refined_intent):
    if "by" in refined_intent:
        artist_name = refined_intent.split("by")[1].strip()
    else:
        artist_name = refined_intent
    return artist_name

def get_song_by_artist(artist_name, latest=True):
    song_response = requests.get(ITUNES_ENDPOINT.format(artist_name, 100, "song"))
    song_data = song_response.json()
    filtered_songs = [song for song in song_data['results'] if song['artistName'].lower() == artist_name.lower()]
    if not filtered_songs:
        return None

    if latest:
        song = sorted(filtered_songs, key=lambda x: x['releaseDate'], reverse=True)[0]
    else:
        song = sorted(filtered_songs, key=lambda x: x['trackPrice'], reverse=True)[0]
    return song

def chatbot(input):
    refined_intent = get_song_intent(input)
    if any(word in refined_intent for word in ["song", "track", "music"]):
        artist_name = extract_artist_name(refined_intent)
        if "latest" in refined_intent:
            song = get_song_by_artist(artist_name, latest=True)
        else:
            song = get_song_by_artist(artist_name, latest=False)
        
        if song:
            track_name = song['trackName']
            artist_name = song['artistName']
            if "latest" in refined_intent:
                reply = f"The latest song by {artist_name} is '{track_name}'."
            else:
                reply = f"One of the most popular songs by {artist_name} is '{track_name}'."
        else:
            reply = f"Sorry, I couldn't find any songs matching your query. However, {refined_intent}"
    else:
        reply = refined_intent

    return reply

inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=True)
