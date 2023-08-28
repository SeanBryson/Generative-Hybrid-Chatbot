# iTunes - Music Version 
import openai
import gradio as gr
import requests
import json

ITUNES_ENDPOINT = "https://itunes.apple.com/search?term={}&limit={}&entity={}"

def extract_artist_name(input):
    if "by" in input:
        artist_name = input.split("by")[1].strip()
    elif "latest" in input and "song" in input:
        artist_name = input.split("latest")[1].split("song")[0].strip()
    else:
        artist_name = input
    return artist_name

def get_latest_song_by_artist(artist_name):
    album_response = requests.get(ITUNES_ENDPOINT.format(artist_name, 100, "album"))
    album_data = album_response.json()
    filtered_albums = [album for album in album_data['results'] if album['artistName'].lower() == artist_name.lower()]
    if not filtered_albums:
        return None

    latest_album = sorted(filtered_albums, key=lambda x: x['releaseDate'], reverse=True)[0]
    latest_album_name = latest_album['collectionName']

    song_response = requests.get(ITUNES_ENDPOINT.format(latest_album_name, 100, "song"))
    song_data = song_response.json()
    filtered_songs = [song for song in song_data['results'] if song['artistName'].lower() == artist_name.lower()]
    if not filtered_songs:
        return None

    latest_song = sorted(filtered_songs, key=lambda x: x['releaseDate'], reverse=True)[0]
    return latest_song

def chatbot(input):
    if any(word in input for word in ["song", "track", "music"]):
        artist_name = extract_artist_name(input)
        latest_song = get_latest_song_by_artist(artist_name)
        if latest_song:
            track_name = latest_song['trackName']
            artist_name = latest_song['artistName']
            reply = f"The latest song by {artist_name} from their most recent album is '{track_name}'."
        else:
            reply = "Sorry, I couldn't find any songs matching your query."
    else:
        reply = "Please specify an artist to search for their latest song."

    return reply

inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=True)

