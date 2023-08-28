import openai
import gradio as gr
import requests
import json
import random

ITUNES_ENDPOINT = "https://itunes.apple.com/search?term={}&limit={}&entity={}"
openai.organization = "org-MYzdbZoWTu1PpVARD32T829L"
file = open('D:\Python_Workspace\Building a Chatbot.txt')
openai.api_key = file.readline()

def extract_artist_name_and_type(input):
    remove_words = ["find", "me", "a", "get", "play", "song", "by", "album", "artist", "track", "music", "latest", "top", 
                    "hit", "popular", "most", "what", "is", "can", "you", "the", "?"]
    artist_name = ' '.join([word for word in input.split() if word.lower() not in remove_words]).strip()
    request_type = "album" if "album" in input else "song"
    return artist_name, request_type

def get_latest_song_by_artist(artist_name):
    album_response = requests.get(ITUNES_ENDPOINT.format(artist_name, 100, "album"))
    filtered_albums = [album for album in album_response.json()['results'] if album['artistName'].lower() == artist_name.lower()]
    if not filtered_albums:
        return None
    latest_album_name = sorted(filtered_albums, key=lambda x: x['releaseDate'], reverse=True)[0]['collectionName']
    song_response = requests.get(ITUNES_ENDPOINT.format(latest_album_name, 100, "song"))
    filtered_songs = [song for song in song_response.json()['results'] if song['artistName'].lower() == artist_name.lower()]
    return sorted(filtered_songs, key=lambda x: x['releaseDate'], reverse=True)[0] if filtered_songs else None

def get_song_suggestions(artist_name, song_name):
    prompt = f"I just listened to '{song_name}' by {artist_name}. What other songs would you recommend?"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a knowledgeable music enthusiast. Provide song recommendations."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()

def get_most_popular_song_by_artist(artist_name):
    song_response = requests.get(ITUNES_ENDPOINT.format(artist_name, 100, "song"))
    filtered_songs = [song for song in song_response.json()['results'] if song['artistName'].lower() == artist_name.lower()]
    return max(filtered_songs, key=lambda x: x.get('trackCount', 0)) if filtered_songs else None

def get_latest_album_by_artist(artist_name):
    album_response = requests.get(ITUNES_ENDPOINT.format(artist_name, 100, "album"))
    filtered_albums = [album for album in album_response.json()['results'] if album['artistName'].lower() == artist_name.lower()]
    return sorted(filtered_albums, key=lambda x: x['releaseDate'], reverse=True)[0] if filtered_albums else None

def get_most_popular_album_by_artist(artist_name):
    album_response = requests.get(ITUNES_ENDPOINT.format(artist_name, 1, "album"))
    return album_response.json()['results'][0] if album_response.json()['results'] else None

def get_favorite_album_justification(artist_name, album_name):
    prompt = f"Why is '{album_name}' by {artist_name} considered one of the best albums?"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a knowledgeable music enthusiast. Provide insights about albums."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()

def chatbot(input):
    artist_name, request_type = extract_artist_name_and_type(input)
    encoded_artist_name = requests.utils.quote(artist_name)
    
    if request_type == "song":
        song = get_latest_song_by_artist(encoded_artist_name) if "latest" in input else get_most_popular_song_by_artist(encoded_artist_name)
        if song:
            track_name = song['trackName']
            song_suggestions = get_song_suggestions(artist_name, track_name)
            return f"The {request_type} song by {artist_name} is '{track_name}'. {song_suggestions}"
    else:
        if "latest" in input:
            album = get_latest_album_by_artist(encoded_artist_name)
            album_type = "latest"
        elif "popular" in input or "most popular" in input:
            album = get_most_popular_album_by_artist(encoded_artist_name)
            album_type = "most popular"
        elif "favorite" in input or "best" in input:
            album_response = requests.get(ITUNES_ENDPOINT.format(encoded_artist_name, 10, "album"))
            album_data = album_response.json()
            if not album_data['results']:
                return "Sorry, I couldn't find any albums for that artist."
            album = random.choice(album_data['results'])
            justification = get_favorite_album_justification(artist_name, album['collectionName'])
            return f"One of the best albums by {artist_name} is '{album['collectionName']}'. {justification}"

        if album:
            return f"The {album_type} album by {artist_name} is '{album['collectionName']}'."

    return "Sorry, I couldn't find any results matching your query."

inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=True)

