import openai
import gradio as gr
import requests
import json
import random

ITUNES_ENDPOINT = "https://itunes.apple.com/search?term={}&limit={}&entity={}"
openai.organization = "org-MYzdbZoWTu1PpVARD32T829L"
file = open('D:\Python_Workspace\Building a Chatbot.txt')
openai.api_key = file.readline()

messages = [
    {"role": "system", "content": "You are a music suggestion bot with API access to iTunes. Please return at least one song"},
]

def extract_artist_name_and_type(input):
    # List of common words to remove
    remove_words = ["find", "me", "a", "get", "play", "song", "by", "album", "artist", "track", "music", "latest", "top", 
                    "hit", "popular", "most", "what", "is", "can", "you", "the", "your", "favorite", "?"]

    # Split the input into words and remove common words
    artist_name_words = [word for word in input.split() if word.lower() not in remove_words]

    # Join the words back together to get the artist's name
    artist_name = ' '.join(artist_name_words).strip()

    # Determine if the user is asking for a song or an album
    if "album" in input:
        request_type = "album"
    else:
        request_type = "song"

    return artist_name, request_type


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
    song_data = song_response.json()
    filtered_songs = [song for song in song_data['results'] if song['artistName'].lower() == artist_name.lower()]
    if not filtered_songs:
        return None

    # Assuming the most popular song is the one with the highest track count (this is a basic heuristic)
    popular_song = max(filtered_songs, key=lambda x: x.get('trackCount', 0))
    return popular_song

def get_latest_album_by_artist(artist_name):
    album_response = requests.get(ITUNES_ENDPOINT.format(artist_name, 100, "album"))
    album_data = album_response.json()
    filtered_albums = [album for album in album_data['results'] if album['artistName'].lower() == artist_name.lower()]
    if not filtered_albums:
        return None

    latest_album = sorted(filtered_albums, key=lambda x: x['releaseDate'], reverse=True)[0]
    return latest_album

def get_song_intent(input):
    messages.append({"role": "user", "content": input})
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
    refined_intent = chat.choices[0].message.content
    messages.append({"role": "assistant", "content": refined_intent})
    return refined_intent

def get_most_popular_album_by_artist(artist_name):
    # For simplicity, we'll assume the first album returned is the most popular, but this might not always be the case.
    album_response = requests.get(ITUNES_ENDPOINT.format(artist_name, 1, "album"))
    album_data = album_response.json()
    if album_data['results']:
        return album_data['results'][0]
    return None

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
    refined_intent = get_song_intent(input)
    artist_name, request_type = extract_artist_name_and_type(input)
    encoded_artist_name = requests.utils.quote(artist_name)  # URL encode the artist's name
    
    if request_type == "song":
        if "latest" in input:
            song = get_latest_song_by_artist(encoded_artist_name)
            song_type = "latest"
        else:
            song = get_most_popular_song_by_artist(encoded_artist_name)
            song_type = "most popular"
        
        if song:
            track_name = song['trackName']
            artist_name = song['artistName']
            song_suggestions = get_song_suggestions(artist_name, track_name)
            reply = f"The {song_type} song by {artist_name} is '{track_name}'. {song_suggestions}"
        else:
            reply = refined_intent
    else:  # request_type == "album"
        if "latest" in input:
            album = get_latest_album_by_artist(encoded_artist_name)
            album_type = "latest"
        elif "popular" in input or "most popular" in input:
            album = get_most_popular_album_by_artist(encoded_artist_name)
            album_type = "most popular"
        elif "favorite" in input or "best" in input:
            # Fetch a list of albums and select a random one as the favorite
            album_response = requests.get(ITUNES_ENDPOINT.format(encoded_artist_name, 10, "album"))
            album_data = album_response.json()
            if not album_data['results']:
                return refined_intent
            album = random.choice(album_data['results'])
            justification = get_favorite_album_justification(artist_name, album['collectionName'])
            return f"One of the best albums by {artist_name} is '{album['collectionName']}'. {justification}"

        if album:
            album_name = album['collectionName']
            artist_name = album['artistName']
            reply = f"The {album_type} album by {artist_name} is '{album_name}'."
        else:
            reply = refined_intent

    return reply

inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")
outputs = gr.outputs.Textbox(label="Reply")

gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=True)
