import string
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

# Initialize a session dictionary to maintain context
session_data = {
    "last_artist": None,
    "last_song": None,
    "last_album": None,
    "interaction_count": 0,
    "previous_recommendations": []
}

# Function to reset the session data
def reset_session():
    """Reset the session data after a certain number of interactions."""
    session_data["last_artist"] = None
    session_data["last_song"] = None
    session_data["last_album"] = None
    session_data["interaction_count"] = 0

def extract_artist_name_and_type(input):
    # Remove punctuation from the input
    input = ''.join(ch for ch in input if ch not in string.punctuation)

    # List of common words to remove
    remove_words = ["find", "me", "a", "get", "play", "song", "by", "album", "artist", "track", "music", "latest", "top", 
                    "hit", "popular", "most", "what", "is", "can", "you", "the", "your", "favorite"]

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
    # Fetch the latest albums by the artist from the iTunes API
    album_response = requests.get(ITUNES_ENDPOINT.format(artist_name, 100, "album"))
    album_data = album_response.json()
    
    # Filter albums to only include those by the specified artist
    filtered_albums = [album for album in album_data['results'] if album['artistName'].lower() == artist_name.lower()]
    
    # If no albums are found, return None
    if not filtered_albums:
        return None

    # Sort the albums by release date and get the latest one
    latest_album = sorted(filtered_albums, key=lambda x: x['releaseDate'], reverse=True)[0]
    latest_album_name = latest_album['collectionName']

    # Fetch songs from the latest album
    song_response = requests.get(ITUNES_ENDPOINT.format(latest_album_name, 100, "song"))
    song_data = song_response.json()
    
    # Filter songs to only include those by the specified artist
    filtered_songs = [song for song in song_data['results'] if song['artistName'].lower() == artist_name.lower()]
    
    # If no songs are found, return None
    if not filtered_songs:
        return None

    # Sort the songs by release date and get the latest one
    latest_song = sorted(filtered_songs, key=lambda x: x['releaseDate'], reverse=True)[0]
    return latest_song

def get_song_suggestions(artist_name, song_name):
    # Construct a prompt for the OpenAI model
    prompt = f"I just listened to '{song_name}' by {artist_name}. What other songs would you recommend?"
    
    # Get song recommendations from the OpenAI model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a knowledgeable music enthusiast. Provide song recommendations."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()

def get_most_popular_song_by_artist(artist_name):
    # Fetch songs by the artist from the iTunes API
    song_response = requests.get(ITUNES_ENDPOINT.format(artist_name, 100, "song"))
    song_data = song_response.json()
    
    # Filter songs to only include those by the specified artist
    filtered_songs = [song for song in song_data['results'] if song['artistName'].lower() == artist_name.lower()]
    
    # If no songs are found, return None
    if not filtered_songs:
        return None

    # Use a heuristic to determine the most popular song (highest track count)
    popular_song = max(filtered_songs, key=lambda x: x.get('trackCount', 0))
    return popular_song

def get_latest_album_by_artist(artist_name):
    # Fetch the latest albums by the artist from the iTunes API
    album_response = requests.get(ITUNES_ENDPOINT.format(artist_name, 100, "album"))
    album_data = album_response.json()
    
    # Filter albums to only include those by the specified artist
    filtered_albums = [album for album in album_data['results'] if album['artistName'].lower() == artist_name.lower()]
    
    # If no albums are found, return None
    if not filtered_albums:
        return None

    # Sort the albums by release date and get the latest one
    latest_album = sorted(filtered_albums, key=lambda x: x['releaseDate'], reverse=True)[0]
    return latest_album

def get_song_intent(input):
    # Append the user's input to the messages list
    messages.append({"role": "user", "content": input})
    
    # Get the refined intent from the OpenAI model
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )
    refined_intent = chat.choices[0].message.content
    
    # Append the model's response to the messages list
    messages.append({"role": "assistant", "content": refined_intent})
    return refined_intent

def get_most_popular_album_by_artist(artist_name):
    # Fetch the most popular album by the artist from the iTunes API (assuming the first album returned is the most popular)
    album_response = requests.get(ITUNES_ENDPOINT.format(artist_name, 1, "album"))
    album_data = album_response.json()
    
    # If there are results, return the first album
    if album_data['results']:
        return album_data['results'][0]
    return None

def get_favorite_album_justification(artist_name, album_name):
    # Construct a prompt for the OpenAI model
    prompt = f"Why is '{album_name}' by {artist_name} considered one of the best albums?"
    
    # Get insights about the album from the OpenAI model
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a knowledgeable music enthusiast. Provide insights about albums."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()


def chatbot(input):
     # Increment interaction count
    session_data["interaction_count"] += 1

    # If the user has interacted multiple times, consider resetting the session for a fresh start
    if session_data["interaction_count"] > 5:
        reset_session()

    # Get the refined intent from the model based on the user input
    refined_intent = get_song_intent(input)

    # Extract the artist's name and determine if the user is asking for a song or an album
    artist_name, request_type = extract_artist_name_and_type(input)

    # If the bot couldn't extract the artist's name and it's the first interaction, ask for clarity
    if not artist_name and session_data["interaction_count"] == 1:
        return "I'm sorry, I couldn't identify the artist. Can you please specify the artist's name more clearly or provide more details?"

    # If the bot couldn't determine if the user is asking for a song or an album, ask for clarity
    if not request_type:
        return f"I understood you're asking about {artist_name}, but are you looking for a song or an album?"
    # Get the refined intent from the model based on the user input
    refined_intent = get_song_intent(input)
    
    # Extract the artist's name and determine if the user is asking for a song or an album
    artist_name, request_type = extract_artist_name_and_type(input)
    
    # URL encode the artist's name to make it suitable for the API request
    encoded_artist_name = requests.utils.quote(artist_name)
    
    # Check if the user is asking for a song
    if request_type == "song":
        # If the user wants the latest song
        if "latest" in input:
            song = get_latest_song_by_artist(encoded_artist_name)
            song_type = "latest"
        # Otherwise, assume they want the most popular song
        else:
            song = get_most_popular_song_by_artist(encoded_artist_name)
            song_type = "most popular"
        
        # If a song is found
        if song:
            track_name = song['trackName']
            artist_name = song['artistName']
            # Get song suggestions based on the found song
            song_suggestions = get_song_suggestions(artist_name, track_name)
            reply = f"The {song_type} song by {artist_name} is '{track_name}'. {song_suggestions}"
        # If no song is found, return the refined intent from the model
        else:
            reply = refined_intent
    # If the user is asking for an album
    else:
        # If the user wants the latest album
        if "latest" in input:
            album = get_latest_album_by_artist(encoded_artist_name)
            album_type = "latest"
        # If the user wants the most popular album
        elif "popular" in input or "most popular" in input:
            album = get_most_popular_album_by_artist(encoded_artist_name)
            album_type = "most popular"
        # If the user asks for a favorite or best album
        elif "favorite" in input or "best" in input:
            # Fetch a list of albums and select a random one as the favorite
            album_response = requests.get(ITUNES_ENDPOINT.format(encoded_artist_name, 10, "album"))
            album_data = album_response.json()
            if not album_data['results']:
                return refined_intent
            album = random.choice(album_data['results'])
            # Get a justification for why this album is considered one of the best
            justification = get_favorite_album_justification(artist_name, album['collectionName'])
            return f"One of the best albums by {artist_name} is '{album['collectionName']}'. {justification}"

        # If an album is found
        if album:
            album_name = album['collectionName']
            artist_name = album['artistName']
            reply = f"The {album_type} album by {artist_name} is '{album_name}'."
        # If no album is found, return the refined intent from the model
        else:
            reply = refined_intent
        
        # If the user asks for a general recommendation without specifying an artist or song
        if "recommend" in input and not artist_name:
            if session_data["last_artist"]:
                # Recommend a song from the last known artist
                song = get_most_popular_song_by_artist(session_data["last_artist"])
                if song and song['trackName'] not in session_data["previous_recommendations"]:
                    session_data["previous_recommendations"].append(song['trackName'])
                    return f"I recommend '{song['trackName']}' by {session_data['last_artist']}."
                else:
                    return "I'm sorry, I couldn't find a new recommendation for that artist. Would you like to try another artist?"
            else:
                return "Can you specify an artist or genre for the recommendation?"

        # Store the artist, song, or album for context in the next interaction
        session_data["last_artist"] = artist_name
        if request_type == "song":
            session_data["last_song"] = refined_intent  # Assuming the song name is in the refined intent
        else:
            session_data["last_album"] = refined_intent  # Assuming the album name is in the refined intent


    return reply


# Define the input interface for the Gradio UI
# This creates a textbox with 7 lines and a label "Chat with AI"
inputs = gr.inputs.Textbox(lines=7, label="Chat with AI")

# Define the output interface for the Gradio UI
# This creates a textbox to display the chatbot's reply with a label "Reply"
outputs = gr.outputs.Textbox(label="Reply")

# Create the Gradio Interface
# - `fn=chatbot`: This specifies that the chatbot function defined earlier will be used to process the input and generate the output.
# - `inputs=inputs` and `outputs=outputs`: These specify the input and output interfaces we defined above.
# - `title`, `description`, and `theme` are used to customize the appearance of the Gradio UI.
# - `launch(share=True)`: This launches the Gradio UI and makes it accessible via a public link.
gr.Interface(fn=chatbot, inputs=inputs, outputs=outputs, title="AI Chatbot",
             description="Ask anything you want",
             theme="compact").launch(share=True)

