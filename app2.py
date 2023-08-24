import openai
import gradio as gr

# Set up OpenAI credentials
openai.organization = "org-MYzdbZoWTu1PpVARD32T829L"
with open('D:\Python_Workspace\Building a Chatbot.txt', 'r') as file:
    openai.api_key = file.readline().strip()

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]

def chatbot(input):
    if "paper" in input or "research" in input:
        search_term = input.replace("paper", "").replace("research", "").strip()
        
        # Use the OpenAI API to interact with the scholarai GPT-4 plugin
        response = openai.Completion.create(
            model="scholarai-gpt-4.0-turbo",
            prompt=f"searchAbstracts: {{\"keywords\": \"{search_term}\", \"query\": \"{search_term}\", \"sort\": \"publication_date\", \"end_year\": \"2023\"}}",
            max_tokens=500
        )
        
        # Parse the response
        scholarai_response = response.choices[0].message['content']
        
        if scholarai_response and 'papers' in scholarai_response and len(scholarai_response['papers']) > 0:
            title = scholarai_response['papers'][0]['title']
            reply = f"I found a paper for you: {title}. [Link to PDF]({scholarai_response['papers'][0]['pdf_url']})"
        else:
            reply = "Sorry, I couldn't find any papers matching your query."
    else:
        if input:
            messages.append({"role": "user", "content": input})
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
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
