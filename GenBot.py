import gradio as gr
import torch
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

def clean_output(text):
    # Remove any repeated punctuation
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove any spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    
    return text.strip()

def generate_response(prompt, temperature, top_p):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, 
                            no_repeat_ngram_size=4, early_stopping=True, 
                            temperature=temperature,  # Use the provided value
                            top_p=top_p)              # Use the provided value
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = clean_output(response)
    return response

# Define Gradio interface
def chat_interface(prompt, temperature, top_p):
    return generate_response(prompt, temperature, top_p)


#########################################################################################################
#########################################################################################################
#########################################################################################################
# Part 2: Fine tuning the model

#Part of fine tuning is cleaning data. For our news summarization bot, we will have to have both news text, and the human generated summaries
#This will teach the model how to summarize

#We do not have time to generate summarize, so we will use a pre made data set found here: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

#We can demo what the cleaning process would be like using the below data set and demo
#In a seperate file, clean the news articles dataset from kaggle: https://www.kaggle.com/datasets/snapcrack/all-the-news?resource=download&select=articles1.csv
#We are only using article1.csv from the kaggle data set
#Cleaning process can be found in DataCleaning.ipynb 

########################################BASE LINE################################################

#Step 1: We need to observe how the model does providing summaries of a text without fine tuning
#Here we are using the first entry in the training data to see how the model does.
import pandas as pd
def baseline_summary():
    #read the CSV
    valid_data = pd.read_csv('C:\Work\Python\Transformer\ChatbotUpdate\sample_test.csv')

    #Get the content of the first article in the data frame
    content = valid_data['article'][0]

    # Tokenize input and generate baseline summary
    input_ids = tokenizer.encode(content, return_tensors="pt")
    baseline_output = model.generate(input_ids, max_length=500, num_return_sequences=1)

    # Decode and print the baseline summary
    baseline_summary = tokenizer.decode(baseline_output[0], skip_special_tokens=True)
    return baseline_summary
'''
    #If the content is too long, we must break it into 3 parts, then summarize each part.

    # Step 1: Split the content into three parts
    #content = articles['content'][0]  # Use the content of the first article
    total_length = len(content)
    part_length = total_length // 3

    part1 = content[:part_length]
    part2 = content[part_length:2*part_length]
    part3 = content[2*part_length:]

    # Step 2: Generate summaries for each part
    
    pre_prompt = "Please summarize the following text:"

    input_ids1 = tokenizer.encode(pre_prompt + " " + part1, return_tensors="pt")
    input_ids2 = tokenizer.encode(pre_prompt + " " + part2, return_tensors="pt")
    input_ids3 = tokenizer.encode(pre_prompt + " " + part3, return_tensors="pt")

    summary1 = model.generate(input_ids1, max_length=100, num_return_sequences=1)
    summary2 = model.generate(input_ids2, max_length=100, num_return_sequences=1)
    summary3 = model.generate(input_ids3, max_length=100, num_return_sequences=1)

    # Step 3: Decode and combine summaries
    summary_text1 = tokenizer.decode(summary1[0], skip_special_tokens=True)
    summary_text2 = tokenizer.decode(summary2[0], skip_special_tokens=True)
    summary_text3 = tokenizer.decode(summary3[0], skip_special_tokens=True)

    combined_summary = summary_text1 + " " + summary_text2 + " " + summary_text3
    print("Combined Summary:", combined_summary)

'''     
    #At this point the model does not perform well. so lets fine tune now

#print(baseline_summary())


########################################FINE TUNE################################################
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW

class SummarizationDataset(Dataset):
    def __init__(self, articles, summaries):
        self.articles = articles
        self.summaries = summaries

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        return self.articles[idx], self.summaries[idx]  # Return input_ids and labels directly

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_rouge(generated_summaries, reference_summaries):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []

    for gen_summary, ref_summary in zip(generated_summaries, reference_summaries):
        scores = scorer.score(ref_summary, gen_summary)
        rouge_scores.append(scores)

    avg_rouge_scores = {
        'rouge1': sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores),
        'rouge2': sum(score['rouge2'].fmeasure for score in rouge_scores) / len(rouge_scores),
        'rougeL': sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)
    }

    avg_rouge_score = sum(avg_rouge_scores.values()) / len(avg_rouge_scores)  # Calculate the average of all Rouge scores
    return avg_rouge_score  # Return the average Rouge score as a single value


def calculate_bleu(generated_summaries, reference_summaries):
    smoothie = SmoothingFunction().method4  # Choose a smoothing method
    bleu_scores = []

    for gen_summary, ref_summary in zip(generated_summaries, reference_summaries):
        gen_tokens = gen_summary.split()
        ref_tokens = ref_summary.split()
        bleu_score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)
        bleu_scores.append(bleu_score)

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu_score



def fine_tuned_summary():
    print('entered fine tuning function')
    special_tokens = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}
    tokenizer.add_special_tokens(special_tokens)

    # Define hyperparameters
    learning_rate = 1e-4
    num_epochs = 3
    batch_size = 1  # Adjust as needed len(train_dataset)


    test_data = pd.read_csv('C:\Work\Python\Transformer\ChatbotUpdate\sample_test.csv')
    train_data = pd.read_csv('C:\Work\Python\Transformer\ChatbotUpdate\sample_training.csv')

    def tokenization_function(data, max_length):
        input_sequences = [tokenizer.encode(text, max_length=max_length, truncation=True) for text in data]
        label_sequences = [sequence[1:] for sequence in input_sequences]  # Shift labels by one position

        return input_sequences, label_sequences

    max_article_length = 1000
    max_summary_length = 300

    tokenized_articles, tokenized_article_labels = tokenization_function(train_data['article'], max_article_length)
    tokenized_summaries, tokenized_summary_labels = tokenization_function(train_data['highlights'], max_summary_length)

    # Create dataset instances
    train_dataset = SummarizationDataset(tokenized_articles, tokenized_summary_labels)  # Use tokenized_summary_labels as labels

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=None)

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fine-tuning loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch[0][0].to(device)  # Access input_ids
            labels = batch[0][1].to(device)     # Access labels

            # Create attention mask
            attention_mask = (input_ids != tokenizer.pad_token_id).float()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    # Save the fine-tuned model
    model.save_pretrained('./fine_tuned_model')

    print('training complete')

    #######TESTING######
    # Tokenize articles and summaries
    test_tokenized_articles = [tokenizer.encode(article, max_length=max_article_length, truncation=True) for article in test_data['article']]
    test_tokenized_summaries = [tokenizer.encode(summary, max_length=max_summary_length, truncation=True) for summary in test_data['highlights']]

    # Create dataset instances
    test_dataset = SummarizationDataset(test_tokenized_articles, test_tokenized_summaries)

    # Create data loaders
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=None)


    #Test results
    model.eval()  # Set the model to evaluation mode

    total_rouge_score = 0
    total_bleu_score = 0
    num_examples = len(test_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(num_examples):
        print(f'This is test #{i}')
        article = test_data['article'][i]
        summary = test_data['highlights'][i]
        
        # Tokenize the article
        input_ids = tokenizer.encode(article, max_length=max_article_length, truncation=True, return_tensors='pt').to(device)

        # Generate summaries using the model
        generated_ids = model.generate(input_ids)

        # Convert generated IDs to text
        generated_summaries = [tokenizer.decode(gen_ids, skip_special_tokens=True) for gen_ids in generated_ids]

        # Calculate ROUGE and BLEU scores
        rouge_score = calculate_rouge(generated_summaries, [summary])  # Pass the summary as a list
        bleu_score = calculate_bleu(generated_summaries, [summary])    # Pass the summary as a list

        total_rouge_score += rouge_score
        total_bleu_score += bleu_score

    # Calculate average metrics
    avg_rouge_score = total_rouge_score / num_examples
    avg_bleu_score = total_bleu_score / num_examples

    print("Average ROUGE Score:", avg_rouge_score)
    print("Average BLEU Score:", avg_bleu_score)


fine_tuned_summary()

# End of Part 2 of demo
#########################################################################################################
#########################################################################################################
#########################################################################################################


# Gradio components
inputs = [
    gr.inputs.Textbox(lines=7, label="Your Prompt"),
    gr.inputs.Slider(minimum=0.1, maximum=1.0, step=0.1, default=0.5, label="Temperature"),
    gr.inputs.Slider(minimum=0.1, maximum=1.0, step=0.1, default=0.9, label="Top P")
]
outputs = gr.outputs.Textbox(label="GPT-2 Response")

# Launch the Gradio interface
gr.Interface(fn=chat_interface, inputs=inputs, outputs=outputs, title="GPT-2 Chatbot", description="Generate responses using GPT-2").launch()
