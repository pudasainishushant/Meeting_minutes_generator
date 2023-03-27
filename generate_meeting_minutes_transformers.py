import warnings
warnings.filterwarnings("ignore")
import os
from dotenv import load_dotenv
import openai
import openai_async
import asyncio
import nest_asyncio
nest_asyncio.apply()
import time
start = time.time()

import torch
from transformers import AutoTokenizer

load_dotenv()

def count_tokens(filename):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    with open(filename, 'r') as f:
        text = f.read()

    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    num_tokens = input_ids.shape[1]
    return num_tokens

filename = "sample_meeting.txt"
token_count = count_tokens(filename)
print(f"Number of tokens: {token_count}")

def break_up_file_to_chunks(filename, chunk_size=2000, overlap=100):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    with open(filename, 'r') as f:
        text = f.read()

    tokens = tokenizer.encode(text)
    num_tokens = len(tokens)
    
    chunks = []
    for i in range(0, num_tokens, chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(chunk)
    
    return chunks

filename = "sample_meeting.txt"

chunks = break_up_file_to_chunks(filename)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {len(chunk)} tokens")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
# print(tokenizer.decode(chunks[0][-100:]))
# print(tokenizer.decode(chunks[1][:100]))
# if tokenizer.decode(chunks[0][-100:]) == tokenizer.decode(chunks[1][:100]):
#     print('Overlap is Good')
# else:
#     print('Overlap is Not Good')


openai.api_key = os.getenv("api_key")

async def summarize_meeting(prompt, timeout, max_tokens):
    
    #timeout = 30
    temperature = 0.5
    #max_tokens = 1000
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0
    
    # Call the OpenAI GPT-3 API
    response = await openai_async.complete(
        openai.api_key,
        timeout=timeout,
        payload={
            "model": "text-davinci-003",
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        },
    )

    # Return the generated text
    return response

filename = "sample_meeting.txt"

prompt_response = []
prompt_tokens = []

chunks = break_up_file_to_chunks(filename)

for i, chunk in enumerate(chunks):
    prompt_request = "Summarize this meeting transcript: " + tokenizer.decode(chunks[i])
    
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(summarize_meeting(prompt = prompt_request, timeout=30, max_tokens = 1000))
    prompt_response.append(response.json()["choices"][0]["text"].strip())
    prompt_tokens.append(response.json()["usage"]["total_tokens"])


prompt_request = "Consoloidate these meeting summaries: " + str(prompt_response)

loop = asyncio.get_event_loop()
response = loop.run_until_complete(summarize_meeting(prompt = prompt_request, timeout=45, max_tokens = 1000))
print(response.json()["choices"][0]["text"].strip())
print(response.json()["usage"]["total_tokens"])


filename = "sample_meeting.txt"

action_response = []
action_tokens = []

chunks = break_up_file_to_chunks(filename)

for i, chunk in enumerate(chunks):
    prompt_request = "Provide a list of action items with a due date from the provided meeting transcript text: " + tokenizer.decode(chunks[i])
    
    loop = asyncio.get_event_loop()
    response = loop.run_until_complete(summarize_meeting(prompt = prompt_request, timeout=30, max_tokens = 1000))
    
    action_response.append(response.json()["choices"][0]["text"].strip())
    action_tokens.append(response.json()["usage"]["total_tokens"])

print(action_response)

prompt_request = "Consoloidate these meeting action items, but exclude action items with Due Date of Immediately: " + str(action_response)

loop = asyncio.get_event_loop()
response = loop.run_until_complete(summarize_meeting(prompt = prompt_request, timeout=45, max_tokens = 1000))

print(response.json()["choices"][0]["text"].strip())
end_time = time.time()
print("Total time",end_time -  start)