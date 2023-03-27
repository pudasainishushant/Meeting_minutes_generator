# from transformers import AutoTokenizer
# from transformers import AutoModelForSeq2SeqLM
# import torch
# checkpoint = "knkarthick/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# text = '''
# Hi, I'm David and I'm supposed to be an industrial designer. 
# '''
# inputs = tokenizer(text, return_tensors="pt").input_ids
# print(tokenizer.tokenize(text))

# tokenized_input = tokenizer.encode(text)
# # breakpoint()
# model = AutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
# # output = model(torch.tensor([tokenized_input]))
# outputs = model.generate(tokenized_input, max_new_tokens=100, do_sample=False)

# print(outputs)

from transformers import T5Tokenizer, T5ForConditionalGeneration

with open('meeting.txt', 'r') as file:
    text = file.read()

tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(summary)
