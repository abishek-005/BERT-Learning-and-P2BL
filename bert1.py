from transformers import BertTokenizer
# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Tokenize a sample text
text = "Hello, my dog is cute"
inputs = tokenizer(text, return_tensors="pt")
# Print the tokenized input
print(inputs)