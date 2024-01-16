from transformers import AutoTokenizer

# Replace 'InstaDeepAI/nucleotide-transformer-500m-human-ref' with the model name or path you want to download
model_name = 'InstaDeepAI/nucleotide-transformer-500m-human-ref'

# Replace 'local_directory' with the directory where you want to save the tokenizer
local_directory = './localtokenizer/'

# Download the tokenizer and save it locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_directory)
