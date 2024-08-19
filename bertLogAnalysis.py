import re
import ollama 
import numpy as np
import faiss
from transformers import BertTokenizer, BertModel
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


log_file_path = 'C:/Apache24/logs/access.log'
logs = []
# Load the tokenizer and model from the Hugging Face model hub
tokenizer = BertTokenizer.from_pretrained("jackaduma/SecBERT")
model = BertModel.from_pretrained("jackaduma/SecBERT")

# Extracting log entries
with open(log_file_path,'r') as file:
    for line in file:
        ip_address = re.search(r'^\S+', line).group()
        timestamp = re.search(r'\[(.*?)\]', line).group(1)
        
        request = re.search(r'\"(\S+)\s+(\S+)\s+HTTP/\d\.\d\"', line)
        if request:
            http_method = request.group(1)
            page_accessed = request.group(2)
        else:
            http_method = "-"
            page_accessed = "-"

        status_code = re.search(r'\s(\d{3})\s', line)
        if status_code:
            status_code = status_code.group(1)
        else:
            status_code = "-"

        byte_size = re.search(r'\s(\d+|-)$', line).group(1)

        log_entry = f"IP {ip_address} accessed {page_accessed} using {http_method} on {timestamp} and received status code {status_code} with {byte_size} bytes"

        logs.append(log_entry)

# Generate embeddings
vectors = []   
for log in logs:
    # Tokenize the input text
    inputs = tokenizer(log, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.pooler_output.numpy()  # Get a single vector embedding
    vectors.append(embedding)
    
    

# Create FAISS index
embedding_matrix = np.vstack(vectors)
d = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embedding_matrix)
#Gemma2 ile queyi düzenlemek için bir aşama daha yapacağız
client = ollama.Client()
prompt_corrector="If possible Fill in the placeholders using the provided data. Leave the placeholders empty if no data is provided. Use the following format: 'IP {write ip_address if you know else leave as blank} accessed {write page_accessed if you know else leave as blank} using {write http_method if you know else leave as blank} on {write day if you know else leave as blank}/{write Name of the month if you know else leave as blank}/{write Year if you know else leave as blank}:{write time if you know else leave as blank}+ {write timezone if you know else leave as blank} and received status code {write status_code if you know else leave as blank} with {write byte_size if you know else leave as blank} bytes'. Do not add, modify, or provide anything outside of the given format.If you can not leave the question as it is do not change anything if you cant fallow the format"

query = "What happened on August 29 "
prompt = f"Content: {prompt_corrector}\n\nQuestion: {query}\n\nAnswer:"
response = client.chat(model="gemma2", messages=[{"role": "user", "content": prompt}])
messageContent = response["message"]["content"]
print("Generated Response:", messageContent)

# Query processing

query_embedding_inputs = tokenizer(messageContent, return_tensors="pt", padding=True, truncation=True)
# Generate embeddings
with torch.no_grad():
    outputs = model(**query_embedding_inputs)
    embeddings = outputs.pooler_output  # or outputs.pooler_output for sentence-level embeddings
query_embedding = embeddings.numpy()

# Search
k = len(logs)
distances, indices = index.search(query_embedding, k)
retrieved_chunks = [logs[i] for i in indices[0]]

# Output results
print("Relevant log entries:")
for num, (chnk, dist) in enumerate(zip(retrieved_chunks, distances[0])):
    print(f"{num+1}) {chnk}\n   Distance: {dist}\n")


context = "\n".join([str(log) for log in retrieved_chunks])
prompt = f"İçerik: {context}\n\nSoru: {query}\n\nCevap:"
response = client.chat(model="gemma2", messages=[{"role": "user", "content": prompt}])

print("Generated Response:", response)

#Bazı verileri özel olarak bulamıyor bunlar için bir distanceları her veride kontrol et 
#Lineları daha belirgin hala getirmek için bir query translation ile uzatmayı düşünüyorum.