import requests
import PyPDF2
import re
from PyPDF2 import PdfReader
import os
import requests
import chromadb
import google.generativeai as genai
import dotenv
from dotenv import load_dotenv
import chromadb.utils.embedding_functions as embedding_functions

## Configure Gemini and API KEY

key_path = r"C:\Users\albfr\Desktop\BeQu\API Key\gemini_api_key1.env"
load_dotenv(key_path)

# Accede a la API key
api_key_1 = os.getenv("gemini_api_key1")

### Function: Exttraction text from the .pdf

file_path=r"C:\Users\albfr\Desktop\BeQu\data\medicalstandard.pdf"

def load_pdf(file_path):
  pdf_reader = PdfReader(file_path)
  text = ""
  
  for page in pdf_reader.pages:
    text += page.extract_text()
  return text


### Checking the .pdf


pdf_text = load_pdf(file_path)

# pdf_text
pdf_text[:1500]

len(pdf_text)
# Function to Split Text into smallers Chuncks

## Setup Max Chunck Lenght and the Chunck Overlap


def pdf_text_splitter(text, max_length=1000, chunk_overlap=0):
  text = re.sub(r'\.\.\.', '', text) # Remove ellipsis (...)
  chunks = []
  start = 0 
  text_length = len(text) # calculate the whole text lenght
  while start < text_length: # keep moving till the end of the full text
    end = start + max_length
    if end < text_length : # When we are not at the end of text
      end = text.rfind(' ', start, end) + 1 # For not cutting words at the middle. Ending chunck between spaces

      if end <= start: # When there is no space, split at the max length
        end = start + max_length

    chunk = text[start:end].strip() # Take text from Stat till End, and remove spaces in between

    if chunk:
      chunks.append(chunk) # When finding a Chucnk, adding to chuncks = []

    start = end - chunk_overlap  # moving start position forward minus any overlaps

    if start >= text_length: # When reaching out the end of the text
      break
  return chunks

chunks = pdf_text_splitter(pdf_text, max_length=2000, chunk_overlap=200)

# Checking Chunks


# len(chunks)

# chunks[0]

## Embedding Chucnks in ChromaDB

google_ef  = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key= os.getenv("gemini_api_key1"))
client = chromadb.PersistentClient(path="embeddings/gemini")


collection = client.get_or_create_collection(name="pdf_rag", embedding_function=google_ef)



for i, d in enumerate(chunks):
  collection.add(documents=[d], ids=[str(i)])


  # Function for Context and Recover Context

def recover_context(context):
  recovered_context = ""
  for item in context:
    recovered_context += item + "\n\n"  ## To aggregate the results in one big answer
  return recovered_context

def text_relevant_in_context(query, db, n_results=3):
  results = db.query(query_texts=[query], n_results=n_results)
  recovered_context = recover_context(results['documents'][0])
  return recovered_context

context = text_relevant_in_context("How to develop a design validation for medical devices", collection )
print(context)

# Prompt Designing 

def give_a_prompt_to_the_agent(query, context):
  prompt = f"""
  You are a consultor agent that answers questions using the text from the context below.
  The question and the context is going to be shared with you, so you can answer the question based on the context.
  If the context does not have enough information to answer the question properly, inform the user about the abscene of relevant  context as an answer

  Question : {query}
  \n
  Context : {context}
  \n
  Answer :
  """
  return prompt

# Function to define the answer based on the prompt

def generate_answer(prompt):
  model = genai.GenerativeModel('gemini-1.5-flash-latest')
  result = model.generate_content(prompt)
  return result

# Prompt


prompt = give_a_prompt_to_the_agent(query="How to develop a design validation for medical devices", context=context)

print(prompt)

answer = generate_answer(prompt)
print(answer.text)