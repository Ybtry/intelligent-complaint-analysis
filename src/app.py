import gradio as gr
import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
vector_store_dir = os.path.abspath(os.path.join(script_dir, '..', 'vector_store'))

try:
    index = faiss.read_index(os.path.join(vector_store_dir, 'faiss_index.bin'))
    with open(os.path.join(vector_store_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    print("Vector store (FAISS index and metadata) loaded successfully.")
except FileNotFoundError:
    print(f"Error: Vector store files not found in {vector_store_dir}.")
    print("Please ensure Task 2 completed successfully and check the directory.")
    exit()

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Embedding model loaded.")

model_name = "distilgpt2"
device = 0 if torch.cuda.is_available() else -1

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    if device != -1:
        model.to(f'cuda:{device}')
    text_generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=200,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2
    )
    print(f"LLM '{model_name}' and text generation pipeline loaded successfully on {'GPU' if device != -1 else 'CPU'}.")
except Exception as e:
    print(f"Error loading or initializing LLM pipeline: {e}")
    print("Please check your internet connection, model name, and available resources.")
    exit()

def retrieve_chunks(query_text, top_k=5):
    query_embedding = embedding_model.encode([query_text]).astype('float32')
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, top_k)

    retrieved_chunks = []
    retrieved_sources_info = []

    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            chunk_info = metadata[idx]
            retrieved_chunks.append(chunk_info['chunk_text'])
            retrieved_sources_info.append(
                f"Product: {chunk_info.get('product', 'N/A')}, Original ID: {chunk_info.get('original_id', 'N/A')}"
            )
        else:
            print(f"Warning: Index {idx} out of bounds for metadata. Skipping.")
    return retrieved_chunks, retrieved_sources_info

def generate_answer(question, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt_template = (
        "You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. "
        "Use the following retrieved complaint excerpts to formulate your answer. "
        "If the context doesn't contain the answer, state that you don't have enough information.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    generated_text = text_generator(prompt_template, num_return_sequences=1)
    answer = generated_text[0]['generated_text'][len(prompt_template):].strip()
    return answer

def rag_chatbot(user_question):
    if not user_question:
        return "Please enter a question.", ""

    retrieved_chunks, retrieved_sources_info = retrieve_chunks(user_question, top_k=5)
    generated_answer = generate_answer(user_question, retrieved_chunks)

    sources_display = "### Retrieved Sources:\n"
    if retrieved_chunks:
        for i, (source_info, chunk_text) in enumerate(zip(retrieved_sources_info, retrieved_chunks)):
            sources_display += f"**Source {i+1} ({source_info}):**\n"
            sources_display += f"> {chunk_text[:300]}...\n\n"
    else:
        sources_display += "No relevant sources found."

    return generated_answer, sources_display

iface = gr.Interface(
    fn=rag_chatbot,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about customer complaints..."),
    outputs=[
        gr.Textbox(label="AI-Generated Answer", interactive=False),
        gr.Markdown(label="Sources Used")
    ],
    title="CrediTrust Complaint Analysis Chatbot",
    description=(
        "Ask questions about customer complaints and get AI-generated answers "
        "with supporting evidence from retrieved complaint narratives."
    ),
    flagging_mode="never",
    examples=[
        ["Why are people unhappy with credit card services?"],
        ["What are the common issues with personal loans?"],
        ["Describe issues related to money transfers."],
        ["Any complaints about fees on savings accounts?"]
    ]
)

if __name__ == "__main__":
    iface.launch()
