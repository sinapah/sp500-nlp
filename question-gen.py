#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:00:02 2025

@author: sinap
"""
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer, PreTrainedTokenizerFast
import torch
import pandas as pd

# Load the knowledge base
with open('knowledge_base.json', 'r') as file:
    kb = json.load(file)

# Load company information
industry_df = pd.read_csv("sp500-companies.csv", encoding='ISO-8859-1')
# Create a lookup dictionary for ticker -> company name mapping
ticker_to_company = {row["Ticker"]: row["Name"] for _, row in industry_df.iterrows()}

# Convert to text pairs
text_pairs = []
for company, details in kb.items():
    # Get company name from ticker, use ticker as fallback if name not found
    company_name = ticker_to_company.get(company, company)
    
    for year, financials in details.items():
        # Check if financials is a dictionary
        if isinstance(financials, dict):
            # Process each financial metric separately
            for metric, value in financials.items():
                fact = f"{company_name}'s {metric} in {year} was {value}"
                formatted_input = f"answer: {value} content: {fact}"
                text_pairs.append((formatted_input, None))
        else:
            # Handle single value entries
            fact = f"{company_name}'s data for {year} is {financials}"
            formatted_input = f"answer: {financials} content: {fact}"
            text_pairs.append((formatted_input, None))

# Print the formatted pairs
#for p, _ in text_pairs:
   # print(p)


# Load T5 model and tokenizer
model_name = "Sehong/t5-large-QuestionGeneration"  # Changed to specialized QG model
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Generate questions
def generate_questions(text_pairs, model, tokenizer, batch_size=8):
    questions = []
    for i in range(0, len(text_pairs), batch_size):
        batch = text_pairs[i:i + batch_size]
        
        try:
            for formatted_input, _ in batch:
                # Tokenize and generate
                input_ids = tokenizer.encode(formatted_input)
                input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
                
                question_ids = model.generate(
                    torch.tensor([input_ids]),
                    num_beams=4,
                    max_length=64,
                    eos_token_id=1
                )
                
                # Decode and clean up the output
                question = tokenizer.decode(question_ids.squeeze().tolist(), skip_special_tokens=True)
                question = question.replace(' # # ', '').replace('  ', ' ').replace(' ##', '')
                questions.append(question)
                print(question)
                with open("questions.txt", "a") as file:  # Open in append mode
                    file.write(question + "\n") 
        except Exception as e:
            print(f"Error processing batch: {e}")
            questions.extend(["Error generating question"] * len(batch))
    
    return questions

# Generate and print questions
# Modified input format to clearly specify the answer
#text_pairs = [("In 2010, Neymar won the world cup. The answer is 2010", None)]

print(len(text_pairs))

generated_questions = generate_questions(text_pairs, model, tokenizer)

for fact, question in zip(text_pairs[:1000], generated_questions):
    print(f"\nFact: {fact[0]}")
    print(f"Generated Question: {question}")
