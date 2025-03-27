#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import spacy
from transformers import pipeline
from fuzzywuzzy import process
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load resources once when server starts
try:
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Load knowledge base
    with open('knowledge_base.json', 'r') as file:
        KNOWLEDGE_BASE = json.load(file)
    
    # Load company mappings
    with open('sp500-companies.csv', 'r', encoding='ISO-8859-1') as file:
        import pandas as pd
        industry_df = pd.read_csv(file)
        company_name_to_ticker = {row["Name"].lower(): row["Ticker"] for _, row in industry_df.iterrows()}
    
    # Initialize QA pipeline
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    print("✅ All resources loaded successfully")
except Exception as e:
    print(f"❌ Error loading resources: {e}")
    raise e

class QuestionRequest(BaseModel):
    question: str
    mode: str

def fuzzy_company_lookup(company_name, company_name_to_ticker):
    match, score = process.extractOne(company_name, company_name_to_ticker.keys())
    return company_name_to_ticker[match] if score > 80 else None

def extract_entities(question):
    """
    Uses NLP to extract company names, tickers, years, and determine if it's an aggregate query.
    """
    doc = nlp(question)
    ###print("////doc", doc.ents)
    ticker = None
    year = None
    aggregate_key = None  # Track if the question is about an aggregate

    for ent in doc.ents:
            
        if (ent.label_ == "DATE") or re.match(r"\b(19\d{2}|20\d{2})\b", ent.text):
            ####print(ent.label_ == "DATE")
            ###print(ent.text)
            year = int(ent.text) if ent.text.lower() not in company_name_to_ticker.keys() else None # Some names like AbbVie are recognized as Spacy as years
        
        possible_ticker = company_name_to_ticker.get(ent.text.lower()) or fuzzy_company_lookup(ent.text.lower(), company_name_to_ticker)
            
        if possible_ticker:
            ticker = possible_ticker
    # Detect aggregate-type questions
    aggregate_mappings = {
        "highest gross profit": "highest_gross_profit",
        "highest revenue": "highest_revenue",
        "highest net income": "highest_net_income",
        "highest operating expense": "highest_operating_expense",
        "highest increase in gross profit": "highest_increase_in_gross_profit",
        "largest industry": "largest_industry",
        "second largest industry": "second_largest_industry",
        "third largest industry": "third_largest_industry",
        "smallest industry": "smallest_industry",
        "second smallest industry": "second_smallest_industry",
        "third smallest industry": "third_smallest_industry",
        "most investment": "industry_with_the_most_invesmtent",
        "second most investment": "industry_with_the_second_most_invesmtent",
        "third most investment": "industry_with_the_third_invesmtent",
        "least investment": "industry_with_the_least_invesmtent",
        "second least investment": "industry_with_the_second_least_invesmtent",
        "third least investment": "industry_with_the_third_least_invesmtent"
    }
    
    ordinal_terms = {
        "second largest": "second_largest_industry",
        "third largest": "third_largest_industry",
        "second smallest": "second_smallest_industry",
        "third smallest": "third_smallest_industry",
        "second most": "industry_with_the_second_most_invesmtent",
        "third most": "industry_with_the_third_invesmtent",
        "second least": "industry_with_the_second_least_invesmtent",
        "third least": "industry_with_the_third_least_invesmtent"
    }
    
    # Use fuzzy matching to determine the best match
    # Try to match ordinal terms like "third largest industry"
    for term, key in ordinal_terms.items():
        if term in question.lower():
            aggregate_key = key
            break
    
    # If no ordinal term is found, fall back to fuzzy matching
    if not aggregate_key:
        match = process.extractOne(question.lower(), aggregate_mappings.keys(), score_cutoff=90)
        if match:
            best_match, _ = match
            aggregate_key = aggregate_mappings[best_match]
    ###print(aggregate_key)
    return ticker, year, aggregate_key

def is_executive_question(question):
    doc = nlp(question)
    has_person = any(ent.label_ == "PERSON" for ent in doc.ents)
    
    for token in doc:
        if token.dep_ == "compound" and token.head.pos_ == "PROPN":
            return True
        if token.pos_ == "VERB" and token.lemma_ in {"serve", "work", "lead", "join", "head"}:
            return True
        if token.text.lower() in {"role", "position", "title", "job", "ceo", "executive", "executives"}:
            return True
    
    if has_person:
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return True
    
    return False

def is_info_question(question):
    keywords = ["headquarters", "located", "office", "based", "industry", "sector", "domain", "field", "date", "founded", "foundation"]
    return any(word in question.lower() for word in keywords)

# Load T5 model and tokenizer
t5_model_name = "t5-small"  # You can use a larger model if you want better performance
tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

def generate_full_sentence(answer):
    """
    Takes an answer and generates a full sentence using the T5 model.
    """
    input_text = f"Answer:{answer}"

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate output using the T5 model
    output_ids = t5_model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)

    # Decode the generated output
    generated_sentence = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_sentence

def answer_question(question, mode="BERT"):
    """
    Main function to process questions and return answers
    """
    ###print("⭐ Processing question:", question)
    
    ticker, year, aggregate_key = extract_entities(question)
    ###print(f"📊 Extracted - Ticker: {ticker}, Year: {year}, Aggregate: {aggregate_key}")
    
    # Handle aggregate queries
    if aggregate_key:
        if not year:
            return "Please specify a year."
        result = KNOWLEDGE_BASE.get("aggregates", {}).get(str(year), {}).get(aggregate_key, "No data available.")
        if mode == "T5":
            return generate_full_sentence(f"The {'company' if 'company' in question else 'industry'} with the {aggregate_key.replace('_', ' ')} in {year} was {result}." if result != "No data available." else result)
        else:
            return f"The {'company' if 'company' in question else 'industry'} with the {aggregate_key.replace('_', ' ')} in {year} was {result}." if result != "No data available." else result
            
    if not ticker:
        ###print("❌ No ticker found")
        return "I couldn't determine the company you're asking about."
    
    # Handle industry and headquarters queries
    if is_info_question(question):
        info = KNOWLEDGE_BASE.get(ticker, {}).get("info", "No company info available.")
        if info != "No company info available.":      
            if mode == "T5":
                return generate_full_sentence(info)
            else:
                response = qa_pipeline(question=question, context=info)
                return response['answer'] if response['score'] > 0.5 else info

    if not year:
        year = max(str(y) for y in KNOWLEDGE_BASE[ticker].keys() if str(y).isdigit())
    
    # Handle executive and financial queries
    if is_executive_question(question):
        ###print("Executive Question")
        context = KNOWLEDGE_BASE.get(ticker, {}).get(str(year), {}).get("executives", "No executive data available.")
    else:
        ###print("Financial Question")
        context = KNOWLEDGE_BASE.get(ticker, {}).get(str(year), {}).get("financials", "No financial data available.")

    if "No data available" in context:
        return context
    
    if mode == "T5":
        return generate_full_sentence(context)
    else:
        response = qa_pipeline(question=question, context=context)
        return response['answer'] 

@app.post("/api/qa/")
async def process_question(request: QuestionRequest):
    """API endpoint for answering questions"""
    try:
        response = answer_question(request.question, request.mode)
        return {"answer": response}
    except Exception as e:
        print("❌ Error:", str(e))
        return {"error": str(e)}

print(answer_question("What company had the highest operating expense in 2023?"))
print(answer_question("What was the revenue of apple in 2020?"))
print(answer_question("Who was the CEO of Apple in 2020?"))
print(answer_question("Who had the highest revenue in 2020?"))
print(answer_question("How much net income did Microsoft have in 2019?"))
print(answer_question("Who was the CEO of Apple in 2021?"))
print(answer_question("What was the revenue of Visa in 2020"))
print(answer_question("What was the gross profit of Equinix in 2021?"))
print(answer_question("What company had the highest gross profit in 2021?"))
print(answer_question("What company had the highest operating expense in 2023?"))
print(answer_question("What company had the highest increase in gross profit in 2022?"))
print(answer_question("Who had the highest revenue in 2020?"))
print(answer_question("What was the increase in net income of Apple in 2023?"))
print(answer_question("What was the gross profit of Adobe in 2021?"))
print(answer_question("What were Agilent Technologies financials in 2023?"))
print(answer_question("What was Michael Tang's place at Agilent Technologies in 2019?"))
print(answer_question("What was Samraat Raha's position at Agilent Technologies in 2020?"))
print(answer_question("What position did Jean Nye hold at Agilent Technologies in 2003?"))
print(answer_question("Where are the headquarters of Apple located?"))
print(answer_question("Where are the headquarters of Adobe?"))
print(answer_question("On what date was Apple founded?"))
print(answer_question("What industry is Agilent Technologies in?"))
print(answer_question("What industry is Federal Realty in?"))
print(answer_question("What industry is American Electric Power in?"))
print(answer_question("Where are the headquarters of American Electric Power ?"))

'''with open('questions.txt', 'r') as file:
    for line in file:
        print(line)
        print(answer_question(line))
        print('\n')'''
'''print(answer_question("What was the largest industry in 2016?"))
print(answer_question("What was the second largest industry in 2016?"))
print(answer_question("What was the third largest industry in 2015?"))
print(answer_question("What was the smallest industry in 2006?"))
print(answer_question("What was the third largest industry in 2004?"))
print(answer_question("What was the third smallest industry in 2018?"))
print(answer_question("Which industry had the most investment in 2018?"))
print(answer_question("Which industry had the second most investment in 2007?"))
print(answer_question("Which industry had the least investment in 2008?"))
print(answer_question("Which industry had the third most investment in 2003?"))
print(answer_question("Which industry had the second least investment in 2004?"))'''

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
            