#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced QA system using NLP-based entity extraction
"""

import pandas as pd
import re
import spacy
from transformers import pipeline
from fuzzywuzzy import process
import json

# Load spaCy's pre-trained English model for NER
nlp = spacy.load("en_core_web_sm")

# Load datasets
industry_df = pd.read_csv("sp500-companies.csv", encoding='ISO-8859-1')
finance_df = pd.read_csv("Financials_SP500.csv", encoding='ISO-8859-1')
executive_df = pd.read_csv("sp500_firm_execu.csv", encoding='ISO-8859-1')
aggregated_data = pd.read_csv("aggregated_stats.csv", encoding='ISO-8859-1')

executive_df = executive_df.rename(columns={'tic': 'Ticker'})

# Standardize tickers
for df in [finance_df, industry_df, executive_df]:
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

# Merge finance_df with industry_df to get company names
merged_df = finance_df.merge(industry_df[['Ticker']], on="Ticker", how="left")
merged_df["Name"] = merged_df["Name"].fillna("")

# Create a lookup dictionary for quick company name -> ticker mapping
company_name_to_ticker = {row["Name"].lower(): row["Ticker"] for _, row in industry_df.iterrows()}
ticker_to_company_name = {row["Ticker"].lower(): row["Name"] for _, row in industry_df.iterrows()}
# Create a structured knowledge base (financials + executives)

with open('knowledge_base.json', 'r') as file:
    knowledge_base = json.load(file)

# Load a pre-trained QA model
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


def fuzzy_company_lookup(company_name, company_name_to_ticker):
    """
    Finds the best match for a given company name using fuzzy matching.
    """
    match, score = process.extractOne(company_name, company_name_to_ticker.keys())
    return company_name_to_ticker[match] if score > 80 else None  # Use a threshold to avoid incorrect matches

def extract_entities(question):
    """
    Uses NLP to extract company names, tickers, years, and determine if it's an aggregate query.
    """
    doc = nlp(question)
    print("////doc", doc.ents)
    ticker = None
    year = None
    aggregate_key = None  # Track if the question is about an aggregate

    for ent in doc.ents:
            
        if (ent.label_ == "DATE") or re.match(r"\b(19\d{2}|20\d{2})\b", ent.text):
            print(ent.label_ == "DATE")
            print(ent.text)
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

    # Use fuzzy matching to determine the best match
    match = process.extractOne(question.lower(), aggregate_mappings.keys(), score_cutoff=90)

    if match:  # Ensure match is not None before unpacking
        best_match, _ = match  # Extract the best matching key
        aggregate_key = aggregate_mappings[best_match]

    return ticker, year, aggregate_key

def is_info_question(question):
    """
    Determines if the question is asking for company industry or headquarters.
    """
    keywords = ["headquarters", "located", "office", "based", "industry", "sector", "domain", "field", "date", "founded", "foundation"]
    lower_question = question.lower()

    return any(word in lower_question for word in keywords)

def is_executive_question(question):
    """
    Determines if a question is about executives using dependency parsing and entity recognition
    """
    doc = nlp(question)
    
    # Check for PERSON entities
    has_person = any(ent.label_ == "PERSON" for ent in doc.ents)
    
    # Check for job title patterns using POS tags
    # Common patterns for executive questions often have proper nouns (PROPN) 
    # followed by or preceded by job-related words
    for token in doc:
        # Check if token is part of a job title compound
        if token.dep_ == "compound" and token.head.pos_ == "PROPN":
            return True
        
        # Check for words like "serve", "work", "lead" that often indicate role questions
        if token.pos_ == "VERB" and token.lemma_ in {"serve", "work", "lead", "join", "head"}:
            return True
        
        # Check for title-related words
        if token.text.lower() in {"role", "position", "title", "job", "ceo", "executive", "executives"}:
            return True
    
    # If we found a PERSON entity, check if the question structure suggests a role query
    if has_person:
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                return True
    
    return False

def answer_question(question):
    """
    Handles financial, executive, and company info queries.
    """
    ticker, year, aggregate_key = extract_entities(question)
    print(ticker, year, aggregate_key)
    # Handle aggregate queries
    if aggregate_key:
        if not year:
            return "Please specify a year."

        result = knowledge_base.get("aggregates", {}).get(str(year), {}).get(aggregate_key, "No data available.")
        return f"The company with the {aggregate_key.replace('_', ' ')} in {year} was {result}." if result != "No data available." else result

    if not ticker:
        return "I couldn't determine the company you're asking about."

    # Handle industry and headquarters queries
    if is_info_question(question):
        info = knowledge_base.get(ticker, {}).get("info", "No company info available.")

        if info != "No company info available.":
            # Use the QA pipeline to extract the specific answer
            response = qa_pipeline(question=question, context=info)

            # Return the extracted answer if the confidence is high
            if response['score'] > 0.5:
                return response['answer']
            else:
                return info  # Fallback to the full sentence if QA confidence is low

    # Default to the latest year if none is provided
    if not year:
        year = max(knowledge_base[ticker].keys())

    # Handle executive and financial queries
    if is_executive_question(question):
        print("Executive Question")
        context = knowledge_base.get(ticker, {}).get(str(year), {}).get("executives", "No executive data available.")
    else:
        print("NOT Executive Question")
        context = knowledge_base.get(ticker, {}).get(str(year), {}).get("financials", "No financial data available.")

    if "No data available" in context:
        return context

    # Use the QA pipeline for the financials and executives
    response = qa_pipeline(question=question, context=context)

    return response['answer']

# Example Queries
#print(knowledge_base)
print(knowledge_base.get('AAPL', {}).get('2020', {}).get("financials", "No financial data available."))
print(answer_question("What was the revenue of Apple in 2020?"))
'''print(answer_question("Who was the CEO of Apple in 2020?"))
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

print(answer_question("What was the largest industry in 2016?"))
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
            
'''with open('questions.txt', 'r') as file:
    for line in file:
        print(line)
        print(answer_question(line))
        print('\n')'''
            