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
knowledge_base = {}

# Add headquarters and industry information to the KB
for _, row in industry_df.iterrows():
    ticker = row["Ticker"]

    if ticker not in knowledge_base:
        knowledge_base[ticker] = {}

    industry = row.get("Industry", "Unknown industry")
    headquarters = row.get("Headquarters Location", "Unknown location")
    founded = row.get("Founded", "Unknown foundation date")

    # Store both industry and headquarters in a single sentence
    knowledge_base[ticker]["info"] = (
        f"{row['Name']} operates in the {industry} industry and is headquartered in {headquarters}. It was founded in {founded}"
    )

# Add financial and executives data
for _, row in merged_df.iterrows():
    ticker = row["Ticker"]
    year = row["Fiscal Year"]

    if ticker not in knowledge_base:
        knowledge_base[ticker] = {}

    knowledge_base[ticker][year] = {
        "financials": (
            f"In {year}, {row['Name']} had {row['Shares (Basic)']} shares outstanding, "
            f"reported a revenue of ${row['Revenue']}, "
            f"a gross profit of ${row['Gross Profit']}, "
            f"and a net income of ${row['Net Income']}, "
            f"and it's increase in gross profit was ${row['increase_in_gross_profit']}, "
            f"and it's increase in operating expense was ${row['increase_in_operating_expense']}, "
            f"and it's increase in net income was ${row['Increase_in_Net_Income']}"
        )
    }

# Add executive information to the knowledge base
for _, row in executive_df.iterrows():
    ticker = row["Ticker"]
    year = row["year"]

    if ticker not in knowledge_base:
        knowledge_base[ticker] = {}

    if year not in knowledge_base[ticker]:
        knowledge_base[ticker][year] = {}

    knowledge_base[ticker][year]["executives"] = (
        f"In {year}, {row['exec_fname']} {row['exec_lname']} served as {row['title']} at {ticker}."
    )

# Initialize an aggregate section in the knowledge base
for _, row in aggregated_data.iterrows():
    year = row["Year"]
    if "aggregates" not in knowledge_base:
        knowledge_base["aggregates"] = {}

    knowledge_base["aggregates"][year] = {
        "highest_gross_profit": row["highest_gross_profit"],  
        "highest_income": row["Company_with_highest_income"],  
        "highest_increase_in_gross_profit": row["highest_increase_in_gross_profit"],
        "highest_operating_expense": row["highest_operating_expense"],
        "highest_revenue": row["Company_with_highest_revenue"]
    }

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
        print("////Possible ticker: ", possible_ticker)
            
        if possible_ticker:
            ticker = possible_ticker
    # Detect aggregate-type questions
    aggregate_mappings = {
        "highest gross profit": "highest_gross_profit",
        "highest revenue": "highest_revenue",
        "highest net income": "highest_net_income",
        "highest operating expense": "highest_operating_expense",
        "highest increase in gross profit": "highest_increase_in_gross_profit"
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

        result = knowledge_base.get("aggregates", {}).get(year, {}).get(aggregate_key, "No data available.")
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
        context = knowledge_base.get(ticker, {}).get(year, {}).get("executives", "No executive data available.")
    else:
        print("NOT Executive Question")
        context = knowledge_base.get(ticker, {}).get(year, {}).get("financials", "No financial data available.")

    if "No data available" in context:
        return context

    # Use the QA pipeline for the financials and executives
    response = qa_pipeline(question=question, context=context)

    return response['answer']

# Define the JSON filename
json_filename = "knowledge_base.json"

# Save knowledge_base as a JSON file
with open(json_filename, "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, indent=4)

# Example Queries
'''print(answer_question("What was the revnue of Apple in 2020?"))
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
print(answer_question("Where are the headquarters of American Electric Power ?"))'''

with open('questions.txt', 'r') as file:
    for line in file:
        print(line)
        print(answer_question(line))
        print('\n')
            
            
            