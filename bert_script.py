#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced QA system using NLP-based entity extraction
"""

import re
import spacy
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from fuzzywuzzy import process

# Load spaCy's pre-trained English model for NLP
nlp = spacy.load("en_core_web_sm")

# Load datasets
industry_df = pd.read_csv("sp500-companies.csv", encoding='ISO-8859-1')
executive_df = pd.read_csv("sp500_firm_execu.csv", encoding='ISO-8859-1').rename(columns={'tic': 'Ticker'})
finance_df = pd.read_csv("Financials_SP500.csv", encoding='ISO-8859-1')

# Standardize tickers
for df in [industry_df, executive_df, finance_df]:
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

# Create a company name-to-ticker lookup
company_name_to_ticker = {row["Name"].lower(): row["Ticker"] for _, row in industry_df.iterrows()}

# Build knowledge base for executive info
executive_kb = {}
for _, row in executive_df.iterrows():
    ticker, year = row["Ticker"], row["year"]
    if ticker not in executive_kb:
        executive_kb[ticker] = {}
    executive_kb[ticker][year] = f"{row['exec_fname']} {row['exec_lname']}"

# Build financial knowledge base
financial_kb = {}
for _, row in finance_df.iterrows():
    ticker, year = row["Ticker"], row["Fiscal Year"]
    if ticker not in financial_kb:
        financial_kb[ticker] = {}

    financial_kb[ticker][year] = {
        "revenue": row["Revenue"],
        "net_income": row["Net Income"],
        "gross_profit": row["Gross Profit"]
    }

# Load T5 Model
t5_model_name = "t5-small"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

def fuzzy_company_lookup(company_name):
    """Finds the best match for a given company name using fuzzy matching."""
    match, score = process.extractOne(company_name.lower(), company_name_to_ticker.keys())
    return company_name_to_ticker[match] if score > 80 else None  # Threshold for accuracy

def extract_entities(question):
    """Uses NLP to extract company names and years from a question."""
    doc = nlp(question)
    ticker, year = None, None

    for ent in doc.ents:
        if ent.label_ == "ORG":  # Organization (Company Name)
            ticker = company_name_to_ticker.get(ent.text.lower()) or fuzzy_company_lookup(ent.text)
        elif ent.label_ == "DATE" or re.match(r"\b(19\d{2}|20\d{2})\b", ent.text):
            year = int(ent.text)

    return ticker, year

def generate_rephrased_answer(question, company, year, answer):
    """Uses T5 to generate a natural response."""
    prompt = f"Rephrase the question: '{question}' into a statement with an answer. {answer}"
    
    input_ids = t5_tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = t5_model.generate(input_ids, max_length=50)
    
    return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

def answer_question(question):
    """Extracts entities, retrieves the answer, and generates a response using T5."""
    ticker, year = extract_entities(question)

    if not ticker:
        return "I couldn't determine the company you're asking about."

    if not year:
        return "Please specify a year for financial or CEO information."

    # Determine if the question is about executives or financials
    is_ceo_question = any(word in question.lower() for word in ["ceo", "executive", "board member", "cfo", "president"])
    
    company_name = next((k for k, v in company_name_to_ticker.items() if v == ticker), ticker)

    if is_ceo_question:
        ceo_name = executive_kb.get(ticker, {}).get(year, "Unknown")
        if ceo_name == "Unknown":
            return f"I couldn't find CEO data for {company_name} in {year}."
        answer = f"The CEO of {company_name} in {year} was {ceo_name}."
    else:
        financials = financial_kb.get(ticker, {}).get(year)
        if not financials:
            return f"I couldn't find financial data for {company_name} in {year}."

        # Determine which financial metric the question asks for
        if "revenue" in question.lower():
            answer = f"The revenue of {company_name} in {year} was ${financials['revenue']} billion."
        elif "net income" in question.lower():
            answer = f"The net income of {company_name} in {year} was ${financials['net_income']} billion."
        elif "gross profit" in question.lower():
            answer = f"The gross profit of {company_name} in {year} was ${financials['gross_profit']} billion."
        else:
            return f"I found financial data for {company_name} in {year}, but I need more details on what you're asking (e.g., revenue, net income, etc.)."

    return generate_rephrased_answer(question, company_name, year, answer)

# Example Queries
print(answer_question("What was the revenue of Microsoft in 2019?"))
print(answer_question("How much net income did Apple have in 2021?"))
print(answer_question("What was the gross profit of Tesla in 2020?"))
print(answer_question("What was the gross profit of Equinix in 2021?"))
print(answer_question("Who was the CEO of Apple in 2020?"))
print(answer_question("Who was the CEO of Oracle Corporation in 2019?"))

