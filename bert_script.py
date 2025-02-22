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

# Create a structured knowledge base (financials + executives)
knowledge_base = {}

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

    ticker = None
    year = None
    aggregate_key = None  # Track if the question is about an aggregate

    for ent in doc.ents:
        if ent.label_ == "ORG":  # Company name
            possible_ticker = company_name_to_ticker.get(ent.text.lower()) or fuzzy_company_lookup(ent.text.lower(), company_name_to_ticker)
            if possible_ticker:
                ticker = possible_ticker
        elif ent.label_ == "DATE" or re.match(r"\b(19\d{2}|20\d{2})\b", ent.text):
            year = int(ent.text)

    # Detect aggregate-type questions
    question_lower = question.lower()
    if "highest gross profit" in question_lower:
        aggregate_key = "highest_gross_profit"
    elif "highest revenue" in question_lower:
        aggregate_key = "highest_revenue"
    elif "highest net income" in question_lower:
        aggregate_key = "highest_net_income"
    elif "highest operating expense" in question_lower:
        aggregate_key = "highest_operating_expense"
    elif "highest increase in gross profit" in question_lower:
        aggregate_key = "highest_increase_in_gross_profit"

    return ticker, year, aggregate_key


def answer_question(question):
    """
    Determines the correct company and year using NLP, retrieves relevant context, and passes it to the QA model.
    """
    ticker, year, aggregate_key = extract_entities(question)
    
    if aggregate_key:  # If it's an aggregate query
        if not year:
            return "Please specify a year."

        result = knowledge_base.get("aggregates", {}).get(year, {}).get(aggregate_key, "No data available.")
        if result == "No data available.":
            return result
        return f"The company with the {aggregate_key.replace('_', ' ')} in {year} was {result}."
    
    if not ticker:
        return "I couldn't determine the company you're asking about."

    if not year:
        year = max(knowledge_base[ticker].keys())  # Use latest available year if not specified

    # Determine whether the question is about financials or executives
    if any(word in question.lower() for word in ["ceo", "executive", "board member", "cfo", "president"]):
        context = knowledge_base.get(ticker, {}).get(year, {}).get("executives", "No executive data available.")
    else:
        context = knowledge_base.get(ticker, {}).get(year, {}).get("financials", "No financial data available.")

    if "No data available" in context:
        return context

    # Use QA model to extract the answer
    response = qa_pipeline(question=question, context=context)

    return response['answer']


# Example Queries
print(answer_question("What was the revnue of Apple in 2020?"))
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
