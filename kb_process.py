#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 18:33:18 2025

@author: sinap
"""
import pandas as pd
import json

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
    
# Define the JSON filename
json_filename = "knowledge_base.json"

# Save knowledge_base as a JSON file
with open(json_filename, "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, indent=4)