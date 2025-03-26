#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 18:33:18 2025

@author: sinap
"""
import pandas as pd
import json

# ===========================
# 📌 Load datasets
# ===========================
industry_df = pd.read_csv("sp500-companies.csv", encoding='ISO-8859-1')
finance_df = pd.read_csv("Financials_SP500.csv", encoding='ISO-8859-1')
executive_df = pd.read_csv("sp500_firm_execu.csv", encoding='ISO-8859-1')
aggregated_data = pd.read_csv("aggregated_stats.csv", encoding='ISO-8859-1')
balance_df = pd.read_csv("us-balance-annual.csv", encoding="ISO-8859-1", sep=';')

# ===========================
# 📌 Preprocessing
# ===========================
# Rename columns for consistency
executive_df = executive_df.rename(columns={'tic': 'Ticker'})

# Standardize tickers
for df in [finance_df, industry_df, executive_df, balance_df]:
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()

# ===========================
# 📌 Create the Knowledge Base
# ===========================
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
        f"{row['Name']} operates in the {industry} industry and is headquartered in {headquarters}. "
        f"It was founded in {founded}."
    )

# Add financial and executive data
merged_df = finance_df.merge(industry_df[['Ticker', 'Industry']], on="Ticker", how="left")

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
            f"with an increase in gross profit of ${row['increase_in_gross_profit']}, "
            f"an increase in operating expense of ${row['increase_in_operating_expense']}, "
            f"and an increase in net income of ${row['Increase_in_Net_Income']}."
        )
    }

# Add executive information
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

# ===========================
# 📌 Find the 1st, 2nd, and 3rd Largest and smallest Industries by Assets and Investments Per Year
# ===========================

# Merge balance and industry data
merged_assets_df = balance_df.merge(industry_df[["Ticker", "Industry"]], on="Ticker", how="left")

# Group by year and industry, summing up the total assets
industry_assets = (
    merged_assets_df.groupby(["Fiscal Year", "Industry"])["Total Assets"]
    .sum()
    .reset_index()
)

industry_investments = (
    merged_assets_df.groupby(["Fiscal Year", "Industry"])["Long Term Investments & Receivables"]
    .sum()
    .reset_index()
)

# Sort by year and descending assets
industry_assets = industry_assets.sort_values(["Fiscal Year", "Total Assets"], ascending=[True, False])
industry_investments = industry_investments.sort_values(["Fiscal Year", "Long Term Investments & Receivables"], ascending=[True, False])


# Collect the top 3 industries for each year
largest_industries_by_year = (
    industry_assets.groupby("Fiscal Year")
    .apply(lambda x: x.nlargest(3, "Total Assets"))
    .reset_index(drop=True)
)

smallest_industries_by_year = (
    industry_assets.groupby("Fiscal Year")
    .apply(lambda x: x.nsmallest(3, "Total Assets"))
    .reset_index(drop=True)
)

most_investments_by_year = (
    industry_investments.groupby("Fiscal Year")
    .apply(lambda x: x.nlargest(3, "Long Term Investments & Receivables"))
    .reset_index(drop=True)
)

least_investments_by_year = (
    industry_investments.groupby("Fiscal Year")
    .apply(lambda x: x.nsmallest(3, "Long Term Investments & Receivables"))
    .reset_index(drop=True)
)

# Prepare a dictionary for easy lookup
largest_industries_dict = {}
smallest_industries_dict = {}
most_investments_dict = {}
least_investments_dict = {}

# Iterate through the DataFrame and store 1st, 2nd, and 3rd largest industries by year
for year in largest_industries_by_year["Fiscal Year"].unique():
    top_industries = largest_industries_by_year[largest_industries_by_year["Fiscal Year"] == year]

    largest = top_industries.iloc[0]["Industry"] if len(top_industries) > 0 else "No data"
    second_largest = top_industries.iloc[1]["Industry"] if len(top_industries) > 1 else "No data"
    third_largest = top_industries.iloc[2]["Industry"] if len(top_industries) > 2 else "No data"
  
    largest_industries_dict[year] = {
        "largest_industry": largest,
        "second_largest_industry": second_largest,
        "third_largest_industry": third_largest,
    }
    
for year in smallest_industries_by_year["Fiscal Year"].unique():
    top_industries = smallest_industries_by_year[smallest_industries_by_year["Fiscal Year"] == year]

    smallest = top_industries.iloc[0]["Industry"] if len(top_industries) > 0 else "No data"
    second_smallest = top_industries.iloc[1]["Industry"] if len(top_industries) > 1 else "No data"
    third_smallest = top_industries.iloc[2]["Industry"] if len(top_industries) > 2 else "No data"
  
    smallest_industries_dict[year] = {
        "smallest_industry": smallest,
        "second_smallest_industry": second_smallest,
        "third_smallest_industry": third_smallest,
    }
    
# Iterate through the DataFrame and store 1st, 2nd, and 3rd smallest industries by year
for year in most_investments_by_year["Fiscal Year"].unique():
    top_industries = most_investments_by_year[most_investments_by_year["Fiscal Year"] == year]

    most = top_industries.iloc[0]["Industry"] if len(top_industries) > 0 else "No data"
    second_most = top_industries.iloc[1]["Industry"] if len(top_industries) > 1 else "No data"
    third_most = top_industries.iloc[2]["Industry"] if len(top_industries) > 2 else "No data"
  
    most_investments_dict[year] = {
        "industry_with_the_most_invesmtent": most,
        "industry_with_the__second_most_invesmtent": second_most,
        "industry_with_the_third_most_invesmtent": third_most,
    }
        
# Iterate through the DataFrame and store 1st, 2nd, and 3rd indutries by year based on investments
for year in least_investments_by_year["Fiscal Year"].unique():
    top_industries = least_investments_by_year[least_investments_by_year["Fiscal Year"] == year]

    least = top_industries.iloc[0]["Industry"] if len(top_industries) > 0 else "No data"
    second_least = top_industries.iloc[1]["Industry"] if len(top_industries) > 1 else "No data"
    third_least = top_industries.iloc[2]["Industry"] if len(top_industries) > 2 else "No data"
  
    least_investments_dict[year] = {
        "industry_with_the_least_invesmtent": least,
        "industry_with_the__second_least_invesmtent": second_least,
        "industry_with_the__third_least_invesmtent": third_least,
    }
    
# ===========================
# 📌 Add Aggregates to the Knowledge Base for All Years
# ===========================
all_years = sorted(set(aggregated_data["Year"]).union(largest_industries_by_year["Fiscal Year"]))

for year in all_years:
    if "aggregates" not in knowledge_base:
        knowledge_base["aggregates"] = {}

    knowledge_base["aggregates"][year] = {
        "highest_gross_profit": aggregated_data.loc[aggregated_data["Year"] == year, "highest_gross_profit"].values[0] if year in aggregated_data["Year"].values else "No data",
        "highest_income": aggregated_data.loc[aggregated_data["Year"] == year, "Company_with_highest_income"].values[0] if year in aggregated_data["Year"].values else "No data",
        "highest_increase_in_gross_profit": aggregated_data.loc[aggregated_data["Year"] == year, "highest_increase_in_gross_profit"].values[0] if year in aggregated_data["Year"].values else "No data",
        "highest_operating_expense": aggregated_data.loc[aggregated_data["Year"] == year, "highest_operating_expense"].values[0] if year in aggregated_data["Year"].values else "No data",
        "highest_revenue": aggregated_data.loc[aggregated_data["Year"] == year, "Company_with_highest_revenue"].values[0] if year in aggregated_data["Year"].values else "No data",
        **largest_industries_dict.get(year, {
            "largest_industry": "No data",
            "second_largest_industry": "No data",
            "third_largest_industry": "No data"
        }),
        **smallest_industries_dict.get(year, {
            "smallest_industry": "No data",
            "second_smallest_industry": "No data",
            "third_smallest_industry": "No data"
        }),
        **most_investments_dict.get(year, {
            "industry_with_the_most_invesmtent": "No data",
            "industry_with_the__second_most_invesmtent": "No data",
            "industry_with_the__third_most_invesmtent": "No data"
        }),
        **least_investments_dict.get(year, {
            "industry_with_the_least_invesmtent": "No data",
            "industry_with_the_second_least_invesmtent": "No data",
            "industry_with_the_thid_least_invesmtent": "No data"
        })
    }
    
# Define the JSON filename
json_filename = "knowledge_base.json"

# Save knowledge_base as a JSON file
with open(json_filename, "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, indent=4)