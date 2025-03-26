#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:26:17 2025

@author: sinap
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bert_script import answer_question

# Initialize the FastAPI app
app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your Vercel domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request format
class QuestionRequest(BaseModel):
    question: str

# API Endpoint
@app.post("/api/qa/")
async def return_answer(request: QuestionRequest):
    """QA endpoint for answering questions"""
    
    # QA model
    response = answer_question(request.question)
    print("✅ API Response:", response)

    
    return {"answer": response}