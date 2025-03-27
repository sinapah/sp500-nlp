from transformers import pipeline

# Load BART model
full_sentence_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")

# Question and answer
question = "What is the capital of France?"
answer = "Paris"

# Optimized input prompt
formatted_input = f"{question} {answer}"

# Generate full sentence with beam search and early stopping
output = full_sentence_pipeline(
    formatted_input,
    max_length=30,               # Limit output length
    num_beams=5,                 # Beam search
    early_stopping=True,         # Stop after valid sentence
    num_return_sequences=1,      # One output
    temperature=1.0,              # Keep it deterministic
    do_sample=False              # No sampling, only beam search
)[0]['generated_text']

print(output)