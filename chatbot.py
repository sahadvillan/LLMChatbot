import re
import pandas as pd
from langchain_ollama import OllamaLLM
from nltk.corpus import stopwords

parts_csv = "Parts_with_similarities.csv"  
df = pd.read_csv(parts_csv)

# Initialize Ollama model
llm = OllamaLLM(model="llama3")

def chat_with_ollama(prompt):
    response = llm.invoke(input=prompt)
    return response.strip()

def find_similar_parts(description):
    match = df[df["DESCRIPTION"].str.contains(description, case=False, na=False)]
    if not match.empty:
        similar_parts = match.iloc[0, 32]  
        print(f"\nBot: Similar parts ID for the given description: {similar_parts}")
    else:
        print("\nBot: No matching descriptions found.")

def chatbot():
    """Interactive chatbot with part similarity and Ollama response generation."""
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if "similar parts" in user_input.lower():
            part_query = re.sub(r"[|/,()]", "", user_input.replace("similar parts", "").strip()).lower()
            part_query = " ".join([word for word in part_query.split() if word not in stopwords.words('english')])  # Remove common words
            common_words = {"please", "find", "search", "show", "tell", "give", "me"}  
            part_query = " ".join([word for word in part_query.split() if word not in common_words])
            # print(part_query)
            find_similar_parts(part_query)
        else:
            print("\nBot: ", end="", flush=True)
            response = chat_with_ollama(user_input)
            for char in response:
                print(char, end="", flush=True)
            print()  

if __name__ == "__main__":
    chatbot()