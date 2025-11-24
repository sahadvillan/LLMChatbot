import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
nltk.download('wordnet')

# This function will create missing description based on other features.
def create_description(row):
    description_parts = []
    description_parts.append("indicator")
    if pd.notna(row[2]) and row[2] != "":
        description_parts.append(str(row[2]) + " movement")
    if pd.notna(row[9]) and row[9] != "":
        description_parts.append(str(row[9]))
    if pd.notna(row[18]) and row[18] != "":
        description_parts.append(str(row[18]))
    if pd.notna(row[11]) and row[11] != "":
        description_parts.append(str(row[11]))
    if pd.notna(row[10]) and row[10] != "":
        description_parts.append(str(row[10]))
    description_parts.append("electric indicator")
    if pd.notna(row[5]) and row[5] != "":
        description_parts.append(str(row[5])+ " blow")
    if pd.notna(row[28]) and row[28] != "":
        description_parts.append(str(row[28]))
    if pd.notna(row[30]) and row[30] != "":
        description_parts.append(str(row[30])+ "ac ")
    if pd.notna(row[27]) and row[27] != "":
        description_parts.append(str(row[27])+ " ir")
    if pd.notna(row[19]) and row[19] != "":
        description_parts.append(str(row[19]))
    if pd.notna(row[11]) and row[11] != "":
        description_parts.append(str(row[11]))
    
    # Join the parts with a space, and return the result
    return " ".join(description_parts)

def find_similar_parts(index, cosine_sim, top_n=5):
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]  # Exclude self
    similar_indices = [i[0] for i in sim_scores]
    similar_parts = df1.iloc[similar_indices]["ID"]
    return similar_parts.values.tolist()

if __name__ == "__main__":

    df1 = pd.read_csv(r"C:\Users\Dell\Desktop\bmw-tasks\Parts.csv", encoding="utf-8", delimiter=";")

    #converted all string values to lowercase
    df = df1.applymap(lambda x: x.lower() if isinstance(x, str) else x) 

    # Remove '|' , ',' and '/' from all columns
    df = df.replace(r"[|/,()]", "", regex=True)

    # Apply the function to fill missing descriptions in column 2
    df.iloc[:, 1] = df.apply(lambda row: create_description(row) if pd.isna(row[1]) or row[1] == "" else row[1], axis=1)

    # Replace the description in column 2 if it matches the specified string
    df.iloc[:, 1] = df.iloc[:, 1].replace("indicator electric indicator", "unknown part")

    # df.to_csv(r"C:\Users\Dell\Desktop\bmw-tasks\Parts_fixed.csv", index=False, encoding="utf-8")

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["DESCRIPTION"])

    # Compute Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    df["SIMILAR_PARTS"] = df.index.map(lambda i: find_similar_parts(i, cosine_sim))

    df.to_csv("Parts_with_similarities.csv", index=False)
    print("Similarity search completed. Results saved!")


    