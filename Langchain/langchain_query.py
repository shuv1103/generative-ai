from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Initialize the OpenAIEmbeddings class with your API key
embeddings = OpenAIEmbeddings(model = "text-embedding-3-large", dimensions = 3072)

document = [
    "Delhi is the capital of India.",
    "Kolkata is the capital of West Bengal.",
    "Hyderabad is the capital of Telangana.",
    "Bangalore is the capital of Karnataka.",
    "Chennai is the capital of Tamil Nadu.",
    "Mumbai is the capital of Maharashtra.",
    "Tirupati is the capital of Andhra Pradesh.",
    "Lucknow is the capital of Uttar Pradesh.",
]

doc_embeddings = embeddings.embed_documents(document)
query_embeddings = embeddings.embed_query("What is the capital of Maharashtra?")

scores = cosine_similarity([query_embeddings], doc_embeddings)[0]

index, score = sorted(enumerate(scores), key=lambda x: x[1])[-1]

print(f"The most similar document is - {document[index]} with a similarity score = {score}")
