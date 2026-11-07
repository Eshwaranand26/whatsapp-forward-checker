#load fact checks from sqlite
#turn each fact check into a document
#create the embeddings
#store into chroma
import sqlite3
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
#langchain_chroma instead of langchain_community.vectorstores(the community version is deprecated)

conn = sqlite3.connect("data/factchecks.db")
df = pd.read_sql("SELECT * FROM factchecks", conn)
conn.close()

print("fact checks loaded :", len(df))

#chroma does not accept empty(None) metadata,3 rows have no rating text
df["rating_raw"] = df["rating_raw"].fillna("")

#one fact check = one document(no splitter needed,a claim is already short)
#claim becomes page_content,all other columns become metadata
data = DataFrameLoader(df, page_content_column = "claim")
docs = data.load()

print(docs[0].page_content)
print(docs[0].metadata)

#qwen3 won the comparison in notebook 02(free,runs on my laptop,no api key)
embedding_model = HuggingFaceEmbeddings(
    model_name = "Qwen/Qwen3-Embedding-0.6B",
    show_progress = True
)

#cosine:chroma's score = 1 - similarity,so it matches the heatmaps in notebook 02
#url as id:the same fact check can never be stored twice
vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    ids = df["url"].tolist(),
    collection_metadata = {"hnsw:space": "cosine"},
    persist_directory = "chroma_db"
)

print("vector database created :", len(vectorstore.get()["ids"]), "fact checks")
