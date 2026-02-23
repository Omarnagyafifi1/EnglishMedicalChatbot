from dotenv import load_dotenv
import os
from src.helper import load_pdf, split_into_mindocus, split_into_chunks, get_embedding_model
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()
PINECONE_API_KEY = os.getenv("PIENCONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

DATA_DIR = os.path.join(os.getcwd(),"data")
extracted_data=load_pdf (DATA_DIR)
print(f"Extracted {len(extracted_data)} documents from PDFs.")
filter_data = split_into_mindocus(extracted_data)
text_chunks=split_into_chunks(filter_data)
print(f'Number of chunks :{len(text_chunks)}')
embeddings = get_embedding_model()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)



index_name = "medical-chatbot"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)