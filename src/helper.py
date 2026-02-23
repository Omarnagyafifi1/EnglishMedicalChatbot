from tqdm.auto import tqdm
from langchain_community.document_loaders import PyPDFLoader
import os
import glob
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings






#extract text from pdf files in the data directory
def load_pdf(data_dir):
    documents = []
    pdf_files = glob.glob(os.path.join(data_dir, "**/*.pdf"), recursive=True)

    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        documents.extend(docs)

    return documents



def split_into_mindocus(docs:list[Document]) -> list[Document]: # remove unnessary metadata and only keep page_content and source
    mindocus = []
    for doc in docs:
        mindocus.append(Document(
            page_content=doc.page_content,
            metadata={"src": doc.metadata.get("source")}
        ))
    return mindocus



def split_into_chunks(docs:list[Document], chunk_size=500, chunk_overlap=20) -> list[Document]:
 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in tqdm(docs, desc="Splitting into chunks"):
        doc_chunks = text_splitter.create_documents([doc.page_content], metadatas=[doc.metadata])
        chunks.extend(doc_chunks)
    return chunks




def get_embedding_model(model_name="all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)