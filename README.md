# 🏥 Medical Chatbot

A production-ready RAG-powered medical question answering chatbot built with **LLMs**, **LangChain**, **Pinecone**, **Groq**, and **Flask**.

## 📋 Features

- **Retrieval-Augmented Generation (RAG)**: Combines document retrieval with LLM inference for accurate, context-aware responses
- **PDF Document Processing**: Automatically loads and processes medical PDFs from the `data/` directory
- **Vector Database**: Uses Pinecone for semantic search across medical documents
- **Fast LLM Inference**: Powered by Groq's API for rapid response generation
- **Flask Web Interface**: User-friendly chat interface for medical Q&A
- **Embeddings**: HuggingFace embeddings (all-MiniLM-L6-v2) for semantic similarity
- **Modular Architecture**: Clean separation of concerns with helper functions and prompt templates

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Git
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- API Keys:
  - **Pinecone**: [Sign up at pinecone.io](https://www.pinecone.io/)
  - **Groq**: [Sign up at groq.com](https://console.groq.com/)
  - **OpenAI** (optional for embeddings): [Sign up at openai.com](https://openai.com/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Omarnagyafifi1/EnglishMedicalChatbot.git
   cd Medical_chatbot
   ```

2. **Create a Python virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # or
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   GROQ_API_KEY=your_groq_api_key
   OPENAI_API_KEY=your_openai_key  # Optional
   ```

5. **Add medical documents**
   Place your PDF files in the `data/` directory:
   ```
   data/
   ├── medical_book_1.pdf
   ├── medical_book_2.pdf
   └── ...
   ```

## 📖 Usage

### 1. Index Medical Documents (Create Vector Database)

```bash
python index_store.py
```

This script:
- Loads all PDFs from the `data/` directory
- Splits documents into chunks
- Generates embeddings
- Uploads to Pinecone index

### 2. Run the Web Application

```bash
python app.py
```

The chatbot will be available at `http://localhost:5000`

### 3. Interactive Research Notebook

Explore the RAG pipeline in action:
```bash
jupyter notebook research/trials.ipynb
```

## 📁 Project Structure

```
Medical_chatbot/
├── app.py                    # Flask web application
├── index_store.py            # Indexing script for Pinecone
├── main.py                   # Entry point / CLI
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata
├── .env                      # Environment variables (not in repo)
│
├── src/
│   ├── __init__.py
│   ├── helper.py             # PDF loading, chunking, embeddings
│   └── prompt.py             # LLM prompt templates
│
├── data/
│   └── Medical_book.pdf      # Sample medical document
│
├── research/
│   └── trials.ipynb          # Jupyter notebook for experimentation
│
├── templates/
│   └── chat.html             # HTML chat interface
│
├── static/
│   └── style.css             # CSS styling
│
└── README.md                 # This file
```

## 🔧 Configuration

### Pinecone Index

The script creates a Pinecone index with:
- **Name**: `medical-chatbot`
- **Dimension**: 384 (HuggingFace embeddings)
- **Metric**: Cosine similarity
- **Infrastructure**: Serverless (AWS us-east-1)

### LLM Models

- **Embedding Model**: `all-MiniLM-L6-v2` (HuggingFace)
- **LLM**: `llama-3.3-70b-versatile` (Groq)
- **Temperature**: 0.1 (deterministic answers)

## 📦 Dependencies

- **langchain**: RAG framework
- **langchain-pinecone**: Pinecone integration
- **langchain-groq**: Groq LLM integration
- **langchain-huggingface**: HuggingFace embeddings
- **pinecone-client**: Pinecone SDK
- **sentence-transformers**: Embedding models
- **pypdf**: PDF processing
- **flask**: Web framework
- **python-dotenv**: Environment variable management

See `requirements.txt` for full list.

## 🤖 How It Works

1. **Document Ingestion**: PDFs are loaded and split into semantic chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings
3. **Vector Storage**: Embeddings are stored in Pinecone for fast retrieval
4. **Query Processing**: User questions are converted to embeddings
5. **Retrieval**: Top-K similar documents are retrieved from Pinecone
6. **LLM Response**: Retrieved context + question are passed to Groq LLM
7. **Answer Generation**: LLM generates concise, medical-focused answers

## 📚 Example Usage

```python
from src.helper import load_pdf, split_into_chunks, get_embedding_model
from langchain_pinecone import PineconeVectorStore

# Load and process documents
docs = load_pdf("data/")
chunks = split_into_chunks(docs)

# Create vector store
embeddings = get_embedding_model()
docsearch = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="medical-chatbot"
)

# Query
results = docsearch.similarity_search("What is diabetes?", k=3)
for doc in results:
    print(doc.page_content)
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `PINECONE_API_KEY` not found | Ensure it's set in `.env` file |
| `GROQ_API_KEY` invalid | Check your Groq API key at [console.groq.com](https://console.groq.com/) |
| Pinecone index not found | Run `python index_store.py` first |
| Low answer quality | Add more medical PDFs to the `data/` directory |
| Slow embeddings | HuggingFace models are cached locally after first download |

## 🔐 Security

- **Never commit `.env` file** to version control
- Use environment variables for all sensitive keys
- Ensure Pinecone and Groq API keys have appropriate permissions
- Validate user inputs in production

## 📝 License

This project is open source. See LICENSE for details.

## 👨‍💻 Author

**Omar Nagy**  
Email: onagy489@gmail.com  
GitHub: [Omarnagyafifi1](https://github.com/Omarnagyafifi1/)

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 Citation

If you use this chatbot in your research or application, please cite:

```bibtex
@software{medical_chatbot_2025,
  title={Medical Chatbot: RAG-Powered Medical Q&A System},
  author={Nagy, Omar},
  year={2025},
  url={https://github.com/Omarnagyafifi1/EnglishMedicalChatbot}
}
```

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - RAG framework
- [Pinecone](https://www.pinecone.io/) - Vector database
- [Groq](https://groq.com/) - Fast LLM inference
- [HuggingFace](https://huggingface.co/) - Pre-trained embeddings
