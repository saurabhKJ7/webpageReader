from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import UnstructuredURLLoader


load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

llm = ChatOpenAI(
    openai_api_base="http://localhost:1234/v1",
    model_name="deepseek-r1-distill-qwen-7b",
    temperature=0.7,
    openai_api_key="sk-anything"
)

vectorstore = None
qa_chain = None

async def load_url_to_vectorstore(page_url: str):
    global vectorstore, qa_chain

    loader = UnstructuredURLLoader(urls=[page_url])
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask_question(query: str) -> str:
    if not qa_chain:
        return "No url present yet."
    return qa_chain.run(query)

