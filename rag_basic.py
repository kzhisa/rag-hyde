import os
import sys
import uuid

import chromadb
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

# ChromaDB
CHROMA_PERSIST_DIRECTORY = os.environ.get("CHROMA_PERSIST_DIRECTORY")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME")

# Retriever settings
TOP_K_VECTOR = 5

# Langchain LangSmith
unique_id = uuid.uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"Tracing RAG Basic - {unique_id}"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# 既存のChromaDBを読み込みVector Retrieverを作成
def vector_retriever(top_k: int = TOP_K_VECTOR):
    """Create base vector retriever from ChromaDB

    Returns:
        Vector Retriever
    """

    # chroma db
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIRECTORY)
    vectordb = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        client=client,
    )

    # base retriever (vector retriever)
    vector_retriever = vectordb.as_retriever(
        search_kwargs={"k": top_k},
    )

    return vector_retriever


# プロンプトテンプレート
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

# Documentsを整形する関数
def doc_to_str(docs):
    return "\n---\n".join(doc.page_content for doc in docs)

# RAGチャットボットを実行
def chat_with_bot(question: str):

    # LLM
    chat_model = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0.0)

    # Vector Retriever
    retriever = vector_retriever()

    # RAG Chain
    rag_chain = (
        {"context": retriever | doc_to_str, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    # プロンプトテンプレートに基づいて応答を生成
    response = rag_chain.invoke(question)

    print(f"AI: {response}")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Error: query required.")
        sys.exit(1)
    
    question = sys.argv[1]

    # チャットセッションの開始
    chat_with_bot(question)
