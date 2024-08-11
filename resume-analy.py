import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import openai
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

if __name__ == '__main__':
    print("Ingesting...")
    loader=TextLoader("resume.txt")
    document=loader.load()
    print("Splitting")
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print("{} {}",texts,len(texts))
    embedding = OpenAIEmbeddings()
    print("ingesting............. save to PineVector store")
    PineconeVectorStore.from_documents(texts,embedding,index_name=os.getenv('INDEX_NAME'))
    print("finished ingesting")

