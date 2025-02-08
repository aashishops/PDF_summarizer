import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
import pandas as pd

load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']


def read_excel_file(file_path):
    try:
        excel_data = pd.ExcelFile(file_path)
        
        full_doc_text = ""
        
   
        for sheet_name in excel_data.sheet_names:
            print(f"Processing sheet: {sheet_name}")
            df = excel_data.parse(sheet_name)  
            
            
            for _, row in df.iterrows():
                formatted_row = ", ".join([f"{col}: {row[col]}" for col in df.columns])
                full_doc_text += formatted_row + "\n"  
        return full_doc_text

    
    except Exception as e:
        print(f"Error extracting text from Excel {file_path}: {e}")
        return "Error extracting text from Excel"


def preprocess_excel_data(text):

    text_splitter = CharacterTextSplitter(
        separator="\n",  
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    
    all_chunks = []
    
    
    lines = text.split("\n")
    
   
    for index, line in enumerate(lines):
        if line.strip():  
            chunks = text_splitter.split_text(line)
            all_chunks.extend([(chunk, {"source": f"Line {index + 1}"}) for chunk in chunks])
    return all_chunks


def extract_pdf_text(pdf_files):
    
    pdf_texts = {}
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        pdf_texts[pdf.name] = text
    return pdf_texts

def split_text_into_chunks(pdf_texts):
   
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    all_chunks = []
    for pdf_name, text in pdf_texts.items():
        chunks = text_splitter.split_text(text)
        
        all_chunks.extend([(chunk, {"source": pdf_name}) for chunk in chunks])
    return all_chunks

def create_knowledge_base(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    
    texts = [chunk[0] for chunk in chunks]
    metadata = [chunk[1] for chunk in chunks]  
    
    knowledge_base = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadata)
    return knowledge_base

def generate_response(user_question, knowledge_base):
   
    docs = knowledge_base.similarity_search(user_question)
    context = []
    sources = []
    for doc in docs:
        text = doc.page_content
        source_pdf = doc.metadata.get("source", "Unknown source")
        sources.append(source_pdf)
        context.append(f"Source: {source_pdf}\n{text}\n")
    
    context_text = "\n\n".join(context)
   
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="deepseek-r1-distill-llama-70b")
    
    prompt_template = ChatPromptTemplate.from_template(""" 
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question. 
        Answer the question as detailed as possible.                                                
        If the question is not related to the context, please respond saying not stated in the given document.                                                                   
        <context>
        {context}
        </context>
        Questions: {input}
        """)

  
    chain = load_qa_chain(
        llm, 
        chain_type="stuff",  
        prompt=prompt_template  
    )

    #
    inputs = {
        "context": context_text,  
        "input": user_question  
    }
    
    print("#" * 100)
    print(context_text)
    
    response = chain.run(input_documents=docs, **inputs)
    
    print("*" * 100)
    print(docs)
    
    unique_sources = pd.Series(sources).unique()

    
    return response, unique_sources.tolist()
