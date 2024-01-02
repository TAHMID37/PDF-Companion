import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
# Set the page configuration
st.set_page_config(page_title="PDF Companion", page_icon=":robot:")

# Sidebar contents
with st.sidebar:
    st.title('📚🔍 PDF Companion')
    st.markdown('''
    ## About
     "PDF Companion" the ultimate PDF reader app that not only brings your documents to life but also provides insightful answers to your queries, revolutionizing the way you interact with your PDF files.
    
 
    ''')
    add_vertical_space(5)
    st.write('Made with 💻❤️ by Tahmid')
 
 
def get_api_key():
    input_text = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text


# Function to save uploaded PDF and get its path
def save_uploaded_pdf(uploaded_pdf):
    # Check if a file was uploaded
    if uploaded_pdf is not None:
        # Check if the directory './pdf' exists, if not, create it
        pdf_dir = "./pdf"
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)

        # Get the filename from the uploaded file
        file_name = uploaded_pdf.name

        # Define the path to save the file
        save_path = os.path.join(pdf_dir, file_name)

        # Save the uploaded file
        with open(save_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        # Get the file path
        file_path = os.path.abspath(save_path)
        return file_path
    else:
        return None

def get_pdf_text(pdf):
    #  pdf_reader = PdfReader(pdf)
    #  text = ""
    #  for page in pdf_reader.pages: 
    #         text += page.extract_text()
    #  return text
    
    loader = PyPDFLoader(pdf)
    pages = loader.load_and_split()
    return pages     

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def main():
    st.header("Chat with PDF 💬")
    
    openai_api_key = get_api_key()
    
    if openai_api_key:
 
 # upload a PDF file
       # Upload a PDF file
        uploaded_pdf = st.file_uploader("Upload your PDF", type='pdf')
        
        if uploaded_pdf is not None:
            # Save the uploaded PDF and get its file path
            file_path = save_uploaded_pdf(uploaded_pdf)

            if file_path:
                # Get text chunks from the PDF
                pages = get_pdf_text(file_path)
                #chunks = get_text_chunks(pages)
                #st.write(pages)

            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            VectorStore = FAISS.from_documents(pages, embedding=embeddings)
            
            # Accept user questions/query
            query = st.text_input("Ask questions about your PDF file:")
 
            if query:
                    
                    docs = VectorStore.similarity_search(query=query, k=3)
            
                    llm = OpenAI(openai_api_key=openai_api_key)
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=query)
                        print(cb)
                    st.write(response)              
            
    else:
       print("Hi")
       st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
       st.stop() 
 
 
 
 
 
if __name__ == '__main__':
    main()
   