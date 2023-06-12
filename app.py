 
import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os



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


def get_pdf_text(pdf):
     pdf_reader = PdfReader(pdf)
     text = ""
     for page in pdf_reader.pages: 
            text += page.extract_text()
     return text     

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def main():
    st.header("Chat with PDF 💬")
    
    openai_api_key = get_api_key()
    
    if openai_api_key:
 
 # upload a PDF file
      pdf = st.file_uploader("Upload your PDF", type='pdf')
  
      if pdf is not None:
             text=get_pdf_text(pdf)
             chunks=get_text_chunks(text)
             
             store_name = pdf.name[:-4]
              
             if os.path.exists(f"{store_name}.pkl"):
               with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
             # st.write('Embeddings Loaded from the Disk')s
             else:
              embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
              VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
              with open(f"{store_name}.pkl", "wb") as f:
                     pickle.dump(VectorStore, f)
       
       # Accept user questions/query
      query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
      if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI(openai_api_key=openai_api_key)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)              
  
 
    else:
       st.warning('Please insert OpenAI API Key. Instructions [here](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)', icon="⚠️")
       st.stop() 
 
 
 
 
 
if __name__ == '__main__':
    main()
 