import os
import pickle
import streamlit as st
from dotenv import load_dotenv
# Removed: from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
# Removed: from langchain.embeddings.openai import OpenAIEmbeddings
# Removed: from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # Updated import
from PyPDF2 import PdfReader # This import might be unused if get_pdf_text is solely using PyPDFLoader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_community.vectorstores import Chroma # Updated import
from langchain_community.document_loaders import PyPDFLoader # Updated import
# Removed: from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
from elevenlabs.client import ElevenLabs

load_dotenv() # Load .env file if present

# Set the page configuration
st.set_page_config(page_title="PDF Companion", page_icon=":robot:")

# Sidebar contents
with st.sidebar:
    st.title('📚🔍 PDF Companion')
    st.markdown('''
    ## About
    "PDF Companion" is an advanced PDF reader that:
    - Answers your questions about PDF content using **Google's Generative AI**.
    - Allows you to ask questions via voice using **Deepgram's ASR**.
    - Provides spoken responses to your queries with **ElevenLabs TTS**.

    To unlock full functionality, please provide API keys for these services in the designated "API Keys" section below.
    ''')
    add_vertical_space(5)
    st.write('Made with 💻❤️ by Tahmid')

# Removed get_openai_api_key() function

def get_google_api_key():
    env_key = os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key
    else:
        st.sidebar.markdown("### API Keys") # This will be the first, so no conditional check needed now
        input_text = st.sidebar.text_input(
            label="Google API Key (Required for Core QA)",
            placeholder="Enter your Google API Key",
            type="password",
            key="google_api_key_input"
        )
        return input_text

def get_deepgram_api_key():
    env_key = os.getenv("DEEPGRAM_API_KEY")
    if env_key:
        return env_key
    else:
        # Ensure "API Keys" heading is only added once by checking if google key input was rendered
        if "google_api_key_input" not in st.session_state:
            st.sidebar.markdown("### API Keys")
        input_text = st.sidebar.text_input(
            label="Deepgram API Key (for Voice Queries)",
            placeholder="Enter your Deepgram API Key",
            type="password",
            key="deepgram_api_key_input"
        )
        return input_text

def get_elevenlabs_api_key():
    env_key = os.getenv("ELEVENLABS_API_KEY")
    if env_key:
        return env_key
    else:
        # Ensure "API Keys" heading is only added once
        if "google_api_key_input" not in st.session_state and "deepgram_api_key_input" not in st.session_state:
            st.sidebar.markdown("### API Keys")
        input_text = st.sidebar.text_input(
            label="ElevenLabs API Key (for Spoken Responses)",
            placeholder="Enter your ElevenLabs API Key",
            type="password",
            key="elevenlabs_api_key_input"
        )
        return input_text

# Function to transcribe audio using Deepgram
def transcribe_audio_deepgram(audio_file_buffer, api_key):
    if not api_key:
        st.error("Deepgram API Key is not provided.")
        return None
    try:
        dg_client = DeepgramClient(api_key=api_key) # Explicitly use keyword argument
        payload: FileSource = {
            "buffer": audio_file_buffer,
        }
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )
        response = dg_client.listen.prerecorded.v("1").transcribe_file(payload, options)
        transcript = response.results.channels[0].alternatives[0].transcript
        return transcript if transcript else None # Return None if transcript is empty
    except Exception as e:
        st.error(f"Deepgram ASR error: {e}")
        return None

# Function to generate audio using ElevenLabs
def generate_audio_elevenlabs(text_to_speak, api_key, voice_id="Rachel"):
    if not api_key:
        st.error("ElevenLabs API Key is not provided.")
        return None
    try:
        client = ElevenLabs(api_key=api_key)
        audio_bytes = client.generate(
            text=text_to_speak,
            voice=voice_id,
            model="eleven_multilingual_v2" # Or your preferred model
        )
        return audio_bytes
    except Exception as e:
        st.error(f"ElevenLabs TTS error: {e}")
        return None

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

    # Get API Keys
    # openai_api_key = get_openai_api_key() # Removed
    google_api_key = get_google_api_key()
    deepgram_api_key = get_deepgram_api_key()
    elevenlabs_api_key = get_elevenlabs_api_key()

    # Display warnings for optional keys if missing (sidebar already handles input prompt)
    if not deepgram_api_key:
        st.sidebar.warning("Deepgram API Key not provided. Voice query feature will be disabled.", icon="⚠️")
    if not elevenlabs_api_key:
        st.sidebar.warning("ElevenLabs API Key not provided. Spoken response feature will be disabled.", icon="⚠️")

    # Core application logic - gated by Google API Key
    if not google_api_key:
        st.error("Google API Key is required for the core PDF Question Answering functionality. Please provide it in the sidebar under 'API Keys'.")
        st.info("Once the Google API Key is provided, you will be able to upload PDFs and ask questions.")
        # Optionally, you can disable other UI elements here or just let them not appear
        # For example, the file uploader will not be shown if we stop here or gate its display
    else:
        # upload a PDF file
        uploaded_pdf = st.file_uploader("Upload your PDF", type='pdf')

        # Initialize query variable
        query = ""

        # Text input for query
        query_text = st.text_input("Ask questions about your PDF file (text):")

        # Audio input for query
        uploaded_audio_file = None
        if deepgram_api_key:
            uploaded_audio_file = st.file_uploader(
                "Or ask by uploading an audio file (e.g., WAV, MP3, M4A):",
                type=['wav', 'mp3', 'm4a']
            )
        # No "else st.info" here, as the sidebar warning for Deepgram key is sufficient.
        # The uploader simply won't appear if key is missing.

        if uploaded_audio_file and deepgram_api_key: # Process if audio uploaded and key exists
            st.audio(uploaded_audio_file, format=uploaded_audio_file.type)
            audio_buffer = uploaded_audio_file.getbuffer()
            transcribed_query = transcribe_audio_deepgram(audio_buffer, deepgram_api_key)
            if transcribed_query:
                st.info(f"Transcribed Query: {transcribed_query}")
                query = transcribed_query  # Use transcribed query
            else:
                st.error("Could not transcribe audio. Please use text input or try another audio file.")
                query = "" # Clear query if transcription failed
        elif query_text:
            query = query_text # Use text query if audio not used or failed

        if uploaded_pdf is None:
            st.info("Please upload a PDF file to begin.")
        elif query: # Only proceed if PDF is uploaded AND a query exists
            file_path = save_uploaded_pdf(uploaded_pdf)
            if file_path: # Ensure PDF was saved correctly
                try:
                    pages = get_pdf_text(file_path) # Attempt to get PDF text first
                    
                    # All subsequent processing should be within this try block
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
                    VectorStore = FAISS.from_documents(pages, embedding=embeddings)
                    
                    docs = VectorStore.similarity_search(query=query, k=3)
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", google_api_key=google_api_key, convert_system_message_to_human=True)
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=query)
                    st.write(response)

                    # ElevenLabs TTS Integration
                    if elevenlabs_api_key:
                        if st.button("🔊 Play Response Audio"):
                            if response: # Ensure there is a response to play
                                audio_data = generate_audio_elevenlabs(response, elevenlabs_api_key)
                                if audio_data:
                                    st.audio(audio_data, format="audio/mpeg")
                                else:
                                    st.error("Could not generate audio for the response.")
                            else:
                                st.info("No response text to play.")
                    # Removed the 'else' that showed "Provide an ElevenLabs API Key..." here, 
                    # as it's already handled by the general API key check section.
                
                except Exception as e:
                    st.error(f"An error occurred during PDF processing or QA: {e}")
                    if "Google" in str(e) or "gemini" in str(e).lower(): # More specific error for Google
                        st.error("Please ensure your Google API key is correct, has the 'Generative Language API' enabled in your Google Cloud console, and that your account has billing enabled.")
                    st.info("For more information on setting up your Google API key, visit: https://ai.google.dev/documentation/text_setup")
                # No specific error message for ElevenLabs here as generate_audio_elevenlabs handles its own errors.

if __name__ == '__main__':
    main()