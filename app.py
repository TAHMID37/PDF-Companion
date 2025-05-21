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
import elevenlabs # Modified import
from elevenlabs.client import ElevenLabs
from streamlit_audiorecorder import audiorecorder # Added import

load_dotenv() # Load .env file if present

# Print ElevenLabs version for debugging
try:
    print(f"DEBUG: elevenlabs.__version__: {elevenlabs.__version__}")
except AttributeError:
    print("DEBUG: elevenlabs.__version__ could not be accessed. The 'elevenlabs' import might not be the package itself.")


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
        # Removed conditional markdown for API Keys header
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
        # Removed conditional markdown for API Keys header
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
        # Removed conditional markdown for API Keys header
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
        audio_bytes_iter = client.text_to_speech.convert(
            voice_id=voice_id, # Changed from voice to voice_id
            text=text_to_speak,
            model_id="eleven_multilingual_v2" # Changed from model to model_id
        )
        # The convert method returns an iterator of audio chunks.
        # We need to concatenate them to get the full audio data.
        audio_bytes_list = []
        for chunk in audio_bytes_iter:
            if chunk:
                audio_bytes_list.append(chunk)
        
        if not audio_bytes_list:
            st.error("ElevenLabs TTS error: No audio data returned.")
            return None
            
        audio_bytes = b"".join(audio_bytes_list)
        return audio_bytes
    except AttributeError as ae:
        st.error(f"ElevenLabs TTS AttributeError: {ae}")
        # Removed detailed debug prints, but keeping the error message.
        return None
    except Exception as e:
        st.error(f"ElevenLabs TTS error (non-AttributeError): {e}")
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

    # API Keys Section in Sidebar
    st.sidebar.markdown("### API Keys") # Added once before all key inputs

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
        st.divider() # Added divider after PDF uploader

        # Initialize query variable
        query = ""

        # Text input for query
        query_text = st.text_input("Ask questions about your PDF file (text):")

        # Voice Query Section
        if deepgram_api_key:
            st.subheader("Voice Query")
            add_vertical_space(1) # Added space
            
            # Option 1: Record Audio
            st.markdown("#### Record Audio:")
            recorded_audio = audiorecorder("Click to record", "Click to stop recording")
            add_vertical_space(1) # Added space
            
            # Option 2: Upload Audio File
            st.markdown("#### Or Upload Audio File:")
            uploaded_audio_file = st.file_uploader(
                "Upload an audio file (e.g., WAV, MP3, M4A):",
                type=['wav', 'mp3', 'm4a'],
                label_visibility="collapsed" 
            )

            if recorded_audio is not None and len(recorded_audio) > 0:
                st.audio(recorded_audio, format="audio/wav")
                audio_buffer = recorded_audio # audiorecorder returns bytes
                transcribed_query = transcribe_audio_deepgram(audio_buffer, deepgram_api_key)
                if transcribed_query:
                    st.info(f"Transcribed Query (from recording): {transcribed_query}")
                    query = transcribed_query
                    query_text = "" # Clear text input
                    uploaded_audio_file = None # Clear file uploader
                else:
                    st.error("Could not transcribe recorded audio. Please try again or use another input method.")
                    query = "" 
            elif uploaded_audio_file is not None:
                st.audio(uploaded_audio_file, format=uploaded_audio_file.type)
                audio_buffer = uploaded_audio_file.getbuffer()
                transcribed_query = transcribe_audio_deepgram(audio_buffer, deepgram_api_key)
                if transcribed_query:
                    st.info(f"Transcribed Query (from file): {transcribed_query}")
                    query = transcribed_query
                    query_text = "" # Clear text input
                else:
                    st.error("Could not transcribe uploaded audio. Please use text input or try another audio file.")
                    query = ""
            # If both are provided, recorded_audio takes precedence due to order of checks
            
        if query_text and not query: # Only use text input if no voice query was processed
            query = query_text

        if uploaded_pdf is None:
            st.info("Please upload a PDF file to begin.")
        elif query: # Only proceed if PDF is uploaded AND a query exists
            st.divider() # Added divider before displaying response
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
                    if elevenlabs_api_key and response: # Check for API key and response
                        audio_data = generate_audio_elevenlabs(response, elevenlabs_api_key)
                        if audio_data:
                            st.audio(audio_data, format="audio/mpeg")
                        else:
                            st.error("Could not generate audio for the response.")
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