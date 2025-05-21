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
from audio_recorder_streamlit import audio_recorder

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
def generate_audio_elevenlabs(text_to_speak, api_key, voice_id="JBFqnCBsd6RMkjVDRZzb"):
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
    # Initialize session state variables
    if 'query' not in st.session_state:
        st.session_state.query = ""
    if 'response' not in st.session_state:
        st.session_state.response = ""
    if 'audio_response' not in st.session_state:
        st.session_state.audio_response = None
    if 'show_response' not in st.session_state:
        st.session_state.show_response = False
    if 'text_output_enabled' not in st.session_state:
        st.session_state.text_output_enabled = True
    if 'voice_output_enabled' not in st.session_state:
        st.session_state.voice_output_enabled = True

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

    # Output Options Section in Sidebar
    st.sidebar.markdown("### Output Options")
    st.session_state.text_output_enabled = st.sidebar.checkbox(
        "Enable Text Output", 
        value=st.session_state.text_output_enabled
    )
    
    voice_output_disabled = not bool(elevenlabs_api_key)
    tooltip_message = "ElevenLabs API Key is required to enable voice output." if voice_output_disabled else ""
    st.session_state.voice_output_enabled = st.sidebar.checkbox(
        "Enable Voice Output", 
        value=st.session_state.voice_output_enabled, 
        disabled=voice_output_disabled,
        help=tooltip_message
    )

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

        # Ensure query_text_input is initialized in session state
        if 'query_text_input' not in st.session_state:
            st.session_state.query_text_input = ""

        # Text input for query
        st.text_input(
            "Ask questions about your PDF file (text):",
            key="query_text_input_widget",  # Widget's own key
            value=st.session_state.query_text_input, # Controlled by this session state variable
            on_change=lambda: setattr(st.session_state, 'query_text_input', st.session_state.query_text_input_widget)
        )

        # Button to submit text query
        if st.button("Submit Text Query", key="submit_text_query_button"):
            if st.session_state.query_text_input.strip():
                st.session_state.query = st.session_state.query_text_input.strip()
                # query_text_input will be cleared after query processing
            else:
                st.warning("Please enter a question in the text box.")

        # Voice Query Section
        if deepgram_api_key:
            st.subheader("Voice Query")
            add_vertical_space(1)
            recorded_audio = audio_recorder("Click to record", "Click to stop recording", key="audio_recorder_widget")
            add_vertical_space(1)

            if recorded_audio is not None and len(recorded_audio) > 0:
                st.audio(recorded_audio, format="audio/wav")
                audio_buffer = recorded_audio
                transcribed_query = transcribe_audio_deepgram(audio_buffer, deepgram_api_key)
                if transcribed_query:
                    st.info(f"Transcribed Query (from recording): {transcribed_query}")
                    st.session_state.query = transcribed_query
                    st.session_state.query_text_input = ""  # Clear text input field as voice query takes precedence
                else:
                    st.error("Could not transcribe recorded audio. Please try again or use another input method.")
        
        # Main query processing logic
        if st.session_state.query:  # Proceed only if a query exists
            if uploaded_pdf is None:
                st.warning("Please upload a PDF file to ask questions about it.")
                st.session_state.query = "" # Clear query as it cannot be processed
                st.session_state.query_text_input = "" # Clear input field too
            else:
                st.divider()
                current_query_to_process = st.session_state.query # Use the active query
                
                # Initialize response states for the current query
                st.session_state.response = ""
                st.session_state.audio_response = None
                st.session_state.show_response = True # Assume we will show something (response or error)
                
                file_path = save_uploaded_pdf(uploaded_pdf)
                if file_path:
                    try:
                        pages = get_pdf_text(file_path)
                        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
                        VectorStore = FAISS.from_documents(pages, embedding=embeddings)
                        docs = VectorStore.similarity_search(query=current_query_to_process, k=3)
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", google_api_key=google_api_key, convert_system_message_to_human=True)
                        chain = load_qa_chain(llm=llm, chain_type="stuff")
                        
                        # Actual QA execution
                        response_text = chain.run(input_documents=docs, question=current_query_to_process)
                        st.session_state.response = response_text # Store text response
                        
                        # Generate audio response if enabled and API key exists and there's a response
                        if st.session_state.voice_output_enabled and elevenlabs_api_key and st.session_state.response:
                            st.session_state.audio_response = generate_audio_elevenlabs(st.session_state.response, elevenlabs_api_key)
                        # generate_audio_elevenlabs handles its own errors by returning None and displaying st.error

                    except Exception as e:
                        st.error(f"An error occurred during PDF processing or QA: {e}")
                        st.session_state.response = f"Error: An error occurred during PDF processing or QA. Details: {e}" # Store error message
                        st.session_state.audio_response = None # Ensure no audio response on error
                        if "Google" in str(e) or "gemini" in str(e).lower():
                             st.error("Please ensure your Google API key is correct, has the 'Generative Language API' enabled in your Google Cloud console, and that your account has billing enabled.")
                
                # Clear the active query and the text input field *after* processing is complete
                st.session_state.query = ""
                st.session_state.query_text_input = "" 
        
        # Conditional display of "Please upload a PDF file to begin."
        # This should only show if no PDF is uploaded AND no response is currently being shown.
        if uploaded_pdf is None and not st.session_state.show_response:
             st.info("Please upload a PDF file to begin.")

        # Response display logic (to be fully refactored in the next subtask)
        if st.session_state.show_response:
            if st.session_state.text_output_enabled and st.session_state.response:
                st.write("### Response")
                st.write(st.session_state.response)
            
            if st.session_state.voice_output_enabled and st.session_state.audio_response: # This variable would be set by response logic
                st.audio(st.session_state.audio_response, format="audio/mpeg")
            elif st.session_state.voice_output_enabled and not st.session_state.audio_response and st.session_state.response and not ("Error:" in st.session_state.response):
                 # This warning logic also belongs to response handling
                st.warning("Voice output is enabled, but could not generate audio. Check ElevenLabs API key and status.")

            if st.button("Clear Chat", key="clear_response_button"): # This button belongs to response handling
                st.session_state.query = ""  # Clear any active query
                st.session_state.response = ""  # Clear previous text response
                st.session_state.audio_response = None  # Clear previous audio response
                st.session_state.show_response = False  # Hide the response area
                st.session_state.query_text_input = ""  # Clear the text input field
                st.session_state.audio_recorder_widget = None # Clear audio recorder state
                # Note: query_text_input_widget will be updated by Streamlit due to binding with query_text_input
                st.rerun() 

if __name__ == '__main__':
    main()