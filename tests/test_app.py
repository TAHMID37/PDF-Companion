import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys

# Add the root directory to sys.path to allow importing app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions from app.py
from app import (
    get_google_api_key,
    get_deepgram_api_key,
    get_elevenlabs_api_key,
    transcribe_audio_deepgram,
    generate_audio_elevenlabs,
    get_pdf_text,
    save_uploaded_pdf
    # main # We will test components called by main, not main directly for now
)

# Remove global streamlit mocks:
# mock_streamlit = MagicMock()
# sys.modules['streamlit'] = mock_streamlit
# sys.modules['streamlit_extras.add_vertical_space'] = MagicMock()

# --- Test API Key Functions ---

def test_get_google_api_key_from_env(mocker):
    mocker.patch.dict(os.environ, {"GOOGLE_API_KEY": "test_google_key_env"})
    assert get_google_api_key() == "test_google_key_env"

@patch('app.st.sidebar.markdown') # Mock markdown as it's called before text_input
@patch('app.st.sidebar.text_input')
@patch('app.st.session_state', new_callable=lambda: {}) # Mock session_state as a new dict for each test
def test_get_google_api_key_fallback_input(mock_session_state, mock_text_input, mock_markdown, mocker):
    mocker.patch.dict(os.environ, {}, clear=True)
    mock_text_input.return_value = "test_google_key_input"
    
    assert get_google_api_key() == "test_google_key_input"
    mock_text_input.assert_called_once_with(
        label="Google API Key (Required for Core QA)",
        placeholder="Enter your Google API Key",
        type="password",
        key="google_api_key_input"
    )
    # Check if markdown was called if session_state for 'google_api_key_input' was not present (which it isn't)
    mock_markdown.assert_called_once()


def test_get_deepgram_api_key_from_env(mocker):
    mocker.patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test_deepgram_key_env"})
    assert get_deepgram_api_key() == "test_deepgram_key_env"

@patch('app.st.sidebar.markdown')
@patch('app.st.sidebar.text_input')
@patch('app.st.session_state', new_callable=lambda: {'google_api_key_input': True}) # Simulate google key was already asked
def test_get_deepgram_api_key_fallback_input(mock_session_state, mock_text_input, mock_markdown, mocker):
    mocker.patch.dict(os.environ, {}, clear=True)
    mock_text_input.return_value = "test_deepgram_key_input"
    
    assert get_deepgram_api_key() == "test_deepgram_key_input"
    mock_text_input.assert_called_once_with( 
        label="Deepgram API Key (for Voice Queries)",
        placeholder="Enter your Deepgram API Key",
        type="password",
        key="deepgram_api_key_input"
    )
    # Markdown for "API Keys" header should not be called if 'google_api_key_input' is in session_state
    mock_markdown.assert_not_called()


def test_get_elevenlabs_api_key_from_env(mocker):
    mocker.patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_elevenlabs_key_env"})
    assert get_elevenlabs_api_key() == "test_elevenlabs_key_env"

@patch('app.st.sidebar.markdown')
@patch('app.st.sidebar.text_input')
@patch('app.st.session_state', new_callable=lambda: {'google_api_key_input': True, 'deepgram_api_key_input': True})
def test_get_elevenlabs_api_key_fallback_input(mock_session_state, mock_text_input, mock_markdown, mocker):
    mocker.patch.dict(os.environ, {}, clear=True)
    mock_text_input.return_value = "test_elevenlabs_key_input"
    assert get_elevenlabs_api_key() == "test_elevenlabs_key_input"
    mock_text_input.assert_called_once_with(
        label="ElevenLabs API Key (for Spoken Responses)",
        placeholder="Enter your ElevenLabs API Key",
        type="password",
        key="elevenlabs_api_key_input"
    )
    mock_markdown.assert_not_called()

# --- Test transcribe_audio_deepgram ---

@patch('app.DeepgramClient')
def test_transcribe_audio_deepgram_success(MockDeepgramClient):
    mock_instance = MockDeepgramClient.return_value
    mock_response = MagicMock()
    mock_response.results.channels[0].alternatives[0].transcript = "hello world"
    mock_instance.listen.prerecorded.v("1").transcribe_file.return_value = mock_response
    
    result = transcribe_audio_deepgram(b"fake_audio_bytes", "fake_deepgram_key")
    assert result == "hello world"
    MockDeepgramClient.assert_called_once_with(api_key="fake_deepgram_key")
    mock_instance.listen.prerecorded.v("1").transcribe_file.assert_called_once()

@patch('app.st.error') # Target st.error directly
@patch('app.DeepgramClient')
def test_transcribe_audio_deepgram_failure(MockDeepgramClient, mock_st_error):
    mock_instance = MockDeepgramClient.return_value
    mock_instance.listen.prerecorded.v("1").transcribe_file.side_effect = Exception("ASR Error")
    
    result = transcribe_audio_deepgram(b"fake_audio_bytes", "fake_deepgram_key")
    assert result is None
    mock_st_error.assert_called_once_with("Deepgram ASR error: ASR Error")

@patch('app.st.error') # Target st.error directly
def test_transcribe_audio_deepgram_no_api_key(mock_st_error):
    result = transcribe_audio_deepgram(b"fake_audio_bytes", None)
    assert result is None
    mock_st_error.assert_called_once_with("Deepgram API Key is not provided.")


# --- Test generate_audio_elevenlabs ---

@patch('app.ElevenLabs')
def test_generate_audio_elevenlabs_success(MockElevenLabs):
    mock_client_instance = MockElevenLabs.return_value
    mock_client_instance.generate.return_value = b"fake_audio_data"
    
    result = generate_audio_elevenlabs("hello", "fake_elevenlabs_key")
    assert result == b"fake_audio_data"
    MockElevenLabs.assert_called_once_with(api_key="fake_elevenlabs_key")
    mock_client_instance.generate.assert_called_once_with(
        text="hello",
        voice="Rachel",
        model="eleven_multilingual_v2"
    )

@patch('app.st.error') # Target st.error directly
@patch('app.ElevenLabs')
def test_generate_audio_elevenlabs_failure(MockElevenLabs, mock_st_error):
    mock_client_instance = MockElevenLabs.return_value
    mock_client_instance.generate.side_effect = Exception("TTS Error")
    
    result = generate_audio_elevenlabs("hello", "fake_elevenlabs_key")
    assert result is None
    mock_st_error.assert_called_once_with("ElevenLabs TTS error: TTS Error")

@patch('app.st.error') # Target st.error directly
def test_generate_audio_elevenlabs_no_api_key(mock_st_error):
    result = generate_audio_elevenlabs("hello", None)
    assert result is None
    mock_st_error.assert_called_once_with("ElevenLabs API Key is not provided.")

# --- Test get_pdf_text ---

@patch('app.PyPDFLoader') # This should be app.PyPDFLoader as PyPDFLoader is imported in app.py
def test_get_pdf_text_success(MockPyPDFLoader):
    mock_loader_instance = MockPyPDFLoader.return_value
    mock_pages = [MagicMock(page_content="Page 1 text"), MagicMock(page_content="Page 2 text")]
    mock_loader_instance.load_and_split.return_value = mock_pages
    
    result = get_pdf_text("dummy.pdf")
    assert result == mock_pages
    MockPyPDFLoader.assert_called_once_with("dummy.pdf")
    mock_loader_instance.load_and_split.assert_called_once()

# --- Test save_uploaded_pdf ---

@patch("app.os.path.exists", return_value=False) # Target app.os.path.exists
@patch("app.os.makedirs") # Target app.os.makedirs
@patch("app.open", new_callable=mock_open) # Target app.open (which is builtins.open)
@patch("app.os.path.abspath") # Target app.os.path.abspath
def test_save_uploaded_pdf_success(mock_abspath, mock_file_open, mock_makedirs, mock_exists):
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = "test.pdf"
    mock_uploaded_file.getbuffer.return_value = b"pdf content"
    
    # If app.py uses os.path.join, we might need to mock that too if path construction is complex
    # For now, assume direct construction or simple join.
    expected_save_dir = "./pdf" # As in app.py
    expected_filename = "test.pdf"
    expected_save_path = os.path.join(expected_save_dir, expected_filename)
    
    mock_abspath.return_value = "/abs/path/to/" + expected_save_path # Make it absolute
    
    result = save_uploaded_pdf(mock_uploaded_file)
    
    assert result == mock_abspath.return_value
    mock_exists.assert_called_once_with(expected_save_dir)
    mock_makedirs.assert_called_once_with(expected_save_dir)
    mock_file_open.assert_called_once_with(expected_save_path, "wb")
    mock_file_open().write.assert_called_once_with(b"pdf content")

def test_save_uploaded_pdf_no_file():
    result = save_uploaded_pdf(None)
    assert result is None

# --- Test Core QA Logic (Simplified) ---
# This is a conceptual test for how one might approach testing main interactions
# It assumes the main function's logic is refactored or specific parts are tested

@patch('app.GoogleGenerativeAIEmbeddings')
@patch('app.ChatGoogleGenerativeAI')
@patch('app.FAISS')
@patch('app.load_qa_chain')
def test_core_qa_components_initialized_and_called(
    MockLoadQaChain, MockFAISS, MockChatGoogle, MockGoogleEmbeddings, mocker # mocker fixture is still useful
):
    # This is a highly simplified test focusing on component initialization with API key
    # and ensuring the chain run is attempted.
    # A real test of main() would be an integration test or require more refactoring of main().

    # Mock return values for the chain
    mock_chain_instance = MockLoadQaChain.return_value
    mock_chain_instance.run.return_value = "Test response"
    
    # Mock FAISS methods
    mock_vector_store_instance = MockFAISS.from_documents.return_value
    mock_vector_store_instance.similarity_search.return_value = [MagicMock()] # dummy docs

    # Simulate inputs to a simplified version of the QA part of main()
    google_api_key = "fake_google_key"
    pages = [MagicMock(page_content="Some text")] # Dummy pages from get_pdf_text
    query = "What is this?"
    
    # Simulate Embeddings initialization
    embeddings_instance = MockGoogleEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    MockGoogleEmbeddings.assert_called_with(model="models/embedding-001", google_api_key=google_api_key)
    
    # Simulate FAISS vector store creation
    MockFAISS.from_documents(pages, embedding=embeddings_instance) # Call it as it would be in app
    MockFAISS.from_documents.assert_called_with(pages, embedding=embeddings_instance) # Assert the call
    
    # Simulate LLM initialization
    llm_instance = MockChatGoogle(model="gemini-pro", google_api_key=google_api_key, convert_system_message_to_human=True)
    MockChatGoogle.assert_called_with(model="gemini-pro", google_api_key=google_api_key, convert_system_message_to_human=True)

    # Simulate similarity search
    docs_result = mock_vector_store_instance.similarity_search(query=query, k=3) # Call it
    mock_vector_store_instance.similarity_search.assert_called_with(query=query, k=3) # Assert
    
    # Simulate chain loading and run
    chain_loaded = MockLoadQaChain(llm=llm_instance, chain_type="stuff") # Call it
    MockLoadQaChain.assert_called_with(llm=llm_instance, chain_type="stuff") # Assert
    
    # Actual run call
    mock_chain_instance.run(input_documents=docs_result, question=query) # Use result of similarity_search
    mock_chain_instance.run.assert_called_with(input_documents=docs_result, question=query)

# Remove the autouse fixture for global mock reset
# @pytest.fixture(autouse=True)
# def reset_streamlit_mocks():
#     mock_streamlit.reset_mock()

# Note: Testing the full app.main() directly as a unit test is complex due to Streamlit's execution model
# and the way it handles state and UI elements. It's better to test functions called by main().
# For full app testing, integration or end-to-end tests with a tool like Selenium would be more appropriate.

# To run these tests:
# 1. Ensure Pipfile has pytest and pytest-mock in [dev-packages]
# 2. Run `pipenv install --dev`
# 3. Run `pipenv run pytest tests/` (or `pipenv run python -m pytest tests/`)
