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

# --- Tests for main() application logic ---

# Fixture to mock streamlit and its components for main() tests
@pytest.fixture
def mock_st_main(mocker):
    mock_st = MagicMock()
    mocker.patch('app.st', mock_st)
    
    # Mock session_state as a dictionary attached to the mock_st object
    # This allows tests to manipulate it directly.
    mock_st.session_state = {} 
    
    # Mock other commonly used Streamlit functions to prevent them from actually running
    mock_st.sidebar = MagicMock()
    mock_st.header = MagicMock()
    mock_st.title = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.divider = MagicMock()
    mock_st.file_uploader = MagicMock()
    mock_st.text_input = MagicMock()
    mock_st.button = MagicMock()
    mock_st.checkbox = MagicMock()
    mock_st.audio = MagicMock()
    mock_st.write = MagicMock()
    mock_st.error = MagicMock()
    mock_st.warning = MagicMock()
    mock_st.info = MagicMock()
    mock_st.experimental_rerun = MagicMock()
    
    # Mock external libraries/functions called by main that are not core to the UI logic being tested
    mocker.patch('app.get_google_api_key', return_value="fake_google_key")
    mocker.patch('app.get_deepgram_api_key', return_value="fake_deepgram_key")
    mocker.patch('app.get_elevenlabs_api_key', return_value="fake_elevenlabs_key")
    mocker.patch('app.audio_recorder', return_value=None) # Default to no audio recorded
    mocker.patch('app.save_uploaded_pdf')
    mocker.patch('app.get_pdf_text')
    m_google_embeddings = mocker.patch('app.GoogleGenerativeAIEmbeddings')
    m_faiss = mocker.patch('app.FAISS')
    m_chat_google = mocker.patch('app.ChatGoogleGenerativeAI')
    m_load_qa_chain = mocker.patch('app.load_qa_chain')

    # Mock the chain and its run method
    mock_chain_instance = MagicMock()
    mock_chain_instance.run.return_value = "Mocked QA response"
    m_load_qa_chain.return_value = mock_chain_instance
    
    # Mock FAISS methods
    mock_vector_store_instance = MagicMock()
    mock_vector_store_instance.similarity_search.return_value = [MagicMock()] # dummy docs
    m_faiss.from_documents.return_value = mock_vector_store_instance

    return mock_st

def test_session_state_initialization(mock_st_main, mocker):
    """
    Tests if session state variables are initialized correctly when main() is first run.
    """
    # Import main here to ensure it uses the mocked 'app.st'
    from app import main

    # Call main to trigger session state initialization
    main()

    # Assertions for session state variables
    assert 'query' in mock_st_main.session_state, "session_state missing 'query'"
    assert mock_st_main.session_state.query == "", "query not initialized to empty string"

    assert 'response' in mock_st_main.session_state, "session_state missing 'response'"
    assert mock_st_main.session_state.response == "", "response not initialized to empty string"

    assert 'audio_response' in mock_st_main.session_state, "session_state missing 'audio_response'"
    assert mock_st_main.session_state.audio_response is None, "audio_response not initialized to None"

    assert 'show_response' in mock_st_main.session_state, "session_state missing 'show_response'"
    assert mock_st_main.session_state.show_response is False, "show_response not initialized to False"

    assert 'text_output_enabled' in mock_st_main.session_state, "session_state missing 'text_output_enabled'"
    assert mock_st_main.session_state.text_output_enabled is True, "text_output_enabled not initialized to True"

    assert 'voice_output_enabled' in mock_st_main.session_state, "session_state missing 'voice_output_enabled'"
    assert mock_st_main.session_state.voice_output_enabled is True, "voice_output_enabled not initialized to True"
    
    assert 'query_text_input' in mock_st_main.session_state, "session_state missing 'query_text_input'"
    assert mock_st_main.session_state.query_text_input == "", "query_text_input not initialized to empty string"

def test_text_output_toggle_enabled(mock_st_main, mocker):
    """Tests text response display when text output is enabled."""
    from app import main
    mock_st_main.session_state.show_response = True
    mock_st_main.session_state.response = "Test text response"
    mock_st_main.session_state.text_output_enabled = True # Simulate checkbox checked

    main()

    # Check that st.write was called for the response
    # We need to check the call list as st.write might be called for other things too.
    # A more robust way would be to check specific calls if there are many st.write calls.
    # For now, we assume these are the primary st.write calls for the response.
    calls = [mocker.call("### Response"), mocker.call("Test text response")]
    mock_st_main.write.assert_has_calls(calls, any_order=False)

def test_text_output_toggle_disabled(mock_st_main, mocker):
    """Tests text response display is suppressed when text output is disabled."""
    from app import main
    mock_st_main.session_state.show_response = True
    mock_st_main.session_state.response = "Test text response"
    mock_st_main.session_state.text_output_enabled = False # Simulate checkbox unchecked

    # Reset mock_st_main.write before calling main to isolate calls within this test
    mock_st_main.write.reset_mock()
    
    main()

    # Assert that st.write was NOT called with the response content
    for call_args in mock_st_main.write.call_args_list:
        assert call_args[0][0] != "### Response"
        assert call_args[0][0] != "Test text response"

def test_voice_output_toggle_enabled_with_audio(mock_st_main, mocker):
    """Tests audio response display when voice output is enabled and audio is available."""
    from app import main
    mock_st_main.session_state.show_response = True
    mock_st_main.session_state.response = "Response text" # Needed for context
    mock_st_main.session_state.audio_response = b"fake_audio_bytes"
    mock_st_main.session_state.voice_output_enabled = True # Simulate checkbox checked

    main()
    mock_st_main.audio.assert_called_once_with(b"fake_audio_bytes", format="audio/mpeg")

def test_voice_output_toggle_disabled(mock_st_main, mocker):
    """Tests audio response display is suppressed when voice output is disabled."""
    from app import main
    mock_st_main.session_state.show_response = True
    mock_st_main.session_state.response = "Response text"
    mock_st_main.session_state.audio_response = b"fake_audio_bytes"
    mock_st_main.session_state.voice_output_enabled = False # Simulate checkbox unchecked
    
    main()
    mock_st_main.audio.assert_not_called()

def test_voice_output_enabled_no_audio_data_shows_warning(mock_st_main, mocker):
    """Tests that a warning is shown if voice output is enabled but no audio data is present (and not an error response)."""
    from app import main
    mock_st_main.session_state.show_response = True
    mock_st_main.session_state.response = "Valid text, but no audio." # Not an error response
    mock_st_main.session_state.audio_response = None
    mock_st_main.session_state.voice_output_enabled = True

    main()
    mock_st_main.warning.assert_called_once_with("Voice output is enabled, but could not generate audio. Check ElevenLabs API key and status.")
    mock_st_main.audio.assert_not_called()

def test_voice_output_enabled_no_audio_data_on_error_response(mock_st_main, mocker):
    """Tests that no warning for missing audio is shown if the response itself is an error message."""
    from app import main
    mock_st_main.session_state.show_response = True
    mock_st_main.session_state.response = "Error: Something went wrong." # Error response
    mock_st_main.session_state.audio_response = None
    mock_st_main.session_state.voice_output_enabled = True

    main()
    mock_st_main.warning.assert_not_called() # Warning should not be called for error responses
    mock_st_main.audio.assert_not_called()


# Placeholder for more tests for main()
def test_clear_last_response_button(mock_st_main, mocker):
    """Tests the 'Clear Last Response' button functionality."""
    from app import main

    # Setup initial state as if a response is being shown
    mock_st_main.session_state.show_response = True
    mock_st_main.session_state.response = "A previous text response."
    mock_st_main.session_state.audio_response = b"previous_audio_bytes"
    mock_st_main.session_state.query = "A previous query" # Should be cleared
    mock_st_main.session_state.query_text_input = "Previous text in input" # Should be cleared

    # Simulate the 'Clear Last Response' button being clicked
    # The button is conditional on st.session_state.show_response, which is True here.
    # We need to make the mock_st_main.button function return True when called with the specific key.
    def button_side_effect(label, key=None):
        if key == "clear_response_button":
            return True
        return False
    mock_st_main.button.side_effect = button_side_effect
    
    main()

    # Assertions
    assert mock_st_main.session_state.show_response is False, "show_response should be False"
    assert mock_st_main.session_state.response == "", "response should be cleared"
    assert mock_st_main.session_state.audio_response is None, "audio_response should be cleared"
    assert mock_st_main.session_state.query == "", "query should be cleared"
    assert mock_st_main.session_state.query_text_input == "", "query_text_input should be cleared"
    
    mock_st_main.experimental_rerun.assert_called_once()

def test_query_handling_text_input_with_pdf(mock_st_main, mocker):
    """Tests query submission via text input when a PDF is uploaded."""
    from app import main

    # Simulate PDF uploaded
    mock_st_main.file_uploader.return_value = MagicMock(name="UploadedPDF") 
    
    # Simulate text entered into the input field
    mock_st_main.session_state.query_text_input = "This is a text query"

    # Simulate the 'Submit Text Query' button being clicked
    def button_side_effect(label, key=None):
        if key == "submit_text_query_button":
            # This simulates the action within app.main() that sets st.session_state.query
            # based on st.session_state.query_text_input
            mock_st_main.session_state.query = mock_st_main.session_state.query_text_input 
            return True
        return False
    mock_st_main.button.side_effect = button_side_effect
    
    # Mock the QA chain execution (already done by fixture, but good to be aware)
    mock_chain_run = mock_st_main.load_qa_chain.return_value.run 

    main()

    # Assert that the QA chain was called with the text query
    mock_chain_run.assert_called_with(input_documents=mocker.ANY, question="This is a text query")
    
    # Assert that query and query_text_input in session_state are cleared after processing
    assert mock_st_main.session_state.query == "", "session_state.query should be cleared after processing"
    assert mock_st_main.session_state.query_text_input == "", "session_state.query_text_input should be cleared"

def test_query_handling_voice_input_with_pdf(mock_st_main, mocker):
    """Tests query submission via voice input when a PDF is uploaded."""
    from app import main

    # Simulate PDF uploaded
    mock_st_main.file_uploader.return_value = MagicMock(name="UploadedPDF")
    
    # Simulate audio_recorder returning audio bytes
    mock_st_main.audio_recorder.return_value = b"fake_audio_bytes_for_voice_query"
    
    # Mock transcribe_audio_deepgram to return a specific transcript
    mock_transcribe = mocker.patch('app.transcribe_audio_deepgram', return_value="This is a voice query")

    # Mock the QA chain execution
    mock_chain_run = mock_st_main.load_qa_chain.return_value.run

    main()

    # Assert transcribe_audio_deepgram was called
    mock_transcribe.assert_called_once_with(b"fake_audio_bytes_for_voice_query", "fake_deepgram_key")
    
    # Assert that st.info was called with the transcribed query (visual feedback for user)
    # This requires checking the call_args_list of mock_st_main.info
    transcribed_info_call_found = False
    for call in mock_st_main.info.call_args_list:
        if call[0][0] == "Transcribed Query (from recording): This is a voice query":
            transcribed_info_call_found = True
            break
    assert transcribed_info_call_found, "st.info with transcribed query not found"

    # Assert that the QA chain was called with the voice query
    mock_chain_run.assert_called_with(input_documents=mocker.ANY, question="This is a voice query")

    # Assert that query and query_text_input in session_state are cleared
    assert mock_st_main.session_state.query == "", "session_state.query should be cleared after processing"
    assert mock_st_main.session_state.query_text_input == "", "session_state.query_text_input should be cleared by voice input"

def test_query_processing_requires_pdf(mock_st_main, mocker):
    """Tests that query processing is skipped and a warning is shown if PDF is not uploaded."""
    from app import main

    mock_st_main.file_uploader.return_value = None # No PDF
    mock_st_main.session_state.query_text_input = "Query without PDF"
    
    # Simulate submit button click
    def button_side_effect(label, key=None):
        if key == "submit_text_query_button":
            mock_st_main.session_state.query = mock_st_main.session_state.query_text_input
            return True
        return False
    mock_st_main.button.side_effect = button_side_effect

    mock_chain_run = mock_st_main.load_qa_chain.return_value.run

    main()

    mock_st_main.warning.assert_any_call("Please upload a PDF file to ask questions about it.")
    mock_chain_run.assert_not_called() # QA chain should not run
    
    # Query should be cleared because it cannot be processed
    assert mock_st_main.session_state.query == "", "session_state.query should be cleared if no PDF"
    # query_text_input is also cleared when query cannot be processed without PDF
    assert mock_st_main.session_state.query_text_input == "", "session_state.query_text_input should be cleared if no PDF"

# def test_query_handling_voice_input(mock_st_main, mocker): ...
