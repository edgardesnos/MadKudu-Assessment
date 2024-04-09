from chat import LLMChatBot
import streamlit as st

# Init session state
if "api_key" not in st.session_state:
    st.session_state.api_key = None
if "model" not in st.session_state:
    st.session_state.model = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("MadKudu Support Assistant")

api_key = st.text_input("OpenAI API Key", type="password")

# Reset chats and model when API key is changed
if api_key != st.session_state.api_key and api_key:
    st.session_state.api_key = api_key
    st.session_state.messages = []
    st.session_state.model = LLMChatBot(api_key)

# Display history of chat messsages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Disable user prompt unless API key has been set
if prompt := st.chat_input("What is your question?", disabled=not api_key):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = st.session_state.model.ask_question(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})