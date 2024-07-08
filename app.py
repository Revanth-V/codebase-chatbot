import os
import gc
import re
import uuid
import textwrap
import subprocess
import nest_asyncio
from dotenv import load_dotenv

import streamlit as st
from client import RAGClient

load_dotenv()

# Initialize the 'id' attribute at the start of your app
if 'id' not in st.session_state:
    st.session_state['id'] = str(uuid.uuid4())  # Use str to store UUID as a string

session_id = st.session_state.id
client = RAGClient()

# utility functions
def parse_github_url(url):
    pattern = r"https://github\.com/([^/]+)/([^/]+)"
    match = re.match(pattern, url)
    return match.groups() if match else (None, None)


def clone_repo(repo_url, dir_path):
    return subprocess.run(["git", "clone", repo_url, dir_path], check=True, text=True, capture_output=True)


def validate_owner_repo(owner, repo):
    return bool(owner) and bool(repo)


def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


with st.sidebar:
    # Input for GitHub URL
    github_url = st.text_input("GitHub Repository URL")

    # Button to load and process the GitHub repository
    process_button = st.button("Load")

    message_container = st.empty()  # Placeholder for dynamic messages

    if process_button and github_url:
        owner, repo = parse_github_url(github_url)
        if validate_owner_repo(owner, repo):
            with st.spinner(f"Loading {repo} repository by {owner}..."):
                try:
                    input_dir_path = f"./repos/{repo}"
                    
                    if not os.path.exists(input_dir_path):
                        clone_repo(github_url, input_dir_path)

                    if os.path.exists(input_dir_path):
                        status = client.read_files(input_dir_path=input_dir_path)
                    else:    
                        st.error('Error occurred while cloning the repository, carefully check the url')
                        st.stop()

                    client.generate_index()
                    client.create_query_engine()

                    if status:
                        message_container.success("Data loaded successfully!!")
                    else:
                        message_container.write(
                            "No data found, check if the repository is not empty!"
                        )
                    st.session_state.query_engine = client.query_engine

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.stop()

                st.success("Ready to Chat!")
        else:
            st.error('Invalid owner or repository')
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header("Chat With Your Code! Powered by LLama3 🦙🚀")

with col2:
    st.button("Clear ↺", on_click=reset_chat)


# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        query_engine = st.session_state.query_engine
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
