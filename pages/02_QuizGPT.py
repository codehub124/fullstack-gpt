import json
import random

import streamlit as st

from langchain_community.retrievers import WikipediaRetriever
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

function_schema = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and return a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {"type": "string"},
                                    "correct": {"type": "boolean"}
                                },
                                "required": ["answer", "correct"]
                            }
                        }
                    },
                    "required": ["question", "answers"],
                }
            }
        },
        "required": ["questions"],
    },
}

def get_llm():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''

    if st.session_state.api_key == '':
        return None
    
    try:
        return ChatOpenAI(
            api_key=st.session_state.get('api_key', ''),
            temperature=0.1,
            model="gpt-4o-mini", #$0.000150 / 1K input tokens
        ).bind(
            function_call = {
                "name": "create_quiz"
            },
            functions=[function_schema]
        )
    except Exception as e:
        st.error(f"""Error initializing LLM: please check your OpenAI API key and try again.""")
        st.session_state.api_key = '' # Clear invalid API key
        return None

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

prompt = PromptTemplate.from_template(
    """
    Make a {difficulty} 2-question quiz about this context: {context}?
    Use 'Easy' for straightforward questions and 'Hard' for questions that demand deeper analysis.
    """
)

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓"
)

st.title("QuizGPT")

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    chain = prompt | get_llm()
    response = chain.invoke({ "context": format_docs(_docs), "difficulty": difficulty })
    quiz_data = json.loads(response.additional_kwargs['function_call']['arguments'])

    # Randomize the order of answers for each question
    for question in quiz_data["questions"]:
        random.shuffle(question["answers"])

    return quiz_data

@st.cache_data(show_spinner="Searching Wikipedia...")
def wikipedia_search(topic):
    retriever = WikipediaRetriever(top_k_results=2)
    return retriever.get_relevant_documents(topic)

with st.sidebar:
    docs = None

    # Difficulty selection
    difficulty = st.selectbox("Choose the difficulty of the quiz", ["Hard", "Easy"])
    
    # Source selection
    choice = st.selectbox("Choose the source of the quiz", ["File", "Wikipedia"])
    
    if choice == "File":
        file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Searching Wikipedia ...")
        if topic:
            docs = wikipedia_search(topic)

    # API key input
    api_key = st.text_input("Enter your OpenAI API key", type="password")

    if api_key:
        st.session_state.api_key = api_key
    elif api_key == '':  # When the input is cleared
        if 'api_key' in st.session_state:
            st.session_state.api_key = ''
            
    st.markdown("[GitHub Source Code Link](https://github.com/codehub124/fullstack-gpt/blob/main/pages/02_QuizGPT.py)")

if not docs or not get_llm():
    st.markdown(
        """
        Welcome to QuizGPT!
        
        I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
        
        Get started by uploading a file or searching for a topic on Wikipedia in the sidebar.
        """
    )

    if not ('api_key' in st.session_state and st.session_state.api_key):
        st.warning("Please enter your OpenAI API key in the sidebar")
        st.stop()

else:
    try:
        response = run_quiz_chain(docs, topic if topic else file.name, difficulty)

        with st.form("questions_form"):
            correct_answers = 0
            for question in response["questions"]:
                st.write(question["question"])
                value = st.radio("Select an option",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                    key=f"question_{question['question']}"
                )
                
                if { "answer": value, "correct": True } in question["answers"]:
                    st.success("Correct!")
                    correct_answers += 1
                elif value is not None:
                    st.error("Wrong!")

            button = st.form_submit_button("Submit")

            if button:          
                if correct_answers == len(response["questions"]):
                    st.info("Congratulations! You got all answers correct.")
                    st.balloons()
                else:
                    st.info(f"You got {correct_answers} out of {len(response['questions'])} questions correct. Please try again!")
    except Exception as e:
        st.error(f"Error running quiz chain: {str(e)}")
        response = None
