import streamlit as st
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
import os

# Create cache directory if it doesn't exist
os.makedirs("./.cache/site_files", exist_ok=True)
set_llm_cache(SQLiteCache(database_path="./.cache/site_files/cache.db"))

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️"
)

st.title("SiteGPT")

def get_llm():
    if 'api_key' not in st.session_state or st.session_state.api_key == '':
        st.warning("Please enter your OpenAI API key")
        st.stop()

    try:
        return ChatOpenAI(
            api_key=st.session_state.get('api_key', ''),
            temperature=0.1,
            model="gpt-4o-mini", #$0.000150 / 1K input tokens
            streaming=True,
            callbacks=[
                StreamingStdOutCallbackHandler()
            ]
        )
    except Exception as e:
        st.error(f"""Error initializing LLM: please check your OpenAI API key and try again.""")
        st.session_state.api_key = '' # Clear invalid API key
        return None

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.
                                                  
    Context: {context}

    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    
    Question: {question}
"""
)

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    
    answers_chain = answers_prompt | get_llm()

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"context": doc.page_content, "question": question}
            ).content,
            "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ]
    }
    
choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)
    
def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | get_llm()
    condenced_answers = "\n\n".join(
        f"{answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\n"
        for answer in answers
    )

    return choose_chain.invoke(
        {
            "question": question,
            "answers": condenced_answers,
        }
    )

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\t", " ")
    )

@st.cache_resource(show_spinner="Loading website...") # Fix for "Cannot serialize the return value" error
def load_sitemap(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/(ai-gateway|vectorize|workers-ai)\/).*"  # Matches any of: /ai-gateway/, /vectorize/ or /workers-ai/
        ],
        parsing_function=parse_page
    )
    try:
        loader.requests_per_second = 1
        docs = loader.load_and_split(text_splitter=splitter)

        if not docs:
            st.warning("No documents were loaded!")
            return None

        vector_store = FAISS.from_documents(
            documents=docs, 
            embedding=OpenAIEmbeddings(api_key=st.session_state.get('api_key', ''))  # Add API key
        )        
        return vector_store.as_retriever()
    except Exception as e:
        st.error(f"Error loading sitemap: {str(e)}")
        return []

with st.sidebar:
    url = st.text_input(
        "The sitemap URL of the website you want to ask questions about.",
        "https://developers.cloudflare.com/sitemap-0.xml",
        disabled=True
    )

    # API key input
    api_key = st.text_input("Enter your OpenAI API key", type="password")

    if api_key:
        st.session_state.api_key = api_key
    elif api_key == '':  # When the input is cleared
        if 'api_key' in st.session_state:
            st.session_state.api_key = ''
            
    st.markdown("[GitHub Source Code Link](https://github.com/codehub124/fullstack-gpt/blob/main/pages/03_SiteGPT.py)")

if not get_llm():
    st.markdown(
        """
        Welcome to SiteGPT!

        Ask questions about the content of ${url}.
        Start by entering the URL of the website you want to ask questions about.
        """
    )

    if not ('api_key' in st.session_state and st.session_state.api_key):
        st.warning("Please enter your OpenAI API key in the sidebar")
        st.stop()
else:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please enter a valid sitemap URL.")
            st.stop()
    else:
        st.markdown(f"Loaded website: {url}")
        retriever = load_sitemap(url)
        query = st.text_input("Ask a question to the website.", placeholder="What's the price for the 100 GB storage plan?")
        
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough()
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))
