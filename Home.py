import streamlit as st
# from langchain.prompts import PromptTemplate

# st.subheader("Welcome to the Streamlit app!")

# st.markdown(
#     """
#     I love it!
# """
# )

# st.write("I love it!")

# st.write([1, 2, 3])

# st.write(PromptTemplate)

# p = PromptTemplate.from_template("What is the capital of {country}?")

# st.write(p)

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🤖"
)

with st.sidebar:
    st.title("Sidebar title")
    st.text_input("Input text")

st.title("FullstackGPT Home")    
    
st.markdown("""
    # Hello!
    
    Welcome to my FullstackGPT Portfolio!
    
    Here are the apps I made:
    
    - ✅ [DocumentGPT](/DocumentGPT)
    - [PrivateGPT](/PrivateGPT)
    - ✅ [QuizGPT](/QuizGPT)
    - [PrivateGPT](/PrivateGPT)
    - ✅ [SiteGPT](/SiteGPT)
    - [MeetingGPT](/MeetingGPT)
    - ✅ [InvestorGPT](/InvestorGPT)
""")