import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent 
from langchain_openai import ChatOpenAI

from src.logger.base import BaseLogger
from src.models.llm import load_llm

from src.utils import execute_plt

#load env vars
load_dotenv()
logger = BaseLogger()
MODEL_NAME = "gpt-3.5-turbo"


def process_query(da_agent,query):
    response = da_agent(query)
    try:
        action = response["intermediate_steps"][-1][0].tool_input["query"]

        if "plt" in action:
            st.write(response["output"])

            fig = execute_plt(action, df=st.session_state.df)
            if fig:
                st.pyplot(fig)

            st.write("**`Executed Code`**")
            st.code(action)

            display_string = response["output"] + "\n\n**`Executed Code`**\n" + action + "\n"
            st.session_state.history.append((query, display_string))
        
        else:
            st.write(response["output"])
            st.session_state.history.append((query, response["output"]))

    except Exception as e:
        # If Indexerror then this is an irrelevant question
        if isinstance(e, IndexError):
            st.error(e)
            st.write("<p style='color:red; font-style:italic'>This question is irrelevant to the dataset")
            st.write(response["output"])
        else:
            st.error(e)

def display_chat_history():
    st.markdown("### Chat History")
    for i, (query, response) in enumerate(st.session_state.history):
        st.markdown(f"**Query {i + 1}:** {query}")
        st.markdown(f"**Response {i + 1}:** {response}")
        st.markdown("---")

def main():
    
    # Set up streamlit interface
    st.set_page_config(
        page_title="Data Analysis Buddy",
        page_icon="📊",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.header("📊 DATA ANALYSIS BUDDY")
    st.write(
        "### Welcome to our data analysis tool, designed to tackle repetitive EDA tasks. Enjoy!"
    )

    # Load LLM model
    llm = load_llm(model_name = MODEL_NAME)
    logger.info(f"### Successfully loaded {MODEL_NAME} model. ###")

    # Upload CSV File
    def file_upload():
        st.session_state['uploaded'] = True

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload CSV file here",
                                         type=["csv"],
                                         on_change=file_upload,
                                         help="Supported file types: csv")
        if st.session_state.get('uploaded', False):
            st.write("File Uploaded")

    # Initiate chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Read csv file 
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write(f"### Your uploaded data:",st.session_state.df.head())

    # Create data analysis agent to query with data
    if uploaded_file is not None and st.session_state.get('df') is not None:
        da_agent = create_pandas_dataframe_agent(llm=llm, 
                                                df=st.session_state.df,
                                                agent_type="tool-calling",
                                                allow_dangerous_code = True,
                                                verbose = True,
                                                return_intermediate_steps = True)
        logger.info(f"### Successfully loaded data analysis agent. ###")

        # Input query and process query
        query = st.text_input("Enter your question here:")

        if st.button("Submit"):
            with st.spinner("Processing..."):
                    process_query(da_agent,query)

            
        # Display chat history
        st.divider()
        display_chat_history()
    else:
        st.info("Please upload a CSV file to get started.") 

if __name__ == "__main__":
    main()