import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent 
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from src.logger.base import BaseLogger
from src.models.llm import load_llm

from src.utils import execute_plt

import re

#load env vars
load_dotenv()
logger = BaseLogger()
MODEL_NAME = "gpt-4o"


def process_query(da_agent, query):
    response = da_agent(query)
    try:
        action = response["intermediate_steps"][-1][0].tool_input["query"]

        if "plt" in action:
            st.write(response["output"])

            # Extract the DataFrame variable name from the action string
            var_name = None
            if 'df_' in action:
                var_name_match = re.search(r'df_(\w+)', action)  # Look for variables like df_xyz
                if var_name_match:
                    var_name = var_name_match.group(0)
            else:
                match = re.search(r'(\w+)\.plot', action)  # Look for df.plot
                if match:
                    var_name = match.group(1)

            # Default to 'df' if no specific variable name is found
            if not var_name:
                var_name = 'df'

            # Retrieve the DataFrame from session state
            df_to_use = None  # Ensure df_to_use is initialized
            if var_name in st.session_state:
                df_to_use = st.session_state[var_name]
            elif "df" not in var_name:
                df_to_use = st.session_state.get("df", None)  # Default to "df" if available
                if df_to_use is None:
                    raise ValueError("No default DataFrame ('df') is available in session state.")
            else:
                raise ValueError(f"Variable '{var_name}' not found in session state. Please check your action string.")

            # Generate the plot using the selected DataFrame
            figure = execute_plt(action, df_to_use)

            # Render the figures in Streamlit
            if figure:
                st.pyplot(figure)
            else:
                st.error("No figures were generated.")

            # Append the query, response, and executed action to session history
            if "history" not in st.session_state:
                st.session_state.history = []  # Initialize session history if not present
            st.session_state.history.append((query, response["output"], action))
        else:
            st.write(response["output"])
            if "history" not in st.session_state:
                st.session_state.history = []  # Initialize session history if not present
            st.session_state.history.append((query, response["output"]))

    except Exception as e:
        # Handle specific exceptions like IndexError
        if isinstance(e, IndexError):
            st.error("This question may be irrelevant to the dataset.")
            st.write(response["output"])
        else:
            st.error(f"Error: {e}")

def display_chat_history():
    st.markdown("### Chat History")
    for i, (query, response, exe_code) in enumerate(st.session_state.history):
        st.markdown(f"**Query {i + 1}:** {query}")
        st.markdown(f"**Response {i + 1}:** {response}")
        if exe_code:
            st.markdown(f"**Execution Code {i + 1}:**")
            st.code(exe_code, language="python")
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
    
    if st.session_state.get('df') is not None:
        st.write(f"### Your uploaded data:",st.session_state.df.head())

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

    # Create data analysis agent to query with data

        da_agent = create_pandas_dataframe_agent(llm=llm, 
                                                df=st.session_state.df,
                                                agent_type="tool-calling",
                                                allow_dangerous_code = True,
                                                verbose = True,
                                                return_intermediate_steps = True
                                                )
        logger.info(f"### Successfully loaded data analysis agent. ###")

        da_agent.memory = memory

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