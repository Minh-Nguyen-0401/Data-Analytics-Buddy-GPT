import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def execute_plt(code: str, df: pd.DataFrame):
    """ Execute the code and return the figure

    Args:
        code (str): action string (containing plt cpde)
        df (pd.DataFrame): session dataframe

    Returns:
        _type_: figure
    """
    try:
        local_vars = {
            "df": df,
            "plt": plt,
            "sns": sns}
        compiled_code = compile(code, "<string>", "exec")
        exec(compiled_code, globals(), local_vars)

        return plt.gcf()
    
    except Exception as e:
        st.error(f"Error executing code: {e}")
        return None