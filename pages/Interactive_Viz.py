import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer

def main():
    
    # Configure Streamlit interface
    st.set_page_config(
        page_title="Interactive Viz",
        page_icon="📈",
        layout="wide"
    )

    st.header("📈 Interactive Viz Tool")
    st.write("## Now, let's visualize your data!")

    # Render pygwalker 
    if st.session_state.get('df') is not None:
        pyg_app = StreamlitRenderer(st.session_state.df)
        pyg_app.explorer()
    
    else:
        st.info("Upload a CSV file to get started with interactive visualizations!")


if __name__ == "__main__":
    main()