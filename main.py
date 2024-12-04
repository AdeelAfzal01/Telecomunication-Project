from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain_openai import ChatOpenAI
import os
import streamlit as st

# Load environment variables
# from dotenv import load_dotenv
# load_dotenv()



def main():
    # Set OpenAI API key
    # os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
    st.set_page_config(page_title="AINA")
    st.header("Ask your AINA 🤖")

    api_key = st.text_input("Enter your OpenAI API Key:", type="password")

    # Ensure the OpenAI API key is available
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        # File uploader for CSV
        csv_file = st.file_uploader("Upload a CSV file", type="csv")
        if csv_file is not None:
            # Create the agent
            agent = create_csv_agent(
                ChatOpenAI(temperature=0, model="gpt-4"),
                csv_file,
                verbose=True,
                allow_dangerous_code=True,
                handle_parsing_errors=True
            )

            # Initialize session state to track conversation
            if "conversation_history" not in st.session_state:
                st.session_state.conversation_history = []

            # Text input for user question
            user_question = st.text_input("Ask a question about the Data:")

            # Button to stop the interaction
            stop_button = st.button("Stop")

            if stop_button:
                st.write("Conversation stopped.")
                st.stop()

            if user_question and user_question.strip() != "":
                # Append question to conversation history
                st.session_state.conversation_history.append({"question": user_question})

                # Get response from the agent
                with st.spinner(text="Processing your question..."):
                    try:
                        response = agent.invoke(user_question)['output']
                        st.session_state.conversation_history[-1]["response"] = response
                        st.write(response)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            # Display conversation history
            if st.session_state.conversation_history:
                st.subheader("Conversation History")
                for i, exchange in enumerate(st.session_state.conversation_history):
                    st.write(f"Q{i+1}: {exchange['question']}")
                    if "response" in exchange:
                        st.write(f"A{i+1}: {exchange['response']}")

if __name__ == "__main__":
    main()
