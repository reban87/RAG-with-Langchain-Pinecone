import streamlit as st
from rag.recruiter_engine import RecruiterRagEngine
from data_processing.document_loader import (
    load_documents
    )


class RecruimentBot():
    def __init__(self):
        print("Initializing Recruitment Bot...")
        try:
            self.engine: RecruiterRagEngine = RecruiterRagEngine()
            # self.monitor: LangSmithMonitor = LangSmithMonitor()
        except Exception as e:
            print(f"Error initializing components: {e}")
        self.documents_ingested: bool = False

    def ingest_documents(self):
        print("Ingesting documents...")
        try:
            docs = load_documents()
            self.engine.process_documents(docs)
            # self.documents_ingested = True
            print(f"Documents ingested: {len(docs)} documents processed")
        except (FileNotFoundError, ValueError) as e:
            print(f"An error occurred while ingesting documents: {e}")

    def query_llm(self,query):
        return self.engine.interpret_query(query)

    def clear_data(self):
        print("Clearing Vectorstore...")
        self.engine.clear_vectorstore()
            


    

def main():
    bot = RecruimentBot()

    with st.sidebar:
        # "Techkraft"
        st.button("Import Docs", type="primary", on_click=bot.ingest_documents)
        st.button("Clear Data",on_click=bot.clear_data)
        
    

    st.title("💬 Lets Recruit!!")
    st.caption("🚀 Helps you find people according to the job description")
    st.markdown(
        r"""
        <style>
        .stDeployButton {
                visibility: hidden;
            }
        </style>
        """, unsafe_allow_html=True
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        response= bot.query_llm(prompt)
        msg = response[0]
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

if __name__ == "__main__":
    main()