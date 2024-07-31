from src.rag.engine import RagEngine
from src.monitoring.langsmith_monitor import LangSmithMonitor
from src.data_processing.document_loader import (
    load_and_split_documents,
    load_and_split_multiple_file_types,
)
from src.config.settings import DATA_DIR


import os
from dotenv import load_dotenv

load_dotenv()

print(
    f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY')}"
)  # Print first 5 characters of the key


class HealthCareBot:
    def __init__(self):
        print("Initializing Health Care Bot...")
        self.engine = RagEngine()
        self.monitor = LangSmithMonitor()
        self.documents_ingested = False
        print("Health Care Bot Initialized")

    def run_interactive(self):
        if not self.documents_ingested:
            print("No documents ingested yet. Please ingest documents first.")
            return
        print("Starting interactive session...")
        try:
            self.engine.run_interactive_session()
        except Exception as e:
            print(f"An error occurred during the interactive session: {e}")

    def run_monitoring(self, start_time, end_time):
        print(f"Running monitoring for period: {start_time} - {end_time}")
        try:
            report = self.monitor.generate_report(start_time, end_time)
            print(report)
        except Exception as e:
            print(f"An error occurred while generating the report: {e}")

    def ingest_documents(self):
        print("Ingesting documents...")
        try:
            docs = load_and_split_documents()
            self.engine.process_documents(docs)
            self.documents_ingested = True
            print(f"Documents ingested: {len(docs)} chunks processed")
        except Exception as e:
            print(f"An error occurred while ingesting documents: {e}")

    def clear_data(self):
        confirmation = input(
            "Are you sure you want to clear all data? This cannot be undone. (y/n): "
        )
        if confirmation.lower() == "y":
            self.engine.clear_vectorstore()
            print("All data has been cleared.")
        else:
            print("Operation cancelled.")


def main():
    print("Starting Health Care Bot...")

    bot = HealthCareBot()
    while True:
        print("\nMain Menu:")
        choice = input(
            "Choose an option: (1) Ingest documents, (2) Run interactive session, (3) Run monitoring, (4) Exit: "
        )
        if choice == "1":
            bot.ingest_documents()
        elif choice == "2":
            bot.run_interactive()
        elif choice == "3":
            start_time = input("Enter start time (YYYY-MM-DD): ")
            end_time = input("Enter end time (YYYY-MM-DD): ")
            bot.run_monitoring(start_time, end_time)
        elif choice == "4":
            print("Exiting...")
            break
        elif choice == "5":
            bot.clear_data()
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
