from langsmith import Client
from config.settings import LANGCHAIN_API_KEY
from datetime import datetime


class LangSmithMonitor:
    def __init__(self, api_key=None):
        self.client = Client(api_key=LANGCHAIN_API_KEY if api_key is None else api_key)
        self.project_name = "engine"

    def ensure_project_exists(self):
        projects = self.client.list_projects()
        if self.project_name not in [p.name for p in projects]:
            print(f"Project '{self.project_name}' does not exist. Creating it now...")
            self.client.create_project(self.project_name)
            print(f"Project '{self.project_name}' created successfully.")
        else:
            print(f"Project '{self.project_name}' already exists.")

    def generate_report(self, start_time, end_time):
        self.ensure_project_exists()

        try:
            print(
                f"type of start_time: {type(start_time)} and type of end_time: {type(end_time)}"
            )
            # Convert string dates to datetime objects
            start_datetime = datetime.strptime(start_time, "%Y-%m-%d")
            end_datetime = datetime.strptime(end_time, "%Y-%m-%d")

            runs = self.client.list_runs(
                start_time=start_datetime,
                end_time=end_datetime,
                project_name=self.project_name,
            )

            total_runs = len(list(runs))
            return f"Total runs between {start_time} and {end_time}: {total_runs}"
        except Exception as e:
            return f"Error generating report: {e}"

    def list_available_projects(self):
        projects = self.client.list_projects()
        return [p.name for p in projects]
