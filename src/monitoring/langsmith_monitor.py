from langsmith import Client
from src.config.settings import LANGSMITH_API_KEY


class LangSmithMonitor:
    def __init__(self, api_key=None):
        if api_key:
            self.client = Client(api_key=LANGSMITH_API_KEY)
        else:
            self.client = Client()

    def generate_report(self, start_time, end_time):
        runs = self.client.list_runs(
            start_time=start_time, end_time=end_time, project_name="engine"
        )

        total_runs = len(list(runs))
        return f"Total runs between {start_time} and {end_time}: {total_runs}"
