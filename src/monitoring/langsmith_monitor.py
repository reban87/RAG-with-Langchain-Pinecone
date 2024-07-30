from langsmith import Client


class LangSmithMonitor:
    def __init__(self):
        self.client = Client()

    def generate_report(self, start_time, end_time):
        runs = self.client.list_runs(
            start_time=start_time, end_time=end_time, project_name="engine"
        )

        total_runs = len(list(runs))
        return f"Total runs between {start_time} and {end_time}: {total_runs}"
