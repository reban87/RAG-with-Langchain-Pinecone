from src.rag.engine import RagEngine
from src.monitoring.langsmith_monitor import LangSmithMonitor


class HealthCareBot:
    def __init__(self):
        print("Initializing Health Care Bot...")
        self.engine = RagEngine()
        self.monitor = LangSmithMonitor()
        print("Health Care Bot Initialized")

    def run_interactive(self):
        print("Starting interactive session...")
        self.engine.run_interactive_session()

    def run_monitoring(self, start_time, end_time):
        print(f"Running monitoring for period: {start_time} - {end_time} ")
        report = self.monitor.generate_report()
        print(report)


def main():
    print("Starting Health Care Bot...")

    bot = HealthCareBot()
    while True:
        print("\nMain Menu:")
        choice = input(
            "Choose an option: (1) Run interactive session, (2) Run monitoring, (3) Exit: "
        )
        if choice == "1":
            bot.run_interactive()
        elif choice == "2":
            start_time = input("Enter start time (YYYY-MM-DD): ")
            end_time = input("Enter end time (YYYY-MM-DD): ")
            bot.run_monitoring(start_time, end_time)
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
