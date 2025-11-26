import os
from pathlib import Path

import questionary
from questionary import Choice, Separator

from modules.state_machine import STATE_MACHINE, State, StateMachine

EXPERIMENTS_PATH = Path("experiments")


def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def select_experiments_with_config(path: Path = EXPERIMENTS_PATH) -> list[str] | None:
    directories = [
        directory.name
        for directory in path.iterdir()
        if directory.is_dir() and (directory / "config.toml").exists()
    ]

    if not directories:
        print("No experiments with config.toml found")
        return None

    selected = questionary.checkbox(
        "Select Directories:",
        choices=[Choice(directory) for directory in directories],
    ).ask()

    return selected


def select_experiment_with_results(path: Path = EXPERIMENTS_PATH) -> str | None:
    directories = [
        directory.name
        for directory in path.iterdir()
        if directory.is_dir() and (directory / "results.json").exists()
    ]

    if not directories:
        print("No experiments with results.json found")
        return None

    selected_experiment = questionary.select(
        "Select experiment:",
        choices=[Choice(directory) for directory in directories]
        + [Separator(), Choice("Back", value="back", shortcut_key="q")],
        use_shortcuts=True,
    ).ask()

    return selected_experiment


def main_menu(state_machine: StateMachine):
    clear_terminal()

    operation = questionary.select(
        "What would you like to do?",
        choices=[
            Choice("Analysis", value="analysis"),
            Choice("Experiment", value="experiment"),
            Choice("Launch Dashboard", value="dashboard"),
            Choice("Launch Demo", value="demo"),
            Choice("Quit", value="quit", shortcut_key="q"),
        ],
        use_shortcuts=True,
    ).ask()

    if operation == "quit":
        state_machine.transition(State.Quit)
    elif operation == "analysis":
        state_machine.transition(State.AnalysisMenu)
    elif operation == "experiment":
        state_machine.transition(State.ExperimentMenu)
    elif operation == "dashboard":
        state_machine.transition(State.LaunchDashboard)
    elif operation == "demo":
        state_machine.transition(State.LaunchDemo)


def analysis_menu(state_machine: StateMachine):
    operation = questionary.select(
        "Select analysis type:",
        choices=[
            Choice("Descriptive", value="descriptive"),
            Choice("Sample Batch", value="sample_batch"),
            Choice("Show Training Graphs", value="training_graphs"),
            Choice("Show Training Results", value="results"),
            Choice("Back", value="back", shortcut_key="q"),
        ],
        use_shortcuts=True,
    ).ask()

    if operation == "back":
        state_machine.transition(State.MainMenu)
    elif operation == "descriptive":
        state_machine.transition(State.AnalyzeDescriptive)
    elif operation == "sample_batch":
        state_machine.transition(State.AnalyzeSampleBatch)
    elif operation == "training_graphs":
        state_machine.transition(State.AnalyzeTrainingGraphs)
    elif operation == "results":
        state_machine.transition(State.AnalyzeResults)


def experiment_menu(state_machine: StateMachine):
    operation = questionary.select(
        "Select experiment type:",
        choices=[
            Choice("All", value="all"),
            Choice("Selected", value="selected"),
            Choice("Back", value="back", shortcut_key="q"),
        ],
        use_shortcuts=True,
    ).ask()

    if operation == "quit":
        state_machine.transition(State.Quit)
    elif operation == "back":
        state_machine.transition(State.MainMenu)
    elif operation == "all":
        state_machine.transition(State.ExperimentAll)
    elif operation == "selected":
        state_machine.transition(State.ExperimentSelected)


def run():
    state_machine = STATE_MACHINE

    while state_machine.get_state() != State.Quit:
        current = state_machine.get_state()

        if current == State.MainMenu:
            main_menu(state_machine)
        elif current == State.AnalysisMenu:
            analysis_menu(state_machine)
        elif current == State.ExperimentMenu:
            experiment_menu(state_machine)
        elif current == State.AnalyzeDescriptive:
            from modules import analysis

            analysis.analyze_descriptive()
            state_machine.transition(State.AnalysisMenu)
        elif current == State.AnalyzeSampleBatch:
            from modules import analysis

            analysis.analyze_sample_batch(pretrained=True)
            state_machine.transition(State.AnalysisMenu)
        elif current == State.AnalyzeTrainingGraphs:
            selected_experiment = select_experiment_with_results()

            if selected_experiment == "back":
                state_machine.transition(State.AnalysisMenu)
                continue

            save_graphs = questionary.confirm(
                "Also save graphs? (Slower)", default=False
            ).ask()

            if save_graphs is None:
                continue

            experiment_directory = EXPERIMENTS_PATH / selected_experiment

            try:
                from modules import analysis

                analysis.analyze_training_graphs(
                    experiment_directory, save_graphs=save_graphs
                )
                state_machine.transition(State.AnalysisMenu)
            except Exception as exception:
                print(f"Error showing graphs for {selected_experiment}: {exception}")
        elif current == State.AnalyzeResults:
            selected_experiment = select_experiment_with_results()

            if selected_experiment == "back":
                state_machine.transition(State.AnalysisMenu)
                continue

            experiment_directory = EXPERIMENTS_PATH / selected_experiment

            try:
                from modules import analysis

                analysis.analyze_results(experiment_directory)
                state_machine.transition(State.AnalysisMenu)
            except Exception as exception:
                print(f"Error showing results for {selected_experiment}: {exception}")
        elif current == State.ExperimentAll:
            directories = [
                directory
                for directory in EXPERIMENTS_PATH.iterdir()
                if directory.is_dir() and (directory / "config.toml").exists()
            ]

            if not directories:
                print("No valid experiments found")

            for directory in directories:
                try:
                    from modules import experiment

                    experiment.run_experiment(directory)
                except Exception as exception:
                    print(f"Error running experiment in {directory}: {exception}")

            state_machine.transition(State.ExperimentMenu)
        elif current == State.ExperimentSelected:
            selected_experiments = select_experiments_with_config()

            if not selected_experiments:
                state_machine.transition(State.ExperimentMenu)
                continue

            for directory in selected_experiments:
                experiment_directory = EXPERIMENTS_PATH / directory

                if not (experiment_directory / "config.toml").exists():
                    print(f"Skipping {directory}: no config.toml found")
                    continue

                try:
                    from modules import experiment

                    experiment.run_experiment(experiment_directory)
                except Exception as exception:
                    print(
                        f"Error running experiment in {experiment_directory}: {exception}"
                    )

            state_machine.transition(State.ExperimentMenu)
        elif current == State.LaunchDashboard:
            from modules import dashboard

            dashboard.make_dashboard().launch()
            state_machine.transition(State.MainMenu)
        elif current == State.LaunchDemo:
            from modules import demo

            demo.make_demo().launch()
            state_machine.transition(State.MainMenu)


run()
