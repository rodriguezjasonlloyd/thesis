import os
from pathlib import Path

import questionary

from modules import analysis, dashboard, experiment
from modules.state_machine import State, StateMachine


def clear_terminal() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def get_directories(path: str) -> list[str]:
    return [
        directory
        for directory in os.listdir(path)
        if os.path.isdir(os.path.join(path, directory))
    ]


def select_directories(path: str = "experiments") -> list[str]:
    directories = get_directories(path)

    if not directories:
        print(f"No directories found in {path}")
        return None

    questions = [
        {
            "type": "checkbox",
            "name": "selected_directories",
            "message": "Select directories:",
            "choices": directories,
        }
    ]

    answer = questionary.prompt(questions)

    return answer.get("selected_directories") if answer else []


def select_experiment_with_results() -> str:
    experiments_path = Path("experiments")

    if not experiments_path.exists():
        print("Experiments directory not found")
        return None

    experiment_directories = [
        directory.name
        for directory in experiments_path.iterdir()
        if directory.is_dir() and (directory / "results.json").exists()
    ]

    if not experiment_directories:
        print("No experiments with results.json found")
        return None

    questions = [
        {
            "type": "select",
            "name": "selected_experiment",
            "message": "Select experiment:",
            "choices": experiment_directories,
        }
    ]

    answer = questionary.prompt(questions)

    return answer.get("selected_experiment") if answer else None


def main_menu(state_machine: StateMachine):
    clear_terminal()

    main_questions = [
        {
            "type": "select",
            "name": "operation",
            "message": "What would you like to do?",
            "choices": ["Analysis", "Experiment", "Launch Dashboard", "Quit"],
        }
    ]

    answer = questionary.prompt(main_questions)

    if not answer:
        return

    operation = answer.get("operation")

    if operation == "Quit":
        state_machine.transition(State.Quit)
    elif operation == "Analysis":
        state_machine.transition(State.AnalyzeMenu)
    elif operation == "Experiment":
        state_machine.transition(State.ExperimentMenu)
    elif operation == "Launch Dashboard":
        state_machine.transition(State.LaunchDashboard)


def analyze_menu(state_machine: StateMachine):
    analyze_questions = [
        {
            "type": "select",
            "name": "analysis_type",
            "message": "Select analysis type:",
            "choices": [
                "Descriptive",
                "Sample Batch",
                "Show Training Graphs",
                "Back",
                "Quit",
            ],
        }
    ]

    answer = questionary.prompt(analyze_questions)

    if not answer:
        return

    analysis_type = answer.get("analysis_type")

    if analysis_type == "Quit":
        state_machine.transition(State.Quit)
    elif analysis_type == "Back":
        state_machine.transition(State.MainMenu)
    elif analysis_type == "Descriptive":
        state_machine.transition(State.AnalyzeDescriptive)
    elif analysis_type == "Sample Batch":
        state_machine.transition(State.AnalyzeSampleBatch)
    elif analysis_type == "Show Training Graphs":
        state_machine.transition(State.AnalyzeTrainingGraphs)


def experiment_menu(state_machine: StateMachine):
    experiment_questions = [
        {
            "type": "select",
            "name": "experiment_type",
            "message": "Select experiment type:",
            "choices": ["All", "Selected", "Back", "Quit"],
        }
    ]

    answer = questionary.prompt(experiment_questions)

    if not answer:
        return

    experiment_type = answer.get("experiment_type")

    if experiment_type == "Quit":
        state_machine.transition(State.Quit)
    elif experiment_type == "Back":
        state_machine.transition(State.MainMenu)
    elif experiment_type == "All":
        state_machine.transition(State.ExperimentAll)
    elif experiment_type == "Selected":
        state_machine.transition(State.ExperimentSelected)


def run():
    state_machine = STATE_MACHINE

    while state_machine.get_state() != State.Quit:
        current = state_machine.get_state()

        if current == State.MainMenu:
            main_menu(state_machine)
        elif current == State.AnalyzeMenu:
            analyze_menu(state_machine)
        elif current == State.ExperimentMenu:
            experiment_menu(state_machine)
        elif current == State.AnalyzeDescriptive:
            # TODO: implement
            state_machine.transition(State.AnalyzeMenu)
        elif current == State.AnalyzeSampleBatch:
            analysis.analyze_sample_batch(pretrained=True)
            state_machine.transition(State.AnalyzeMenu)
        elif current == State.AnalyzeTrainingGraphs:
            selected_experiment = select_experiment_with_results()

            if selected_experiment:
                experiment_directory = Path("experiments") / selected_experiment

                try:
                    analysis.show_training_graphs(
                        experiment_directory, save_graphs=True
                    )
                except Exception as exception:
                    print(
                        f"Error showing graphs for {selected_experiment}: {exception}"
                    )

            state_machine.transition(State.AnalyzeMenu)
        elif current == State.ExperimentAll:
            experiments_path = Path("experiments")

            if not experiments_path.exists():
                print("Experiments directory not found")
                continue

            experiment_directories = [
                directory
                for directory in experiments_path.iterdir()
                if directory.is_dir() and (directory / "config.toml").exists()
            ]

            if not experiment_directories:
                print("No valid experiments found")
                continue

            for experiment_directory in experiment_directories:
                try:
                    results = experiment.run_experiment(experiment_directory)
                    print(f"Experiment completed: {results['experiment_name']}")
                except Exception as exception:
                    print(
                        f"Error running experiment in {experiment_directory}: {exception}"
                    )

            state_machine.transition(State.ExperimentMenu)
        elif current == State.ExperimentSelected:
            selected_directories = select_directories()

            if not selected_directories:
                print("No experiments selected")
                continue

            experiments_path = Path("experiments")

            for directory_name in selected_directories:
                experiment_directory = experiments_path / directory_name

                if not (experiment_directory / "config.toml").exists():
                    print(f"Skipping {directory_name}: no config.toml found")
                    continue

                try:
                    results = experiment.run_experiment(experiment_directory)
                    print(f"Experiment completed: {results['experiment_name']}")
                except Exception as exception:
                    print(
                        f"Error running experiment in {experiment_directory}: {exception}"
                    )

            state_machine.transition(State.ExperimentMenu)
        elif current == State.LaunchDashboard:
            dashboard.make_dashboard().launch()
            state_machine.transition(State.MainMenu)

    print("Goodbye!")


run()
