from os import listdir as os_listdir
from os import name as os_name
from os import path as os_path
from os import system as os_system
from pathlib import Path
from traceback import print_exc

from questionary import prompt

from modules.experiment import run_experiment
from modules.state_machine import State, StateMachine


def clear_terminal() -> None:
    os_system("cls" if os_name == "nt" else "clear")


def get_directories(path: str) -> list[str]:
    return [
        directory
        for directory in os_listdir(path)
        if os_path.isdir(os_path.join(path, directory))
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

    answer = prompt(questions)

    return answer.get("selected_directories") if answer else []


def main_menu(state_machine: StateMachine):
    clear_terminal()

    main_questions = [
        {
            "type": "select",
            "name": "operation",
            "message": "What would you like to do?",
            "choices": ["Analysis", "Experiment", "Quit"],
        }
    ]

    answer = prompt(main_questions)

    if not answer:
        return

    operation = answer.get("operation")

    if operation == "Quit":
        state_machine.transition(State.Quit)
    elif operation == "Analysis":
        state_machine.transition(State.AnalyzeMenu)
    elif operation == "Experiment":
        state_machine.transition(State.ExperimentMenu)


def analyze_menu(state_machine: StateMachine):
    analyze_questions = [
        {
            "type": "select",
            "name": "analysis_type",
            "message": "Select analysis type:",
            "choices": [
                "Descriptive",
                "Sample Batch",
                "Sample Batch Normalized",
                "Back",
                "Quit",
            ],
        }
    ]

    answer = prompt(analyze_questions)

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
    elif analysis_type == "Sample Batch Normalized":
        state_machine.transition(State.AnalyzeSampleBatchNormalized)


def experiment_menu(state_machine: StateMachine):
    experiment_questions = [
        {
            "type": "select",
            "name": "experiment_type",
            "message": "Select experiment type:",
            "choices": ["All", "Selected", "Back", "Quit"],
        }
    ]

    answer = prompt(experiment_questions)

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
    state_machine = StateMachine(
        State.MainMenu,
        {
            State.MainMenu: [State.Quit, State.AnalyzeMenu, State.ExperimentMenu],
            State.AnalyzeMenu: [
                State.Quit,
                State.MainMenu,
                State.AnalyzeDescriptive,
                State.AnalyzeSampleBatch,
                State.AnalyzeSampleBatchNormalized,
            ],
            State.AnalyzeDescriptive: [State.AnalyzeMenu],
            State.AnalyzeSampleBatch: [State.AnalyzeMenu],
            State.AnalyzeSampleBatchNormalized: [State.AnalyzeMenu],
            State.ExperimentMenu: [
                State.Quit,
                State.MainMenu,
                State.ExperimentAll,
                State.ExperimentSelected,
            ],
            State.ExperimentAll: [State.ExperimentMenu],
            State.ExperimentSelected: [State.ExperimentMenu],
            State.Quit: [],
        },
    )

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
            # TODO: implement
            state_machine.transition(State.AnalyzeMenu)
        elif current == State.AnalyzeSampleBatchNormalized:
            # TODO: implement
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
                    results = run_experiment(experiment_directory)
                    print(f"Experiment completed: {results['experiment_name']}")
                    print(
                        f"Average validation accuracy: {results['average_validation_acc']:.2f}%"
                    )
                except Exception:
                    print_exc()
                    print(f"Error running experiment in {experiment_directory}")

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
                    results = run_experiment(experiment_directory)
                    print(f"Experiment completed: {results['experiment_name']}")
                    print(
                        f"Average validation accuracy: {results['average_validation_accuracy']:.2f}%"
                    )
                except Exception:
                    print_exc()
                    print(f"Error running experiment in {experiment_directory}")

            state_machine.transition(State.ExperimentMenu)

    print("Goodbye!")


run()
