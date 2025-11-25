import enum
from enum import Enum


class State(Enum):
    MainMenu = enum.auto()
    Quit = enum.auto()

    AnalysisMenu = enum.auto()
    AnalyzeDescriptive = enum.auto()
    AnalyzeSampleBatch = enum.auto()
    AnalyzeTrainingGraphs = enum.auto()
    AnalyzeResults = enum.auto()

    ExperimentMenu = enum.auto()
    ExperimentAll = enum.auto()
    ExperimentSelected = enum.auto()

    LaunchDashboard = enum.auto()


class StateMachine:
    def __init__(
        self, initial_state: State, transitions: dict[State, list[State]]
    ) -> None:
        self.current_state: State = initial_state
        self.transitions: dict[State, list[State]] = transitions

    def transition(self, new_state: State) -> None:
        if new_state in self.transitions[self.current_state]:
            self.current_state = new_state
        else:
            raise ValueError(
                f"Invalid transition from {self.current_state} to {new_state}"
            )

    def get_state(self) -> State:
        return self.current_state


STATE_MACHINE = StateMachine(
    State.MainMenu,
    {
        State.MainMenu: [
            State.Quit,
            State.AnalysisMenu,
            State.ExperimentMenu,
            State.LaunchDashboard,
        ],
        State.AnalysisMenu: [
            State.MainMenu,
            State.AnalyzeDescriptive,
            State.AnalyzeSampleBatch,
            State.AnalyzeTrainingGraphs,
            State.AnalyzeResults,
        ],
        State.AnalyzeDescriptive: [State.AnalysisMenu],
        State.AnalyzeSampleBatch: [State.AnalysisMenu],
        State.AnalyzeTrainingGraphs: [State.AnalysisMenu],
        State.AnalyzeResults: [State.AnalysisMenu],
        State.ExperimentMenu: [
            State.MainMenu,
            State.ExperimentAll,
            State.ExperimentSelected,
        ],
        State.ExperimentAll: [State.ExperimentMenu],
        State.ExperimentSelected: [State.ExperimentMenu],
        State.LaunchDashboard: [State.MainMenu],
        State.Quit: [],
    },
)
