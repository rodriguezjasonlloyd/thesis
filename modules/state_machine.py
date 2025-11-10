from enum import Enum, auto


class State(Enum):
    MainMenu = auto()
    Quit = auto()

    AnalyzeMenu = auto()
    AnalyzeDescriptive = auto()
    AnalyzeSampleBatch = auto()
    AnalyzeTrainingGraphs = auto()

    ExperimentMenu = auto()
    ExperimentAll = auto()
    ExperimentSelected = auto()

    LaunchDashboard = auto()


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
