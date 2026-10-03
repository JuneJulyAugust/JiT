"""A recognizable failure for intentionally unfinished learning exercises."""


class ExerciseNotImplemented(NotImplementedError):
    def __init__(self, exercise: str, description: str) -> None:
        super().__init__(
            f"{exercise}: {description}. Implement this exercise; see toy_experiment/TODO.md."
        )
