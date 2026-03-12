def require_graph(problem):
    """Verify the problem exposes all attributes needed by graph-search algorithms."""
    for attr in ('start', 'goal', 'adj', 'weights'):
        if not hasattr(problem, attr):
            raise ValueError(
                f"Graph-search algorithms require a problem with "
                f"'start', 'goal', 'adj', and 'weights'. Missing: '{attr}'."
            )


def require_heuristic(problem, algo_name: str):
    """Verify the problem exposes a heuristic callable (needed by Greedy BFS / A*)."""
    if not hasattr(problem, 'heuristic') or not callable(problem.heuristic):
        raise ValueError(
            f"{algo_name} requires problem.heuristic(node) -> float."
        )
