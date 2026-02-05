import pickle


class Solver:
    """Base solver class with save/load functionality."""
    @staticmethod
    def save(structure, filename):
        with open(filename + '.pkl', 'wb') as file:
            pickle.dump(structure, file)
