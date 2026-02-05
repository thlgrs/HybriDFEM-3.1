from abc import ABC, abstractmethod


class BaseFE(ABC):
    """Abstract base class for finite elements."""

    @abstractmethod
    def make_connect(self, connect, node_number):
        pass

    @abstractmethod
    def get_mass(self):
        pass

    @abstractmethod
    def get_k_glob(self):
        pass

    @abstractmethod
    def get_k_glob0(self):
        pass

    @abstractmethod
    def get_k_glob_LG(self):
        pass

    @abstractmethod
    def get_p_glob(self, q_glob):
        pass
