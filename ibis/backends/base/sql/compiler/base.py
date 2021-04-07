import abc


class DML(abc.ABC):
    @abc.abstractmethod
    def compile(self):
        pass


class DDL(abc.ABC):
    @abc.abstractmethod
    def compile(self):
        pass
