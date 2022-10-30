import pickle
from typing import Any


class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, __module_name: str, __global_name: str) -> Any:
        if __module_name == "__main__":
            __module_name = "HMMTrainer"
        return super().find_class(__module_name, __global_name)
