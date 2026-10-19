from typing import Any


class PreprocessorBase(object):

    def __init__(self):
        super().__init__() 
        
    def forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError("Subclasses must implement this method")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def to(self, device):
        pass

    @property
    def device(self):
        raise NotImplementedError("Subclasses must implement this method")

