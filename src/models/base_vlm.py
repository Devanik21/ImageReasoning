"""Base class for Vision-Language Models."""
from abc import ABC, abstractmethod


class BaseVLM(ABC):
    @abstractmethod
    def generate(self, image, prompt):
        pass
