"""Base class for Vision-Language Models."""

class BaseVLM:
    def generate(self, image, prompt):
        raise NotImplementedError
