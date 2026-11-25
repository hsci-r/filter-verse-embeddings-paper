"""FILTER verse representations for Finnic oral poetry."""

__version__ = '1.0.0'


def __getattr__(name):
    if name == 'FilterVerseEncoder':
        from .inference import FilterVerseEncoder
        return FilterVerseEncoder
    raise AttributeError(name)
