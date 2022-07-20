import logging
import contextlib
from typing import Any, List, Tuple, Union


class Logger(logging.Logger):
    def __init__(
        self,
        name,
        level,
        filename=None,
        filemode="a",
        stream=None,
        format_str=None,
        dateformat=None,
        style="%",
    ):
        super().__init__(name, level)
        if filename is not None:
            handler = logging.FileHandler(filename, filemode)  # type:ignore
        else:
            handler = logging.StreamHandler(stream)  # type:ignore
        self._formatter = logging.Formatter(format_str, dateformat, style)
        handler.setFormatter(self._formatter)
        super().addHandler(handler)

    def add_filehandler(self, log_filename):
        filehandler = logging.FileHandler(log_filename)
        filehandler.setFormatter(self._formatter)
        self.addHandler(filehandler)


logger = Logger(
    name="tex_gen", level=logging.INFO, format_str="%(asctime)-15s %(message)s"
)


@contextlib.contextmanager
def dummy_context_mgr():
    yield None



class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax.

    Ref: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d4b2afe9c27e3c305b721bc886d2cb5229458eba/dnnlib/util.py#L40
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]