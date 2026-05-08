# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections import namedtuple
from keyword import iskeyword
from textwrap import dedent, indent
from typing import Any, Callable, Dict, Iterable, TypeVar

T = TypeVar("T")


def unsqueeze_right(arr: T, ndim: int) -> T:
    """Append 1-sized dimensions to `arr` to create a result with `ndim` dimensions."""
    return arr[(...,) + (None,) * (ndim - arr.ndim)]


def unsqueeze_left(arr: T, ndim: int) -> T:
    """Preppend 1-sized dimensions to `arr` to create a result with `ndim` dimensions."""
    return arr[(None,) * (ndim - arr.ndim)]

T = TypeVar("T")


def is_variable(name):
    """Returns True if `name` is a valid Python variable name and also not a keyword."""
    return name.isidentifier() and not iskeyword(name)


class ComponentStore:
    """
    Represents a storage object for other objects (specifically functions) keyed to a name with a description.
    """

    _Component = namedtuple("Component", ("description", "value"))

    def __init__(self, name: str, description: str) -> None:
        self.components: Dict[str, self._Component] = {}
        self.name: str = name
        self.description: str = description

        self.__doc__ = f"Component Store '{name}': {description}\n{self.__doc__ or ''}".strip()

    def add(self, name: str, desc: str, value: T) -> T:
        if not is_variable(name):
            raise ValueError("Name of component must be valid Python identifier")

        self.components[name] = self._Component(desc, value)
        return value

    def add_def(self, name: str, desc: str) -> Callable:
        def deco(func):
            return self.add(name, desc, func)
        return deco

    def __contains__(self, name: str) -> bool:
        return name in self.components

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Iterable:
        for k, v in self.components.items():
            yield k, v.value

    def __str__(self):
        result = f"Component Store '{self.name}': {self.description}\nAvailable components:"
        for k, v in self.components.items():
            result += f"\n* {k}:"
            if hasattr(v.value, "__doc__"):
                doc = indent(dedent(v.value.__doc__.lstrip("\n").rstrip()), "    ")
                result += f"\n{doc}\n"
            else:
                result += f" {v.description}"
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self.components:
            return self.components[name].value
        else:
            return self.__getattribute__(name)

    def __getitem__(self, name: str) -> Any:
        if name in self.components:
            return self.components[name].value
        else:
            raise ValueError(f"Component '{name}' not found")
