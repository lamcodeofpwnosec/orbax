# Copyright 2024 The Orbax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Basic type definitions for working with arrays and fragments."""

from typing import Union


Shape = tuple[int, ...]

# Indexing an np.ndarray with an empty tuple gives an array of the same shape,
# *unless* the array is zero-dimensional in which case the result is a scalar.
# Indexing an np.ndarray with Ellipsis always gives an array of the same shape.
# For that reason we use Ellipsis instead of an empty tuple, to avoid needing
# a bunch of special-case code to deal with zero-dimensional arrays.
NdSlice = Union[tuple[slice, ...], type(Ellipsis)]

Index = tuple[slice, ...]