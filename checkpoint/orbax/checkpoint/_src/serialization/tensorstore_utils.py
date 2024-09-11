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

"""TensorStore serialization helper functions."""

import os
import re
from typing import  Any, Optional, Union


DEFAULT_DRIVER = 'file'

PROCESS_SUBDIR_PREFIX = 'ocdbt.process_'
_OCDBT_PROCESS_ID_RE = r'[A-Za-z0-9]+'
DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2**31  # 2 GiB

ZARR_VER2 = 'zarr'
ZARR_VER3 = 'zarr3'

_GCS_PATH_RE = r'^gs://([^/]*)/(.*)$'


JsonSpec = dict[str, Any]


def _get_kvstore_for_gcs(ckpt_path: str) -> JsonSpec:
  m = re.fullmatch(_GCS_PATH_RE, ckpt_path, re.DOTALL)
  if m is None:
    raise ValueError(
        'The ckpt_path should contain the bucket name and the '
        f'file path inside the bucket. Got: {ckpt_path}'
    )
  gcs_bucket = m.group(1)
  path_without_bucket = m.group(2)
  return {'driver': 'gcs', 'bucket': gcs_bucket, 'path': path_without_bucket}


def build_kvstore_tspec(
    directory: str,
    name: Optional[str] = None,
    *,
    use_ocdbt: bool = True,
    process_id: Optional[Union[int, str]] = None,
    ocdbt_target_data_file_size: Optional[int] = None,
) -> JsonSpec:
  """Constructs a spec for a Tensorstore KvStore.

  Args:
    directory: Base path (key prefix) of the KvStore, used by the underlying
      file driver.
    name: Name (filename) of the parameter.
    use_ocdbt: Whether to use OCDBT driver.
    process_id: [only used with OCDBT driver] If provided,
      `{directory}/ocdbt.process_{process_id}` path is used as the base path.
      If a string, must conform to [A-Za-z0-9]+ pattern.
    ocdbt_target_data_file_size: [only used with OCDBT driver] Specifies the
      target size (in bytes) of each OCDBT data file.

  Returns:
    A Tensorstore KvStore spec in dictionary form.
  """
  default_driver = DEFAULT_DRIVER
  # Normalize path to exclude trailing '/'. In GCS path case, we will need to
  # fix the path prefix to add back the stripped '/'.
  directory = os.path.normpath(directory).replace('gs:/', 'gs://')
  is_gcs_path = directory.startswith('gs://')
  kv_spec = {}

  if use_ocdbt:
    if not is_gcs_path and not os.path.isabs(directory):
      raise ValueError(f'Checkpoint path should be absolute. Got {directory}')
    if process_id is not None:
      process_id = str(process_id)
      if re.fullmatch(_OCDBT_PROCESS_ID_RE, process_id) is None:
        raise ValueError(
            f'process_id must conform to {_OCDBT_PROCESS_ID_RE} pattern'
            f', got {process_id}'
        )
      directory = os.path.join(
          directory, f'{PROCESS_SUBDIR_PREFIX}{process_id}'
      )
    base_driver_spec = (
        directory
        if is_gcs_path
        else {'driver': default_driver, 'path': str(directory)}
    )
    kv_spec.update({
        'driver': 'ocdbt',
        'base': base_driver_spec,
    })
    if name is not None:
      kv_spec['path'] = name

    kv_spec.update({  # pytype: disable=attribute-error
        # Enable read coalescing.  This feature merges adjacent read_ops into
        # one, which could reduce I/O ops by a factor of 10. This is especially
        # beneficial for unstacked models.
        'experimental_read_coalescing_threshold_bytes': 1000000,
        'experimental_read_coalescing_merged_bytes': 500000000000,
        'experimental_read_coalescing_interval': '1ms',
        # References the cache specified in ts.Context.
        'cache_pool': 'cache_pool#ocdbt',
    })
    # TODO: b/354139177 - double-check this option and its default value are
    # taking effect as expected.
    if ocdbt_target_data_file_size:
      kv_spec['target_data_file_size'] = ocdbt_target_data_file_size
  else:
    if name is None:
      path = directory
    else:
      path = os.path.join(directory, name)
    if is_gcs_path:
      kv_spec = _get_kvstore_for_gcs(path)
    else:
      kv_spec = {'driver': default_driver, 'path': path}

  return kv_spec


def add_ocdbt_write_options(kvstore_tspec: JsonSpec) -> None:
  """Adds write-specific options to a TensorStore OCDBT KVStore spec."""
  kvstore_tspec['config'] = {
      # Store .zarray metadata inline but not large chunks.
      'max_inline_value_bytes': 1024,
      # Large value allows a single root node to support faster traversal.
      'max_decoded_node_bytes': 100000000,
      # There won't be any concurrent writes by multiple machines to the same
      # OCDBT database.  Therefore, we can use the simpler and more efficient
      # single-file manifest format in all cases.
      'manifest_kind': 'single',
  }
  # assume_config avoids writing an initial empty manifest to ensure a
  # consistent configuration, since Orbax never writes to the same OCDBT
  # database concurrently from multiple processes.
  kvstore_tspec.update(assume_config=True)