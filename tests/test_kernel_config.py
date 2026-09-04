import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / 'native_sparse_attention_pytorch' / 'kernel_config.py'
SPEC = importlib.util.spec_from_file_location('kernel_config_under_test', MODULE_PATH)
kernel_config = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kernel_config)


def test_head_dimension_128_uses_safe_tile_on_96kib_gpu():
    assert kernel_config.select_kernel_config(128, 101376) == (8, 4, 1)
    assert kernel_config.select_kernel_config(96, 101376) == (8, 4, 1)


def test_large_shared_memory_gpu_keeps_default_tile():
    assert kernel_config.select_kernel_config(128, 163840) == (16, 8, 1)
    assert kernel_config.select_kernel_config(64, 101376) == (16, 4, 1)


def test_unknown_shared_memory_limit_is_conservative():
    assert kernel_config.select_kernel_config(128, None) == (8, 4, 1)


def test_shared_memory_limit_reads_driver_properties(monkeypatch):
    class Properties:
        shared_memory_per_block = 49152
        shared_memory_per_block_optin = 101376

    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: Properties())

    assert kernel_config.get_shared_memory_limit(torch.device('cuda')) == 101376
    assert kernel_config.get_shared_memory_limit(torch.device('cpu')) is None


def test_launch_retries_resource_error_with_safe_tile():
    class OutOfResources(Exception):
        pass

    calls = []

    def launch(*config):
        calls.append(config)
        if len(calls) == 1:
            raise OutOfResources('out of resource: shared memory')
        return 'launched'

    result = kernel_config.launch_with_shared_memory_fallback(launch, (16, 8, 1))

    assert result == 'launched'
    assert calls == [(16, 8, 1), (8, 4, 1)]


def test_launch_does_not_hide_non_resource_errors():
    calls = []

    def launch(*config):
        calls.append(config)
        raise RuntimeError('kernel compilation failed')

    with pytest.raises(RuntimeError, match='kernel compilation failed'):
        kernel_config.launch_with_shared_memory_fallback(launch, (16, 8, 1))

    assert calls == [(16, 8, 1)]
