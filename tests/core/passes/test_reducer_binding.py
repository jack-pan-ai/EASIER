# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import contextmanager
from unittest.mock import patch

import torch

import easier as esr
from easier.core import passes
from easier.core.jit import EasierTracer
from easier.core.passes.reducer_binding import (
    reducers_have_equivalent_index_layout,
)
from easier.core.runtime.dist_env import CommBackendConfig, DummyDistEnv


@contextmanager
def _dummy_dist_env():
    with (
        patch(
            "easier.core.runtime.dist_env._get_or_init_dist_env",
            new=lambda _device_type: DummyDistEnv("cpu"),
        ),
        patch(
            "easier.core.runtime.dist_env._comm_backend_config",
            CommBackendConfig("gloo"),
        ),
        patch(
            "easier.core.runtime.dist_env._runtime_device_type",
            "cpu",
        ),
    ):
        yield


def _trace_and_group(*modules):
    graphs = [EasierTracer().trace(module) for module in modules]
    return passes.group_tensors(list(modules), graphs)


def test_equivalent_reducer_index_layout_requires_exact_values():
    idx = torch.tensor([0, 1, 0, 2], dtype=torch.int64)
    lhs = esr.Reducer(idx.clone(), n=3, reduce="sum")
    rhs = esr.Reducer(idx.clone(), n=3, reduce="sum")

    with _dummy_dist_env():
        assert reducers_have_equivalent_index_layout(lhs, rhs)

    different_values = esr.Reducer(
        torch.tensor([0, 1, 2, 0], dtype=torch.int64),
        n=3,
        reduce="sum",
    )
    different_n = esr.Reducer(idx.clone(), n=4, reduce="sum")
    different_reduce = esr.Reducer(idx.clone(), n=3, reduce="amax")
    different_dtype = esr.Reducer(idx.to(torch.int32), n=3, reduce="sum")

    with _dummy_dist_env():
        assert not reducers_have_equivalent_index_layout(
            lhs, different_values
        )
        assert not reducers_have_equivalent_index_layout(lhs, different_n)
        # Reduction operation changes arithmetic, not the CSR input layout.
        assert reducers_have_equivalent_index_layout(lhs, different_reduce)
        assert not reducers_have_equivalent_index_layout(
            lhs, different_dtype
        )


def _make_phases(first_idx, second_idx, *, share_output_group):
    shared_input = esr.Tensor(torch.arange(4.0), mode="partition")
    shared_output = esr.Tensor(torch.zeros(3), mode="partition")

    class Phase(esr.Module):
        def __init__(self, reducer_idx):
            super().__init__()
            self.shared_input = shared_input
            self.reducer = esr.Reducer(reducer_idx, n=3)
            if share_output_group:
                self.shared_output = shared_output

        def forward(self):
            values = self.shared_input * 2.0
            if share_output_group:
                self.reducer(values, out=self.shared_output)
            else:
                self.reducer(values)

    first = Phase(first_idx)
    second = Phase(second_idx)
    first.easier_hint_name = "first"
    second.easier_hint_name = "second"
    first.reducer.easier_hint_name = "first.reducer"
    second.reducer.easier_hint_name = "second.reducer"
    return first, second


def _run_full_sparse_aot(first, second):
    modules, graphs = _trace_and_group(first, second)
    passes.bind_reducer(modules, graphs)
    passes.group_tensors(modules, graphs)
    for module in modules:
        module.partition_mode = "evenly"
    passes.partition_tensor_groups(modules, graphs)
    passes.encode_sparsity(modules, graphs)
    return modules, graphs


def _count_csr_selectors(modules):
    return sum(
        name.startswith("csr_selector")
        for module in modules
        for name, _submodule in module.named_children()
    )


def test_equivalent_reducers_with_shared_output_group_skip_csr_selector():
    idx = torch.tensor([0, 1, 0, 2], dtype=torch.int64)
    first, second = _make_phases(
        idx.clone(), idx.clone(), share_output_group=True
    )
    with _dummy_dist_env():
        modules, _graphs = _run_full_sparse_aot(first, second)

    assert _count_csr_selectors(modules) == 0
    assert first.reducer.easier_index_status == "rewritten"
    assert second.reducer.easier_index_status == "rewritten"


def test_equivalent_reducers_with_distinct_output_groups_keep_selector():
    idx = torch.tensor([0, 1, 0, 2], dtype=torch.int64)
    first, second = _make_phases(
        idx.clone(), idx.clone(), share_output_group=False
    )
    with _dummy_dist_env():
        modules, _graphs = _run_full_sparse_aot(first, second)

    assert _count_csr_selectors(modules) == 1


def test_different_reducers_keep_the_csr_selector():
    first, second = _make_phases(
        torch.tensor([0, 1, 0, 2], dtype=torch.int64),
        torch.tensor([0, 1, 2, 0], dtype=torch.int64),
        share_output_group=True,
    )
    with _dummy_dist_env():
        modules, _graphs = _run_full_sparse_aot(first, second)

    assert _count_csr_selectors(modules) == 1
