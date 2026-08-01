# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
from typing import Dict, List, Sequence, Set, Tuple

import torch
from torch import nn
from torch.fx.graph import Graph
from torch.fx.node import Node

from easier.core.utils import logger
import easier.core.module as esr

from easier.core.passes.tensor_grouping import \
    EasierTensorGroup, get_node_tensor_group
from easier.core.passes.utils import \
    EasierInterpreter, SubmodNameAllocator, \
    normalize_reducer_call_into_args
from easier.core.runtime.data_loader import InMemoryTensorLoader
from easier.core.runtime.dist_env import get_runtime_dist_env


def reducers_have_equivalent_index_layout(
    lhs: esr.Reducer,
    rhs: esr.Reducer,
    cache: Dict[Tuple[int, int], bool] | None = None,
) -> bool:
    """Return whether two Reducers provably impose the same CSR layout.

    Reducer binding normally chooses one Reducer to define a tensor group's
    CSR layout and inserts an identity Selector before every other Reducer.
    Two separate Reducer instances may nevertheless describe exactly the
    same mapping (for example, the same physical operator in two program
    phases).  In that case the Selector is redundant and, more importantly,
    prevents the surrounding select/map/reduce chain from fusing.

    Be deliberately conservative here.  Shared loader identity is an exact
    proof.  Otherwise require identical metadata and an exact decoded-value
    comparison; source-path equality alone does not prove immutable contents.
    """
    if lhs is rhs:
        return True
    if lhs.n != rhs.n:
        return False

    lhs_loader = lhs.easier_data_loader
    rhs_loader = rhs.easier_data_loader
    if lhs_loader is rhs_loader:
        return True
    if (
        lhs_loader.shape != rhs_loader.shape
        or lhs_loader.dtype != rhs_loader.dtype
        or lhs_loader.device != rhs_loader.device
    ):
        return False

    cache_key = tuple(sorted((id(lhs_loader), id(rhs_loader))))
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    # Rank 0 makes the exact decision and broadcasts it because even a tiny
    # worker-local discrepancy must not produce divergent compiled graphs.
    dist_env = get_runtime_dist_env()
    if dist_env.rank == 0:
        if (
            isinstance(lhs_loader, InMemoryTensorLoader)
            and isinstance(rhs_loader, InMemoryTensorLoader)
        ):
            # Avoid a full scan when two operators literally share the same
            # tensor; otherwise torch.equal is exact and allocates no copy.
            equivalent = (
                lhs_loader.tensor is rhs_loader.tensor
                or torch.equal(lhs_loader.tensor, rhs_loader.tensor)
            )
        else:
            # Other loaders may refer to mutable external data.  Path/repr/
            # hash identity is insufficient, so compare decoded chunks.
            equivalent = False
            try:
                missing = object()
                chunk_pairs = itertools.zip_longest(
                    lhs_loader.partially_load_by_chunk(8 * 1024 * 1024),
                    rhs_loader.partially_load_by_chunk(8 * 1024 * 1024),
                    fillvalue=missing,
                )
                equivalent = all(
                    lhs_chunk is not missing
                    and rhs_chunk is not missing
                    and torch.equal(lhs_chunk, rhs_chunk)
                    for lhs_chunk, rhs_chunk in chunk_pairs
                )
            except (NotImplementedError, RuntimeError, TypeError, ValueError):
                # A custom loader that cannot prove equality retains the
                # existing conservative CSR-Selector insertion.
                equivalent = False
        dist_env.broadcast_object_list(0, [equivalent])
    else:
        [equivalent] = dist_env.broadcast_object_list(0)

    if cache is not None:
        cache[cache_key] = equivalent
    return equivalent


class ReducerBinder(EasierInterpreter[None]):
    def __init__(self, modules: Sequence[esr.Module], graphs: Sequence[Graph]):
        super().__init__(modules, graphs)

        # Not all TensorGroup is bound to a Reducer.
        self.tengrp2reducer: Dict[
            EasierTensorGroup, Dict[esr.Reducer, int]
        ] = {}

    def if_call_module(self, submod: nn.Module) -> None:
        if not isinstance(submod, esr.Reducer):
            return

        args = self.current_node.args
        kwargs = self.current_node.kwargs
        input_node, opt_inplace_out_node = \
            normalize_reducer_call_into_args(*args, **kwargs)
        assert isinstance(input_node, Node)

        tgrp = get_node_tensor_group(input_node)
        assert tgrp is not None

        nnodes = self.tengrp2reducer.setdefault(tgrp, {})
        nnodes[submod] = nnodes.get(submod, 0) + 1


class CsrSelectorInserter(EasierInterpreter[None]):
    def __init__(
        self, modules: Sequence[esr.Module], graphs: Sequence[Graph],
        tgrp2reducer: Dict[EasierTensorGroup, esr.Reducer],
        equivalent_reducers: Set[esr.Reducer]
    ) -> None:
        super().__init__(modules, graphs)

        self.tgrp2reducer = tgrp2reducer
        self.equivalent_reducers = equivalent_reducers
        self.selector_name_allocator = SubmodNameAllocator('csr_selector')

    def if_call_module(self, submod: nn.Module) -> None:
        if not isinstance(submod, esr.Reducer):
            return

        args = self.current_node.args
        kwargs = self.current_node.kwargs
        input_node, opt_inplace_out_node = \
            normalize_reducer_call_into_args(*args, **kwargs)
        assert isinstance(input_node, Node)

        tgrp = get_node_tensor_group(input_node)
        assert tgrp is not None

        bound_reducer = self.tgrp2reducer[tgrp]
        if (
            bound_reducer is not submod
            and submod not in self.equivalent_reducers
        ):

            # TODO reuse selector instance if <tgrp, reducer> met.

            selector_attrname = self.selector_name_allocator.alloc_name(
                self.current_module, hint=self.current_node.name
            )

            # Collectively create and insert.
            # During module dumping, this Selector will be dumped
            # as normal Selectors, and during loading this Selector will be
            # created again -- it's ok as this is merely a data loader,
            # till its `.idx` get directly overwritten with the loaded data.
            csr_selector = esr.Selector(esr.arange(
                submod.easier_data_loader.shape[0],
                dtype=submod.easier_data_loader.dtype,
                device=submod.easier_data_loader.device
            ))
            csr_selector.easier_hint_name = \
                f"{submod.easier_hint_name}.{selector_attrname}"
            # TODO if we reuse the Selector instance the naming will be
            # as consistent as dataflow_distribution
            # f"{submod.easier_hint_name}.reorderingSelector"

            self.current_module.add_module(selector_attrname, csr_selector)

            with self.current_graph.inserting_before(self.current_node):
                csr_selector_node = self.current_graph.call_module(
                    selector_attrname, (input_node,)
                )
                self.current_node.replace_input_with(
                    input_node, csr_selector_node
                )

            logger.info(f"Insert arange-Selector for {self.current_node.name}")


def bind_reducer(modules: List[esr.Module], graphs: List[Graph]):
    """
    Analyze which Reducer decides (CSR-encoded) layout of each tensor.
    If one tensor is used by multiple Reducers, insert Selectors for
    extra Reducers.
    """
    reducer_binder = ReducerBinder(modules, graphs)
    reducer_binder.run()

    # pick one Reducer instance with the maximum number of Nodes.
    target_reducers: Dict[EasierTensorGroup, esr.Reducer] = {}
    equivalent_reducers: Set[esr.Reducer] = set()
    equivalence_cache: Dict[Tuple[int, int], bool] = {}
    for grp, reducer2nnodes in reducer_binder.tengrp2reducer.items():

        # If multiple Reducers are reducing the same input tensor group,
        # we first sort them by "fullness" i.e. how many percentage of
        # the OUTPUT tensor group gets written.
        # (however, we ignore the sizes of those OUTPUT tensor groups for now,
        # which are the `Reducer.n`s)
        # Which could be simply calculated as `len(unique(R.idx)) / R.n`
        #
        # tuple (fullness, nnodes) are ordered lexicographically
        weighted_reducers = [
            ((float(r.easier_data_loader.count_unique()) / r.n, nnodes), r)
            for r, nnodes in reducer2nnodes.items()
        ]
        _maxweight, target = max(weighted_reducers, key=lambda tp: tp[0])

        target_reducers[grp] = target
        for reducer in reducer2nnodes:
            if (
                reducer is not target
                # Two identical Reducer mappings can share the input CSR
                # layout only when their output TensorDefs already share one
                # layout group.  Otherwise sparse reordering still needs the
                # synthetic Selector to resolve competing reducer edges.
                and reducer.easier_tensor_group
                is target.easier_tensor_group
                and reducers_have_equivalent_index_layout(
                    target, reducer, equivalence_cache
                )
            ):
                equivalent_reducers.add(reducer)
                logger.info(
                    "Reducer %s has the same index layout as bound Reducer "
                    "%s; skip the redundant CSR Selector",
                    reducer.easier_hint_name,
                    target.easier_hint_name,
                )

    selector_inserter = CsrSelectorInserter(
        modules, graphs, target_reducers, equivalent_reducers
    )
    selector_inserter.run()

    return modules, graphs
