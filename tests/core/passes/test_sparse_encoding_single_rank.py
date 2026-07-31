# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import unittest
from unittest import mock

import torch

from easier.core.module import Reducer, Selector
from easier.core.passes.sparse_encoding.sparse_encoding import (
    reorder_input_by_reducer,
    rewrite_reducer_instance,
    rewrite_selector_instance,
)
from easier.core.passes.tensor_group_partition import (
    ElemPart,
    ElemPartArangeIdx,
    ElemPartReorderedArangeIdx,
)
from easier.core.passes.utils import get_selector_reducer_idx_partition
from easier.core.runtime.dist_env import DummyDistEnv


def _arange_elempart(start: int, end: int, hint: str) -> ElemPart:
    return ElemPart(
        ElemPartArangeIdx(start, end),
        torch.arange(start, end, dtype=torch.int64),
        [end - start],
        hint,
    )


class SingleRankSparseEncodingTests(unittest.TestCase):
    def setUp(self):
        self.dist_env = DummyDistEnv("cpu")
        self.env = mock.patch.dict(
            os.environ, {"EASIER_SINGLE_RANK_FAST_PATH": "1"}
        )
        self.runtime = mock.patch(
            "easier.core.passes.sparse_encoding.sparse_encoding."
            "get_runtime_dist_env",
            return_value=self.dist_env,
        )
        self.loader_runtime = mock.patch(
            "easier.core.runtime.data_loader.get_runtime_dist_env",
            return_value=self.dist_env,
        )
        self.env.start()
        self.runtime.start()
        self.loader_runtime.start()

    def tearDown(self):
        self.loader_runtime.stop()
        self.runtime.stop()
        self.env.stop()

    def test_reducer_stable_receiver_order_needs_no_runtime_reordering(self):
        reducer = Reducer(torch.tensor([2, 0, 2, 1], dtype=torch.int64), 3)
        input_elempart = _arange_elempart(0, 4, "edges")
        output_elempart = _arange_elempart(0, 3, "receivers")

        (input_gidx, output_gidx), reordered_input = \
            reorder_input_by_reducer(reducer, input_elempart, output_elempart)
        self.assertTrue(torch.equal(input_gidx, torch.tensor([1, 3, 0, 2])))
        self.assertTrue(torch.equal(output_gidx, torch.tensor([0, 1, 2, 2])))
        self.assertTrue(torch.equal(reordered_input, input_gidx))

        reordered_elempart = ElemPart(
            ElemPartReorderedArangeIdx(0, 4), reordered_input, [4], "edges:r"
        )
        rewrite_reducer_instance(
            reducer,
            input_gidx,
            output_gidx,
            reordered_elempart,
            output_elempart,
        )
        self.assertTrue(torch.equal(reducer.idx, torch.tensor([0, 1, 2, 2])))
        self.assertIsNone(reducer.easier_reordering_selector_idx)
        self.assertEqual(reducer.runtime_halos_recv_lengths, [4])
        self.assertEqual(reducer.runtime_halos_local_idxes[0].numel(), 0)

    def test_selector_directly_follows_reordered_edge_elempart(self):
        selector = Selector(torch.tensor([2, 0, 1, 2], dtype=torch.int64))
        src, _ = get_selector_reducer_idx_partition(selector)
        node_elempart = _arange_elempart(0, 3, "nodes")
        edge_elempart = ElemPart(
            ElemPartReorderedArangeIdx(0, 4),
            torch.tensor([1, 3, 0, 2], dtype=torch.int64),
            [4],
            "edges:r",
        )

        rewrite_selector_instance(
            selector,
            src,
            torch.empty((0,), dtype=torch.int64),
            node_elempart,
            edge_elempart,
        )
        self.assertTrue(torch.equal(selector.idx, torch.tensor([0, 2, 2, 1])))
        self.assertEqual(selector.runtime_halos_recv_lengths, [3])
        self.assertEqual(selector.runtime_halos_local_idxes[0].numel(), 0)


if __name__ == "__main__":
    unittest.main()
