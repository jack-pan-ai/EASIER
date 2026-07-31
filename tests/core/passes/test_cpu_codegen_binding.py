# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from easier.core.passes.code_gen import merged_gen_binding_cpu


class CpuCodegenBindingTest(unittest.TestCase):
    def test_forwarded_reducer_output_uses_caller_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template_dir = root / "cpu_template"
            template_dir.mkdir()
            codegen_dir = Path(merged_gen_binding_cpu.__file__).parent
            shutil.copyfile(
                codegen_dir
                / "cpu_template"
                / "merged_binding_template_cpu.cpp",
                template_dir / "merged_binding_template_cpu.cpp",
            )
            inputs = [{
                "name": "input_x",
                "target": "input_x",
                "dtype": "scalar_t",
                "op": "placeholder",
                "args": (),
                "dtype_data": torch.float64,
                "shape_ahead": 2,
            }]
            outputs = [{
                "name": "reduce_out",
                "target": "reducer",
                "shape": (4,),
            }]

            merged_gen_binding_cpu.generate_cpu_binding_code(
                root,
                inputs,
                outputs,
                [],
                forwarded_output_indices=[0],
            )
            binding = (root / "merged_binding_cpu.cpp").read_text()

        destination = "forward_out_0_reduce_out"
        self.assertIn(f"torch::Tensor {destination}", binding)
        self.assertNotIn("torch::empty({num_rows, 4}", binding)
        self.assertIn(
            f"reinterpret_cast<ValueT*>({destination}.data_ptr())", binding
        )
        self.assertIn(f"{destination}.numel() == num_rows * 4", binding)
        self.assertIn(f"{destination} must be a CPU tensor", binding)
        self.assertIn(f"{destination} must be contiguous", binding)
        self.assertIn(f"{destination} dtype mismatch", binding)
        self.assertIn(f"return std::make_tuple({destination});", binding)


if __name__ == "__main__":
    unittest.main()
