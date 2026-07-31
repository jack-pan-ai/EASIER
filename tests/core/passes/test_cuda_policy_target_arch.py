# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from easier.core.passes import codegen


def _cuda_capability(major: int, minor: int):
    return (
        mock.patch.object(codegen.torch.cuda, "is_available", return_value=True),
        mock.patch.object(
            codegen.torch.cuda,
            "get_device_capability",
            return_value=(major, minor),
        ),
    )


class CudaPolicyTargetArchTests(unittest.TestCase):
    def test_cuda_build_uses_one_target_for_nvcc_policy_and_module_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "binding.cu"
            source.write_text("// binding\n")
            include = root / "include"
            include.mkdir()
            available, capability = _cuda_capability(9, 0)
            extension = object()
            with (
                available,
                capability,
                mock.patch.object(codegen, "load", return_value=extension) as load,
                mock.patch.object(codegen.shutil, "which", return_value=None),
            ):
                actual = codegen._build_cuda_extension_from_src(
                    "region", str(source), str(include), "abc123"
                )

        self.assertIs(actual, extension)
        call = load.call_args
        self.assertEqual(call.kwargs["name"], "new_region_sm_90_abc123")
        flags = call.kwargs["extra_cuda_cflags"]
        self.assertIn("-arch=sm_90", flags)
        self.assertIn("-DEASIER_CUDA_TARGET_ARCH=900", flags)
        self.assertIn("-DEASIER_CUDA_F64_ITEMS_PER_THREAD=6", flags)

    def test_cuda_build_accepts_validated_f64_items_per_thread_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "binding.cu"
            source.write_text("// binding\n")
            include = root / "include"
            include.mkdir()
            available, capability = _cuda_capability(9, 0)
            with (
                available,
                capability,
                mock.patch.dict(
                    os.environ,
                    {"EASIER_CUDA_F64_ITEMS_PER_THREAD": "4"},
                ),
                mock.patch.object(codegen, "load", return_value=object()) as load,
                mock.patch.object(codegen.shutil, "which", return_value=None),
            ):
                codegen._build_cuda_extension_from_src(
                    "region", str(source), str(include), "abc123"
                )

        self.assertIn(
            "-DEASIER_CUDA_F64_ITEMS_PER_THREAD=4",
            load.call_args.kwargs["extra_cuda_cflags"],
        )

    def test_invalid_f64_items_per_thread_is_rejected(self):
        with mock.patch.dict(
            os.environ,
            {"EASIER_CUDA_F64_ITEMS_PER_THREAD": "0"},
        ):
            with self.assertRaisesRegex(ValueError, "between 1 and 16"):
                codegen._cuda_f64_items_per_thread()

    def test_cuda_build_fingerprint_includes_f64_items_per_thread(self):
        with mock.patch.dict(
            os.environ,
            {"EASIER_CUDA_F64_ITEMS_PER_THREAD": "4"},
        ):
            items_four = codegen._build_fingerprint("cuda")
        with mock.patch.dict(
            os.environ,
            {"EASIER_CUDA_F64_ITEMS_PER_THREAD": "6"},
        ):
            items_six = codegen._build_fingerprint("cuda")

        self.assertNotEqual(items_four, items_six)

    def test_cuda_persistent_cache_is_namespaced_by_target_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(codegen, "_cache_root_dir", return_value=directory):
                available, capability = _cuda_capability(9, 0)
                with available, capability:
                    sm90 = Path(codegen._cache_entry_dir("cuda", "region", "key"))
                available, capability = _cuda_capability(8, 0)
                with available, capability:
                    sm80 = Path(codegen._cache_entry_dir("cuda", "region", "key"))
                cpu = Path(codegen._cache_entry_dir("cpu", "region", "key"))

            self.assertEqual(
                sm90, root / "v2" / "cuda" / "sm_90" / "region" / "key"
            )
            self.assertEqual(
                sm80, root / "v2" / "cuda" / "sm_80" / "region" / "key"
            )
            self.assertNotEqual(sm90, sm80)
            self.assertEqual(cpu, root / "v2" / "cpu" / "region" / "key")

    def test_cache_key_changes_with_source_and_build_fingerprint(self):
        with mock.patch.object(codegen, "_build_fingerprint", return_value="build-a"):
            key_source_a = codegen._verified_cache_key("cpu", "source-a")
            key_source_b = codegen._verified_cache_key("cpu", "source-b")
        with mock.patch.object(codegen, "_build_fingerprint", return_value="build-b"):
            key_build_b = codegen._verified_cache_key("cpu", "source-a")

        self.assertNotEqual(key_source_a, key_source_b)
        self.assertNotEqual(key_source_a, key_build_b)

    def test_build_fingerprint_includes_cache_driver_source(self):
        with mock.patch.object(
            codegen, "_driver_source_hash", return_value="driver-a"
        ):
            driver_a = codegen._build_fingerprint("cpu")
        with mock.patch.object(
            codegen, "_driver_source_hash", return_value="driver-b"
        ):
            driver_b = codegen._build_fingerprint("cpu")

        self.assertNotEqual(driver_a, driver_b)

    def test_policy_selector_uses_stable_build_target_not_cub_host_macro(self):
        policy = (
            Path(codegen.__file__).parent
            / "code_gen"
            / "include"
            / "merged_policy.cuh"
        ).read_text()
        selector = policy.split("// Define the policy", 1)[1]

        self.assertIn("#ifndef EASIER_CUDA_TARGET_ARCH", selector)
        self.assertIn("#if (EASIER_CUDA_TARGET_ARCH >= 900)", selector)
        self.assertIn("#elif (EASIER_CUDA_TARGET_ARCH >= 600)", selector)
        self.assertIn("using PtxPolicy = merged::Policy900<ValueT>;", selector)
        self.assertIn("using PtxPolicy = merged::Policy600<ValueT>;", selector)
        self.assertNotIn("#if (CUB_PTX_ARCH", selector)


if __name__ == "__main__":
    unittest.main()
