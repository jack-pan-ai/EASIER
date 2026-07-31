# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from unittest import mock

from easier.core.passes import codegen


def test_cpu_snapshot_excludes_mutable_cuda_include_tree(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "merged_binding_cpu.cpp").write_text("// cpu binding\n")
    (project / "merged_spmv.h").write_text("// generated cpu header\n")
    cuda_include = project / "include"
    cuda_include.mkdir()
    (cuda_include / "data_struct_shared.cuh").write_text("// shared tensor\n")
    (cuda_include / "merged_spmv.cuh").write_text("// mutable cuda output A\n")

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    with (
        mock.patch.object(codegen, "_repo_root", return_value=str(project)),
        mock.patch.object(codegen.tempfile, "mkdtemp", return_value=str(snapshot)),
    ):
        source, include, first_hash = codegen._snapshot_generated_source_cpu("region")
    assert Path(source).read_text() == "// cpu binding\n"
    assert (snapshot / "merged_spmv.h").read_text() == "// generated cpu header\n"
    assert Path(include).is_dir()
    assert [path.name for path in Path(include).iterdir()] == ["data_struct_shared.cuh"]
    assert (Path(include) / "data_struct_shared.cuh").read_text() == "// shared tensor\n"

    # A preceding CUDA generation may rewrite its include tree.  A fresh CPU
    # snapshot must retain the same hash and bytes.
    (cuda_include / "merged_spmv.cuh").write_text("// mutable cuda output B\n")
    second_snapshot = tmp_path / "snapshot-second"
    second_snapshot.mkdir()
    with (
        mock.patch.object(codegen, "_repo_root", return_value=str(project)),
        mock.patch.object(
            codegen.tempfile, "mkdtemp", return_value=str(second_snapshot)
        ),
    ):
        _, second_include, second_hash = codegen._snapshot_generated_source_cpu("region")
    assert [path.name for path in Path(second_include).iterdir()] == [
        "data_struct_shared.cuh"
    ]
    assert second_hash == first_hash
