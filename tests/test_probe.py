

def test_patch_config_reads_a_pimm_export_directory(tmp_path):
    """`patch_config_from_checkpoint` must accept a `pimm export` DIRECTORY.

    load_probe_model learned about export dirs when pimm training landed, but
    this did not, so probing a pimm-trained model died in torch.load with
    `IsADirectoryError` before reaching the loader that would have coped. The
    tokenizer geometry lives in config.json's transform list, not the model
    section, because it describes how coefficients become tokens rather than the
    architecture.
    """
    import json

    from helix.model.checkpoint import patch_config_from_checkpoint

    d = tmp_path / "export"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"")          # only presence matters here
    (d / "config.json").write_text(json.dumps({
        "model": {"d": 32},
        "transform": [
            {"type": "CoeffCollect"},
            {"type": "CoeffTokenize", "cfg": {"cell_t": "grid_center", "pw": 16, "pt": 8}},
        ],
    }))
    pc = patch_config_from_checkpoint(str(d))
    assert pc is not None, "export dir returned no patch config"
    assert pc.cell_t == "grid_center", (
        f"cell_t must come from the export, got {pc.cell_t!r} - falling back to "
        f"PatchConfig()'s 'centroid' default silently feeds the model a time "
        f"coordinate it never trained on")


def test_patch_config_export_without_tokenizer_returns_none(tmp_path):
    """'not recorded' must stay distinguishable from 'recorded as the default'."""
    import json

    from helix.model.checkpoint import patch_config_from_checkpoint

    d = tmp_path / "export"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"")
    (d / "config.json").write_text(json.dumps({"model": {"d": 32}, "transform": []}))
    assert patch_config_from_checkpoint(str(d)) is None
