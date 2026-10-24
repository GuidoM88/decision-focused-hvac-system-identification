from dfl_hvac.config import load_yaml


def test_yaml_loader_requires_mapping(tmp_path):
    valid = tmp_path / "valid.yaml"
    valid.write_text("a: 1\n", encoding="utf-8")
    assert load_yaml(valid) == {"a": 1}

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- a\n- b\n", encoding="utf-8")

    try:
        load_yaml(invalid)
    except ValueError as exc:
        assert "mapping" in str(exc)
    else:
        raise AssertionError("non-mapping YAML root was accepted")
