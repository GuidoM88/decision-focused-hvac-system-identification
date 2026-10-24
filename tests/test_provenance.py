import json

from dfl_hvac.provenance import describe_external_file, write_run_manifest


def test_manifest_records_file_checksum(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("energyplus-input", encoding="utf-8")

    item = describe_external_file(source, source="test")
    manifest = tmp_path / "manifest.json"
    write_run_manifest(manifest, files=[item], metadata={"energyplus": "test-version"})

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["files"][0]["sha256"] == item.sha256
    assert payload["metadata"]["energyplus"] == "test-version"
