"""CPU-only data integrity checks; no remote datasets or training required."""
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "prepare_data.py"
SPEC = importlib.util.spec_from_file_location("rfdetr_prepare_data", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
prepare = module.prepare
DataError = module.DataError


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    for split, colors, ids in (("train", (10, 20, 30, 40), (7, 0)),
                               ("valid", (50, 60), (101, 22)),
                               ("test", (70, 80), (101, 22))):
        folder = source / split
        folder.mkdir()
        images, annotations = [], []
        for i, color in enumerate(colors, 1):
            filename = f"frame_{i}.png"
            Image.new("RGB", (8, 8), (color, 0, 0)).save(folder / filename)
            images.append({"id": i, "file_name": filename, "width": 8, "height": 8})
            annotations.append({"id": i, "image_id": i, "category_id": ids[(i - 1) % 2],
                                "bbox": [1, 1, 2, 3], "area": 6, "iscrowd": 0})
        write(folder / "_annotations.coco.json", {
            "categories": [{"id": 900, "name": "unused-parent", "supercategory": "none"},
                           {"id": ids[0], "name": "pole", "supercategory": "obstacle"},
                           {"id": ids[1], "name": "stairs", "supercategory": "obstacle"}],
            "images": images, "annotations": annotations})
    return source, tmp_path / "prepared"


def modify(source, split, callback):
    path = source / split / "_annotations.coco.json"
    data = read(path)
    callback(data)
    write(path, data)


def test_full_dataset_keeps_real_class_zero_and_remaps_each_split(dataset):
    source, output = dataset
    report = prepare(source, output)
    assert read(output / "class_names.json") == ["pole", "stairs"]
    assert report["removed_unannotated_category_names"] == ["unused-parent"]
    for split, count in (("train", 4), ("valid", 2), ("test", 2)):
        normalized = read(output / split / "_annotations.coco.json")
        assert len(normalized["images"]) == count
        assert [a["category_id"] for a in normalized["annotations"][:2]] == [1, 2]
        assert normalized["categories"] == report["categories"]
        assert report["splits"][split]["class_instance_counts"]["stairs"] > 0


def test_exact_pixel_duplicates_removed_from_training_not_test(dataset):
    source, output = dataset
    Image.open(source / "test/frame_1.png").save(source / "train/duplicate.bmp")
    def add(data):
        data["images"].append({"id": 9, "file_name": "duplicate.bmp", "width": 8, "height": 8})
        data["annotations"].append({"id": 9, "image_id": 9, "category_id": 7, "bbox": [1, 1, 2, 3]})
    modify(source, "train", add)
    report = prepare(source, output)
    assert report["splits"]["train"]["images"] == 4
    assert report["splits"]["test"]["images"] == 2
    assert report["duplicates_removed"][0]["kept_split"] == "test"


def test_duplicate_with_conflicting_labels_rejected(dataset):
    source, output = dataset
    Image.open(source / "test/frame_1.png").save(source / "train/frame_2.png")
    with pytest.raises(DataError, match="Conflicting labels"):
        prepare(source, output)
    assert not output.exists()


@pytest.mark.parametrize("box", [[-1, 1, 2, 3], [7, 1, 2, 3], [1, 1, 0, 3], [1, 1, float("nan"), 3]])
def test_bad_bbox_fails_without_writing_output(dataset, box):
    source, output = dataset
    modify(source, "train", lambda data: data["annotations"][0].update(bbox=box))
    with pytest.raises(DataError, match="bbox|Bounding box"):
        prepare(source, output)
    assert not output.exists()


def test_drop_policy_removes_entire_image_not_only_bad_box(dataset):
    source, output = dataset
    def add_invalid(data):
        data["annotations"].append({"id": 99, "image_id": 1, "category_id": 0, "bbox": [-1, 1, 2, 3]})
    modify(source, "train", add_invalid)
    report = prepare(source, output, invalid_images="drop")
    assert report["splits"]["train"]["images"] == 3
    assert report["splits"]["train"]["annotations"] == 3
    assert report["dropped_invalid_images"][0]["image_id"] == 1


def test_missing_image_rejected(dataset):
    source, output = dataset
    modify(source, "valid", lambda data: data["images"][0].update(file_name="missing.png"))
    with pytest.raises(DataError, match="Missing image"):
        prepare(source, output)


def test_dimension_mismatch_rejected(dataset):
    source, output = dataset
    modify(source, "valid", lambda data: data["images"][0].update(width=9))
    with pytest.raises(DataError, match="dimensions"):
        prepare(source, output)


def test_traversal_rejected(dataset):
    source, output = dataset
    modify(source, "train", lambda data: data["images"][0].update(file_name="../outside.png"))
    with pytest.raises(DataError, match="escapes"):
        prepare(source, output)


def test_subset_preserves_class_coverage_and_seed(dataset):
    source, output = dataset
    first = prepare(source, output, max_train=2, seed=9)
    second_output = output.with_name("prepared_2")
    prepare(source, second_output, max_train=2, seed=9)
    assert all(first["splits"]["train"]["class_instance_counts"].values())
    assert read(output / "train_manifest.json") == read(second_output / "train_manifest.json")
    with pytest.raises(DataError, match="cover every"):
        prepare(source, output.with_name("too_small"), max_train=1)


def test_refuses_overwrite_and_nested_source(dataset):
    source, output = dataset
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("keep")
    with pytest.raises(DataError, match="overwrite"):
        prepare(source, output)
    assert marker.read_text() == "keep"
    with pytest.raises(DataError, match="non-nested"):
        prepare(source, source / "prepared")


def test_orphan_annotation_rejected(dataset):
    source, output = dataset
    modify(source, "train", lambda data: data["annotations"][0].update(image_id=999))
    with pytest.raises(DataError, match="missing image IDs"):
        prepare(source, output)


def test_eval_only_class_rejected(dataset):
    source, output = dataset
    def extra_class(data):
        data["categories"].append({"id": 999, "name": "drop-off"})
        data["annotations"][0]["category_id"] = 999
    modify(source, "test", extra_class)
    with pytest.raises(DataError, match="No usable training labels.*drop-off"):
        prepare(source, output)
