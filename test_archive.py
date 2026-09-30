from evops.pareto import BoundedParetoArchive


def test_archive_bounding():
    archive = BoundedParetoArchive(max_capacity=3, max_flags=[True, True])
    candidates = [
        {"chromosome": [64, 64, 32, 0, 0.0, 0.001, 0.0, 0.5, 0.0, 3], "obj_vector": [0.9, 0.9]},
        {"chromosome": [128, 64, 32, 0, 0.0, 0.001, 0.0, 0.5, 0.0, 3], "obj_vector": [0.85, 0.95]},
        {"chromosome": [256, 64, 32, 0, 0.0, 0.001, 0.0, 0.5, 0.0, 3], "obj_vector": [0.95, 0.85]},
        {"chromosome": [64, 128, 32, 0, 0.0, 0.001, 0.0, 0.5, 0.0, 3], "obj_vector": [0.92, 0.92]},
    ]
    archive.update(candidates)
    assert len(archive.archive) <= 3, f"Archive exceeded max capacity: {len(archive.archive)}"
