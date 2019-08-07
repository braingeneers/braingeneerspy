import braingeneers.datasets


def test_load_batch():
    batch = braingeneers.datasets.load_batch("test-datasets")
    assert batch["uuid"] == "test-datasets"
    assert len(batch["experiments"]) == 1
    assert batch["experiments"][0] == "experiment.json"


def test_load_experiment():
    experiment = braingeneers.datasets.load_experiment("test-datasets", 0)
    assert experiment["name"] == "test"
    assert experiment["blocks"][0]["timestamp"] == "2019-02-05T19:38:42.314159-08:00"
    assert len(experiment["blocks"]) == 2


def test_load_blocks():
    X, t, fs = braingeneers.datasets.load_blocks("test-datasets", 0)
    assert X.shape == (90, 4)
    assert (X[0, :] == [-10., -9.5, -9., -8.5]).all()

    assert t[0] == 0.0

    X, t, fs = braingeneers.datasets.load_blocks("test-datasets", 0, 1)
    assert X.shape == (30, 4)

    experiment = braingeneers.datasets.load_experiment("test-datasets", 0)
    assert t[0] == fs * 1000 * experiment["blocks"][0]["num_frames"]

    X, t, fs = braingeneers.datasets.load_blocks("test-datasets", 0, 0, 2)
    assert X.shape == (90, 4)
