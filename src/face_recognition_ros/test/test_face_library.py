import numpy as np
import pytest

from face_recognition_ros.face_library import UNKNOWN, FaceLibrary

rng = np.random.default_rng(0)


def random_identity(dim=128):
    return rng.normal(size=dim).astype(np.float32)


def noisy(vec, scale=0.1):
    return vec + rng.normal(scale=scale * np.linalg.norm(vec) / np.sqrt(vec.size), size=vec.size)


def test_empty_library_returns_unknown():
    match = FaceLibrary().identify(random_identity())
    assert match.label == UNKNOWN
    assert not match.is_known


def test_identifies_enrolled_person_from_noisy_sample():
    ada, grace = random_identity(), random_identity()
    lib = FaceLibrary()
    lib.add('ada', ada)
    lib.add('grace', grace)

    assert lib.identify(noisy(ada)).label == 'ada'
    assert lib.identify(noisy(grace)).label == 'grace'


def test_stranger_is_unknown():
    lib = FaceLibrary()
    lib.add('ada', random_identity())
    # Random high-dimensional vectors are almost orthogonal (cosine ~ 0).
    assert lib.identify(random_identity()).label == UNKNOWN


def test_score_is_cosine_similarity_and_scale_invariant():
    ada = random_identity()
    lib = FaceLibrary()
    lib.add('ada', ada)
    assert lib.identify(ada * 42).score == pytest.approx(1.0, abs=1e-5)


def test_threshold_controls_acceptance():
    ada = random_identity()
    query = noisy(ada, scale=1.0)  # cosine around 0.7
    strict, lenient = FaceLibrary(threshold=0.99), FaceLibrary(threshold=0.3)
    for lib in (strict, lenient):
        lib.add('ada', ada)
    assert strict.identify(query).label == UNKNOWN
    assert lenient.identify(query).label == 'ada'


def test_identify_batch_matches_single_calls():
    people = {name: random_identity() for name in ('a', 'b', 'c')}
    lib = FaceLibrary()
    for name, vec in people.items():
        lib.add(name, vec)
    queries = np.stack([noisy(v) for v in people.values()])
    assert [m.label for m in lib.identify_batch(queries)] == list(people)


def test_rejects_bad_input():
    lib = FaceLibrary()
    with pytest.raises(ValueError):
        lib.add('', random_identity())
    with pytest.raises(ValueError):
        lib.add(UNKNOWN, random_identity())
    lib.add('ada', random_identity(128))
    with pytest.raises(ValueError):
        lib.add('grace', random_identity(64))
    with pytest.raises(ValueError):
        lib.identify(random_identity(64))


def test_remove():
    lib = FaceLibrary()
    lib.add('ada', random_identity())
    lib.add('ada', random_identity())
    lib.add('grace', random_identity())
    assert lib.remove('ada') == 2
    assert lib.identities == ['grace']


def test_save_and_load_roundtrip(tmp_path):
    ada = random_identity()
    lib = FaceLibrary(threshold=0.5)
    lib.add('ada', ada)
    lib.add('grace', random_identity())
    path = tmp_path / 'nested' / 'library.npz'
    lib.save(path)

    loaded = FaceLibrary.load(path, threshold=0.5)
    assert len(loaded) == 2
    assert loaded.identities == ['ada', 'grace']
    assert loaded.identify(noisy(ada)).label == 'ada'
    assert list(path.parent.iterdir()) == [path]  # no temp files left behind


def test_load_or_create_missing_file(tmp_path):
    lib = FaceLibrary.load_or_create(tmp_path / 'missing.npz')
    assert len(lib) == 0
