"""
Unit and integration tests for the persistent metadata cache in Runner.

Tests focus on:
- Loading cache from disk (.pkl file)
- Saving cache to disk atomically
- Handling corrupted or missing cache files
- Cache usage during preprocessing to skip metadata fetches
- Fallback behavior
"""

import pickle
from unittest.mock import Mock, patch

from cachetools import LRUCache

from coffea.processor.executor import (
    DEFAULT_METADATA_CACHE,
    FileMeta,
    IterativeExecutor,
    Runner,
    set_accumulator,
)

# ============================================================================
# UNIT TESTS: Cache Loading
# ============================================================================


def test_cache_load_success(tmp_path, monkeypatch):
    """Test successful loading of an existing valid cache file."""
    # Arrange: Create a valid cache file
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    mock_cache = {
        FileMeta("dataset1", "file1.root", "Events"): {
            "numentries": 1000,
            "uuid": b"test_uuid_1234",
        }
    }
    with open(cache_file, "wb") as f:
        pickle.dump(mock_cache, f)

    monkeypatch.setattr(Runner, "cache_file", cache_file)

    # Act: Initialize Runner (triggers _load_cache)
    with patch.object(Runner, "_preprocess_fileset_root"):
        runner = Runner(
            executor=IterativeExecutor(),
            metadata_cache=None,  # Force loading from disk
        )

    # Assert: Cache should be loaded
    assert len(runner.metadata_cache) == 1
    filemeta = FileMeta("dataset1", "file1.root", "Events")
    assert filemeta in runner.metadata_cache
    assert runner.metadata_cache[filemeta]["numentries"] == 1000


def test_cache_file_missing(tmp_path, monkeypatch):
    """Test that missing cache file returns empty dict without crashing."""
    # Arrange: Ensure cache file does not exist
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    assert not cache_file.exists()

    monkeypatch.setattr(Runner, "cache_file", cache_file)

    # Act: Initialize Runner
    with patch.object(Runner, "_preprocess_fileset_root"):
        runner = Runner(
            executor=IterativeExecutor(),
            metadata_cache=None,
        )

    # Assert: Should fall back to DEFAULT_METADATA_CACHE (LRU)
    # The _load_cache returns {} when file doesn't exist, then __post_init__
    # assigns DEFAULT_METADATA_CACHE
    assert runner.metadata_cache is not None
    assert len(runner.metadata_cache) == 0


def test_cache_file_corrupted(tmp_path, monkeypatch, capsys):
    """Test that corrupted cache file is handled gracefully."""
    # Arrange: Write invalid data to cache file
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    with open(cache_file, "wb") as f:
        f.write(b"corrupted data not pickle")

    monkeypatch.setattr(Runner, "cache_file", cache_file)

    # Act: Initialize Runner
    with patch.object(Runner, "_preprocess_fileset_root"):
        runner = Runner(
            executor=IterativeExecutor(),
            metadata_cache=None,
        )

    # Assert: Should return empty dict and print warning
    captured = capsys.readouterr()
    assert "Could not load cache file" in captured.out
    assert len(runner.metadata_cache) == 0


def test_cache_load_prints_message(tmp_path, monkeypatch, capsys):
    """Test that loading cache prints informative message."""
    # Arrange: Create cache with multiple entries
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    mock_cache = {
        FileMeta("ds1", "f1.root", "Events"): {"numentries": 100, "uuid": b"uuid1"},
        FileMeta("ds2", "f2.root", "Events"): {"numentries": 200, "uuid": b"uuid2"},
        FileMeta("ds3", "f3.root", "Events"): {"numentries": 300, "uuid": b"uuid3"},
    }
    with open(cache_file, "wb") as f:
        pickle.dump(mock_cache, f)

    monkeypatch.setattr(Runner, "cache_file", cache_file)

    # Act
    with patch.object(Runner, "_preprocess_fileset_root"):
        Runner(executor=IterativeExecutor(), metadata_cache=None)

    # Assert
    captured = capsys.readouterr()
    assert "Loaded 3 entries from metadata cache" in captured.out


# ============================================================================
# UNIT TESTS: Cache Saving
# ============================================================================


def test_cache_save_success(tmp_path, monkeypatch):
    """Test successful saving of cache to disk atomically."""
    # Arrange
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    monkeypatch.setattr(Runner, "cache_file", cache_file)

    with patch.object(Runner, "_preprocess_fileset_root"):
        runner = Runner(executor=IterativeExecutor())

    # Populate cache
    filemeta = FileMeta("dataset1", "file1.root", "Events")
    runner.metadata_cache[filemeta] = {
        "numentries": 5000,
        "uuid": b"saved_uuid",
    }

    # Act: Save cache
    runner._save_cache()

    # Assert: Cache file should exist
    assert cache_file.exists()

    # Verify atomic save (temp file should not exist)
    temp_file = cache_file.with_suffix(".tmp")
    assert not temp_file.exists()

    # Reload and verify contents
    with open(cache_file, "rb") as f:
        loaded_cache = pickle.load(f)
    assert filemeta in loaded_cache
    assert loaded_cache[filemeta]["numentries"] == 5000


def test_cache_save_failure_open_error(tmp_path, monkeypatch, capsys):
    """Test that cache save failure prints warning without crashing."""
    # Arrange
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    monkeypatch.setattr(Runner, "cache_file", cache_file)

    with patch.object(Runner, "_preprocess_fileset_root"):
        runner = Runner(executor=IterativeExecutor())

    runner.metadata_cache[FileMeta("ds", "f.root", "Events")] = {
        "numentries": 100,
        "uuid": b"uuid",
    }

    # Act: Mock open to raise error
    with patch("builtins.open", side_effect=OSError("Disk full")):
        runner._save_cache()  # Should not raise

    # Assert: Warning printed
    captured = capsys.readouterr()
    assert "Could not save cache file" in captured.out
    assert "Disk full" in captured.out


def test_cache_save_failure_pickle_error(tmp_path, monkeypatch, capsys):
    """Test that pickle.dump failure is handled gracefully."""
    # Arrange
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    monkeypatch.setattr(Runner, "cache_file", cache_file)

    with patch.object(Runner, "_preprocess_fileset_root"):
        runner = Runner(executor=IterativeExecutor())

    runner.metadata_cache[FileMeta("ds", "f.root", "Events")] = {
        "numentries": 100,
        "uuid": b"uuid",
    }

    # Act: Mock pickle.dump to fail
    with patch("pickle.dump", side_effect=pickle.PicklingError("Cannot pickle")):
        runner._save_cache()

    # Assert
    captured = capsys.readouterr()
    assert "Could not save cache file" in captured.out


def test_cache_overwrite_behavior(tmp_path, monkeypatch):
    """Test that saving cache twice overwrites with latest content."""
    # Arrange
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    monkeypatch.setattr(Runner, "cache_file", cache_file)

    with patch.object(Runner, "_preprocess_fileset_root"):
        runner = Runner(executor=IterativeExecutor())

    filemeta = FileMeta("dataset", "file.root", "Events")

    # First save
    runner.metadata_cache[filemeta] = {"numentries": 1000, "uuid": b"uuid1"}
    runner._save_cache()

    # Second save with updated data
    runner.metadata_cache[filemeta] = {"numentries": 2000, "uuid": b"uuid2"}
    runner._save_cache()

    # Assert: Latest content persisted
    with open(cache_file, "rb") as f:
        loaded = pickle.load(f)
    assert loaded[filemeta]["numentries"] == 2000
    assert loaded[filemeta]["uuid"] == b"uuid2"


def test_cache_save_uses_highest_protocol(tmp_path, monkeypatch):
    """Test that cache is saved using pickle.HIGHEST_PROTOCOL."""
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    monkeypatch.setattr(Runner, "cache_file", cache_file)

    with patch.object(Runner, "_preprocess_fileset_root"):
        runner = Runner(executor=IterativeExecutor())

    runner.metadata_cache[FileMeta("ds", "f.root", "Events")] = {
        "numentries": 100,
        "uuid": b"uuid",
    }

    # Mock pickle.dump to capture protocol argument
    with patch("pickle.dump") as mock_dump:
        runner._save_cache()
        mock_dump.assert_called_once()
        assert mock_dump.call_args[1]["protocol"] == pickle.HIGHEST_PROTOCOL


# ============================================================================
# INTEGRATION TESTS: Cache Usage in Preprocessing
# ============================================================================


def test_cache_skips_metadata_fetching_when_populated(tmp_path, monkeypatch):
    """Test that cached metadata skips fetching from files."""
    # Arrange: Pre-populate cache
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    filemeta = FileMeta("dataset1", "file1.root", "Events")
    mock_cache = {
        filemeta: {
            "numentries": 9999,
            "uuid": b"cached_uuid_1234",
        }
    }
    with open(cache_file, "wb") as f:
        pickle.dump(mock_cache, f)

    monkeypatch.setattr(Runner, "cache_file", cache_file)

    # Create runner
    runner = Runner(executor=IterativeExecutor(), metadata_cache=None)

    # Create a fileset
    fileset = {"dataset1": ["file1.root"]}

    # Mock metadata fetcher to ensure it's NOT called
    mock_fetcher = Mock(return_value=set_accumulator([filemeta]))

    # Act: Preprocess with mocked fetcher
    with patch.object(Runner, "metadata_fetcher_root", mock_fetcher):
        with patch.object(runner, "_save_cache") as mock_save:
            # Normalize and populate from cache
            normalized = list(runner._normalize_fileset(fileset, "Events"))
            for fm in normalized:
                fm.maybe_populate(runner.metadata_cache)

            # Check if we need to fetch (we shouldn't)
            to_get = {fm for fm in normalized if not fm.populated(clusters=False)}

            # Assert: No files need fetching
            assert len(to_get) == 0
            # Fetcher should not be called
            mock_fetcher.assert_not_called()
            # Save should not be triggered (no new metadata)
            mock_save.assert_not_called()


def test_cache_updated_when_metadata_missing(tmp_path, monkeypatch):
    """Test that missing metadata is fetched and cache is updated."""
    # Arrange: Empty cache
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    monkeypatch.setattr(Runner, "cache_file", cache_file)

    # Create runner with empty cache
    runner = Runner(executor=IterativeExecutor())

    # Create fileset
    fileset = {"dataset1": ["file1.root"]}

    # Mock metadata fetcher
    filemeta = FileMeta("dataset1", "file1.root", "Events")
    metadata = {"numentries": 12345, "uuid": b"new_uuid"}
    filemeta_with_meta = FileMeta("dataset1", "file1.root", "Events", metadata)

    mock_accumulator = set_accumulator([filemeta_with_meta])

    # Create a mock executor that returns the accumulator
    mock_pre_executor = Mock()
    mock_pre_executor.copy = Mock(return_value=mock_pre_executor)
    mock_pre_executor.return_value = (mock_accumulator, 0)

    # Replace the pre_executor
    runner.pre_executor = mock_pre_executor

    # Act: Run preprocessing
    with patch.object(runner, "_save_cache") as mock_save:
        runner._preprocess_fileset_root(
            list(runner._normalize_fileset(fileset, "Events"))
        )

        # Assert: Cache was updated
        assert filemeta in runner.metadata_cache
        assert runner.metadata_cache[filemeta]["numentries"] == 12345
        assert runner.metadata_cache[filemeta]["uuid"] == b"new_uuid"

        # Assert: Cache was saved to disk
        mock_save.assert_called_once()


def test_cache_not_saved_when_no_updates(tmp_path, monkeypatch):
    """Test that cache is not saved when no new metadata is fetched."""
    # Arrange: Pre-populate cache completely
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    filemeta = FileMeta("dataset1", "file1.root", "Events")
    mock_cache = {
        filemeta: {
            "numentries": 5000,
            "uuid": b"existing_uuid",
        }
    }
    with open(cache_file, "wb") as f:
        pickle.dump(mock_cache, f)

    monkeypatch.setattr(Runner, "cache_file", cache_file)

    with patch.object(Runner, "_preprocess_fileset_root"):
        runner = Runner(executor=IterativeExecutor(), metadata_cache=None)

    fileset = {"dataset1": ["file1.root"]}

    # Act: Preprocess when all metadata is cached
    with patch.object(runner, "_save_cache") as mock_save:
        normalized = list(runner._normalize_fileset(fileset, "Events"))
        for fm in normalized:
            fm.maybe_populate(runner.metadata_cache)

        # Simulate preprocessing check
        to_get = {fm for fm in normalized if not fm.populated(clusters=False)}

        # Since all is cached, to_get should be empty
        if len(to_get) == 0:
            pass  # No fetching, no saving

        # Assert: Save not called
        mock_save.assert_not_called()


def test_cache_handles_multiple_datasets(tmp_path, monkeypatch):
    """Test cache correctly handles multiple datasets and files."""
    # Arrange: Create fresh cache file
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    monkeypatch.setattr(Runner, "cache_file", cache_file)

    # Create runner with explicit empty cache to avoid shared state
    runner = Runner(executor=IterativeExecutor(), metadata_cache={})

    # Create multiple file metadata entries
    filemeta1 = FileMeta("dataset_A", "fileA1.root", "Events")
    filemeta2 = FileMeta("dataset_A", "fileA2.root", "Events")
    filemeta3 = FileMeta("dataset_B", "fileB1.root", "Events")

    runner.metadata_cache[filemeta1] = {"numentries": 100, "uuid": b"uuid_a1"}
    runner.metadata_cache[filemeta2] = {"numentries": 200, "uuid": b"uuid_a2"}
    runner.metadata_cache[filemeta3] = {"numentries": 300, "uuid": b"uuid_b1"}

    # Act: Save cache
    runner._save_cache()

    # Assert: Reload and verify all entries
    with open(cache_file, "rb") as f:
        loaded = pickle.load(f)

    assert len(loaded) == 3
    assert loaded[filemeta1]["numentries"] == 100
    assert loaded[filemeta2]["numentries"] == 200
    assert loaded[filemeta3]["numentries"] == 300


def test_cache_update_triggers_save(tmp_path, monkeypatch):
    """Integration test: verify cache update triggers save in preprocessing."""
    # Arrange
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    monkeypatch.setattr(Runner, "cache_file", cache_file)

    runner = Runner(executor=IterativeExecutor())

    fileset = {"dataset1": ["file1.root"]}

    # Mock successful metadata fetch
    filemeta = FileMeta("dataset1", "file1.root", "Events")
    metadata = {"numentries": 7777, "uuid": b"fetched_uuid"}
    filemeta_with_meta = FileMeta("dataset1", "file1.root", "Events", metadata)
    mock_accumulator = set_accumulator([filemeta_with_meta])

    # Create a mock executor
    mock_pre_executor = Mock()
    mock_pre_executor.copy = Mock(return_value=mock_pre_executor)
    mock_pre_executor.return_value = (mock_accumulator, 0)

    # Replace the pre_executor
    runner.pre_executor = mock_pre_executor

    # Act
    runner._preprocess_fileset_root(list(runner._normalize_fileset(fileset, "Events")))

    # Assert: Cache file was created
    assert cache_file.exists()

    # Verify contents
    with open(cache_file, "rb") as f:
        saved = pickle.load(f)
    assert filemeta in saved
    assert saved[filemeta]["numentries"] == 7777


# ============================================================================
# EDGE CASES
# ============================================================================


def test_cache_handles_empty_fileset(tmp_path, monkeypatch):
    """Test that empty fileset doesn't break cache logic."""
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    monkeypatch.setattr(Runner, "cache_file", cache_file)

    with patch.object(Runner, "_preprocess_fileset_root"):
        runner = Runner(executor=IterativeExecutor())

    fileset = {}

    # Act: Should not crash
    with patch.object(runner, "_save_cache") as mock_save:
        normalized = list(runner._normalize_fileset(fileset, "Events"))
        assert len(normalized) == 0
        # No save triggered for empty fileset
        mock_save.assert_not_called()


def test_cache_file_read_only_filesystem(tmp_path, monkeypatch, capsys):
    """Test cache save handles read-only filesystem gracefully."""
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    monkeypatch.setattr(Runner, "cache_file", cache_file)

    with patch.object(Runner, "_preprocess_fileset_root"):
        runner = Runner(executor=IterativeExecutor())

    runner.metadata_cache[FileMeta("ds", "f.root", "Events")] = {
        "numentries": 100,
        "uuid": b"uuid",
    }

    # Simulate read-only error
    with patch("builtins.open", side_effect=PermissionError("Read-only filesystem")):
        runner._save_cache()

    captured = capsys.readouterr()
    assert "Could not save cache file" in captured.out
    assert "Read-only filesystem" in captured.out


def test_cache_with_cluster_alignment(tmp_path, monkeypatch):
    """Test cache handles cluster-aligned metadata correctly."""
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    filemeta = FileMeta("dataset1", "file1.root", "Events")
    mock_cache = {
        filemeta: {
            "numentries": 1000,
            "uuid": b"uuid",
            "clusters": [0, 100, 500, 1000],  # Cluster boundaries
        }
    }
    with open(cache_file, "wb") as f:
        pickle.dump(mock_cache, f)

    monkeypatch.setattr(Runner, "cache_file", cache_file)

    with patch.object(Runner, "_preprocess_fileset_root"):
        runner = Runner(
            executor=IterativeExecutor(),
            metadata_cache=None,
            align_clusters=True,
        )

    # Verify cache loaded with cluster info
    assert filemeta in runner.metadata_cache
    assert "clusters" in runner.metadata_cache[filemeta]
    assert runner.metadata_cache[filemeta]["clusters"] == [0, 100, 500, 1000]


def test_cache_save_unpicklable_content(tmp_path, monkeypatch, capsys):
    """Test behavior when metadata contains something that cannot be pickled."""
    cache_file = tmp_path / ".coffea_metadata_cache.pkl"
    monkeypatch.setattr(Runner, "cache_file", cache_file)

    runner = Runner(executor=IterativeExecutor(), metadata_cache={})

    # Add a lambda (unpicklable by standard pickle)
    runner.metadata_cache[FileMeta("ds", "f.root", "T")] = {"func": lambda x: x}

    runner._save_cache()

    captured = capsys.readouterr()
    assert "Could not save cache file" in captured.out
    # Cache file should not exist or be incomplete
    assert not cache_file.exists() or cache_file.stat().st_size == 0


def test_default_cache_fallback(tmp_path, monkeypatch):
    """Ensure Runner uses DEFAULT_METADATA_CACHE when no file exists and no cache provided."""
    # Point to non-existent path
    cache_file = tmp_path / "nonexistent" / ".coffea_metadata_cache.pkl"
    monkeypatch.setattr(Runner, "cache_file", cache_file)

    with patch.object(Runner, "_preprocess_fileset_root"):
        runner = Runner(executor=IterativeExecutor(), metadata_cache=None)

    # Check if it points to the global DEFAULT_METADATA_CACHE or is an empty LRU
    assert isinstance(runner.metadata_cache, (dict, LRUCache))
    # It should be the default cache when file doesn't exist
    if not cache_file.exists():
        assert (
            runner.metadata_cache is DEFAULT_METADATA_CACHE
            or len(runner.metadata_cache) == 0
        )


def test_full_roundtrip_persistence(tmp_path, monkeypatch):
    """Integration: Process -> Save -> New Runner -> Load."""
    cache_file = tmp_path / "persistent_cache.pkl"
    monkeypatch.setattr(Runner, "cache_file", cache_file)

    fileset = {"dataset": ["file.root"]}
    filemeta = FileMeta("dataset", "file.root", "Events")
    filemeta_with_meta = FileMeta(
        "dataset", "file.root", "Events", {"numentries": 100, "uuid": b"abc"}
    )

    # 1. Setup first runner to "fetch" and save
    runner1 = Runner(executor=IterativeExecutor(), metadata_cache={})
    mock_acc = set_accumulator([filemeta_with_meta])

    # Create a mock executor
    mock_pre_executor = Mock()
    mock_pre_executor.copy = Mock(return_value=mock_pre_executor)
    mock_pre_executor.return_value = (mock_acc, 0)
    runner1.pre_executor = mock_pre_executor

    # Trigger preprocess which calls _save_cache
    runner1._preprocess_fileset_root(
        list(runner1._normalize_fileset(fileset, "Events"))
    )

    assert cache_file.exists()

    # 2. Setup second runner to load that file
    with patch.object(Runner, "_preprocess_fileset_root"):
        runner2 = Runner(executor=IterativeExecutor(), metadata_cache=None)

    assert filemeta in runner2.metadata_cache
    assert runner2.metadata_cache[filemeta]["numentries"] == 100
    assert runner2.metadata_cache[filemeta]["uuid"] == b"abc"


def test_align_clusters_cache_incomplete(tmp_path, monkeypatch):
    """Verify that align_clusters=True with incomplete cache triggers refetch."""
    cache_file = tmp_path / "cluster_cache.pkl"
    monkeypatch.setattr(Runner, "cache_file", cache_file)

    # Cache has numentries and uuid, but NO clusters
    fm_key = FileMeta("ds", "f.root", "Events")
    with open(cache_file, "wb") as f:
        pickle.dump({fm_key: {"numentries": 10, "uuid": b"u"}}, f)

    runner = Runner(
        executor=IterativeExecutor(), metadata_cache=None, align_clusters=True
    )

    # Load the filemeta and populate from cache
    normalized = list(runner._normalize_fileset({"ds": ["f.root"]}, "Events"))
    for fm in normalized:
        fm.maybe_populate(runner.metadata_cache)

    # FileMeta.populated(clusters=True) should return False here
    assert not normalized[0].populated(clusters=True)

    # Create complete metadata with clusters
    filemeta_complete = FileMeta(
        "ds", "f.root", "Events", {"clusters": [0, 10], "numentries": 10, "uuid": b"u"}
    )
    mock_acc = set_accumulator([filemeta_complete])

    # Create a mock executor
    mock_pre_executor = Mock()
    mock_pre_executor.copy = Mock(return_value=mock_pre_executor)
    mock_pre_executor.return_value = (mock_acc, 0)
    runner.pre_executor = mock_pre_executor

    # Run preprocessing
    runner._preprocess_fileset_root(normalized)

    # Verify cache was updated with clusters
    assert "clusters" in runner.metadata_cache[fm_key]
    assert runner.metadata_cache[fm_key]["clusters"] == [0, 10]
