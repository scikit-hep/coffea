import sys
import types as _types


def _install_fake_rucio():
    if "rucio" in sys.modules:
        return
    rucio = _types.ModuleType("rucio")
    client_mod = _types.ModuleType("rucio.client")

    class Client:
        pass

    client_mod.Client = Client
    rucio.client = client_mod
    sys.modules["rucio"] = rucio
    sys.modules["rucio.client"] = client_mod


_install_fake_rucio()

from coffea.dataset_tools import rucio_utils  # noqa: E402


class FakeClient:
    def __init__(self, replicas):
        self._replicas = replicas

    def list_replicas(self, dids):
        return self._replicas


def _filedata(name, sites):
    rses = {}
    pfns = {}
    states = {}
    for site, available in sites.items():
        key = f"root://{site}/{name}"
        rses[site] = [key]
        pfns[key] = {"type": "DISK", "volatile": False}
        states[site] = "AVAILABLE" if available else "UNAVAILABLE"
    return {"name": name, "rses": rses, "pfns": pfns, "states": states}


_PREFIXES = {"T2_A": "root://a/", "T2_B": "root://b/"}


def test_first_partial_allowed_skips_unviable(monkeypatch):
    monkeypatch.setattr(rucio_utils, "get_xrootd_sites_map", lambda: _PREFIXES)
    client = FakeClient(
        [
            _filedata("/f1.root", {"T2_A": True}),
            _filedata("/f2.root", {"T2_B": False}),
        ]
    )
    files, sites, counts = rucio_utils.get_dataset_files_replicas(
        "ds", client=client, mode="first", partial_allowed=True
    )
    assert files == ["root://a//f1.root"]
    assert sites == ["T2_A"]
    assert dict(counts) == {"T2_A": 1}


def test_first_sites_counts_not_stale(monkeypatch):
    monkeypatch.setattr(rucio_utils, "get_xrootd_sites_map", lambda: _PREFIXES)
    client = FakeClient(
        [
            _filedata("/f1.root", {"T2_A": True}),
            _filedata("/f2.root", {"T2_B": True}),
        ]
    )
    files, sites, counts = rucio_utils.get_dataset_files_replicas(
        "ds", client=client, mode="first"
    )
    assert files == ["root://a//f1.root", "root://b//f2.root"]
    assert sites == ["T2_A", "T2_B"]
    assert dict(counts) == {"T2_A": 1, "T2_B": 1}
