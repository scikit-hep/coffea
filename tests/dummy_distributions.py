import awkward as ak
import numpy as np


def dummy_jagged_eta_pt():
    np.random.seed(42)
    counts = np.random.exponential(2, size=50).astype(int)
    entries = np.sum(counts)
    test_eta = np.random.uniform(-3.0, 3.0, size=entries)
    test_pt = np.random.exponential(10.0, size=entries) + np.random.exponential(
        10, size=entries
    )
    return (counts, test_eta, test_pt)


def dummy_four_momenta():
    np.random.seed(12345)
    nrows = 1000
    counts = np.minimum(np.random.exponential(0.5, size=nrows).astype(int), 20)

    px = np.random.normal(loc=20.0, scale=5.0, size=np.sum(counts))
    py = np.random.normal(loc=20.0, scale=5.0, size=np.sum(counts))
    pz = np.random.normal(loc=0, scale=55, size=np.sum(counts))
    m_pi = np.full_like(px, fill_value=0.135)
    energy = np.sqrt(px * px + py * py + pz * pz + m_pi * m_pi)
    return (counts, px, py, pz, energy)


def dummy_events():
    counts, px, py, pz, energy = dummy_four_momenta()
    thep4 = np.stack((px, py, pz, energy)).T

    class obj:
        def __init__(self):
            self.p4 = thep4
            self.px = px
            self.py = py
            self.pz = pz
            self.en = energy
            self.pt = np.hypot(px, py)
            self.phi = np.arctan2(py, px)
            self.eta = np.arctanh(pz / np.sqrt(px * px + py * py + pz * pz))
            self.mass = np.sqrt(np.abs(energy * energy - (px * px + py * py + pz * pz)))
            self.blah = energy * px
            self.count = counts

    class events:
        def __init__(self):
            self.thing = obj()

    return events()


def gen_reco_TLV():
    from coffea.nanoevents.methods import vector

    ak.behavior.update(vector.behavior)

    gen_pt = ak.Array([[10.0, 20.0, 30.0], [], [40.0, 50.0]])
    reco_pt = ak.Array([[20.2, 10.1, 30.3, 50.5], [50.5], [60]])

    gen_eta = ak.Array([[-3.0, -2.0, 2.0], [], [-1.0, 1.0]])
    reco_eta = ak.Array([[-2.2, -3.3, 2.2, 0.0], [0.0], [1.1]])

    gen_phi = ak.Array([[-1.5, 0.0, 1.5], [], [0.78, -0.78]])
    reco_phi = ak.Array([[0.1, -1.4, 1.4, 0.78], [0.78], [-0.77]])

    gen = ak.zip(
        {
            "pt": gen_pt,
            "eta": gen_eta,
            "phi": gen_phi,
            "mass": ak.full_like(gen_pt, 0.2),
        },
        with_name="PtEtaPhiMLorentzVector",
    )
    reco = ak.zip(
        {
            "pt": reco_pt,
            "eta": reco_eta,
            "phi": reco_phi,
            "mass": ak.full_like(reco_pt, 0.2),
        },
        with_name="PtEtaPhiMLorentzVector",
    )

    return (gen, reco)
