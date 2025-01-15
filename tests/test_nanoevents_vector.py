import awkward as ak
import numpy as np
import pytest
from numpy.testing import assert_allclose

from coffea.nanoevents.methods import vector

ATOL = 1e-8


def assert_record_arrays_equal(a, b, check_type=False):
    if check_type:
        assert type(a) is type(b)
    assert ak.fields(a) == ak.fields(b)
    assert all(ak.all(ak.isclose(a[f], b[f])) for f in ak.fields(a))


def assert_awkward_allclose(actual, desired):
    flat_actual = ak.flatten(actual, axis=None)
    flat_desired = ak.flatten(desired, axis=None)
    # we should check None values, but not used in these tests
    assert_allclose(flat_actual, flat_desired)


def test_two_vector():
    a = ak.zip(
        {"x": [[1, 2], [], [3], [4]], "y": [[5, 6], [], [7], [8]]},
        with_name="TwoVector",
        behavior=vector.behavior,
    )
    b = ak.zip(
        {"x": [[11, 12], [], [13], [14]], "y": [[15, 16], [], [17], [18]]},
        with_name="TwoVector",
        behavior=vector.behavior,
    )

    assert_record_arrays_equal(
        -a, ak.zip({"x": [[-1, -2], [], [-3], [-4]], "y": [[-5, -6], [], [-7], [-8]]})
    )

    assert_record_arrays_equal(
        a + b,
        ak.zip({"x": [[12, 14], [], [16], [18]], "y": [[20, 22], [], [24], [26]]}),
    )
    assert_record_arrays_equal(
        a - b,
        ak.zip(
            {"x": [[-10, -10], [], [-10], [-10]], "y": [[-10, -10], [], [-10], [-10]]}
        ),
    )

    assert_record_arrays_equal(
        a * 2, ak.zip({"x": [[2, 4], [], [6], [8]], "y": [[10, 12], [], [14], [16]]})
    )
    assert_record_arrays_equal(
        a / 2,
        ak.zip({"x": [[0.5, 1], [], [1.5], [2]], "y": [[2.5, 3], [], [3.5], [4]]}),
    )

    assert_awkward_allclose(a.dot(b), ak.Array([[86, 120], [], [158], [200]]))
    assert_awkward_allclose(b.dot(a), ak.Array([[86, 120], [], [158], [200]]))

    assert ak.all(abs(a.unit.r - 1) < ATOL)
    assert ak.all(abs(a.unit.phi - a.phi) < ATOL)


def test_polar_two_vector():
    a = ak.zip(
        {
            "rho": [[1, 2], [], [3], [4]],
            "phi": [[0.3, 0.4], [], [0.5], [0.6]],
        },
        with_name="PolarTwoVector",
        behavior=vector.behavior,
    )

    assert_record_arrays_equal(
        a * 2,
        ak.zip({"rho": [[2, 4], [], [6], [8]], "phi": [[0.3, 0.4], [], [0.5], [0.6]]}),
    )
    assert ak.all((a * (-2)).rho == [[2, 4], [], [6], [8]])
    assert ak.all(
        (a * (-2)).phi
        - ak.Array(
            [[-2.8415926535, -2.7415926535], [], [-2.6415926535], [-2.5415926535]]
        )
        < ATOL
    )
    assert_record_arrays_equal(
        a / 2,
        ak.zip(
            {"rho": [[0.5, 1], [], [1.5], [2]], "phi": [[0.3, 0.4], [], [0.5], [0.6]]}
        ),
    )

    assert ak.all(abs((-a).x + a.x) < ATOL)
    assert ak.all(abs((-a).y + a.y) < ATOL)
    assert_record_arrays_equal(a * (-1), -a)

    assert ak.all(ak.isclose(a.unit.phi, a.phi))


def test_three_vector():
    a = ak.zip(
        {
            "x": [[1, 2], [], [3], [4]],
            "y": [[5, 6], [], [7], [8]],
            "z": [[9, 10], [], [11], [12]],
        },
        with_name="ThreeVector",
        behavior=vector.behavior,
    )
    b = ak.zip(
        {
            "x": [[4, 1], [], [10], [11]],
            "y": [[17, 7], [], [11], [6]],
            "z": [[9, 11], [], [5], [16]],
        },
        with_name="ThreeVector",
        behavior=vector.behavior,
    )

    assert_record_arrays_equal(
        -a,
        ak.zip(
            {
                "x": [[-1, -2], [], [-3], [-4]],
                "y": [[-5, -6], [], [-7], [-8]],
                "z": [[-9, -10], [], [-11], [-12]],
            }
        ),
    )

    assert_record_arrays_equal(
        a + b,
        ak.zip(
            {
                "x": [[5, 3], [], [13], [15]],
                "y": [[22, 13], [], [18], [14]],
                "z": [[18, 21], [], [16], [28]],
            }
        ),
    )
    assert_record_arrays_equal(
        a - b,
        ak.zip(
            {
                "x": [[-3, 1], [], [-7], [-7]],
                "y": [[-12, -1], [], [-4], [2]],
                "z": [[0, -1], [], [6], [-4]],
            }
        ),
    )
    assert_record_arrays_equal(
        b - a,
        ak.zip(
            {
                "x": [[3, -1], [], [7], [7]],
                "y": [[12, 1], [], [4], [-2]],
                "z": [[0, 1], [], [-6], [4]],
            }
        ),
    )

    assert_record_arrays_equal(
        a * 2,
        ak.zip(
            {
                "x": [[2, 4], [], [6], [8]],
                "y": [[10, 12], [], [14], [16]],
                "z": [[18, 20], [], [22], [24]],
            }
        ),
    )
    assert_record_arrays_equal(
        a / 2,
        ak.zip(
            {
                "x": [[0.5, 1], [], [1.5], [2]],
                "y": [[2.5, 3], [], [3.5], [4]],
                "z": [[4.5, 5], [], [5.5], [6]],
            }
        ),
    )

    assert ak.all(a.dot(b) == ak.Array([[170, 154], [], [162], [284]]))
    assert ak.all(b.dot(a) == ak.Array([[170, 154], [], [162], [284]]))

    assert_record_arrays_equal(
        a.cross(b),
        ak.zip(
            {
                "x": [[-108, -4], [], [-86], [56]],
                "y": [[27, -12], [], [95], [68]],
                "z": [[-3, 8], [], [-37], [-64]],
            }
        ),
    )
    assert_record_arrays_equal(
        b.cross(a),
        ak.zip(
            {
                "x": [[108, 4], [], [86], [-56]],
                "y": [[-27, 12], [], [-95], [-68]],
                "z": [[3, -8], [], [37], [64]],
            }
        ),
    )

    assert ak.all(abs(a.unit.rho - 1) < ATOL)
    assert ak.all(abs(a.unit.phi - a.phi) < ATOL)


def test_spherical_three_vector():
    a = ak.zip(
        {
            "rho": [[1.0, 2.0], [], [3.0], [4.0]],
            "theta": [[1.2, 0.7], [], [1.8], [1.9]],
            "phi": [[0.3, 0.4], [], [0.5], [0.6]],
        },
        with_name="SphericalThreeVector",
        behavior=vector.behavior,
    )

    assert ak.all(abs((-a).x + a.x) < ATOL)
    assert ak.all(abs((-a).y + a.y) < ATOL)
    assert ak.all(abs((-a).z + a.z) < ATOL)
    assert_record_arrays_equal(a * (-1), -a, check_type=True)


def test_lorentz_vector():
    a = ak.zip(
        {
            "x": [[1, 2], [], [3], [4]],
            "y": [[5, 6], [], [7], [8]],
            "z": [[9, 10], [], [11], [12]],
            "t": [[50, 51], [], [52], [53]],
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )
    b = ak.zip(
        {
            "x": [[4, 1], [], [10], [11]],
            "y": [[17, 7], [], [11], [6]],
            "z": [[9, 11], [], [5], [16]],
            "t": [[60, 61], [], [62], [63]],
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )

    assert_record_arrays_equal(
        -a,
        ak.zip(
            {
                "x": [[-1, -2], [], [-3], [-4]],
                "y": [[-5, -6], [], [-7], [-8]],
                "z": [[-9, -10], [], [-11], [-12]],
                "t": [[-50, -51], [], [-52], [-53]],
            }
        ),
    )

    assert_record_arrays_equal(
        a + b,
        ak.zip(
            {
                "x": [[5, 3], [], [13], [15]],
                "y": [[22, 13], [], [18], [14]],
                "z": [[18, 21], [], [16], [28]],
                "t": [[110, 112], [], [114], [116]],
            }
        ),
    )
    assert_record_arrays_equal(
        a - b,
        ak.zip(
            {
                "x": [[-3, 1], [], [-7], [-7]],
                "y": [[-12, -1], [], [-4], [2]],
                "z": [[0, -1], [], [6], [-4]],
                "t": [[-10, -10], [], [-10], [-10]],
            }
        ),
    )

    assert_record_arrays_equal(
        a * 2,
        ak.zip(
            {
                "x": [[2, 4], [], [6], [8]],
                "y": [[10, 12], [], [14], [16]],
                "z": [[18, 20], [], [22], [24]],
                "t": [[100, 102], [], [104], [106]],
            }
        ),
    )
    assert_record_arrays_equal(
        a / 2,
        ak.zip(
            {
                "x": [[0.5, 1], [], [1.5], [2]],
                "y": [[2.5, 3], [], [3.5], [4]],
                "z": [[4.5, 5], [], [5.5], [6]],
                "t": [[25, 25.5], [], [26], [26.5]],
            }
        ),
    )

    assert_record_arrays_equal(
        a.pvec,
        ak.zip(
            {
                "x": [[1, 2], [], [3], [4]],
                "y": [[5, 6], [], [7], [8]],
                "z": [[9, 10], [], [11], [12]],
            }
        ),
    )

    boosted = a.boost(-a.boostvec)
    assert ak.all(abs(boosted.x) < ATOL)
    assert ak.all(abs(boosted.y) < ATOL)
    assert ak.all(abs(boosted.z) < ATOL)


def test_pt_eta_phi_m_lorentz_vector():
    a = ak.zip(
        {
            "pt": [[1, 2], [], [3], [4]],
            "eta": [[1.2, 1.4], [], [1.6], [3.4]],
            "phi": [[0.3, 0.4], [], [0.5], [0.6]],
            "mass": [[0.5, 0.9], [], [1.3], [4.5]],
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )
    a = ak.Array(a, behavior=vector.behavior)

    assert ak.all((a * (-2)).pt == ak.Array([[2, 4], [], [6], [8]]))
    assert ak.all(
        (a * (-2)).theta
        - ak.Array(
            [[2.556488570968, 2.65804615357], [], [2.74315571762], [3.07487087733]]
        )
        < ATOL
    )
    assert ak.all(
        (a * (-2)).phi
        - ak.Array(
            [[-2.8415926535, -2.7415926535], [], [-2.6415926535], [-2.5415926535]]
        )
        < ATOL
    )
    assert_record_arrays_equal(
        a / 2,
        ak.zip(
            {
                "pt": [[0.5, 1], [], [1.5], [2]],
                "eta": [[1.2, 1.4], [], [1.6], [3.4]],
                "phi": [[0.3, 0.4], [], [0.5], [0.6]],
                "mass": [[0.25, 0.45], [], [0.65], [2.25]],
            }
        ),
    )
    assert_record_arrays_equal(a * (-1), -a, check_type=True)

    boosted = a.boost(-a.boostvec)
    assert ak.all(abs(boosted.x) < ATOL)
    assert ak.all(abs(boosted.y) < ATOL)
    assert ak.all(abs(boosted.z) < ATOL)


def test_pt_eta_phi_e_lorentz_vector():
    a = ak.zip(
        {
            "pt": [[1, 2], [], [3], [4]],
            "eta": [[1.2, 1.4], [], [1.6], [3.4]],
            "phi": [[0.3, 0.4], [], [0.5], [0.6]],
            "energy": [[50, 51], [], [52], [60]],
        },
        with_name="PtEtaPhiELorentzVector",
        behavior=vector.behavior,
    )

    assert ak.all((a * (-2)).pt == ak.Array([[2, 4], [], [6], [8]]))
    assert ak.all(
        (a * (-2)).theta
        - ak.Array(
            [[2.556488570968, 2.65804615357], [], [2.74315571762], [3.07487087733]]
        )
        < ATOL
    )
    assert ak.all(
        (a * (-2)).phi
        - ak.Array(
            [[-2.8415926535, -2.7415926535], [], [-2.6415926535], [-2.5415926535]]
        )
        < ATOL
    )
    assert_record_arrays_equal(
        a / 2,
        ak.zip(
            {
                "pt": [[0.5, 1], [], [1.5], [2]],
                "eta": [[1.2, 1.4], [], [1.6], [3.4]],
                "phi": [[0.3, 0.4], [], [0.5], [0.6]],
                "energy": [[25, 25.5], [], [26], [30]],
            }
        ),
    )
    assert_record_arrays_equal(a * (-1), -a, check_type=True)

    boosted = a.boost(-a.boostvec)
    assert ak.all(abs(boosted.x) < ATOL)
    assert ak.all(abs(boosted.y) < ATOL)
    assert ak.all(abs(boosted.z) < ATOL)


@pytest.mark.parametrize("a_dtype", ["i4", "f4", "f8"])
@pytest.mark.parametrize("b_dtype", ["i4", "f4", "f8"])
def test_lorentz_vector_numba(a_dtype, b_dtype):
    a = ak.zip(
        {
            "x": np.array([1, 2, 3, 4], dtype=a_dtype),
            "y": np.array([5, 6, 7, 8], dtype=a_dtype),
            "z": np.array([9, 10, 11, 12], dtype=a_dtype),
            "t": np.array([50, 51, 52, 53], dtype=b_dtype),  # b on purpose
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )
    b = ak.zip(
        {
            "x": np.array([4, 1, 10, 11], dtype=b_dtype),
            "y": np.array([17, 7, 11, 6], dtype=b_dtype),
            "z": np.array([9, 11, 5, 16], dtype=b_dtype),
            "t": np.array([60, 61, 62, 63], dtype=b_dtype),
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )
    assert pytest.approx(a.mass) == [
        48.91829923454004,
        49.60846701924985,
        50.24937810560445,
        50.84289527554464,
    ]
    assert pytest.approx((a + b).mass) == [
        106.14612569472331,
        109.20164833920778,
        110.66616465749593,
        110.68423555321688,
    ]

    computed_dphi = a.delta_phi(b).to_numpy()

    assert pytest.approx(computed_dphi, abs=1e-6) == np.array(
        [
            0.03369510734601633,
            -0.1798534997924781,
            0.33292327383538156,
            0.6078019961139605,
        ],
        dtype=computed_dphi.dtype,
    )


@pytest.mark.parametrize(
    "lcoord", ["LorentzVector", "PtEtaPhiMLorentzVector", "PtEtaPhiELorentzVector"]
)
@pytest.mark.parametrize("threecoord", ["ThreeVector", "SphericalThreeVector"])
@pytest.mark.parametrize("twocoord", ["TwoVector", "PolarTwoVector"])
def test_inherited_method_transpose(lcoord, threecoord, twocoord):
    if lcoord == "LorentzVector":
        a = ak.zip(
            {
                "x": [10.0, 20.0, 30.0],
                "y": [-10.0, 20.0, 30.0],
                "z": [5.0, 10.0, 15.0],
                "t": [16.0, 31.0, 46.0],
            },
            with_name=lcoord,
            behavior=vector.behavior,
        )
    elif lcoord == "PtEtaPhiMLorentzVector":
        a = ak.zip(
            {
                "pt": [10.0, 20.0, 30.0],
                "eta": [0.0, 1.1, 2.2],
                "phi": [0.1, 0.9, -1.1],
                "mass": [1.0, 1.0, 1.0],
            },
            with_name=lcoord,
            behavior=vector.behavior,
        )
    elif lcoord == "PtEtaPhiELorentzVector":
        a = ak.zip(
            {
                "pt": [10.0, 20.0, 30.0],
                "eta": [0.0, 1.1, 2.2],
                "phi": [0.1, 0.9, -1.1],
                "energy": [11.0, 21.0, 31.0],
            },
            with_name=lcoord,
            behavior=vector.behavior,
        )
    if threecoord == "ThreeVector":
        b = ak.zip(
            {
                "x": [-10.0, 20.0, -30.0],
                "y": [-10.0, -20.0, 30.0],
                "z": [5.0, -10.0, 15.0],
            },
            with_name=threecoord,
            behavior=vector.behavior,
        )
    elif threecoord == "SphericalThreeVector":
        b = ak.zip(
            {
                "rho": [10.0, 20.0, 30.0],
                "theta": [0.3, 0.6, 1.1],
                "phi": [-3.0, 1.1, 0.2],
            },
            with_name=threecoord,
            behavior=vector.behavior,
        )
    if twocoord == "TwoVector":
        c = ak.zip(
            {"x": [-10.0, 13.0, 15.0], "y": [12.0, -4.0, 41.0]},
            with_name=twocoord,
            behavior=vector.behavior,
        )
    elif twocoord == "PolarTwoVector":
        c = ak.zip(
            {"rho": [-10.0, 13.0, 15.0], "phi": [1.22, -1.0, 1.0]},
            with_name=twocoord,
            behavior=vector.behavior,
        )

    assert_record_arrays_equal(a.like(b) + b, b + a.like(b), check_type=True)
    assert_record_arrays_equal(a.like(c) + c, c + a.like(c), check_type=True)
    assert_record_arrays_equal(b.like(c) + c, c + b.like(c), check_type=True)

    with pytest.raises(TypeError):
        a + b == b + a
    with pytest.raises(TypeError):
        a + c == c + a
    with pytest.raises(TypeError):
        b + c == c + b

    assert_allclose(a.delta_phi(b), -b.delta_phi(a))
    assert_allclose(a.delta_phi(c), -c.delta_phi(a))
    assert_allclose(b.delta_phi(c), -c.delta_phi(b))

    assert_record_arrays_equal((a.like(b) - b), -(b - a.like(b)), check_type=True)
    assert_record_arrays_equal((a.like(c) - c), -(c - a.like(c)), check_type=True)
    assert_record_arrays_equal((b.like(c) - c), -(c - b.like(c)), check_type=True)

    with pytest.raises(TypeError):
        a - b == -(b - a)
    with pytest.raises(TypeError):
        a - c == -(c - a)
    with pytest.raises(TypeError):
        b - c == -(c - b)


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_dask_metric_table_and_nearest(optimization_enabled):
    import dask
    from dask_awkward.lib.testutils import assert_eq

    from coffea.nanoevents import NanoEventsFactory

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        eagerevents = NanoEventsFactory.from_root(
            {"tests/samples/nano_dy.root": "Events"},
            delayed=False,
        ).events()

        daskevents = NanoEventsFactory.from_root(
            {"tests/samples/nano_dy.root": "Events"},
            delayed=True,
        ).events()

        mval_eager, (a_eager, b_eager) = eagerevents.Electron.metric_table(
            eagerevents.TrigObj, return_combinations=True
        )
        mval_dask, (a_dask, b_dask) = dask.compute(
            *daskevents.Electron.metric_table(
                daskevents.TrigObj, return_combinations=True
            )
        )
        assert_eq(mval_eager, mval_dask)
        assert_eq(a_eager, a_dask)
        assert_eq(b_eager, b_dask)

        out_eager, metric_eager = eagerevents.Electron.nearest(
            eagerevents.TrigObj, return_metric=True
        )
        out_dask, metric_dask = dask.compute(
            *daskevents.Electron.nearest(daskevents.TrigObj, return_metric=True)
        )
        assert_eq(out_eager, out_dask)
        assert_eq(metric_eager, metric_dask)

        out_eager_thresh, metric_eager_thresh = eagerevents.Electron.nearest(
            eagerevents.TrigObj, return_metric=True, threshold=0.4
        )
        out_dask_thresh, metric_dask_thresh = dask.compute(
            *daskevents.Electron.nearest(
                daskevents.TrigObj, return_metric=True, threshold=0.4
            )
        )
        assert_eq(out_eager_thresh, out_dask_thresh)
        assert_eq(metric_eager_thresh, metric_dask_thresh)


@pytest.mark.parametrize("optimization_enabled", [True, False])
def test_photon_zero_mass_charge(optimization_enabled):
    import dask
    from dask_awkward.lib.testutils import assert_eq

    from coffea.nanoevents import NanoEventsFactory

    with dask.config.set({"awkward.optimization.enabled": optimization_enabled}):
        eagerevents = NanoEventsFactory.from_root(
            {"tests/samples/nano_dy.root": "Events"},
            delayed=False,
        ).events()

        daskevents = NanoEventsFactory.from_root(
            {"tests/samples/nano_dy.root": "Events"},
            delayed=True,
        ).events()

        np.testing.assert_allclose(ak.flatten(eagerevents.Photon.mass), 0.0, atol=1e-5)
        np.testing.assert_allclose(
            ak.flatten(daskevents.Photon.mass).compute(), 0.0, atol=1e-5
        )
        np.testing.assert_allclose(
            ak.flatten(eagerevents.Photon.charge), 0.0, atol=1e-5
        )
        np.testing.assert_allclose(
            ak.flatten(daskevents.Photon.charge).compute(), 0.0, atol=1e-5
        )

        eagerdiphotonevents = eagerevents[ak.num(eagerevents.Photon) == 2]
        daskdiphotonevents = daskevents[ak.num(daskevents.Photon) == 2]
        eagerdiphotons = ak.zip(
            {
                "tag": eagerdiphotonevents.Photon[:, 0],
                "probe": eagerdiphotonevents.Photon[:, 1],
            }
        )
        daskdiphotons = ak.zip(
            {
                "tag": daskdiphotonevents.Photon[:, 0],
                "probe": daskdiphotonevents.Photon[:, 1],
            }
        )
        eagerdiphotons["mass"] = (eagerdiphotons.tag + eagerdiphotons.probe).mass
        daskdiphotons["mass"] = (daskdiphotons.tag + daskdiphotons.probe).mass
        eagermll = np.sqrt(
            2
            * eagerdiphotons.tag.pt
            * eagerdiphotons.probe.pt
            * (
                np.cosh(eagerdiphotons.tag.eta - eagerdiphotons.probe.eta)
                - np.cos(eagerdiphotons.tag.phi - eagerdiphotons.probe.phi)
            )
        )
        daskmll = np.sqrt(
            2
            * daskdiphotons.tag.pt
            * daskdiphotons.probe.pt
            * (
                np.cosh(daskdiphotons.tag.eta - daskdiphotons.probe.eta)
                - np.cos(daskdiphotons.tag.phi - daskdiphotons.probe.phi)
            )
        )
        assert_eq(eagerdiphotons.mass, eagermll)
        assert_eq(daskdiphotons.mass, daskmll)
        assert_eq(eagerdiphotons.mass, daskdiphotons.mass)
