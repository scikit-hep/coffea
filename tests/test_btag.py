import awkward as ak
import numpy
import pytest
from dummy_distributions import dummy_jagged_eta_pt

from coffea.btag_tools import BTagScaleFactor


def test_BTagScalefactor():
    sf1 = BTagScaleFactor("tests/samples/testBTagSF.btag.csv", "medium")
    sf2 = BTagScaleFactor(
        "tests/samples/DeepCSV_102XSF_V1.btag.csv.gz",
        BTagScaleFactor.RESHAPE,
        "iterativefit",
    )
    sf3 = BTagScaleFactor(
        "tests/samples/DeepCSV_102XSF_V1.btag.csv.gz", BTagScaleFactor.TIGHT
    )
    sf4 = BTagScaleFactor(
        "tests/samples/DeepCSV_2016LegacySF_V1_TuneCP5.btag.csv.gz",
        BTagScaleFactor.RESHAPE,
        "iterativefit",
        keep_df=True,
    )
    # import pdb; pdb.set_trace()

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    test_flavor = numpy.random.choice([0, 4, 5], size=len(test_eta))
    test_allb = numpy.ones_like(test_flavor) * 5
    test_discr = numpy.random.rand(len(test_eta))
    offsets = numpy.zeros(len(counts) + 1)
    offsets[1:] = numpy.cumsum(counts)
    test_jets = ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(offsets),
            ak.zip(
                {
                    "pt": test_pt,
                    "eta": test_eta,
                    "flavor": test_flavor,
                    "btag": test_discr,
                },
                highlevel=False,
            ),
        )
    )

    expected = numpy.array(
        [
            0.93724101,
            0.89943609,
            1.0671185,
            1.06846618,
            0.94530984,
            1.06645614,
            0.91862676,
            1.06645614,
            0.94372127,
            0.94505261,
            1.06645614,
            1.06645614,
            1.06645614,
            1.06645614,
            0.91385676,
            1.06738093,
            0.89943609,
            0.92593492,
            1.06960044,
            0.89943609,
            1.06645614,
            1.06645614,
            0.94290361,
            1.06892548,
            0.92440686,
            0.92046542,
            1.06645614,
            0.93676041,
            0.93392431,
            0.91694353,
            0.89943609,
            0.89943609,
            0.89943609,
            0.89943609,
            0.89943609,
            0.89943609,
            0.89943609,
            0.89943609,
            0.93371251,
            0.89943609,
            0.89943609,
            0.89943609,
            0.94767034,
            1.06645614,
            1.0670672,
            1.07136352,
            0.89943609,
            0.90445481,
            0.89943609,
            1.06645614,
            0.89943609,
            0.89943609,
            0.93745389,
            0.90949125,
            0.91778825,
            1.06645614,
            1.06645614,
            0.89943609,
            0.89943609,
            1.06645614,
            1.06645614,
            1.06645614,
        ]
    )
    result = sf1.eval("central", test_flavor, test_eta, test_pt, test_discr)
    assert numpy.allclose(result, expected)

    sf1.eval("up", test_flavor, test_eta, test_pt)
    sf2.eval("central", test_allb, test_eta, test_pt, test_discr)
    with pytest.raises(ValueError):
        sf2.eval("up", test_allb, test_eta, test_pt)
    sf3.eval("central", test_flavor, test_eta, test_pt, test_discr)
    sf3.eval("up", test_flavor, test_eta, test_pt)
    with pytest.raises(ValueError):
        sf4.eval("central", test_flavor, test_eta, test_pt, test_discr)

    expected = numpy.array(
        [
            1.2185781,
            1.03526095,
            1.14997077,
            0.91933821,
            1.2185781,
            1.08865945,
            0.99422718,
            1.01943199,
            1.01025089,
            1.20312875,
            0.84198391,
            0.91726759,
            0.93501452,
            1.31649974,
            1.14997077,
            1.02107876,
            1.06150099,
            1.06063444,
            0.90508972,
            1.20768481,
            0.8484613,
            0.99217259,
            0.98333802,
            1.31302575,
            1.0104926,
            1.00474285,
            1.24375693,
            1.20949677,
            0.91714979,
            0.99533782,
            1.14997077,
            1.02871797,
            0.99619147,
            0.97543142,
            1.31518787,
            1.30700837,
            1.14997077,
            0.99879282,
            0.98961045,
            1.14997077,
            0.88343516,
            0.9930647,
            1.17767042,
            1.14997077,
            1.30594256,
            0.91888068,
            1.04737201,
            1.03583147,
            1.02833176,
            0.99527427,
            0.98546895,
            1.14997077,
            1.04815223,
            1.28007547,
            1.1970858,
            1.12892238,
            1.14997077,
            1.14997077,
            1.01656481,
            0.84198391,
            1.2996388,
            1.14997077,
        ]
    )
    result = sf4.eval("central", test_allb, test_eta, test_pt, test_discr)
    assert numpy.allclose(result, expected)

    sf1.eval("down", test_jets.flavor, test_jets.eta, test_jets.pt)
