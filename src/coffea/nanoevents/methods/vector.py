"""2D, 3D, and Lorentz vector class mixins

These mixins will eventually be superseded by the `vector <https://github.com/scikit-hep/vector>`__ library,
which will hopefully be feature-compatible. The 2D vector provides cartesian and polar coordinate attributes,
where ``r`` represents the polar distance from the origin..  The 3D vector provides cartesian and spherical coordinates,
where ``rho`` represents the 3D distance from the origin and ``r`` is the axial distance from the z axis, so that it can
subclass the 2D vector. The Lorentz vector also subclasses the 3D vector, adding ``t`` as the fourth
cartesian coordinate. Aliases typical of momentum vectors are also provided.

A small example::

    import numpy as np
    import awkward as ak
    from coffea.nanoevents.methods import vector

    n = 1000

    vec = ak.zip(
        {
            "x": np.random.normal(size=n),
            "y": np.random.normal(size=n),
            "z": np.random.normal(size=n),
        },
        with_name="ThreeVector",
        behavior=vector.behavior,
    )

    vec4 = ak.zip(
        {
            "pt": vec.r,
            "eta": -np.log(np.tan(vec.theta/2)),
            "phi": vec.phi,
            "mass": np.full(n, 1.),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

    assert np.allclose(np.array(vec4.x), np.array(vec.x))
    assert np.allclose(np.array(vec4.y), np.array(vec.y))
    assert np.allclose(np.array(vec4.z), np.array(vec.z))
    assert np.allclose(np.array(abs(2*vec + vec4) / abs(vec)), 3)

"""

import numbers
from datetime import datetime

import awkward
import numba
import numpy
import pytz
import vector
from dask_awkward import dask_method
from vector.backends.awkward import (
    MomentumAwkward2D,
    MomentumAwkward3D,
    MomentumAwkward4D,
)

from coffea.util import deprecate

_cst = pytz.timezone("US/Central")
_depttime = _cst.localize(datetime(2024, 12, 31, 11, 59, 59))
deprecate(
    (
        "coffea.nanoevents.methods.vector will be removed and replaced with scikit-hep vector. "
        "Nanoevents schemas internal to coffea will be migrated. "
        "Otherwise please consider using that package!"
    ),
    version="2025.1.0",
    date=str(_depttime),
    category=FutureWarning,
)


@numba.vectorize(
    [
        numba.float32(numba.float32, numba.float32, numba.float32, numba.float32),
        numba.float64(numba.float64, numba.float64, numba.float64, numba.float64),
    ]
)
def _mass2_kernel(t, x, y, z):
    return t * t - x * x - y * y - z * z


@numba.vectorize(
    [
        numba.float32(numba.float32, numba.float32),
        numba.float64(numba.float64, numba.float64),
    ]
)
def delta_phi(a, b):
    """Compute difference in angle given two angles a and b

    Returns a value within [-pi, pi)
    """
    return (a - b + numpy.pi) % (2 * numpy.pi) - numpy.pi


@numba.vectorize(
    [
        numba.float32(numba.float32, numba.float32, numba.float32, numba.float32),
        numba.float64(numba.float64, numba.float64, numba.float64, numba.float64),
    ]
)
def delta_r(eta1, phi1, eta2, phi2):
    r"""Distance in (eta,phi) plane given two pairs of (eta,phi)

    :math:`\sqrt{\Delta\eta^2 + \Delta\phi^2}`
    """
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return numpy.hypot(deta, dphi)


behavior = {}
behavior.update(vector.backends.awkward.behavior)


@awkward.mixin_class(behavior)
class TwoVector(MomentumAwkward2D):
    """A cartesian 2-dimensional vector

    A heavy emphasis towards a momentum vector interpretation is assumed, hence
    properties like `px` and `py` are provided in addition to `x` and `y`.

    This mixin class requires the parent class to provide items `x` and `y`.
    """

    @property
    def r(self):
        r"""Distance from origin in XY plane

        :math:`\sqrt{x^2+y^2}`
        """
        return self.rho

    @property
    def r2(self):
        """Squared `r`"""
        return self.rho2

    @awkward.mixin_class_method(numpy.absolute)
    def absolute(self):
        """Returns magnitude of the 2D vector

        Alias for `r`
        """
        return self.r

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return self.scale(-1)

    def sum(self, axis=-1):
        """Sum an array of vectors elementwise using `x` and `y` components"""
        return awkward.zip(
            {
                "x": awkward.sum(self.x, axis=axis),
                "y": awkward.sum(self.y, axis=axis),
            },
            with_name="TwoVector",
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using `x` and `y` components"""
        return self.scale(other)

    @awkward.mixin_class_method(numpy.divide, {numbers.Number})
    def divide(self, other):
        """Divide this vector by a scalar elementwise using its cartesian components

        This is realized by using the multiplication functionality"""
        return self.scale(1 / other)

    def delta_phi(self, other):
        """Compute difference in angle between two vectors

        Returns a value within [-pi, pi)
        """
        return self.deltaphi(other)

    @property
    def unit(self):
        """Unit vector, a vector of length 1 pointing in the same direction"""
        return self / self.r


@awkward.mixin_class(behavior)
class PolarTwoVector(TwoVector):
    """A polar coordinate 2-dimensional vector

    This mixin class requires the parent class to provide items `rho` and `phi`.
    Some additional properties are overridden for performance
    """

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using using `x` and `y` components

        In reality, this directly adjusts `r` and `phi` for performance
        """
        return self.scale(other)

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return self.scale(-1)


@awkward.mixin_class(behavior)
class ThreeVector(MomentumAwkward3D):
    """A cartesian 3-dimensional vector

    A heavy emphasis towards a momentum vector interpretation is assumed.
    This mixin class requires the parent class to provide items `x`, `y`, and `z`.
    """

    @property
    def r(self):
        r"""Distance from origin in XY plane

        :math:`\sqrt{x^2+y^2}`
        """
        return self.rho

    @property
    def r2(self):
        """Squared `r`"""
        return self.rho2

    @awkward.mixin_class_method(numpy.absolute)
    def absolute(self):
        """Returns magnitude of the 3D vector

        Alias for `rho`
        """
        return self.p

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return self.scale(-1)

    @awkward.mixin_class_method(numpy.divide, {numbers.Number})
    def divide(self, other):
        """Divide this vector by a scalar elementwise using its cartesian components
        This is realized by using the multiplication functionality"""
        return self.scale(1 / other)

    def sum(self, axis=-1):
        """Sum an array of vectors elementwise using `x`, `y`, and `z` components"""
        return awkward.zip(
            {
                "x": awkward.sum(self.x, axis=axis),
                "y": awkward.sum(self.y, axis=axis),
                "z": awkward.sum(self.z, axis=axis),
            },
            with_name="ThreeVector",
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using `x`, `y`, and `z` components"""
        return self.scale(other)

    def delta_phi(self, other):
        """Compute difference in angle between two vectors

        Returns a value within [-pi, pi)
        """
        return self.deltaphi(other)

    @property
    def unit(self):
        """Unit vector, a vector of length 1 pointing in the same direction"""
        return self / self.rho


@awkward.mixin_class(behavior)
class SphericalThreeVector(ThreeVector):
    """A spherical coordinate 3-dimensional vector

    This mixin class requires the parent class to provide items `rho`, `theta`, and `phi`.
    Some additional properties are overridden for performance
    """

    @property
    def r(self):
        r"""Distance from origin in XY plane

        :math:`\sqrt{x^2+y^2} = \rho \sin(\theta)`
        """
        return self.rho

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using `x`, `y`, and `z` components

        In reality, this directly adjusts `r`, `theta` and `phi` for performance
        """
        return self.scale(other)

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return self.scale(-1)


def _metric_table_core(a, b, axis, metric, return_combinations):
    if axis is None:
        a, b = a, b
    else:
        a, b = awkward.unzip(awkward.cartesian([a, b], axis=axis, nested=True))
    mval = metric(a, b)
    if return_combinations:
        return mval, (a, b)
    return mval


def _nearest_core(x, y, axis, metric, return_metric, threshold):
    mval, (a, b) = x.metric_table(y, axis, metric, return_combinations=True)
    if axis is None:
        # NotImplementedError: awkward.firsts with axis=-1
        axis = y.layout.purelist_depth - 2
    mmin = awkward.argmin(mval, axis=axis + 1, keepdims=True)
    out = awkward.firsts(b[mmin], axis=axis + 1)
    metric = awkward.firsts(mval[mmin], axis=axis + 1)
    if threshold is not None:
        out = awkward.mask(out, metric <= threshold)
    if return_metric:
        return out, metric
    return out


@awkward.mixin_class(behavior)
class LorentzVector(MomentumAwkward4D):
    """A cartesian Lorentz vector

    A heavy emphasis towards a momentum vector interpretation is assumed.
    (+, -, -, -) metric
    This mixin class requires the parent class to provide items `x`, `y`, `z`, and `t`.
    """

    @awkward.mixin_class_method(numpy.absolute)
    def absolute(self):
        """Magnitude of this Lorentz vector

        Alias for `mass`
        """
        return self.mass

    def sum(self, axis=-1):
        """Sum an array of vectors elementwise using `x`, `y`, `z`, and `t` components"""
        return awkward.zip(
            {
                "x": awkward.sum(self.x, axis=axis),
                "y": awkward.sum(self.y, axis=axis),
                "z": awkward.sum(self.z, axis=axis),
                "t": awkward.sum(self.t, axis=axis),
            },
            with_name="LorentzVector",
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using `x`, `y`, `z`, and `t` components"""
        return self.scale(other)

    @awkward.mixin_class_method(numpy.divide, {numbers.Number})
    def divide(self, other):
        """Divide this vector by a scalar elementwise using its cartesian components
        This is realized by using the multiplication functionality"""
        return self.scale(1 / other)

    def delta_r2(self, other):
        """Squared `delta_r`"""
        return self.deltaR2(other)

    def delta_r(self, other):
        r"""Distance between two Lorentz vectors in (eta,phi) plane

        :math:`\sqrt{\Delta\eta^2 + \Delta\phi^2}`
        """
        return self.deltaR(other)

    def delta_phi(self, other):
        """Compute difference in angle between two vectors

        Returns a value within [-pi, pi)
        """
        return self.deltaphi(other)

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return self.scale(-1)

    @property
    def pvec(self):
        """The `x`, `y` and `z` components as a `ThreeVector`"""
        return awkward.zip(
            {"x": self.x, "y": self.y, "z": self.z},
            with_name="ThreeVector",
            behavior=self.behavior,
        )

    @property
    def boostvec(self):
        """The `x`, `y` and `z` components divided by `t` as a `ThreeVector`

        This can be used for boosting. For cases where `|t| <= r`, this
        returns the unit vector.
        """
        return self.to_beta3()

    @dask_method
    def metric_table(
        self,
        other,
        axis=1,
        metric=lambda a, b: a.delta_r(b),
        return_combinations=False,
    ):
        """Return a list of a metric evaluated between this object and another.

        The two arrays should be broadcast-compatible on all axes other than the specified
        axis, which will be used to form a cartesian product. If axis=None, broadcast arrays directly.
        The return shape will be that of ``self`` with a new axis with shape of ``other`` appended
        at the specified axis depths.

        Parameters
        ----------
            other : awkward.Array
                Another array with same shape in all but ``axis``
            axis : int, optional
                The axis to form the cartesian product (default 1). If None, the metric
                is directly evaluated on the input arrays (i.e. they should broadcast)
            metric : callable
                A function of two arguments, returning a scalar. The default metric is `delta_r`.
            return_combinations : bool
                If True return the combinations of inputs as well as an unzipped tuple
        """
        return _metric_table_core(self, other, axis, metric, return_combinations)

    @metric_table.dask
    def metric_table(
        self,
        dask_array,
        other,
        axis=1,
        metric=lambda a, b: a.delta_r(b),
        return_combinations=False,
    ):
        return _metric_table_core(dask_array, other, axis, metric, return_combinations)

    @dask_method
    def nearest(
        self,
        other,
        axis=1,
        metric=lambda a, b: a.delta_r(b),
        return_metric=False,
        threshold=None,
    ):
        """Return nearest object to this one

        Finds item in ``other`` satisfying ``min(metric(self, other))``.
        The two arrays should be broadcast-compatible on all axes other than the specified
        axis, which will be used to form a cartesian product. If axis=None, broadcast arrays directly.
        The return shape will be that of ``self``.

        Parameters
        ----------
            other : awkward.Array
                Another array with same shape in all but ``axis``
            axis : int, optional
                The axis to form the cartesian product (default 1). If None, the metric
                is directly evaluated on the input arrays (i.e. they should broadcast)
            metric : callable
                A function of two arguments, returning a scalar. The default metric is `delta_r`.
            return_metric : bool, optional
                If true, return both the closest object and its metric (default false)
            threshold : Number, optional
                If set, any objects with ``metric > threshold`` will be masked from the result
        """
        return _nearest_core(self, other, axis, metric, return_metric, threshold)

    @nearest.dask
    def nearest(
        self,
        dask_array,
        other,
        axis=1,
        metric=lambda a, b: a.delta_r(b),
        return_metric=False,
        threshold=None,
    ):
        return _nearest_core(dask_array, other, axis, metric, return_metric, threshold)


@awkward.mixin_class(behavior)
class PtEtaPhiMLorentzVector(LorentzVector):
    """A Lorentz vector using pseudorapidity and mass

    This mixin class requires the parent class to provide items `pt`, `eta`, `phi`, and `mass`.
    Some additional properties are overridden for performance
    """

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using `x`, `y`, `z`, and `t` components

        In reality, this directly adjusts `pt`, `eta`, `phi` and `mass` for performance
        """
        absother = abs(other)
        return awkward.zip(
            {
                "pt": self.pt * absother,
                "eta": self.eta * numpy.sign(other),
                "phi": self.phi % (2 * numpy.pi) - (numpy.pi * (other < 0)),
                "mass": self.mass * absother,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return awkward.zip(
            {
                "pt": self.pt,
                "eta": -self.eta,
                "phi": self.phi % (2 * numpy.pi) - numpy.pi,
                "mass": self.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(numpy.divide, {numbers.Number})
    def divide(self, other):
        """Divide this vector by a scalar elementwise using its cartesian components
        This is realized by using the multiplication functionality"""
        return self.multiply(1 / other)


@awkward.mixin_class(behavior)
class PtEtaPhiELorentzVector(LorentzVector):
    """A Lorentz vector using pseudorapidity and energy

    This mixin class requires the parent class to provide items `pt`, `eta`, `phi`, and `energy`.
    Some additional properties are overridden for performance
    """

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using `x`, `y`, `z`, and `t` components

        In reality, this directly adjusts `pt`, `eta`, `phi` and `energy` for performance
        """
        return awkward.zip(
            {
                "pt": self.pt * abs(other),
                "eta": self.eta * numpy.sign(other),
                "phi": self.phi % (2 * numpy.pi) - (numpy.pi * (other < 0)),
                "energy": self.energy * other,
            },
            with_name="PtEtaPhiELorentzVector",
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return awkward.zip(
            {
                "pt": self.pt,
                "eta": -self.eta,
                "phi": self.phi % (2 * numpy.pi) - numpy.pi,
                "energy": -self.energy,
            },
            with_name="PtEtaPhiELorentzVector",
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(numpy.divide, {numbers.Number})
    def divide(self, other):
        """Divide this vector by a scalar elementwise using its cartesian components
        This is realized by using the multiplication functionality"""
        return self.multiply(1 / other)


_binary_dispatch_cls = {
    "TwoVector": TwoVector,
    "PolarTwoVector": TwoVector,
    "ThreeVector": ThreeVector,
    "SphericalThreeVector": ThreeVector,
    "LorentzVector": LorentzVector,
    "PtEtaPhiMLorentzVector": LorentzVector,
    "PtEtaPhiELorentzVector": LorentzVector,
}
_rank = [TwoVector, ThreeVector, LorentzVector]

for lhs, lhs_to in _binary_dispatch_cls.items():
    for rhs, rhs_to in _binary_dispatch_cls.items():
        out_to = min(lhs_to, rhs_to, key=_rank.index)
        behavior[(numpy.add, lhs, rhs)] = out_to.add
        behavior[(numpy.subtract, lhs, rhs)] = out_to.subtract


TwoVectorArray.ProjectionClass2D = TwoVectorArray  # noqa: F821
TwoVectorArray.ProjectionClass3D = ThreeVectorArray  # noqa: F821
TwoVectorArray.ProjectionClass4D = LorentzVectorArray  # noqa: F821
TwoVectorArray.MomentumClass = PolarTwoVectorArray  # noqa: F821

PolarTwoVectorArray.ProjectionClass2D = PolarTwoVectorArray  # noqa: F821
PolarTwoVectorArray.ProjectionClass3D = SphericalThreeVectorArray  # noqa: F821
PolarTwoVectorArray.ProjectionClass4D = LorentzVectorArray  # noqa: F821
PolarTwoVectorArray.MomentumClass = PolarTwoVectorArray  # noqa: F821

ThreeVectorArray.ProjectionClass2D = TwoVectorArray  # noqa: F821
ThreeVectorArray.ProjectionClass3D = ThreeVectorArray  # noqa: F821
ThreeVectorArray.ProjectionClass4D = LorentzVectorArray  # noqa: F821
ThreeVectorArray.MomentumClass = SphericalThreeVectorArray  # noqa: F821

SphericalThreeVectorArray.ProjectionClass2D = PolarTwoVectorArray  # noqa: F821
SphericalThreeVectorArray.ProjectionClass3D = SphericalThreeVectorArray  # noqa: F821
SphericalThreeVectorArray.ProjectionClass4D = LorentzVectorArray  # noqa: F821
SphericalThreeVectorArray.MomentumClass = SphericalThreeVectorArray  # noqa: F821

LorentzVectorArray.ProjectionClass2D = TwoVectorArray  # noqa: F821
LorentzVectorArray.ProjectionClass3D = ThreeVectorArray  # noqa: F821
LorentzVectorArray.ProjectionClass4D = LorentzVectorArray  # noqa: F821
LorentzVectorArray.MomentumClass = LorentzVectorArray  # noqa: F821

PtEtaPhiMLorentzVectorArray.ProjectionClass2D = TwoVectorArray  # noqa: F821
PtEtaPhiMLorentzVectorArray.ProjectionClass3D = ThreeVectorArray  # noqa: F821
PtEtaPhiMLorentzVectorArray.ProjectionClass4D = LorentzVectorArray  # noqa: F821
PtEtaPhiMLorentzVectorArray.MomentumClass = LorentzVectorArray  # noqa: F821

PtEtaPhiELorentzVectorArray.ProjectionClass2D = TwoVectorArray  # noqa: F821
PtEtaPhiELorentzVectorArray.ProjectionClass3D = ThreeVectorArray  # noqa: F821
PtEtaPhiELorentzVectorArray.ProjectionClass4D = LorentzVectorArray  # noqa: F821
PtEtaPhiELorentzVectorArray.MomentumClass = LorentzVectorArray  # noqa: F821

__all__ = [
    "TwoVector",
    "PolarTwoVector",
    "ThreeVector",
    "SphericalThreeVector",
    "LorentzVector",
    "PtEtaPhiMLorentzVector",
    "PtEtaPhiELorentzVector",
]
