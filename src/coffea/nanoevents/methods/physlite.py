"""Mixins for the ATLAS PHYSLITE schema - work in progress."""

from numbers import Number

import awkward
import dask_awkward
import numpy
from dask_awkward import dask_property

from coffea.nanoevents.methods import base, vector

behavior = {}
behavior.update(base.behavior)
behavior.update(vector.behavior)


def _set_repr_name(classname):
    def namefcn(self):
        return classname

    behavior[("__typestr__", classname)] = classname[0].lower() + classname[1:]
    behavior[classname].__repr__ = namefcn


# from MetaData/EventFormat
_hash_to_target_name = {
    13267281: "TruthPhotons",
    342174277: "TruthMuons",
    368360608: "TruthNeutrinos",
    375408000: "TruthTaus",
    394100163: "TruthElectrons",
    614719239: "TruthBoson",
    660928181: "TruthTop",
    779635413: "TruthBottom",
}


def _element_link(target_collection, eventindex, index, key):
    # check to make sure both the target_collection and the eventindex are both dask-awkward arrays or both awkward arrays
    if isinstance(target_collection, dask_awkward.Array) != isinstance(
        eventindex, dask_awkward.Array
    ):
        raise ValueError(
            "element linking must be done on two dask_awkward arrays or two akward arrays not a mix of the two"
        )

    global_index = _get_global_index(target_collection, eventindex, index)
    global_index = awkward.where(key != 0, global_index, -1)
    return target_collection._apply_global_index(global_index)


def _element_link_method(self, link_name, target_name, _dask_array_):
    if _dask_array_ is not None:
        target = _dask_array_.attrs["@original_array"][target_name]
        links = _dask_array_[link_name]
        return _element_link(
            target,
            _dask_array_._eventindex,
            links.m_persIndex,
            links.m_persKey,
        )
    links = self[link_name]
    return _element_link(
        self._events()[target_name],
        self._eventindex,
        links.m_persIndex,
        links.m_persKey,
    )


def _element_link_multiple(events, obj, link_field, with_name=None):
    # currently not working in dask because:
    # - we don't know the resulting type beforehand
    # - also not the targets, so no way to find out which columns to load?
    # - could consider to treat the case of truth collections by just loading all truth columns
    link = obj[link_field]
    key = link.m_persKey
    index = link.m_persIndex
    unique_keys = [
        i
        for i in numpy.unique(awkward.to_numpy(awkward.flatten(key, axis=None)))
        if i != 0
    ]

    def where(unique_keys):
        target_name = _hash_to_target_name[unique_keys[0]]
        mask = key == unique_keys[0]
        global_index = _get_global_index(events[target_name], obj._eventindex, index)
        global_index = awkward.where(mask, global_index, -1)
        links = events[target_name]._apply_global_index(global_index)
        if len(unique_keys) == 1:
            return links
        return awkward.where(mask, links, where(unique_keys[1:]))

    out = where(unique_keys).mask[key != 0]
    if with_name is not None:
        out = awkward.with_parameter(out, "__record__", with_name)
    return out


def _get_target_offsets(load_column, event_index):
    if isinstance(load_column, dask_awkward.Array) and isinstance(
        event_index, dask_awkward.Array
    ):
        # wrap in map_partitions if dask arrays
        return dask_awkward.map_partitions(
            _get_target_offsets, load_column, event_index, label="get_target_offsets"
        )

    offsets = load_column.layout.offsets.data

    if isinstance(event_index, Number):
        return offsets[event_index]

    # let the necessary column optimization know that we need to load this
    # column to get the offsets
    if awkward.backend(load_column) == "typetracer":
        awkward.typetracer.touch_data(load_column)

    # necessary to stick it into the `NumpyArray` constructor
    # if typetracer is passed through
    offsets = awkward.typetracer.length_zero_if_typetracer(
        load_column.layout.offsets.data
    )

    def descend(layout, depth, **kwargs):
        if layout.purelist_depth == 1:
            return awkward.contents.NumpyArray(offsets)[layout]

    return awkward.transform(descend, event_index.layout)


def _get_global_index(target, eventindex, index):
    for field in target.fields:
        # fetch first column to get offsets from
        # (but try to avoid the double-jagged ones if possible)
        load_column = target[field]
        if load_column.ndim < 3:
            break
    target_offsets = _get_target_offsets(load_column, eventindex)
    return target_offsets + index


behavior.update(
    awkward._util.copy_behaviors("PtEtaPhiMLorentzVector", "Particle", behavior)
)


@awkward.mixin_class(behavior)
class Particle(vector.PtEtaPhiMLorentzVector, base.NanoCollection):
    """Generic particle collection that has Lorentz vector properties"""

    @property
    def mass(self):
        return self.m


_set_repr_name("Particle")

ParticleArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ParticleArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ParticleArray.ProjectionClass4D = ParticleArray  # noqa: F821
ParticleArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(
    awkward._util.copy_behaviors("LorentzVector", "TrackParticle", behavior)
)


@awkward.mixin_class(behavior)
class TrackParticle(vector.LorentzVector, base.NanoCollection):
    """Collection of track particles, following `xAOD::TrackParticle_v1
    <https://gitlab.cern.ch/atlas/athena/-/blob/21.2/Event/xAOD/xAODTracking/Root/TrackParticle_v1.cxx#L82>`_.
    """

    @property
    def theta(self):
        return self["theta"]

    @property
    def phi(self):
        return self["phi"]

    @property
    def p(self):
        return 1.0 / numpy.abs(self.qOverP)

    @property
    def x(self):
        return self.p * numpy.sin(self.theta) * numpy.cos(self.phi)

    @property
    def y(self):
        return self.p * numpy.sin(self.theta) * numpy.sin(self.phi)

    @property
    def z(self):
        return self.p * numpy.cos(self.theta)

    @property
    def t(self):
        return numpy.sqrt(139.570**2 + self.x**2 + self.y**2 + self.z**2)


_set_repr_name("TrackParticle")

TrackParticleArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
TrackParticleArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
TrackParticleArray.ProjectionClass4D = TrackParticleArray  # noqa: F821
TrackParticleArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("Particle", "Muon", behavior))


@awkward.mixin_class(behavior)
class Muon(Particle):
    """Muon collection, following `xAOD::Muon_v1
    <https://gitlab.cern.ch/atlas/athena/-/blob/21.2/Event/xAOD/xAODMuon/Root/Muon_v1.cxx>`_.
    """

    @dask_property
    def trackParticle(self):
        return _element_link_method(
            self,
            "combinedTrackParticleLink",
            "CombinedMuonTrackParticles",
            None,
        )

    @trackParticle.dask
    def trackParticle(self, dask_array):
        return _element_link_method(
            self,
            "combinedTrackParticleLink",
            "CombinedMuonTrackParticles",
            dask_array,
        )


_set_repr_name("Muon")

MuonArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
MuonArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
MuonArray.ProjectionClass4D = MuonArray  # noqa: F821
MuonArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("Particle", "Electron", behavior))


@awkward.mixin_class(behavior)
class Electron(Particle):
    """Electron collection, following `xAOD::Electron_v1
    <https://gitlab.cern.ch/atlas/athena/-/blob/21.2/Event/xAOD/xAODEgamma/Root/Electron_v1.cxx>`_.
    """

    @dask_property
    def trackParticles(self):
        return _element_link_method(
            self, "trackParticleLinks", "GSFTrackParticles", None
        )

    @trackParticles.dask
    def trackParticles(self, dask_array):
        return _element_link_method(
            self, "trackParticleLinks", "GSFTrackParticles", dask_array
        )

    @dask_property
    def trackParticle(self):
        trackParticles = _element_link_method(
            self, "trackParticleLinks", "GSFTrackParticles", None
        )
        # Ellipsis (..., 0) slicing not supported yet by dask_awkward
        slicer = tuple([slice(None) for i in range(trackParticles.ndim - 1)] + [0])
        return trackParticles[slicer]

    @trackParticle.dask
    def trackParticle(self, dask_array):
        trackParticles = _element_link_method(
            self, "trackParticleLinks", "GSFTrackParticles", dask_array
        )
        # Ellipsis (..., 0) slicing not supported yet by dask_awkward
        slicer = tuple([slice(None) for i in range(trackParticles.ndim - 1)] + [0])
        return trackParticles[slicer]

    @dask_property
    def caloClusters(self):
        return _element_link_method(self, "caloClusterLinks", "egammaClusters", None)

    @caloClusters.dask
    def caloClusters(self, dask_array):
        return _element_link_method(
            self, "caloClusterLinks", "egammaClusters", dask_array
        )


_set_repr_name("Electron")

ElectronArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ElectronArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ElectronArray.ProjectionClass4D = ElectronArray  # noqa: F821
ElectronArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(
    awkward._util.copy_behaviors("LorentzVector", "TruthParticle", behavior)
)


@awkward.mixin_class(behavior)
class TruthParticle(vector.LorentzVector, base.NanoCollection):
    """Truth particle collection, following `xAOD::TruthParticle_v1
    <https://gitlab.cern.ch/atlas/athena/-/blob/21.2/Event/xAOD/xAODTruth/Root/TruthParticle_v1.cxx>`_.
    """

    @property
    def x(self):
        return self["px"]

    @property
    def y(self):
        return self["py"]

    @property
    def z(self):
        return self["pz"]

    @property
    def t(self):
        return self["e"]

    @property
    def mass(self):
        return self["m"]

    @property
    def children(self):
        return _element_link_multiple(
            self._events(), self, "childLinks", with_name="TruthParticle"
        )

    @property
    def parents(self):
        return _element_link_multiple(
            self._events(), self, "parentLinks", with_name="TruthParticle"
        )


_set_repr_name("TruthParticle")

TruthParticleArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
TruthParticleArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
TruthParticleArray.ProjectionClass4D = TruthParticleArray  # noqa: F821
TruthParticleArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821
