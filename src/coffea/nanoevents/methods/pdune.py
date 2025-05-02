"""Mixins for the ATLAS PHYSLITE schema - work in progress."""

import awkward
import numpy

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
    global_index = _get_global_index(target_collection, eventindex, index)
    global_index = awkward.where(key != 0, global_index, -1)
    return target_collection._apply_global_index(global_index)


def _element_link_multiple(events, obj, link_field, with_name=None):
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


def _get_target_offsets(offsets, event_index):
    if isinstance(event_index, int):
        return offsets[event_index]

    def descend(layout, depth):
        if layout.purelist_depth == 1:
            return lambda: awkward.layout.NumpyArray(offsets)[layout]

    return awkward._util.recursively_apply(event_index.layout, descend)


def _get_global_index(target, eventindex, index):
    load_column = awkward.materialized(
        target[target.fields[0]]
    )  # need to load one column to extract the offsets
    target_offsets = _get_target_offsets(load_column.layout.offsets, eventindex)
    return target_offsets + index


behavior.update(awkward._util.copy_behaviors("LorentzVector", "Particle", behavior))


@awkward.mixin_class(behavior)
class Particle(vector.LorentzVector, base.NanoCollection):
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

    @property
    def trackParticle(self):
        return _element_link(
            self._events().CombinedMuonTrackParticles,
            self._eventindex,
            self["combinedTrackParticleLink.m_persIndex"],
            self["combinedTrackParticleLink.m_persKey"],
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

    @property
    def trackParticles(self):
        links = self.trackParticleLinks
        return _element_link(
            self._events().GSFTrackParticles,
            self._eventindex,
            links.m_persIndex,
            links.m_persKey,
        )

    @property
    def trackParticle(self):
        trackParticles = self.trackParticles
        return self.trackParticles[
            tuple([slice(None) for i in range(trackParticles.ndim - 1)] + [0])
        ]


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
