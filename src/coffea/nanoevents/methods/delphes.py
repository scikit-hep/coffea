"""Mixins for the Delphes schema

See https://cp3.irmp.ucl.ac.be/projects/delphes/wiki/WorkBook/RootTreeDescription for details.
"""

import awkward
import numpy

from coffea.nanoevents.methods import base, candidate, vector

behavior = {}
behavior.update(base.behavior)
# vector behavior is included in candidate behavior
behavior.update(candidate.behavior)


class DelphesEvents(behavior["NanoEvents"]):
    def __repr__(self):
        return f"<Delphes event {self.Event.Number}>"


behavior["NanoEvents"] = DelphesEvents


def _set_repr_name(classname):
    def namefcn(self):
        return classname

    behavior[("__typestr__", classname)] = classname[0].lower() + classname[1:]
    behavior[classname].__repr__ = namefcn


@awkward.mixin_class(behavior)
class Event: ...


_set_repr_name("Event")


@awkward.mixin_class(behavior)
class LHEFEvent(Event): ...


_set_repr_name("LHEFEvent")


@awkward.mixin_class(behavior)
class HepMCEvent(Event): ...


_set_repr_name("HepMCEvent")


@awkward.mixin_class(behavior)
class LHCOEvent(Event): ...


_set_repr_name("LHCOEvent")


@awkward.mixin_class(behavior)
class Weight(base.NanoCollection): ...


_set_repr_name("Weight")


@awkward.mixin_class(behavior)
class WeightLHEF(Event): ...


_set_repr_name("WeightLHEF")


@awkward.mixin_class(behavior)
class Rho(base.NanoCollection): ...


_set_repr_name("Rho")


@awkward.mixin_class(behavior)
class ScalarHT(base.NanoCollection): ...


_set_repr_name("ScalarHT")


@awkward.mixin_class(behavior)
class MissingET(vector.SphericalThreeVector, base.NanoCollection):
    @property
    def rho(self):
        return self["MET"] * numpy.cosh(self.eta)

    @property
    def theta(self):
        return 2 * numpy.arctan(numpy.exp(-self.eta))

    @property
    def phi(self):
        return self["Phi"]

    @property
    def eta(self):
        return self["Eta"]


_set_repr_name("MissingET")

behavior.update(awkward._util.copy_behaviors("LorentzVector", "Vertex", behavior))


@awkward.mixin_class(behavior)
class Vertex(vector.LorentzVector):
    """Generic vertex collection that has Lorentz vector properties"""

    @property
    def t(self):
        return self["T"]

    @property
    def x(self):
        return self["X"]

    @property
    def y(self):
        return self["Y"]

    @property
    def z(self):
        return self["Z"]


_set_repr_name("Vertex")

VertexArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
VertexArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
VertexArray.ProjectionClass4D = VertexArray  # noqa: F821
VertexArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(
    awkward._util.copy_behaviors("PtEtaPhiMLorentzVector", "Particle", behavior)
)


@awkward.mixin_class(behavior)
class Particle(vector.PtEtaPhiMLorentzVector):
    """Generic particle collection that has Lorentz vector properties

    The following branches are not used:

     - E: particle energy
     - Px: particle momentum vector (x component)
     - Py: particle momentum vector (y component)
     - Pz: particle momentum vector (z component)

     - P: particle momentum
     - Rapidity: particle rapidity

     - T: particle vertex position (t component)
     - X: particle vertex position (x component)
     - Y: particle vertex position (y component)
     - Z: particle vertex position (z component)
    """

    @property
    def pt(self):
        return self["PT"]

    @property
    def eta(self):
        return self["Eta"]

    @property
    def phi(self):
        return self["Phi"]

    @property
    def mass(self):
        return self["Mass"]


_set_repr_name("Particle")

ParticleArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ParticleArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ParticleArray.ProjectionClass4D = ParticleArray  # noqa: F821
ParticleArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("Particle", "MasslessParticle", behavior))


@awkward.mixin_class(behavior)
class MasslessParticle(Particle, base.NanoCollection):
    @property
    def mass(self):
        return awkward.zeros_like(self.pt)


_set_repr_name("MasslessParticle")

MasslessParticleArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
MasslessParticleArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
MasslessParticleArray.ProjectionClass4D = MasslessParticleArray  # noqa: F821
MasslessParticleArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("MasslessParticle", "Photon", behavior))


@awkward.mixin_class(behavior)
class Photon(MasslessParticle, base.NanoCollection): ...


_set_repr_name("Photon")

PhotonArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
PhotonArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
PhotonArray.ProjectionClass4D = PhotonArray  # noqa: F821
PhotonArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("MasslessParticle", "Electron", behavior))


@awkward.mixin_class(behavior)
class Electron(MasslessParticle, base.NanoCollection): ...


_set_repr_name("Electron")

ElectronArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ElectronArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ElectronArray.ProjectionClass4D = ElectronArray  # noqa: F821
ElectronArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("MasslessParticle", "Muon", behavior))


@awkward.mixin_class(behavior)
class Muon(MasslessParticle, base.NanoCollection): ...


_set_repr_name("Muon")

MuonArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
MuonArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
MuonArray.ProjectionClass4D = MuonArray  # noqa: F821
MuonArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("Particle", "Jet", behavior))


@awkward.mixin_class(behavior)
class Jet(Particle, base.NanoCollection): ...


_set_repr_name("Jet")

JetArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
JetArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
JetArray.ProjectionClass4D = JetArray  # noqa: F821
JetArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("Particle", "Track", behavior))


@awkward.mixin_class(behavior)
class Track(Particle, base.NanoCollection): ...


_set_repr_name("Track")

TrackArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
TrackArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
TrackArray.ProjectionClass4D = TrackArray  # noqa: F821
TrackArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821

behavior.update(awkward._util.copy_behaviors("MasslessParticle", "Tower", behavior))


@awkward.mixin_class(behavior)
class Tower(MasslessParticle, base.NanoCollection):
    @property
    def pt(self):
        return self["ET"]


_set_repr_name("Tower")

TowerArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
TowerArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
TowerArray.ProjectionClass4D = TowerArray  # noqa: F821
TowerArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


__all__ = [
    "DelphesEvents",
    "Event",
    "LHEFEvent",
    "HepMCEvent",
    "LHCOEvent",
    "Weight",
    "WeightLHEF",
    "Rho",
    "ScalarHT",
    "MissingET",
    "Vertex",
    "Particle",
    "Photon",
    "Electron",
    "Muon",
    "Jet",
    "Track",
    "Tower",
]
