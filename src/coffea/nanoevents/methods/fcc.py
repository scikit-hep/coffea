import awkward
import numpy
from dask_awkward.lib.core import dask_property

from coffea.nanoevents.methods import base, edm4hep, vector

behavior = {}
behavior.update(base.behavior)


class _FCCEvents(behavior["NanoEvents"]):
    """FCCEvents"""

    def __repr__(self):
        return "FCC Events"


behavior["NanoEvents"] = _FCCEvents


def _set_repr_name(classname):
    def namefcn(self):
        return classname

    behavior[classname].__repr__ = namefcn


@awkward.mixin_class(behavior)
class MomentumCandidate(vector.LorentzVector):
    """A Lorentz vector with charge

    This mixin class requires the parent class to provide the items `px`, `py`, `pz`, `E`, and `charge`.
    """

    @awkward.mixin_class_method(numpy.add, {"MomentumCandidate"})
    def add(self, other):
        """Add two candidates together elementwise using `px`, `py`, `pz`, `E`, and `charge` components"""
        return awkward.zip(
            {
                "px": self.px + other.px,
                "py": self.py + other.py,
                "pz": self.pz + other.pz,
                "E": self.E + other.E,
                "charge": self.charge + other.charge,
            },
            with_name="MomentumCandidate",
            behavior=self.behavior,
        )

    def sum(self, axis=-1):
        """Sum an array of vectors elementwise using `px`, `py`, `pz`, `E`, and `charge` components"""
        return awkward.zip(
            {
                "px": awkward.sum(self.px, axis=axis),
                "py": awkward.sum(self.py, axis=axis),
                "pz": awkward.sum(self.pz, axis=axis),
                "E": awkward.sum(self.E, axis=axis),
                "charge": awkward.sum(self.charge, axis=axis),
            },
            with_name="MomentumCandidate",
            behavior=self.behavior,
        )

    @property
    def absolute_mass(self):
        return numpy.sqrt(numpy.abs(self.mass2))


behavior.update(
    awkward._util.copy_behaviors(vector.LorentzVector, MomentumCandidate, behavior)
)

MomentumCandidateArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
MomentumCandidateArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
MomentumCandidateArray.ProjectionClass4D = vector.LorentzVectorArray  # noqa: F821
MomentumCandidateArray.MomentumClass = MomentumCandidateArray  # noqa: F821


@awkward.mixin_class(behavior)
class MCParticle(MomentumCandidate, base.NanoCollection):
    """Generated Monte Carlo particles"""

    # Daughters
    @dask_property
    def get_daughters_index(self):
        """
        Obtain the global indices of the daughters of each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx1 collection
        - The Particleidx1.index contains info about the daughters
        """
        return self.daughters.Particleidx1_rangesG

    @get_daughters_index.dask
    def get_daughters_index(self, dask_array):
        """
        Obtain the global indices of the daughters of each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx1 collection
        - The Particleidx1.index contains info about the daughters
        """
        return dask_array.daughters.Particleidx1_rangesG

    @dask_property
    def get_daughters(self):
        """
        Obtain the actual MCParticle daughters to each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx1 collection
        - The Particleidx1.index contains info about the daughters
        """
        return self._events().Particle._apply_global_index(self.get_daughters_index)

    @get_daughters.dask
    def get_daughters(self, dask_array):
        """
        Obtain the actual MCParticle daughters to each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx1 collection
        - The Particleidx1.index contains info about the daughters
        """
        return dask_array._events().Particle._apply_global_index(
            dask_array.get_daughters_index
        )

    # Parents
    @dask_property
    def get_parents_index(self):
        """
        Obtain the global indices of the parents of each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx0 collection
        - The Particleidx0.index contains info about the parents
        """
        return self.parents.Particleidx0_rangesG

    @get_parents_index.dask
    def get_parents_index(self, dask_array):
        """
        Obtain the indices of the parents of each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx0 collection
        - The Particleidx0.index contains info about the parents
        """
        return dask_array.parents.Particleidx0_rangesG

    @dask_property
    def get_parents(self):
        """
        Obtain the actual MCParticle parents to each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx0 collection
        - The Particleidx0.index contains info about the parents
        """
        return self._events().Particle._apply_global_index(self.get_parents_index)

    @get_parents.dask
    def get_parents(self, dask_array):
        """
        Obtain the actual MCParticle parents to each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx0 collection
        - The Particleidx0.index contains info about the parents
        """
        return dask_array._events().Particle._apply_global_index(
            dask_array.get_parents_index
        )


_set_repr_name("MCParticle")
behavior.update(awkward._util.copy_behaviors(MomentumCandidate, MCParticle, behavior))

MCParticleArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
MCParticleArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
MCParticleArray.ProjectionClass4D = MCParticleArray  # noqa: F821
MCParticleArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


@awkward.mixin_class(behavior)
class ReconstructedParticle(MomentumCandidate, base.NanoCollection):
    """Reconstructed particle"""

    def match_collection(self, idx):
        """Returns matched particles; pass on an ObjectID array."""
        return self[idx.index]

    @dask_property
    def match_muons(self):
        """Returns matched muons; drops none values"""
        m = self._events().ReconstructedParticles._apply_global_index(
            self.Muonidx0_indexGlobal
        )
        return awkward.drop_none(m, behavior=self.behavior)

    @match_muons.dask
    def match_muons(self, dask_array):
        """Returns matched muons; drops none values"""
        m = dask_array._events().ReconstructedParticles._apply_global_index(
            dask_array.Muonidx0_indexGlobal
        )
        return awkward.drop_none(m, behavior=self.behavior)

    @dask_property
    def match_electrons(self):
        """Returns matched electrons; drops none values"""
        e = self._events().ReconstructedParticles._apply_global_index(
            self.Electronidx0_indexGlobal
        )
        return awkward.drop_none(e, behavior=self.behavior)

    @match_electrons.dask
    def match_electrons(self, dask_array):
        """Returns matched electrons; drops none values"""
        e = dask_array._events().ReconstructedParticles._apply_global_index(
            dask_array.Electronidx0_indexGlobal
        )
        return awkward.drop_none(e, behavior=self.behavior)

    @dask_property
    def match_gen(self):
        """Returns the Gen level (MC) particle corresponding to the ReconstructedParticle"""
        prepared = self._events().Particle[self._events().MCRecoAssociations.mc.index]
        return prepared._apply_global_index(self.MCRecoAssociationsidx0_indexGlobal)

    @match_gen.dask
    def match_gen(self, dask_array):
        """Returns the Gen level (MC) particle corresponding to the ReconstructedParticle"""
        prepared = dask_array._events().Particle[
            dask_array._events().MCRecoAssociations.mc.index
        ]
        return prepared._apply_global_index(
            dask_array.MCRecoAssociationsidx0_indexGlobal
        )


_set_repr_name("ReconstructedParticle")
behavior.update(
    awkward._util.copy_behaviors(MomentumCandidate, ReconstructedParticle, behavior)
)

ReconstructedParticleArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ReconstructedParticleArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ReconstructedParticleArray.ProjectionClass4D = ReconstructedParticleArray  # noqa: F821
ReconstructedParticleArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


@awkward.mixin_class(behavior)
class MCRecoParticleAssociation(base.NanoCollection):
    """MCRecoParticleAssociation objects."""

    @property
    def reco_mc_index(self):
        """
        Returns an array of indices mapping to generator particles for each reconstructed particle
        """
        arr_reco = self.reco.index[:, :, numpy.newaxis]
        arr_mc = self.mc.index[:, :, numpy.newaxis]

        joined_indices = awkward.concatenate((arr_reco, arr_mc), axis=2)

        return joined_indices

    @dask_property
    def reco_mc(self):
        """
        Returns an array of records mapping to generator particle record for each reconstructed particle record
        - Needs 'ReconstructedParticles' and 'Particle' collections
        """
        reco_index = self.reco.index
        mc_index = self.mc.index
        r = self._events().ReconstructedParticles[reco_index][:, :, numpy.newaxis]
        m = self._events().Particle[mc_index][:, :, numpy.newaxis]

        return awkward.concatenate((r, m), axis=2)

    @reco_mc.dask
    def reco_mc(self, dask_array):
        """
        Returns an array of records mapping to generator particle record for each reconstructed particle record
        - Needs 'ReconstructedParticles' and 'Particle' collections
        """
        reco_index = dask_array.reco.index
        mc_index = dask_array.mc.index
        r = dask_array._events().ReconstructedParticles[reco_index][:, :, numpy.newaxis]
        m = dask_array._events().Particle[mc_index][:, :, numpy.newaxis]

        return awkward.concatenate((r, m), axis=2)


_set_repr_name("MCRecoParticleAssociation")

MCRecoParticleAssociationArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
MCRecoParticleAssociationArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
MCRecoParticleAssociationArray.ProjectionClass4D = (  # noqa: F821
    MCRecoParticleAssociationArray  # noqa: F821  # noqa: F821
)
MCRecoParticleAssociationArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


@awkward.mixin_class(behavior)
class ParticleID(base.NanoCollection):
    """ParticleID collection"""


_set_repr_name("ParticleID")

ParticleIDArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ParticleIDArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ParticleIDArray.ProjectionClass4D = ParticleIDArray  # noqa: F821
ParticleIDArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


@awkward.mixin_class(behavior)
class ObjectID(base.NanoCollection):
    """
    Generic Object ID storage, pointing to another collection
    - All the idx collections are assigned this mixin

    Note: The Hash tagged '#' branches have the <podio::ObjectID> type
    """


_set_repr_name("ObjectID")

ObjectIDArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ObjectIDArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ObjectIDArray.ProjectionClass4D = ObjectIDArray  # noqa: F821
ObjectIDArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


@awkward.mixin_class(behavior)
class Cluster(base.NanoCollection):
    """
    Clusters

    Note: I could not find much info on this, to build its methods
    """


_set_repr_name("Cluster")

ClusterArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ClusterArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ClusterArray.ProjectionClass4D = ClusterArray  # noqa: F821
ClusterArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


@awkward.mixin_class(behavior)
class Track(base.NanoCollection):
    """
    Tracks

    Note: I could not find much info on this, to build its methods
    """


_set_repr_name("Track")

TrackArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
TrackArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
TrackArray.ProjectionClass4D = TrackArray  # noqa: F821
TrackArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


###########################################
# Overloads for FCC_edm4hep1
###########################################

behavior_edm4hep1 = {}
behavior_edm4hep1.update(base.behavior)
behavior_edm4hep1.update(edm4hep.behavior)


def _set_repr_name_edm4hep1(classname):
    def namefcn(self):
        return classname

    behavior_edm4hep1[classname].__repr__ = namefcn


@awkward.mixin_class(behavior_edm4hep1)
class MCParticle(edm4hep.MCParticle):  # noqa: F811
    """EDM4HEP Datatype: MCParticle; Modified for FCC"""

    # Get MC Daughters
    @dask_property
    def get_daughters(self):
        return self.Map_Relation("daughters", "Particle")

    @get_daughters.dask
    def get_daughters(self, dask_array):
        return dask_array.Map_Relation("daughters", "Particle")

    # Get MC Parents
    @dask_property
    def get_parents(self):
        return self.Map_Relation("parents", "Particle")

    @get_parents.dask
    def get_parents(self, dask_array):
        return dask_array.Map_Relation("parents", "Particle")


_set_repr_name_edm4hep1("MCParticle")
behavior_edm4hep1.update(
    awkward._util.copy_behaviors(MomentumCandidate, MCParticle, behavior)
)
MCParticleArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
MCParticleArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
MCParticleArray.ProjectionClass4D = MCParticleArray  # noqa: F821
MCParticleArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


@awkward.mixin_class(behavior_edm4hep1)
class ReconstructedParticle(edm4hep.ReconstructedParticle):  # noqa: F811
    """EDM4HEP Datatype: Reconstructed particle; Modified for FCC"""

    # Get MC counterpart
    @dask_property
    def match_gen(self):
        return self.Map_Link("to", "Particle")

    @match_gen.dask
    def match_gen(self, dask_array):
        return dask_array.Map_Link("to", "Particle")

    # Get cluster EFlowPhoton
    @dask_property
    def get_cluster_photons(self):
        return self.Map_Relation("clusters", "EFlowPhoton")

    @get_cluster_photons.dask
    def get_cluster_photons(self, dask_array):
        return dask_array.Map_Relation("clusters", "EFlowPhoton")

    # Get ReconstructedParticle
    @dask_property
    def get_reconstructedparticles(self):
        return self.Map_Relation("particles", "ReconstructedParticles")

    @get_reconstructedparticles.dask
    def get_reconstructedparticles(self, dask_array):
        return dask_array.Map_Relation("particles", "ReconstructedParticles")

    # Get Tracks
    @dask_property
    def get_tracks(self):
        return self.Map_Relation("tracks", "EFlowTrack")

    @get_tracks.dask
    def get_tracks(self, dask_array):
        return dask_array.Map_Relation("tracks", "EFlowTrack")


_set_repr_name_edm4hep1("ReconstructedParticle")
behavior_edm4hep1.update(
    awkward._util.copy_behaviors(
        MomentumCandidate, ReconstructedParticle, behavior_edm4hep1
    )
)
ReconstructedParticleArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ReconstructedParticleArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ReconstructedParticleArray.ProjectionClass4D = ReconstructedParticleArray  # noqa: F821
ReconstructedParticleArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


@awkward.mixin_class(behavior_edm4hep1)
class ParticleID(edm4hep.ParticleID):  # noqa: F811
    """EDM4HEP Datatype: ParticleID; Modified for FCC"""

    # Get ReconstructedParticle
    @dask_property
    def get_reconstructedparticles(self):
        return self.Map_Relation("particle", "ReconstructedParticles")

    @get_reconstructedparticles.dask
    def get_reconstructedparticles(self, dask_array):
        return dask_array.Map_Relation("particle", "ReconstructedParticles")


_set_repr_name_edm4hep1("ParticleID")


@awkward.mixin_class(behavior_edm4hep1)
class Cluster(edm4hep.Cluster):  # noqa: F811
    """EDM4HEP Datatype: Cluster; Modified for FCC"""

    # Get cluster EFlowPhoton
    @dask_property
    def get_cluster_photons(self):
        return self.Map_Relation("clusters", "EFlowPhoton")

    @get_cluster_photons.dask
    def get_cluster_photons(self, dask_array):
        return dask_array.Map_Relation("clusters", "EFlowPhoton")

    # Get calorimeter hits
    @dask_property
    def get_hits(self):
        return self.Map_Relation("hits", "CalorimeterHits")

    @get_hits.dask
    def get_hits(self, dask_array):
        return dask_array.Map_Relation("hits", "CalorimeterHits")


_set_repr_name_edm4hep1("Cluster")


@awkward.mixin_class(behavior_edm4hep1)
class Track(edm4hep.Track):  # noqa: F811
    """EDM4HEP Datatype: Track; Modified for FCC"""

    # Get Tracks
    @dask_property
    def get_tracks(self):
        return self.Map_Relation("tracks", "EFlowTrack")

    @get_tracks.dask
    def get_tracks(self, dask_array):
        return dask_array.Map_Relation("tracks", "EFlowTrack")

    # Get tracker hits
    @dask_property
    def get_trackerhits(self):
        return self.Map_Relation("trackerHits", "TrackerHits")

    @get_trackerhits.dask
    def get_trackerhits(self, dask_array):
        return dask_array.Map_Relation("trackerHits", "TrackerHits")


_set_repr_name_edm4hep1("Track")
