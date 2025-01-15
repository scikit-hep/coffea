//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Sat Jul 31 09:34:42 2021 by ROOT version 6.20/02
// from TTree beamana/beam analysis tree
// found on file: pduneana.root
//////////////////////////////////////////////////////////

#ifndef beamana_h
#define beamana_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>

// Header file for the classes stored in the TTree if any.
#include "vector"
#include "vector"
#include "vector"
#include "vector"
#include "vector"
#include "string"

class beamana {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Int_t           run;
   Int_t           subrun;
   Int_t           event;
   Int_t           MC;
   Int_t           reco_beam_type;
   Double_t        reco_beam_startX;
   Double_t        reco_beam_startY;
   Double_t        reco_beam_startZ;
   Double_t        reco_beam_endX;
   Double_t        reco_beam_endY;
   Double_t        reco_beam_endZ;
   Double_t        reco_beam_len;
   Double_t        reco_beam_alt_len;
   Double_t        reco_beam_calo_startX;
   Double_t        reco_beam_calo_startY;
   Double_t        reco_beam_calo_startZ;
   Double_t        reco_beam_calo_endX;
   Double_t        reco_beam_calo_endY;
   Double_t        reco_beam_calo_endZ;
   vector<double>  *reco_beam_calo_startDirX;
   vector<double>  *reco_beam_calo_startDirY;
   vector<double>  *reco_beam_calo_startDirZ;
   vector<double>  *reco_beam_calo_endDirX;
   vector<double>  *reco_beam_calo_endDirY;
   vector<double>  *reco_beam_calo_endDirZ;
   Double_t        reco_beam_trackDirX;
   Double_t        reco_beam_trackDirY;
   Double_t        reco_beam_trackDirZ;
   Double_t        reco_beam_trackEndDirX;
   Double_t        reco_beam_trackEndDirY;
   Double_t        reco_beam_trackEndDirZ;
   Int_t           reco_beam_vertex_nHits;
   Double_t        reco_beam_vertex_michel_score;
   Int_t           reco_beam_trackID;
   Int_t           n_beam_slices;
   Int_t           n_beam_particles;
   vector<int>     *beam_track_IDs;
   vector<double>  *beam_particle_scores;
   vector<double>  *reco_beam_dQdX_SCE;
   vector<double>  *reco_beam_EField_SCE;
   vector<double>  *reco_beam_calo_X;
   vector<double>  *reco_beam_calo_Y;
   vector<double>  *reco_beam_calo_Z;
   vector<double>  *reco_beam_dQ;
   vector<double>  *reco_beam_dEdX_SCE;
   vector<double>  *reco_beam_calibrated_dEdX_SCE;
   vector<double>  *reco_beam_calibrated_dQdX_SCE;
   vector<double>  *reco_beam_resRange_SCE;
   vector<double>  *reco_beam_TrkPitch_SCE;
   vector<double>  *reco_beam_dQdX_NoSCE;
   vector<double>  *reco_beam_dQ_NoSCE;
   vector<double>  *reco_beam_dEdX_NoSCE;
   vector<double>  *reco_beam_calibrated_dEdX_NoSCE;
   vector<double>  *reco_beam_resRange_NoSCE;
   vector<double>  *reco_beam_TrkPitch_NoSCE;
   vector<double>  *reco_beam_calo_wire;
   vector<double>  *reco_beam_calo_wire_z;
   vector<double>  *reco_beam_calo_wire_NoSCE;
   vector<double>  *reco_beam_calo_wire_z_NoSCE;
   vector<double>  *reco_beam_calo_tick;
   vector<int>     *reco_beam_calo_TPC;
   vector<int>     *reco_beam_calo_TPC_NoSCE;
   Bool_t          reco_beam_flipped;
   Bool_t          reco_beam_passes_beam_cuts;
   Int_t           reco_beam_PFP_ID;
   Int_t           reco_beam_PFP_nHits;
   Double_t        reco_beam_PFP_trackScore;
   Double_t        reco_beam_PFP_emScore;
   Double_t        reco_beam_PFP_michelScore;
   Double_t        reco_beam_PFP_trackScore_collection;
   Double_t        reco_beam_PFP_emScore_collection;
   Double_t        reco_beam_PFP_michelScore_collection;
   Int_t           reco_beam_allTrack_ID;
   Bool_t          reco_beam_allTrack_beam_cuts;
   Bool_t          reco_beam_allTrack_flipped;
   Double_t        reco_beam_allTrack_len;
   Double_t        reco_beam_allTrack_startX;
   Double_t        reco_beam_allTrack_startY;
   Double_t        reco_beam_allTrack_startZ;
   Double_t        reco_beam_allTrack_endX;
   Double_t        reco_beam_allTrack_endY;
   Double_t        reco_beam_allTrack_endZ;
   Double_t        reco_beam_allTrack_trackDirX;
   Double_t        reco_beam_allTrack_trackDirY;
   Double_t        reco_beam_allTrack_trackDirZ;
   Double_t        reco_beam_allTrack_trackEndDirX;
   Double_t        reco_beam_allTrack_trackEndDirY;
   Double_t        reco_beam_allTrack_trackEndDirZ;
   vector<double>  *reco_beam_allTrack_resRange;
   vector<double>  *reco_beam_allTrack_calibrated_dEdX;
   Double_t        reco_beam_allTrack_Chi2_proton;
   Int_t           reco_beam_allTrack_Chi2_ndof;
   vector<double>  *reco_track_startX;
   vector<double>  *reco_track_startY;
   vector<double>  *reco_track_startZ;
   vector<double>  *reco_track_endX;
   vector<double>  *reco_track_endY;
   vector<double>  *reco_track_endZ;
   vector<double>  *reco_track_michel_score;
   vector<int>     *reco_track_ID;
   vector<int>     *reco_track_nHits;
   vector<int>     *reco_daughter_PFP_true_byHits_PDG;
   vector<int>     *reco_daughter_PFP_true_byHits_ID;
   vector<int>     *reco_daughter_PFP_true_byHits_origin;
   vector<int>     *reco_daughter_PFP_true_byHits_parID;
   vector<int>     *reco_daughter_PFP_true_byHits_parPDG;
   vector<string>  *reco_daughter_PFP_true_byHits_process;
   vector<unsigned long> *reco_daughter_PFP_true_byHits_sharedHits;
   vector<unsigned long> *reco_daughter_PFP_true_byHits_emHits;
   vector<double>  *reco_daughter_PFP_true_byHits_len;
   vector<double>  *reco_daughter_PFP_true_byHits_startX;
   vector<double>  *reco_daughter_PFP_true_byHits_startY;
   vector<double>  *reco_daughter_PFP_true_byHits_startZ;
   vector<double>  *reco_daughter_PFP_true_byHits_endX;
   vector<double>  *reco_daughter_PFP_true_byHits_endY;
   vector<double>  *reco_daughter_PFP_true_byHits_endZ;
   vector<double>  *reco_daughter_PFP_true_byHits_startPx;
   vector<double>  *reco_daughter_PFP_true_byHits_startPy;
   vector<double>  *reco_daughter_PFP_true_byHits_startPz;
   vector<double>  *reco_daughter_PFP_true_byHits_startP;
   vector<double>  *reco_daughter_PFP_true_byHits_startE;
   vector<string>  *reco_daughter_PFP_true_byHits_endProcess;
   vector<double>  *reco_daughter_PFP_true_byHits_purity;
   vector<double>  *reco_daughter_PFP_true_byHits_completeness;
   vector<int>     *reco_daughter_PFP_true_byE_PDG;
   vector<double>  *reco_daughter_PFP_true_byE_len;
   vector<double>  *reco_daughter_PFP_true_byE_completeness;
   vector<double>  *reco_daughter_PFP_true_byE_purity;
   vector<int>     *reco_daughter_allTrack_ID;
   vector<vector<double> > *reco_daughter_allTrack_dQdX_SCE;
   vector<vector<double> > *reco_daughter_allTrack_calibrated_dQdX_SCE;
   vector<vector<double> > *reco_daughter_allTrack_EField_SCE;
   vector<vector<double> > *reco_daughter_allTrack_dEdX_SCE;
   vector<vector<double> > *reco_daughter_allTrack_resRange_SCE;
   vector<vector<double> > *reco_daughter_allTrack_calibrated_dEdX_SCE;
   vector<double>  *reco_daughter_allTrack_Chi2_proton;
   vector<double>  *reco_daughter_allTrack_Chi2_pion;
   vector<double>  *reco_daughter_allTrack_Chi2_muon;
   vector<int>     *reco_daughter_allTrack_Chi2_ndof;
   vector<int>     *reco_daughter_allTrack_Chi2_ndof_pion;
   vector<int>     *reco_daughter_allTrack_Chi2_ndof_muon;
   vector<double>  *reco_daughter_allTrack_Chi2_proton_plane0;
   vector<double>  *reco_daughter_allTrack_Chi2_proton_plane1;
   vector<int>     *reco_daughter_allTrack_Chi2_ndof_plane0;
   vector<int>     *reco_daughter_allTrack_Chi2_ndof_plane1;
   vector<vector<double> > *reco_daughter_allTrack_calibrated_dEdX_SCE_plane0;
   vector<vector<double> > *reco_daughter_allTrack_calibrated_dEdX_SCE_plane1;
   vector<vector<double> > *reco_daughter_allTrack_resRange_plane0;
   vector<vector<double> > *reco_daughter_allTrack_resRange_plane1;
   vector<double>  *reco_daughter_allTrack_Theta;
   vector<double>  *reco_daughter_allTrack_Phi;
   vector<double>  *reco_daughter_allTrack_len;
   vector<double>  *reco_daughter_allTrack_alt_len;
   vector<double>  *reco_daughter_allTrack_startX;
   vector<double>  *reco_daughter_allTrack_startY;
   vector<double>  *reco_daughter_allTrack_startZ;
   vector<double>  *reco_daughter_allTrack_endX;
   vector<double>  *reco_daughter_allTrack_endY;
   vector<double>  *reco_daughter_allTrack_endZ;
   vector<double>  *reco_daughter_allTrack_dR;
   vector<vector<double> > *reco_daughter_allTrack_calo_X;
   vector<vector<double> > *reco_daughter_allTrack_calo_Y;
   vector<vector<double> > *reco_daughter_allTrack_calo_Z;
   vector<double>  *reco_daughter_allTrack_to_vertex;
   vector<double>  *reco_daughter_allTrack_vertex_michel_score;
   vector<int>     *reco_daughter_allTrack_vertex_nHits;
   vector<int>     *reco_daughter_allShower_ID;
   vector<double>  *reco_daughter_allShower_len;
   vector<double>  *reco_daughter_allShower_startX;
   vector<double>  *reco_daughter_allShower_startY;
   vector<double>  *reco_daughter_allShower_startZ;
   vector<double>  *reco_daughter_allShower_dirX;
   vector<double>  *reco_daughter_allShower_dirY;
   vector<double>  *reco_daughter_allShower_dirZ;
   vector<double>  *reco_daughter_allShower_energy;
   vector<int>     *reco_daughter_PFP_ID;
   vector<int>     *reco_daughter_PFP_nHits;
   vector<int>     *reco_daughter_PFP_nHits_collection;
   vector<double>  *reco_daughter_PFP_trackScore;
   vector<double>  *reco_daughter_PFP_emScore;
   vector<double>  *reco_daughter_PFP_michelScore;
   vector<double>  *reco_daughter_PFP_trackScore_collection;
   vector<double>  *reco_daughter_PFP_emScore_collection;
   vector<double>  *reco_daughter_PFP_michelScore_collection;
   Int_t           true_beam_PDG;
   Double_t        true_beam_mass;
   Int_t           true_beam_ID;
   string          *true_beam_endProcess;
   Double_t        true_beam_endX;
   Double_t        true_beam_endY;
   Double_t        true_beam_endZ;
   Double_t        true_beam_endX_SCE;
   Double_t        true_beam_endY_SCE;
   Double_t        true_beam_endZ_SCE;
   Double_t        true_beam_startX;
   Double_t        true_beam_startY;
   Double_t        true_beam_startZ;
   Double_t        true_beam_startPx;
   Double_t        true_beam_startPy;
   Double_t        true_beam_startPz;
   Double_t        true_beam_startP;
   Double_t        true_beam_endPx;
   Double_t        true_beam_endPy;
   Double_t        true_beam_endPz;
   Double_t        true_beam_endP;
   Double_t        true_beam_endP2;
   Double_t        true_beam_last_len;
   Double_t        true_beam_startDirX;
   Double_t        true_beam_startDirY;
   Double_t        true_beam_startDirZ;
   Int_t           true_beam_nElasticScatters;
   vector<double>  *true_beam_elastic_costheta;
   vector<double>  *true_beam_elastic_X;
   vector<double>  *true_beam_elastic_Y;
   vector<double>  *true_beam_elastic_Z;
   vector<double>  *true_beam_elastic_deltaE;
   vector<double>  *true_beam_elastic_IDE_edep;
   Double_t        true_beam_IDE_totalDep;
   Int_t           true_beam_nHits;
   vector<vector<int> > *true_beam_reco_byHits_PFP_ID;
   vector<vector<int> > *true_beam_reco_byHits_PFP_nHits;
   vector<vector<int> > *true_beam_reco_byHits_allTrack_ID;
   Int_t           true_daughter_nPi0;
   Int_t           true_daughter_nPiPlus;
   Int_t           true_daughter_nProton;
   Int_t           true_daughter_nNeutron;
   Int_t           true_daughter_nPiMinus;
   Int_t           true_daughter_nNucleus;
   Int_t           reco_beam_vertex_slice;
   vector<int>     *true_beam_daughter_PDG;
   vector<int>     *true_beam_daughter_ID;
   vector<double>  *true_beam_daughter_len;
   vector<double>  *true_beam_daughter_startX;
   vector<double>  *true_beam_daughter_startY;
   vector<double>  *true_beam_daughter_startZ;
   vector<double>  *true_beam_daughter_startPx;
   vector<double>  *true_beam_daughter_startPy;
   vector<double>  *true_beam_daughter_startPz;
   vector<double>  *true_beam_daughter_startP;
   vector<double>  *true_beam_daughter_endX;
   vector<double>  *true_beam_daughter_endY;
   vector<double>  *true_beam_daughter_endZ;
   vector<string>  *true_beam_daughter_Process;
   vector<string>  *true_beam_daughter_endProcess;
   vector<int>     *true_beam_daughter_nHits;
   vector<vector<int> > *true_beam_daughter_reco_byHits_PFP_ID;
   vector<vector<int> > *true_beam_daughter_reco_byHits_PFP_nHits;
   vector<vector<double> > *true_beam_daughter_reco_byHits_PFP_trackScore;
   vector<vector<int> > *true_beam_daughter_reco_byHits_allTrack_ID;
   vector<vector<double> > *true_beam_daughter_reco_byHits_allTrack_startX;
   vector<vector<double> > *true_beam_daughter_reco_byHits_allTrack_startY;
   vector<vector<double> > *true_beam_daughter_reco_byHits_allTrack_startZ;
   vector<vector<double> > *true_beam_daughter_reco_byHits_allTrack_endX;
   vector<vector<double> > *true_beam_daughter_reco_byHits_allTrack_endY;
   vector<vector<double> > *true_beam_daughter_reco_byHits_allTrack_endZ;
   vector<vector<double> > *true_beam_daughter_reco_byHits_allTrack_len;
   vector<vector<int> > *true_beam_daughter_reco_byHits_allShower_ID;
   vector<vector<double> > *true_beam_daughter_reco_byHits_allShower_startX;
   vector<vector<double> > *true_beam_daughter_reco_byHits_allShower_startY;
   vector<vector<double> > *true_beam_daughter_reco_byHits_allShower_startZ;
   vector<vector<double> > *true_beam_daughter_reco_byHits_allShower_len;
   vector<int>     *true_beam_Pi0_decay_ID;
   vector<int>     *true_beam_Pi0_decay_parID;
   vector<int>     *true_beam_Pi0_decay_PDG;
   vector<double>  *true_beam_Pi0_decay_startP;
   vector<double>  *true_beam_Pi0_decay_startPx;
   vector<double>  *true_beam_Pi0_decay_startPy;
   vector<double>  *true_beam_Pi0_decay_startPz;
   vector<double>  *true_beam_Pi0_decay_startX;
   vector<double>  *true_beam_Pi0_decay_startY;
   vector<double>  *true_beam_Pi0_decay_startZ;
   vector<double>  *true_beam_Pi0_decay_len;
   vector<int>     *true_beam_Pi0_decay_nHits;
   vector<vector<int> > *true_beam_Pi0_decay_reco_byHits_PFP_ID;
   vector<vector<int> > *true_beam_Pi0_decay_reco_byHits_PFP_nHits;
   vector<vector<double> > *true_beam_Pi0_decay_reco_byHits_PFP_trackScore;
   vector<vector<int> > *true_beam_Pi0_decay_reco_byHits_allTrack_ID;
   vector<vector<double> > *true_beam_Pi0_decay_reco_byHits_allTrack_startX;
   vector<vector<double> > *true_beam_Pi0_decay_reco_byHits_allTrack_startY;
   vector<vector<double> > *true_beam_Pi0_decay_reco_byHits_allTrack_startZ;
   vector<vector<double> > *true_beam_Pi0_decay_reco_byHits_allTrack_endX;
   vector<vector<double> > *true_beam_Pi0_decay_reco_byHits_allTrack_endY;
   vector<vector<double> > *true_beam_Pi0_decay_reco_byHits_allTrack_endZ;
   vector<vector<double> > *true_beam_Pi0_decay_reco_byHits_allTrack_len;
   vector<vector<int> > *true_beam_Pi0_decay_reco_byHits_allShower_ID;
   vector<vector<double> > *true_beam_Pi0_decay_reco_byHits_allShower_startX;
   vector<vector<double> > *true_beam_Pi0_decay_reco_byHits_allShower_startY;
   vector<vector<double> > *true_beam_Pi0_decay_reco_byHits_allShower_startZ;
   vector<vector<double> > *true_beam_Pi0_decay_reco_byHits_allShower_len;
   vector<int>     *true_beam_grand_daughter_ID;
   vector<int>     *true_beam_grand_daughter_parID;
   vector<int>     *true_beam_grand_daughter_PDG;
   vector<int>     *true_beam_grand_daughter_nHits;
   vector<string>  *true_beam_grand_daughter_Process;
   vector<string>  *true_beam_grand_daughter_endProcess;
   string          *reco_beam_true_byE_endProcess;
   string          *reco_beam_true_byE_process;
   Int_t           reco_beam_true_byE_origin;
   Int_t           reco_beam_true_byE_PDG;
   Int_t           reco_beam_true_byE_ID;
   string          *reco_beam_true_byHits_endProcess;
   string          *reco_beam_true_byHits_process;
   Int_t           reco_beam_true_byHits_origin;
   Int_t           reco_beam_true_byHits_PDG;
   Int_t           reco_beam_true_byHits_ID;
   Bool_t          reco_beam_true_byE_matched;
   Bool_t          reco_beam_true_byHits_matched;
   Double_t        reco_beam_true_byHits_purity;
   vector<string>  *true_beam_processes;
   Double_t        beam_inst_P;
   vector<double>  *beam_inst_TOF;
   vector<int>     *beam_inst_TOF_Chan;
   Double_t        beam_inst_X;
   Double_t        beam_inst_Y;
   Double_t        beam_inst_Z;
   Double_t        beam_inst_dirX;
   Double_t        beam_inst_dirY;
   Double_t        beam_inst_dirZ;
   Int_t           beam_inst_nFibersP1;
   Int_t           beam_inst_nFibersP2;
   Int_t           beam_inst_nFibersP3;
   vector<int>     *beam_inst_PDG_candidates;
   Int_t           beam_inst_nTracks;
   Int_t           beam_inst_nMomenta;
   Bool_t          beam_inst_valid;
   Int_t           beam_inst_trigger;
   Double_t        reco_beam_Chi2_proton;
   Int_t           reco_beam_Chi2_ndof;
   vector<double>  *reco_daughter_allTrack_momByRange_proton;
   vector<double>  *reco_daughter_allTrack_momByRange_muon;
   Double_t        reco_beam_momByRange_proton;
   Double_t        reco_beam_momByRange_muon;
   vector<double>  *reco_daughter_allTrack_momByRange_alt_proton;
   vector<double>  *reco_daughter_allTrack_momByRange_alt_muon;
   Double_t        reco_beam_momByRange_alt_proton;
   Double_t        reco_beam_momByRange_alt_muon;
   Double_t        reco_beam_true_byE_endPx;
   Double_t        reco_beam_true_byE_endPy;
   Double_t        reco_beam_true_byE_endPz;
   Double_t        reco_beam_true_byE_endE;
   Double_t        reco_beam_true_byE_endP;
   Double_t        reco_beam_true_byE_startPx;
   Double_t        reco_beam_true_byE_startPy;
   Double_t        reco_beam_true_byE_startPz;
   Double_t        reco_beam_true_byE_startE;
   Double_t        reco_beam_true_byE_startP;
   Double_t        reco_beam_true_byHits_endPx;
   Double_t        reco_beam_true_byHits_endPy;
   Double_t        reco_beam_true_byHits_endPz;
   Double_t        reco_beam_true_byHits_endE;
   Double_t        reco_beam_true_byHits_endP;
   Double_t        reco_beam_true_byHits_startPx;
   Double_t        reco_beam_true_byHits_startPy;
   Double_t        reco_beam_true_byHits_startPz;
   Double_t        reco_beam_true_byHits_startE;
   Double_t        reco_beam_true_byHits_startP;
   vector<double>  *reco_beam_incidentEnergies;
   Double_t        reco_beam_interactingEnergy;
   vector<double>  *true_beam_incidentEnergies;
   Double_t        true_beam_interactingEnergy;
   vector<int>     *true_beam_slices;
   vector<int>     *true_beam_slices_found;
   vector<double>  *true_beam_slices_deltaE;
   Double_t        em_energy;
   vector<double>  *true_beam_traj_X;
   vector<double>  *true_beam_traj_Y;
   vector<double>  *true_beam_traj_Z;
   vector<double>  *true_beam_traj_Px;
   vector<double>  *true_beam_traj_Py;
   vector<double>  *true_beam_traj_Pz;
   vector<double>  *true_beam_traj_KE;
   vector<double>  *true_beam_traj_X_SCE;
   vector<double>  *true_beam_traj_Y_SCE;
   vector<double>  *true_beam_traj_Z_SCE;
   vector<double>  *g4rw_primary_weights;
   vector<double>  *g4rw_primary_plus_sigma_weight;
   vector<double>  *g4rw_primary_minus_sigma_weight;
   vector<string>  *g4rw_primary_var;
   vector<double>  *g4rw_alt_primary_plus_sigma_weight;
   vector<double>  *g4rw_alt_primary_minus_sigma_weight;
   vector<double>  *g4rw_full_primary_plus_sigma_weight;
   vector<double>  *g4rw_full_primary_minus_sigma_weight;
   vector<vector<double> > *g4rw_full_grid_weights;
   vector<vector<double> > *g4rw_full_grid_piplus_weights;
   vector<vector<double> > *g4rw_full_grid_piplus_weights_fake_data;
   vector<vector<double> > *g4rw_full_grid_piminus_weights;
   vector<vector<double> > *g4rw_full_grid_proton_weights;
   vector<vector<double> > *g4rw_primary_grid_weights;
   vector<double>  *g4rw_primary_grid_pair_weights;
   vector<double>  *reco_beam_spacePts_X;
   vector<double>  *reco_beam_spacePts_Y;
   vector<double>  *reco_beam_spacePts_Z;
   vector<vector<double> > *reco_daughter_spacePts_X;
   vector<vector<double> > *reco_daughter_spacePts_Y;
   vector<vector<double> > *reco_daughter_spacePts_Z;
   vector<vector<double> > *reco_daughter_shower_spacePts_X;
   vector<vector<double> > *reco_daughter_shower_spacePts_Y;
   vector<vector<double> > *reco_daughter_shower_spacePts_Z;

   // List of branches
   TBranch        *b_run;   //!
   TBranch        *b_subrun;   //!
   TBranch        *b_event;   //!
   TBranch        *b_MC;   //!
   TBranch        *b_reco_beam_type;   //!
   TBranch        *b_reco_beam_startX;   //!
   TBranch        *b_reco_beam_startY;   //!
   TBranch        *b_reco_beam_startZ;   //!
   TBranch        *b_reco_beam_endX;   //!
   TBranch        *b_reco_beam_endY;   //!
   TBranch        *b_reco_beam_endZ;   //!
   TBranch        *b_reco_beam_len;   //!
   TBranch        *b_reco_beam_alt_len;   //!
   TBranch        *b_reco_beam_calo_startX;   //!
   TBranch        *b_reco_beam_calo_startY;   //!
   TBranch        *b_reco_beam_calo_startZ;   //!
   TBranch        *b_reco_beam_calo_endX;   //!
   TBranch        *b_reco_beam_calo_endY;   //!
   TBranch        *b_reco_beam_calo_endZ;   //!
   TBranch        *b_reco_beam_calo_startDirX;   //!
   TBranch        *b_reco_beam_calo_startDirY;   //!
   TBranch        *b_reco_beam_calo_startDirZ;   //!
   TBranch        *b_reco_beam_calo_endDirX;   //!
   TBranch        *b_reco_beam_calo_endDirY;   //!
   TBranch        *b_reco_beam_calo_endDirZ;   //!
   TBranch        *b_reco_beam_trackDirX;   //!
   TBranch        *b_reco_beam_trackDirY;   //!
   TBranch        *b_reco_beam_trackDirZ;   //!
   TBranch        *b_reco_beam_trackEndDirX;   //!
   TBranch        *b_reco_beam_trackEndDirY;   //!
   TBranch        *b_reco_beam_trackEndDirZ;   //!
   TBranch        *b_reco_beam_vertex_nHits;   //!
   TBranch        *b_reco_beam_vertex_michel_score;   //!
   TBranch        *b_reco_beam_trackID;   //!
   TBranch        *b_n_beam_slices;   //!
   TBranch        *b_n_beam_particles;   //!
   TBranch        *b_beam_track_IDs;   //!
   TBranch        *b_beam_particle_scores;   //!
   TBranch        *b_reco_beam_dQdX_SCE;   //!
   TBranch        *b_reco_beam_EField_SCE;   //!
   TBranch        *b_reco_beam_calo_X;   //!
   TBranch        *b_reco_beam_calo_Y;   //!
   TBranch        *b_reco_beam_calo_Z;   //!
   TBranch        *b_reco_beam_dQ;   //!
   TBranch        *b_reco_beam_dEdX_SCE;   //!
   TBranch        *b_reco_beam_calibrated_dEdX_SCE;   //!
   TBranch        *b_reco_beam_calibrated_dQdX_SCE;   //!
   TBranch        *b_reco_beam_resRange_SCE;   //!
   TBranch        *b_reco_beam_TrkPitch_SCE;   //!
   TBranch        *b_reco_beam_dQdX_NoSCE;   //!
   TBranch        *b_reco_beam_dQ_NoSCE;   //!
   TBranch        *b_reco_beam_dEdX_NoSCE;   //!
   TBranch        *b_reco_beam_calibrated_dEdX_NoSCE;   //!
   TBranch        *b_reco_beam_resRange_NoSCE;   //!
   TBranch        *b_reco_beam_TrkPitch_NoSCE;   //!
   TBranch        *b_reco_beam_calo_wire;   //!
   TBranch        *b_reco_beam_calo_wire_z;   //!
   TBranch        *b_reco_beam_calo_wire_NoSCE;   //!
   TBranch        *b_reco_beam_calo_wire_z_NoSCE;   //!
   TBranch        *b_reco_beam_calo_tick;   //!
   TBranch        *b_reco_beam_calo_TPC;   //!
   TBranch        *b_reco_beam_calo_TPC_NoSCE;   //!
   TBranch        *b_reco_beam_flipped;   //!
   TBranch        *b_reco_beam_passes_beam_cuts;   //!
   TBranch        *b_reco_beam_PFP_ID;   //!
   TBranch        *b_reco_beam_PFP_nHits;   //!
   TBranch        *b_reco_beam_PFP_trackScore;   //!
   TBranch        *b_reco_beam_PFP_emScore;   //!
   TBranch        *b_reco_beam_PFP_michelScore;   //!
   TBranch        *b_reco_beam_PFP_trackScore_collection;   //!
   TBranch        *b_reco_beam_PFP_emScore_collection;   //!
   TBranch        *b_reco_beam_PFP_michelScore_collection;   //!
   TBranch        *b_reco_beam_allTrack_ID;   //!
   TBranch        *b_reco_beam_allTrack_beam_cuts;   //!
   TBranch        *b_reco_beam_allTrack_flipped;   //!
   TBranch        *b_reco_beam_allTrack_len;   //!
   TBranch        *b_reco_beam_allTrack_startX;   //!
   TBranch        *b_reco_beam_allTrack_startY;   //!
   TBranch        *b_reco_beam_allTrack_startZ;   //!
   TBranch        *b_reco_beam_allTrack_endX;   //!
   TBranch        *b_reco_beam_allTrack_endY;   //!
   TBranch        *b_reco_beam_allTrack_endZ;   //!
   TBranch        *b_reco_beam_allTrack_trackDirX;   //!
   TBranch        *b_reco_beam_allTrack_trackDirY;   //!
   TBranch        *b_reco_beam_allTrack_trackDirZ;   //!
   TBranch        *b_reco_beam_allTrack_trackEndDirX;   //!
   TBranch        *b_reco_beam_allTrack_trackEndDirY;   //!
   TBranch        *b_reco_beam_allTrack_trackEndDirZ;   //!
   TBranch        *b_reco_beam_allTrack_resRange;   //!
   TBranch        *b_reco_beam_allTrack_calibrated_dEdX;   //!
   TBranch        *b_reco_beam_allTrack_Chi2_proton;   //!
   TBranch        *b_reco_beam_allTrack_Chi2_ndof;   //!
   TBranch        *b_reco_track_startX;   //!
   TBranch        *b_reco_track_startY;   //!
   TBranch        *b_reco_track_startZ;   //!
   TBranch        *b_reco_track_endX;   //!
   TBranch        *b_reco_track_endY;   //!
   TBranch        *b_reco_track_endZ;   //!
   TBranch        *b_reco_track_michel_score;   //!
   TBranch        *b_reco_track_ID;   //!
   TBranch        *b_reco_track_nHits;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_PDG;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_ID;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_origin;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_parID;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_parPDG;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_process;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_sharedHits;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_emHits;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_len;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_startX;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_startY;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_startZ;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_endX;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_endY;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_endZ;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_startPx;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_startPy;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_startPz;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_startP;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_startE;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_endProcess;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_purity;   //!
   TBranch        *b_reco_daughter_PFP_true_byHits_completeness;   //!
   TBranch        *b_reco_daughter_PFP_true_byE_PDG;   //!
   TBranch        *b_reco_daughter_PFP_true_byE_len;   //!
   TBranch        *b_reco_daughter_PFP_true_byE_completeness;   //!
   TBranch        *b_reco_daughter_PFP_true_byE_purity;   //!
   TBranch        *b_reco_daughter_allTrack_ID;   //!
   TBranch        *b_reco_daughter_allTrack_dQdX_SCE;   //!
   TBranch        *b_reco_daughter_allTrack_calibrated_dQdX_SCE;   //!
   TBranch        *b_reco_daughter_allTrack_EField_SCE;   //!
   TBranch        *b_reco_daughter_allTrack_dEdX_SCE;   //!
   TBranch        *b_reco_daughter_allTrack_resRange_SCE;   //!
   TBranch        *b_reco_daughter_allTrack_calibrated_dEdX_SCE;   //!
   TBranch        *b_reco_daughter_allTrack_Chi2_proton;   //!
   TBranch        *b_reco_daughter_allTrack_Chi2_pion;   //!
   TBranch        *b_reco_daughter_allTrack_Chi2_muon;   //!
   TBranch        *b_reco_daughter_allTrack_Chi2_ndof;   //!
   TBranch        *b_reco_daughter_allTrack_Chi2_ndof_pion;   //!
   TBranch        *b_reco_daughter_allTrack_Chi2_ndof_muon;   //!
   TBranch        *b_reco_daughter_allTrack_Chi2_proton_plane0;   //!
   TBranch        *b_reco_daughter_allTrack_Chi2_proton_plane1;   //!
   TBranch        *b_reco_daughter_allTrack_Chi2_ndof_plane0;   //!
   TBranch        *b_reco_daughter_allTrack_Chi2_ndof_plane1;   //!
   TBranch        *b_reco_daughter_allTrack_calibrated_dEdX_SCE_plane0;   //!
   TBranch        *b_reco_daughter_allTrack_calibrated_dEdX_SCE_plane1;   //!
   TBranch        *b_reco_daughter_allTrack_resRange_plane0;   //!
   TBranch        *b_reco_daughter_allTrack_resRange_plane1;   //!
   TBranch        *b_reco_daughter_allTrack_Theta;   //!
   TBranch        *b_reco_daughter_allTrack_Phi;   //!
   TBranch        *b_reco_daughter_allTrack_len;   //!
   TBranch        *b_reco_daughter_allTrack_alt_len;   //!
   TBranch        *b_reco_daughter_allTrack_startX;   //!
   TBranch        *b_reco_daughter_allTrack_startY;   //!
   TBranch        *b_reco_daughter_allTrack_startZ;   //!
   TBranch        *b_reco_daughter_allTrack_endX;   //!
   TBranch        *b_reco_daughter_allTrack_endY;   //!
   TBranch        *b_reco_daughter_allTrack_endZ;   //!
   TBranch        *b_reco_daughter_allTrack_dR;   //!
   TBranch        *b_reco_daughter_allTrack_calo_X;   //!
   TBranch        *b_reco_daughter_allTrack_calo_Y;   //!
   TBranch        *b_reco_daughter_allTrack_calo_Z;   //!
   TBranch        *b_reco_daughter_allTrack_to_vertex;   //!
   TBranch        *b_reco_daughter_allTrack_vertex_michel_score;   //!
   TBranch        *b_reco_daughter_allTrack_vertex_nHits;   //!
   TBranch        *b_reco_daughter_allShower_ID;   //!
   TBranch        *b_reco_daughter_allShower_len;   //!
   TBranch        *b_reco_daughter_allShower_startX;   //!
   TBranch        *b_reco_daughter_allShower_startY;   //!
   TBranch        *b_reco_daughter_allShower_startZ;   //!
   TBranch        *b_reco_daughter_allShower_dirX;   //!
   TBranch        *b_reco_daughter_allShower_dirY;   //!
   TBranch        *b_reco_daughter_allShower_dirZ;   //!
   TBranch        *b_reco_daughter_allShower_energy;   //!
   TBranch        *b_reco_daughter_PFP_ID;   //!
   TBranch        *b_reco_daughter_PFP_nHits;   //!
   TBranch        *b_reco_daughter_PFP_nHits_collection;   //!
   TBranch        *b_reco_daughter_PFP_trackScore;   //!
   TBranch        *b_reco_daughter_PFP_emScore;   //!
   TBranch        *b_reco_daughter_PFP_michelScore;   //!
   TBranch        *b_reco_daughter_PFP_trackScore_collection;   //!
   TBranch        *b_reco_daughter_PFP_emScore_collection;   //!
   TBranch        *b_reco_daughter_PFP_michelScore_collection;   //!
   TBranch        *b_true_beam_PDG;   //!
   TBranch        *b_true_beam_mass;   //!
   TBranch        *b_true_beam_ID;   //!
   TBranch        *b_true_beam_endProcess;   //!
   TBranch        *b_true_beam_endX;   //!
   TBranch        *b_true_beam_endY;   //!
   TBranch        *b_true_beam_endZ;   //!
   TBranch        *b_true_beam_endX_SCE;   //!
   TBranch        *b_true_beam_endY_SCE;   //!
   TBranch        *b_true_beam_endZ_SCE;   //!
   TBranch        *b_true_beam_startX;   //!
   TBranch        *b_true_beam_startY;   //!
   TBranch        *b_true_beam_startZ;   //!
   TBranch        *b_true_beam_startPx;   //!
   TBranch        *b_true_beam_startPy;   //!
   TBranch        *b_true_beam_startPz;   //!
   TBranch        *b_true_beam_startP;   //!
   TBranch        *b_true_beam_endPx;   //!
   TBranch        *b_true_beam_endPy;   //!
   TBranch        *b_true_beam_endPz;   //!
   TBranch        *b_true_beam_endP;   //!
   TBranch        *b_true_beam_endP2;   //!
   TBranch        *b_true_beam_last_len;   //!
   TBranch        *b_true_beam_startDirX;   //!
   TBranch        *b_true_beam_startDirY;   //!
   TBranch        *b_true_beam_startDirZ;   //!
   TBranch        *b_true_beam_nElasticScatters;   //!
   TBranch        *b_true_beam_elastic_costheta;   //!
   TBranch        *b_true_beam_elastic_X;   //!
   TBranch        *b_true_beam_elastic_Y;   //!
   TBranch        *b_true_beam_elastic_Z;   //!
   TBranch        *b_true_beam_elastic_deltaE;   //!
   TBranch        *b_true_beam_elastic_IDE_edep;   //!
   TBranch        *b_true_beam_IDE_totalDep;   //!
   TBranch        *b_true_beam_nHits;   //!
   TBranch        *b_true_beam_reco_byHits_PFP_ID;   //!
   TBranch        *b_true_beam_reco_byHits_PFP_nHits;   //!
   TBranch        *b_true_beam_reco_byHits_allTrack_ID;   //!
   TBranch        *b_true_daughter_nPi0;   //!
   TBranch        *b_true_daughter_nPiPlus;   //!
   TBranch        *b_true_daughter_nProton;   //!
   TBranch        *b_true_daughter_nNeutron;   //!
   TBranch        *b_true_daughter_nPiMinus;   //!
   TBranch        *b_true_daughter_nNucleus;   //!
   TBranch        *b_reco_beam_vertex_slice;   //!
   TBranch        *b_true_beam_daughter_PDG;   //!
   TBranch        *b_true_beam_daughter_ID;   //!
   TBranch        *b_true_beam_daughter_len;   //!
   TBranch        *b_true_beam_daughter_startX;   //!
   TBranch        *b_true_beam_daughter_startY;   //!
   TBranch        *b_true_beam_daughter_startZ;   //!
   TBranch        *b_true_beam_daughter_startPx;   //!
   TBranch        *b_true_beam_daughter_startPy;   //!
   TBranch        *b_true_beam_daughter_startPz;   //!
   TBranch        *b_true_beam_daughter_startP;   //!
   TBranch        *b_true_beam_daughter_endX;   //!
   TBranch        *b_true_beam_daughter_endY;   //!
   TBranch        *b_true_beam_daughter_endZ;   //!
   TBranch        *b_true_beam_daughter_Process;   //!
   TBranch        *b_true_beam_daughter_endProcess;   //!
   TBranch        *b_true_beam_daughter_nHits;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_PFP_ID;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_PFP_nHits;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_PFP_trackScore;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_allTrack_ID;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_allTrack_startX;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_allTrack_startY;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_allTrack_startZ;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_allTrack_endX;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_allTrack_endY;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_allTrack_endZ;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_allTrack_len;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_allShower_ID;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_allShower_startX;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_allShower_startY;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_allShower_startZ;   //!
   TBranch        *b_true_beam_daughter_reco_byHits_allShower_len;   //!
   TBranch        *b_true_beam_Pi0_decay_ID;   //!
   TBranch        *b_true_beam_Pi0_decay_parID;   //!
   TBranch        *b_true_beam_Pi0_decay_PDG;   //!
   TBranch        *b_true_beam_Pi0_decay_startP;   //!
   TBranch        *b_true_beam_Pi0_decay_startPx;   //!
   TBranch        *b_true_beam_Pi0_decay_startPy;   //!
   TBranch        *b_true_beam_Pi0_decay_startPz;   //!
   TBranch        *b_true_beam_Pi0_decay_startX;   //!
   TBranch        *b_true_beam_Pi0_decay_startY;   //!
   TBranch        *b_true_beam_Pi0_decay_startZ;   //!
   TBranch        *b_true_beam_Pi0_decay_len;   //!
   TBranch        *b_true_beam_Pi0_decay_nHits;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_PFP_ID;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_PFP_nHits;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_PFP_trackScore;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_allTrack_ID;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_allTrack_startX;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_allTrack_startY;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_allTrack_startZ;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_allTrack_endX;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_allTrack_endY;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_allTrack_endZ;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_allTrack_len;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_allShower_ID;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_allShower_startX;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_allShower_startY;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_allShower_startZ;   //!
   TBranch        *b_true_beam_Pi0_decay_reco_byHits_allShower_len;   //!
   TBranch        *b_true_beam_grand_daughter_ID;   //!
   TBranch        *b_true_beam_grand_daughter_parID;   //!
   TBranch        *b_true_beam_grand_daughter_PDG;   //!
   TBranch        *b_true_beam_grand_daughter_nHits;   //!
   TBranch        *b_true_beam_grand_daughter_Process;   //!
   TBranch        *b_true_beam_grand_daughter_endProcess;   //!
   TBranch        *b_reco_beam_true_byE_endProcess;   //!
   TBranch        *b_reco_beam_true_byE_process;   //!
   TBranch        *b_reco_beam_true_byE_origin;   //!
   TBranch        *b_reco_beam_true_byE_PDG;   //!
   TBranch        *b_reco_beam_true_byE_ID;   //!
   TBranch        *b_reco_beam_true_byHits_endProcess;   //!
   TBranch        *b_reco_beam_true_byHits_process;   //!
   TBranch        *b_reco_beam_true_byHits_origin;   //!
   TBranch        *b_reco_beam_true_byHits_PDG;   //!
   TBranch        *b_reco_beam_true_byHits_ID;   //!
   TBranch        *b_reco_beam_true_byE_matched;   //!
   TBranch        *b_reco_beam_true_byHits_matched;   //!
   TBranch        *b_reco_beam_true_byHits_purity;   //!
   TBranch        *b_true_beam_processes;   //!
   TBranch        *b_beam_inst_P;   //!
   TBranch        *b_beam_inst_TOF;   //!
   TBranch        *b_beam_inst_TOF_Chan;   //!
   TBranch        *b_beam_inst_X;   //!
   TBranch        *b_beam_inst_Y;   //!
   TBranch        *b_beam_inst_Z;   //!
   TBranch        *b_beam_inst_dirX;   //!
   TBranch        *b_beam_inst_dirY;   //!
   TBranch        *b_beam_inst_dirZ;   //!
   TBranch        *b_beam_inst_nFibersP1;   //!
   TBranch        *b_beam_inst_nFibersP2;   //!
   TBranch        *b_beam_inst_nFibersP3;   //!
   TBranch        *b_beam_inst_PDG_candidates;   //!
   TBranch        *b_beam_inst_nTracks;   //!
   TBranch        *b_beam_inst_nMomenta;   //!
   TBranch        *b_beam_inst_valid;   //!
   TBranch        *b_beam_inst_trigger;   //!
   TBranch        *b_reco_beam_Chi2_proton;   //!
   TBranch        *b_reco_beam_Chi2_ndof;   //!
   TBranch        *b_reco_daughter_allTrack_momByRange_proton;   //!
   TBranch        *b_reco_daughter_allTrack_momByRange_muon;   //!
   TBranch        *b_reco_beam_momByRange_proton;   //!
   TBranch        *b_reco_beam_momByRange_muon;   //!
   TBranch        *b_reco_daughter_allTrack_momByRange_alt_proton;   //!
   TBranch        *b_reco_daughter_allTrack_momByRange_alt_muon;   //!
   TBranch        *b_reco_beam_momByRange_alt_proton;   //!
   TBranch        *b_reco_beam_momByRange_alt_muon;   //!
   TBranch        *b_reco_beam_true_byE_endPx;   //!
   TBranch        *b_reco_beam_true_byE_endPy;   //!
   TBranch        *b_reco_beam_true_byE_endPz;   //!
   TBranch        *b_reco_beam_true_byE_endE;   //!
   TBranch        *b_reco_beam_true_byE_endP;   //!
   TBranch        *b_reco_beam_true_byE_startPx;   //!
   TBranch        *b_reco_beam_true_byE_startPy;   //!
   TBranch        *b_reco_beam_true_byE_startPz;   //!
   TBranch        *b_reco_beam_true_byE_startE;   //!
   TBranch        *b_reco_beam_true_byE_startP;   //!
   TBranch        *b_reco_beam_true_byHits_endPx;   //!
   TBranch        *b_reco_beam_true_byHits_endPy;   //!
   TBranch        *b_reco_beam_true_byHits_endPz;   //!
   TBranch        *b_reco_beam_true_byHits_endE;   //!
   TBranch        *b_reco_beam_true_byHits_endP;   //!
   TBranch        *b_reco_beam_true_byHits_startPx;   //!
   TBranch        *b_reco_beam_true_byHits_startPy;   //!
   TBranch        *b_reco_beam_true_byHits_startPz;   //!
   TBranch        *b_reco_beam_true_byHits_startE;   //!
   TBranch        *b_reco_beam_true_byHits_startP;   //!
   TBranch        *b_reco_beam_incidentEnergies;   //!
   TBranch        *b_reco_beam_interactingEnergy;   //!
   TBranch        *b_true_beam_incidentEnergies;   //!
   TBranch        *b_true_beam_interactingEnergy;   //!
   TBranch        *b_true_beam_slices;   //!
   TBranch        *b_true_beam_slices_found;   //!
   TBranch        *b_true_beam_slices_deltaE;   //!
   TBranch        *b_em_energy;   //!
   TBranch        *b_true_beam_traj_X;   //!
   TBranch        *b_true_beam_traj_Y;   //!
   TBranch        *b_true_beam_traj_Z;   //!
   TBranch        *b_true_beam_traj_Px;   //!
   TBranch        *b_true_beam_traj_Py;   //!
   TBranch        *b_true_beam_traj_Pz;   //!
   TBranch        *b_true_beam_traj_KE;   //!
   TBranch        *b_true_beam_traj_X_SCE;   //!
   TBranch        *b_true_beam_traj_Y_SCE;   //!
   TBranch        *b_true_beam_traj_Z_SCE;   //!
   TBranch        *b_g4rw_primary_weights;   //!
   TBranch        *b_g4rw_primary_plus_sigma_weight;   //!
   TBranch        *b_g4rw_primary_minus_sigma_weight;   //!
   TBranch        *b_g4rw_primary_var;   //!
   TBranch        *b_g4rw_alt_primary_plus_sigma_weight;   //!
   TBranch        *b_g4rw_alt_primary_minus_sigma_weight;   //!
   TBranch        *b_g4rw_full_primary_plus_sigma_weight;   //!
   TBranch        *b_g4rw_full_primary_minus_sigma_weight;   //!
   TBranch        *b_g4rw_full_grid_weights;   //!
   TBranch        *b_g4rw_full_grid_piplus_weights;   //!
   TBranch        *b_g4rw_full_grid_piplus_weights_fake_data;   //!
   TBranch        *b_g4rw_full_grid_piminus_weights;   //!
   TBranch        *b_g4rw_full_grid_proton_weights;   //!
   TBranch        *b_g4rw_primary_grid_weights;   //!
   TBranch        *b_g4rw_primary_grid_pair_weights;   //!
   TBranch        *b_reco_beam_spacePts_X;   //!
   TBranch        *b_reco_beam_spacePts_Y;   //!
   TBranch        *b_reco_beam_spacePts_Z;   //!
   TBranch        *b_reco_daughter_spacePts_X;   //!
   TBranch        *b_reco_daughter_spacePts_Y;   //!
   TBranch        *b_reco_daughter_spacePts_Z;   //!
   TBranch        *b_reco_daughter_shower_spacePts_X;   //!
   TBranch        *b_reco_daughter_shower_spacePts_Y;   //!
   TBranch        *b_reco_daughter_shower_spacePts_Z;   //!

   beamana(TTree *tree=0);
   virtual ~beamana();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual Bool_t   Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef beamana_cxx
beamana::beamana(TTree *tree) : fChain(0)
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("pduneana.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("pduneana.root");
      }
      TDirectory * dir = (TDirectory*)f->Get("pduneana.root:/pduneana");
      dir->GetObject("beamana",tree);

   }
   Init(tree);
}

beamana::~beamana()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t beamana::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t beamana::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void beamana::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   reco_beam_calo_startDirX = 0;
   reco_beam_calo_startDirY = 0;
   reco_beam_calo_startDirZ = 0;
   reco_beam_calo_endDirX = 0;
   reco_beam_calo_endDirY = 0;
   reco_beam_calo_endDirZ = 0;
   beam_track_IDs = 0;
   beam_particle_scores = 0;
   reco_beam_dQdX_SCE = 0;
   reco_beam_EField_SCE = 0;
   reco_beam_calo_X = 0;
   reco_beam_calo_Y = 0;
   reco_beam_calo_Z = 0;
   reco_beam_dQ = 0;
   reco_beam_dEdX_SCE = 0;
   reco_beam_calibrated_dEdX_SCE = 0;
   reco_beam_calibrated_dQdX_SCE = 0;
   reco_beam_resRange_SCE = 0;
   reco_beam_TrkPitch_SCE = 0;
   reco_beam_dQdX_NoSCE = 0;
   reco_beam_dQ_NoSCE = 0;
   reco_beam_dEdX_NoSCE = 0;
   reco_beam_calibrated_dEdX_NoSCE = 0;
   reco_beam_resRange_NoSCE = 0;
   reco_beam_TrkPitch_NoSCE = 0;
   reco_beam_calo_wire = 0;
   reco_beam_calo_wire_z = 0;
   reco_beam_calo_wire_NoSCE = 0;
   reco_beam_calo_wire_z_NoSCE = 0;
   reco_beam_calo_tick = 0;
   reco_beam_calo_TPC = 0;
   reco_beam_calo_TPC_NoSCE = 0;
   reco_beam_allTrack_resRange = 0;
   reco_beam_allTrack_calibrated_dEdX = 0;
   reco_track_startX = 0;
   reco_track_startY = 0;
   reco_track_startZ = 0;
   reco_track_endX = 0;
   reco_track_endY = 0;
   reco_track_endZ = 0;
   reco_track_michel_score = 0;
   reco_track_ID = 0;
   reco_track_nHits = 0;
   reco_daughter_PFP_true_byHits_PDG = 0;
   reco_daughter_PFP_true_byHits_ID = 0;
   reco_daughter_PFP_true_byHits_origin = 0;
   reco_daughter_PFP_true_byHits_parID = 0;
   reco_daughter_PFP_true_byHits_parPDG = 0;
   reco_daughter_PFP_true_byHits_process = 0;
   reco_daughter_PFP_true_byHits_sharedHits = 0;
   reco_daughter_PFP_true_byHits_emHits = 0;
   reco_daughter_PFP_true_byHits_len = 0;
   reco_daughter_PFP_true_byHits_startX = 0;
   reco_daughter_PFP_true_byHits_startY = 0;
   reco_daughter_PFP_true_byHits_startZ = 0;
   reco_daughter_PFP_true_byHits_endX = 0;
   reco_daughter_PFP_true_byHits_endY = 0;
   reco_daughter_PFP_true_byHits_endZ = 0;
   reco_daughter_PFP_true_byHits_startPx = 0;
   reco_daughter_PFP_true_byHits_startPy = 0;
   reco_daughter_PFP_true_byHits_startPz = 0;
   reco_daughter_PFP_true_byHits_startP = 0;
   reco_daughter_PFP_true_byHits_startE = 0;
   reco_daughter_PFP_true_byHits_endProcess = 0;
   reco_daughter_PFP_true_byHits_purity = 0;
   reco_daughter_PFP_true_byHits_completeness = 0;
   reco_daughter_PFP_true_byE_PDG = 0;
   reco_daughter_PFP_true_byE_len = 0;
   reco_daughter_PFP_true_byE_completeness = 0;
   reco_daughter_PFP_true_byE_purity = 0;
   reco_daughter_allTrack_ID = 0;
   reco_daughter_allTrack_dQdX_SCE = 0;
   reco_daughter_allTrack_calibrated_dQdX_SCE = 0;
   reco_daughter_allTrack_EField_SCE = 0;
   reco_daughter_allTrack_dEdX_SCE = 0;
   reco_daughter_allTrack_resRange_SCE = 0;
   reco_daughter_allTrack_calibrated_dEdX_SCE = 0;
   reco_daughter_allTrack_Chi2_proton = 0;
   reco_daughter_allTrack_Chi2_pion = 0;
   reco_daughter_allTrack_Chi2_muon = 0;
   reco_daughter_allTrack_Chi2_ndof = 0;
   reco_daughter_allTrack_Chi2_ndof_pion = 0;
   reco_daughter_allTrack_Chi2_ndof_muon = 0;
   reco_daughter_allTrack_Chi2_proton_plane0 = 0;
   reco_daughter_allTrack_Chi2_proton_plane1 = 0;
   reco_daughter_allTrack_Chi2_ndof_plane0 = 0;
   reco_daughter_allTrack_Chi2_ndof_plane1 = 0;
   reco_daughter_allTrack_calibrated_dEdX_SCE_plane0 = 0;
   reco_daughter_allTrack_calibrated_dEdX_SCE_plane1 = 0;
   reco_daughter_allTrack_resRange_plane0 = 0;
   reco_daughter_allTrack_resRange_plane1 = 0;
   reco_daughter_allTrack_Theta = 0;
   reco_daughter_allTrack_Phi = 0;
   reco_daughter_allTrack_len = 0;
   reco_daughter_allTrack_alt_len = 0;
   reco_daughter_allTrack_startX = 0;
   reco_daughter_allTrack_startY = 0;
   reco_daughter_allTrack_startZ = 0;
   reco_daughter_allTrack_endX = 0;
   reco_daughter_allTrack_endY = 0;
   reco_daughter_allTrack_endZ = 0;
   reco_daughter_allTrack_dR = 0;
   reco_daughter_allTrack_calo_X = 0;
   reco_daughter_allTrack_calo_Y = 0;
   reco_daughter_allTrack_calo_Z = 0;
   reco_daughter_allTrack_to_vertex = 0;
   reco_daughter_allTrack_vertex_michel_score = 0;
   reco_daughter_allTrack_vertex_nHits = 0;
   reco_daughter_allShower_ID = 0;
   reco_daughter_allShower_len = 0;
   reco_daughter_allShower_startX = 0;
   reco_daughter_allShower_startY = 0;
   reco_daughter_allShower_startZ = 0;
   reco_daughter_allShower_dirX = 0;
   reco_daughter_allShower_dirY = 0;
   reco_daughter_allShower_dirZ = 0;
   reco_daughter_allShower_energy = 0;
   reco_daughter_PFP_ID = 0;
   reco_daughter_PFP_nHits = 0;
   reco_daughter_PFP_nHits_collection = 0;
   reco_daughter_PFP_trackScore = 0;
   reco_daughter_PFP_emScore = 0;
   reco_daughter_PFP_michelScore = 0;
   reco_daughter_PFP_trackScore_collection = 0;
   reco_daughter_PFP_emScore_collection = 0;
   reco_daughter_PFP_michelScore_collection = 0;
   true_beam_endProcess = 0;
   true_beam_elastic_costheta = 0;
   true_beam_elastic_X = 0;
   true_beam_elastic_Y = 0;
   true_beam_elastic_Z = 0;
   true_beam_elastic_deltaE = 0;
   true_beam_elastic_IDE_edep = 0;
   true_beam_reco_byHits_PFP_ID = 0;
   true_beam_reco_byHits_PFP_nHits = 0;
   true_beam_reco_byHits_allTrack_ID = 0;
   true_beam_daughter_PDG = 0;
   true_beam_daughter_ID = 0;
   true_beam_daughter_len = 0;
   true_beam_daughter_startX = 0;
   true_beam_daughter_startY = 0;
   true_beam_daughter_startZ = 0;
   true_beam_daughter_startPx = 0;
   true_beam_daughter_startPy = 0;
   true_beam_daughter_startPz = 0;
   true_beam_daughter_startP = 0;
   true_beam_daughter_endX = 0;
   true_beam_daughter_endY = 0;
   true_beam_daughter_endZ = 0;
   true_beam_daughter_Process = 0;
   true_beam_daughter_endProcess = 0;
   true_beam_daughter_nHits = 0;
   true_beam_daughter_reco_byHits_PFP_ID = 0;
   true_beam_daughter_reco_byHits_PFP_nHits = 0;
   true_beam_daughter_reco_byHits_PFP_trackScore = 0;
   true_beam_daughter_reco_byHits_allTrack_ID = 0;
   true_beam_daughter_reco_byHits_allTrack_startX = 0;
   true_beam_daughter_reco_byHits_allTrack_startY = 0;
   true_beam_daughter_reco_byHits_allTrack_startZ = 0;
   true_beam_daughter_reco_byHits_allTrack_endX = 0;
   true_beam_daughter_reco_byHits_allTrack_endY = 0;
   true_beam_daughter_reco_byHits_allTrack_endZ = 0;
   true_beam_daughter_reco_byHits_allTrack_len = 0;
   true_beam_daughter_reco_byHits_allShower_ID = 0;
   true_beam_daughter_reco_byHits_allShower_startX = 0;
   true_beam_daughter_reco_byHits_allShower_startY = 0;
   true_beam_daughter_reco_byHits_allShower_startZ = 0;
   true_beam_daughter_reco_byHits_allShower_len = 0;
   true_beam_Pi0_decay_ID = 0;
   true_beam_Pi0_decay_parID = 0;
   true_beam_Pi0_decay_PDG = 0;
   true_beam_Pi0_decay_startP = 0;
   true_beam_Pi0_decay_startPx = 0;
   true_beam_Pi0_decay_startPy = 0;
   true_beam_Pi0_decay_startPz = 0;
   true_beam_Pi0_decay_startX = 0;
   true_beam_Pi0_decay_startY = 0;
   true_beam_Pi0_decay_startZ = 0;
   true_beam_Pi0_decay_len = 0;
   true_beam_Pi0_decay_nHits = 0;
   true_beam_Pi0_decay_reco_byHits_PFP_ID = 0;
   true_beam_Pi0_decay_reco_byHits_PFP_nHits = 0;
   true_beam_Pi0_decay_reco_byHits_PFP_trackScore = 0;
   true_beam_Pi0_decay_reco_byHits_allTrack_ID = 0;
   true_beam_Pi0_decay_reco_byHits_allTrack_startX = 0;
   true_beam_Pi0_decay_reco_byHits_allTrack_startY = 0;
   true_beam_Pi0_decay_reco_byHits_allTrack_startZ = 0;
   true_beam_Pi0_decay_reco_byHits_allTrack_endX = 0;
   true_beam_Pi0_decay_reco_byHits_allTrack_endY = 0;
   true_beam_Pi0_decay_reco_byHits_allTrack_endZ = 0;
   true_beam_Pi0_decay_reco_byHits_allTrack_len = 0;
   true_beam_Pi0_decay_reco_byHits_allShower_ID = 0;
   true_beam_Pi0_decay_reco_byHits_allShower_startX = 0;
   true_beam_Pi0_decay_reco_byHits_allShower_startY = 0;
   true_beam_Pi0_decay_reco_byHits_allShower_startZ = 0;
   true_beam_Pi0_decay_reco_byHits_allShower_len = 0;
   true_beam_grand_daughter_ID = 0;
   true_beam_grand_daughter_parID = 0;
   true_beam_grand_daughter_PDG = 0;
   true_beam_grand_daughter_nHits = 0;
   true_beam_grand_daughter_Process = 0;
   true_beam_grand_daughter_endProcess = 0;
   reco_beam_true_byE_endProcess = 0;
   reco_beam_true_byE_process = 0;
   reco_beam_true_byHits_endProcess = 0;
   reco_beam_true_byHits_process = 0;
   true_beam_processes = 0;
   beam_inst_TOF = 0;
   beam_inst_TOF_Chan = 0;
   beam_inst_PDG_candidates = 0;
   reco_daughter_allTrack_momByRange_proton = 0;
   reco_daughter_allTrack_momByRange_muon = 0;
   reco_daughter_allTrack_momByRange_alt_proton = 0;
   reco_daughter_allTrack_momByRange_alt_muon = 0;
   reco_beam_incidentEnergies = 0;
   true_beam_incidentEnergies = 0;
   true_beam_slices = 0;
   true_beam_slices_found = 0;
   true_beam_slices_deltaE = 0;
   true_beam_traj_X = 0;
   true_beam_traj_Y = 0;
   true_beam_traj_Z = 0;
   true_beam_traj_Px = 0;
   true_beam_traj_Py = 0;
   true_beam_traj_Pz = 0;
   true_beam_traj_KE = 0;
   true_beam_traj_X_SCE = 0;
   true_beam_traj_Y_SCE = 0;
   true_beam_traj_Z_SCE = 0;
   g4rw_primary_weights = 0;
   g4rw_primary_plus_sigma_weight = 0;
   g4rw_primary_minus_sigma_weight = 0;
   g4rw_primary_var = 0;
   g4rw_alt_primary_plus_sigma_weight = 0;
   g4rw_alt_primary_minus_sigma_weight = 0;
   g4rw_full_primary_plus_sigma_weight = 0;
   g4rw_full_primary_minus_sigma_weight = 0;
   g4rw_full_grid_weights = 0;
   g4rw_full_grid_piplus_weights = 0;
   g4rw_full_grid_piplus_weights_fake_data = 0;
   g4rw_full_grid_piminus_weights = 0;
   g4rw_full_grid_proton_weights = 0;
   g4rw_primary_grid_weights = 0;
   g4rw_primary_grid_pair_weights = 0;
   reco_beam_spacePts_X = 0;
   reco_beam_spacePts_Y = 0;
   reco_beam_spacePts_Z = 0;
   reco_daughter_spacePts_X = 0;
   reco_daughter_spacePts_Y = 0;
   reco_daughter_spacePts_Z = 0;
   reco_daughter_shower_spacePts_X = 0;
   reco_daughter_shower_spacePts_Y = 0;
   reco_daughter_shower_spacePts_Z = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("run", &run, &b_run);
   fChain->SetBranchAddress("subrun", &subrun, &b_subrun);
   fChain->SetBranchAddress("event", &event, &b_event);
   fChain->SetBranchAddress("MC", &MC, &b_MC);
   fChain->SetBranchAddress("reco_beam_type", &reco_beam_type, &b_reco_beam_type);
   fChain->SetBranchAddress("reco_beam_startX", &reco_beam_startX, &b_reco_beam_startX);
   fChain->SetBranchAddress("reco_beam_startY", &reco_beam_startY, &b_reco_beam_startY);
   fChain->SetBranchAddress("reco_beam_startZ", &reco_beam_startZ, &b_reco_beam_startZ);
   fChain->SetBranchAddress("reco_beam_endX", &reco_beam_endX, &b_reco_beam_endX);
   fChain->SetBranchAddress("reco_beam_endY", &reco_beam_endY, &b_reco_beam_endY);
   fChain->SetBranchAddress("reco_beam_endZ", &reco_beam_endZ, &b_reco_beam_endZ);
   fChain->SetBranchAddress("reco_beam_len", &reco_beam_len, &b_reco_beam_len);
   fChain->SetBranchAddress("reco_beam_alt_len", &reco_beam_alt_len, &b_reco_beam_alt_len);
   fChain->SetBranchAddress("reco_beam_calo_startX", &reco_beam_calo_startX, &b_reco_beam_calo_startX);
   fChain->SetBranchAddress("reco_beam_calo_startY", &reco_beam_calo_startY, &b_reco_beam_calo_startY);
   fChain->SetBranchAddress("reco_beam_calo_startZ", &reco_beam_calo_startZ, &b_reco_beam_calo_startZ);
   fChain->SetBranchAddress("reco_beam_calo_endX", &reco_beam_calo_endX, &b_reco_beam_calo_endX);
   fChain->SetBranchAddress("reco_beam_calo_endY", &reco_beam_calo_endY, &b_reco_beam_calo_endY);
   fChain->SetBranchAddress("reco_beam_calo_endZ", &reco_beam_calo_endZ, &b_reco_beam_calo_endZ);
   fChain->SetBranchAddress("reco_beam_calo_startDirX", &reco_beam_calo_startDirX, &b_reco_beam_calo_startDirX);
   fChain->SetBranchAddress("reco_beam_calo_startDirY", &reco_beam_calo_startDirY, &b_reco_beam_calo_startDirY);
   fChain->SetBranchAddress("reco_beam_calo_startDirZ", &reco_beam_calo_startDirZ, &b_reco_beam_calo_startDirZ);
   fChain->SetBranchAddress("reco_beam_calo_endDirX", &reco_beam_calo_endDirX, &b_reco_beam_calo_endDirX);
   fChain->SetBranchAddress("reco_beam_calo_endDirY", &reco_beam_calo_endDirY, &b_reco_beam_calo_endDirY);
   fChain->SetBranchAddress("reco_beam_calo_endDirZ", &reco_beam_calo_endDirZ, &b_reco_beam_calo_endDirZ);
   fChain->SetBranchAddress("reco_beam_trackDirX", &reco_beam_trackDirX, &b_reco_beam_trackDirX);
   fChain->SetBranchAddress("reco_beam_trackDirY", &reco_beam_trackDirY, &b_reco_beam_trackDirY);
   fChain->SetBranchAddress("reco_beam_trackDirZ", &reco_beam_trackDirZ, &b_reco_beam_trackDirZ);
   fChain->SetBranchAddress("reco_beam_trackEndDirX", &reco_beam_trackEndDirX, &b_reco_beam_trackEndDirX);
   fChain->SetBranchAddress("reco_beam_trackEndDirY", &reco_beam_trackEndDirY, &b_reco_beam_trackEndDirY);
   fChain->SetBranchAddress("reco_beam_trackEndDirZ", &reco_beam_trackEndDirZ, &b_reco_beam_trackEndDirZ);
   fChain->SetBranchAddress("reco_beam_vertex_nHits", &reco_beam_vertex_nHits, &b_reco_beam_vertex_nHits);
   fChain->SetBranchAddress("reco_beam_vertex_michel_score", &reco_beam_vertex_michel_score, &b_reco_beam_vertex_michel_score);
   fChain->SetBranchAddress("reco_beam_trackID", &reco_beam_trackID, &b_reco_beam_trackID);
   fChain->SetBranchAddress("n_beam_slices", &n_beam_slices, &b_n_beam_slices);
   fChain->SetBranchAddress("n_beam_particles", &n_beam_particles, &b_n_beam_particles);
   fChain->SetBranchAddress("beam_track_IDs", &beam_track_IDs, &b_beam_track_IDs);
   fChain->SetBranchAddress("beam_particle_scores", &beam_particle_scores, &b_beam_particle_scores);
   fChain->SetBranchAddress("reco_beam_dQdX_SCE", &reco_beam_dQdX_SCE, &b_reco_beam_dQdX_SCE);
   fChain->SetBranchAddress("reco_beam_EField_SCE", &reco_beam_EField_SCE, &b_reco_beam_EField_SCE);
   fChain->SetBranchAddress("reco_beam_calo_X", &reco_beam_calo_X, &b_reco_beam_calo_X);
   fChain->SetBranchAddress("reco_beam_calo_Y", &reco_beam_calo_Y, &b_reco_beam_calo_Y);
   fChain->SetBranchAddress("reco_beam_calo_Z", &reco_beam_calo_Z, &b_reco_beam_calo_Z);
   fChain->SetBranchAddress("reco_beam_dQ", &reco_beam_dQ, &b_reco_beam_dQ);
   fChain->SetBranchAddress("reco_beam_dEdX_SCE", &reco_beam_dEdX_SCE, &b_reco_beam_dEdX_SCE);
   fChain->SetBranchAddress("reco_beam_calibrated_dEdX_SCE", &reco_beam_calibrated_dEdX_SCE, &b_reco_beam_calibrated_dEdX_SCE);
   fChain->SetBranchAddress("reco_beam_calibrated_dQdX_SCE", &reco_beam_calibrated_dQdX_SCE, &b_reco_beam_calibrated_dQdX_SCE);
   fChain->SetBranchAddress("reco_beam_resRange_SCE", &reco_beam_resRange_SCE, &b_reco_beam_resRange_SCE);
   fChain->SetBranchAddress("reco_beam_TrkPitch_SCE", &reco_beam_TrkPitch_SCE, &b_reco_beam_TrkPitch_SCE);
   fChain->SetBranchAddress("reco_beam_dQdX_NoSCE", &reco_beam_dQdX_NoSCE, &b_reco_beam_dQdX_NoSCE);
   fChain->SetBranchAddress("reco_beam_dQ_NoSCE", &reco_beam_dQ_NoSCE, &b_reco_beam_dQ_NoSCE);
   fChain->SetBranchAddress("reco_beam_dEdX_NoSCE", &reco_beam_dEdX_NoSCE, &b_reco_beam_dEdX_NoSCE);
   fChain->SetBranchAddress("reco_beam_calibrated_dEdX_NoSCE", &reco_beam_calibrated_dEdX_NoSCE, &b_reco_beam_calibrated_dEdX_NoSCE);
   fChain->SetBranchAddress("reco_beam_resRange_NoSCE", &reco_beam_resRange_NoSCE, &b_reco_beam_resRange_NoSCE);
   fChain->SetBranchAddress("reco_beam_TrkPitch_NoSCE", &reco_beam_TrkPitch_NoSCE, &b_reco_beam_TrkPitch_NoSCE);
   fChain->SetBranchAddress("reco_beam_calo_wire", &reco_beam_calo_wire, &b_reco_beam_calo_wire);
   fChain->SetBranchAddress("reco_beam_calo_wire_z", &reco_beam_calo_wire_z, &b_reco_beam_calo_wire_z);
   fChain->SetBranchAddress("reco_beam_calo_wire_NoSCE", &reco_beam_calo_wire_NoSCE, &b_reco_beam_calo_wire_NoSCE);
   fChain->SetBranchAddress("reco_beam_calo_wire_z_NoSCE", &reco_beam_calo_wire_z_NoSCE, &b_reco_beam_calo_wire_z_NoSCE);
   fChain->SetBranchAddress("reco_beam_calo_tick", &reco_beam_calo_tick, &b_reco_beam_calo_tick);
   fChain->SetBranchAddress("reco_beam_calo_TPC", &reco_beam_calo_TPC, &b_reco_beam_calo_TPC);
   fChain->SetBranchAddress("reco_beam_calo_TPC_NoSCE", &reco_beam_calo_TPC_NoSCE, &b_reco_beam_calo_TPC_NoSCE);
   fChain->SetBranchAddress("reco_beam_flipped", &reco_beam_flipped, &b_reco_beam_flipped);
   fChain->SetBranchAddress("reco_beam_passes_beam_cuts", &reco_beam_passes_beam_cuts, &b_reco_beam_passes_beam_cuts);
   fChain->SetBranchAddress("reco_beam_PFP_ID", &reco_beam_PFP_ID, &b_reco_beam_PFP_ID);
   fChain->SetBranchAddress("reco_beam_PFP_nHits", &reco_beam_PFP_nHits, &b_reco_beam_PFP_nHits);
   fChain->SetBranchAddress("reco_beam_PFP_trackScore", &reco_beam_PFP_trackScore, &b_reco_beam_PFP_trackScore);
   fChain->SetBranchAddress("reco_beam_PFP_emScore", &reco_beam_PFP_emScore, &b_reco_beam_PFP_emScore);
   fChain->SetBranchAddress("reco_beam_PFP_michelScore", &reco_beam_PFP_michelScore, &b_reco_beam_PFP_michelScore);
   fChain->SetBranchAddress("reco_beam_PFP_trackScore_collection", &reco_beam_PFP_trackScore_collection, &b_reco_beam_PFP_trackScore_collection);
   fChain->SetBranchAddress("reco_beam_PFP_emScore_collection", &reco_beam_PFP_emScore_collection, &b_reco_beam_PFP_emScore_collection);
   fChain->SetBranchAddress("reco_beam_PFP_michelScore_collection", &reco_beam_PFP_michelScore_collection, &b_reco_beam_PFP_michelScore_collection);
   fChain->SetBranchAddress("reco_beam_allTrack_ID", &reco_beam_allTrack_ID, &b_reco_beam_allTrack_ID);
   fChain->SetBranchAddress("reco_beam_allTrack_beam_cuts", &reco_beam_allTrack_beam_cuts, &b_reco_beam_allTrack_beam_cuts);
   fChain->SetBranchAddress("reco_beam_allTrack_flipped", &reco_beam_allTrack_flipped, &b_reco_beam_allTrack_flipped);
   fChain->SetBranchAddress("reco_beam_allTrack_len", &reco_beam_allTrack_len, &b_reco_beam_allTrack_len);
   fChain->SetBranchAddress("reco_beam_allTrack_startX", &reco_beam_allTrack_startX, &b_reco_beam_allTrack_startX);
   fChain->SetBranchAddress("reco_beam_allTrack_startY", &reco_beam_allTrack_startY, &b_reco_beam_allTrack_startY);
   fChain->SetBranchAddress("reco_beam_allTrack_startZ", &reco_beam_allTrack_startZ, &b_reco_beam_allTrack_startZ);
   fChain->SetBranchAddress("reco_beam_allTrack_endX", &reco_beam_allTrack_endX, &b_reco_beam_allTrack_endX);
   fChain->SetBranchAddress("reco_beam_allTrack_endY", &reco_beam_allTrack_endY, &b_reco_beam_allTrack_endY);
   fChain->SetBranchAddress("reco_beam_allTrack_endZ", &reco_beam_allTrack_endZ, &b_reco_beam_allTrack_endZ);
   fChain->SetBranchAddress("reco_beam_allTrack_trackDirX", &reco_beam_allTrack_trackDirX, &b_reco_beam_allTrack_trackDirX);
   fChain->SetBranchAddress("reco_beam_allTrack_trackDirY", &reco_beam_allTrack_trackDirY, &b_reco_beam_allTrack_trackDirY);
   fChain->SetBranchAddress("reco_beam_allTrack_trackDirZ", &reco_beam_allTrack_trackDirZ, &b_reco_beam_allTrack_trackDirZ);
   fChain->SetBranchAddress("reco_beam_allTrack_trackEndDirX", &reco_beam_allTrack_trackEndDirX, &b_reco_beam_allTrack_trackEndDirX);
   fChain->SetBranchAddress("reco_beam_allTrack_trackEndDirY", &reco_beam_allTrack_trackEndDirY, &b_reco_beam_allTrack_trackEndDirY);
   fChain->SetBranchAddress("reco_beam_allTrack_trackEndDirZ", &reco_beam_allTrack_trackEndDirZ, &b_reco_beam_allTrack_trackEndDirZ);
   fChain->SetBranchAddress("reco_beam_allTrack_resRange", &reco_beam_allTrack_resRange, &b_reco_beam_allTrack_resRange);
   fChain->SetBranchAddress("reco_beam_allTrack_calibrated_dEdX", &reco_beam_allTrack_calibrated_dEdX, &b_reco_beam_allTrack_calibrated_dEdX);
   fChain->SetBranchAddress("reco_beam_allTrack_Chi2_proton", &reco_beam_allTrack_Chi2_proton, &b_reco_beam_allTrack_Chi2_proton);
   fChain->SetBranchAddress("reco_beam_allTrack_Chi2_ndof", &reco_beam_allTrack_Chi2_ndof, &b_reco_beam_allTrack_Chi2_ndof);
   fChain->SetBranchAddress("reco_track_startX", &reco_track_startX, &b_reco_track_startX);
   fChain->SetBranchAddress("reco_track_startY", &reco_track_startY, &b_reco_track_startY);
   fChain->SetBranchAddress("reco_track_startZ", &reco_track_startZ, &b_reco_track_startZ);
   fChain->SetBranchAddress("reco_track_endX", &reco_track_endX, &b_reco_track_endX);
   fChain->SetBranchAddress("reco_track_endY", &reco_track_endY, &b_reco_track_endY);
   fChain->SetBranchAddress("reco_track_endZ", &reco_track_endZ, &b_reco_track_endZ);
   fChain->SetBranchAddress("reco_track_michel_score", &reco_track_michel_score, &b_reco_track_michel_score);
   fChain->SetBranchAddress("reco_track_ID", &reco_track_ID, &b_reco_track_ID);
   fChain->SetBranchAddress("reco_track_nHits", &reco_track_nHits, &b_reco_track_nHits);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_PDG", &reco_daughter_PFP_true_byHits_PDG, &b_reco_daughter_PFP_true_byHits_PDG);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_ID", &reco_daughter_PFP_true_byHits_ID, &b_reco_daughter_PFP_true_byHits_ID);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_origin", &reco_daughter_PFP_true_byHits_origin, &b_reco_daughter_PFP_true_byHits_origin);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_parID", &reco_daughter_PFP_true_byHits_parID, &b_reco_daughter_PFP_true_byHits_parID);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_parPDG", &reco_daughter_PFP_true_byHits_parPDG, &b_reco_daughter_PFP_true_byHits_parPDG);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_process", &reco_daughter_PFP_true_byHits_process, &b_reco_daughter_PFP_true_byHits_process);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_sharedHits", &reco_daughter_PFP_true_byHits_sharedHits, &b_reco_daughter_PFP_true_byHits_sharedHits);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_emHits", &reco_daughter_PFP_true_byHits_emHits, &b_reco_daughter_PFP_true_byHits_emHits);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_len", &reco_daughter_PFP_true_byHits_len, &b_reco_daughter_PFP_true_byHits_len);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_startX", &reco_daughter_PFP_true_byHits_startX, &b_reco_daughter_PFP_true_byHits_startX);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_startY", &reco_daughter_PFP_true_byHits_startY, &b_reco_daughter_PFP_true_byHits_startY);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_startZ", &reco_daughter_PFP_true_byHits_startZ, &b_reco_daughter_PFP_true_byHits_startZ);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_endX", &reco_daughter_PFP_true_byHits_endX, &b_reco_daughter_PFP_true_byHits_endX);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_endY", &reco_daughter_PFP_true_byHits_endY, &b_reco_daughter_PFP_true_byHits_endY);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_endZ", &reco_daughter_PFP_true_byHits_endZ, &b_reco_daughter_PFP_true_byHits_endZ);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_startPx", &reco_daughter_PFP_true_byHits_startPx, &b_reco_daughter_PFP_true_byHits_startPx);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_startPy", &reco_daughter_PFP_true_byHits_startPy, &b_reco_daughter_PFP_true_byHits_startPy);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_startPz", &reco_daughter_PFP_true_byHits_startPz, &b_reco_daughter_PFP_true_byHits_startPz);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_startP", &reco_daughter_PFP_true_byHits_startP, &b_reco_daughter_PFP_true_byHits_startP);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_startE", &reco_daughter_PFP_true_byHits_startE, &b_reco_daughter_PFP_true_byHits_startE);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_endProcess", &reco_daughter_PFP_true_byHits_endProcess, &b_reco_daughter_PFP_true_byHits_endProcess);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_purity", &reco_daughter_PFP_true_byHits_purity, &b_reco_daughter_PFP_true_byHits_purity);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byHits_completeness", &reco_daughter_PFP_true_byHits_completeness, &b_reco_daughter_PFP_true_byHits_completeness);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byE_PDG", &reco_daughter_PFP_true_byE_PDG, &b_reco_daughter_PFP_true_byE_PDG);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byE_len", &reco_daughter_PFP_true_byE_len, &b_reco_daughter_PFP_true_byE_len);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byE_completeness", &reco_daughter_PFP_true_byE_completeness, &b_reco_daughter_PFP_true_byE_completeness);
   fChain->SetBranchAddress("reco_daughter_PFP_true_byE_purity", &reco_daughter_PFP_true_byE_purity, &b_reco_daughter_PFP_true_byE_purity);
   fChain->SetBranchAddress("reco_daughter_allTrack_ID", &reco_daughter_allTrack_ID, &b_reco_daughter_allTrack_ID);
   fChain->SetBranchAddress("reco_daughter_allTrack_dQdX_SCE", &reco_daughter_allTrack_dQdX_SCE, &b_reco_daughter_allTrack_dQdX_SCE);
   fChain->SetBranchAddress("reco_daughter_allTrack_calibrated_dQdX_SCE", &reco_daughter_allTrack_calibrated_dQdX_SCE, &b_reco_daughter_allTrack_calibrated_dQdX_SCE);
   fChain->SetBranchAddress("reco_daughter_allTrack_EField_SCE", &reco_daughter_allTrack_EField_SCE, &b_reco_daughter_allTrack_EField_SCE);
   fChain->SetBranchAddress("reco_daughter_allTrack_dEdX_SCE", &reco_daughter_allTrack_dEdX_SCE, &b_reco_daughter_allTrack_dEdX_SCE);
   fChain->SetBranchAddress("reco_daughter_allTrack_resRange_SCE", &reco_daughter_allTrack_resRange_SCE, &b_reco_daughter_allTrack_resRange_SCE);
   fChain->SetBranchAddress("reco_daughter_allTrack_calibrated_dEdX_SCE", &reco_daughter_allTrack_calibrated_dEdX_SCE, &b_reco_daughter_allTrack_calibrated_dEdX_SCE);
   fChain->SetBranchAddress("reco_daughter_allTrack_Chi2_proton", &reco_daughter_allTrack_Chi2_proton, &b_reco_daughter_allTrack_Chi2_proton);
   fChain->SetBranchAddress("reco_daughter_allTrack_Chi2_pion", &reco_daughter_allTrack_Chi2_pion, &b_reco_daughter_allTrack_Chi2_pion);
   fChain->SetBranchAddress("reco_daughter_allTrack_Chi2_muon", &reco_daughter_allTrack_Chi2_muon, &b_reco_daughter_allTrack_Chi2_muon);
   fChain->SetBranchAddress("reco_daughter_allTrack_Chi2_ndof", &reco_daughter_allTrack_Chi2_ndof, &b_reco_daughter_allTrack_Chi2_ndof);
   fChain->SetBranchAddress("reco_daughter_allTrack_Chi2_ndof_pion", &reco_daughter_allTrack_Chi2_ndof_pion, &b_reco_daughter_allTrack_Chi2_ndof_pion);
   fChain->SetBranchAddress("reco_daughter_allTrack_Chi2_ndof_muon", &reco_daughter_allTrack_Chi2_ndof_muon, &b_reco_daughter_allTrack_Chi2_ndof_muon);
   fChain->SetBranchAddress("reco_daughter_allTrack_Chi2_proton_plane0", &reco_daughter_allTrack_Chi2_proton_plane0, &b_reco_daughter_allTrack_Chi2_proton_plane0);
   fChain->SetBranchAddress("reco_daughter_allTrack_Chi2_proton_plane1", &reco_daughter_allTrack_Chi2_proton_plane1, &b_reco_daughter_allTrack_Chi2_proton_plane1);
   fChain->SetBranchAddress("reco_daughter_allTrack_Chi2_ndof_plane0", &reco_daughter_allTrack_Chi2_ndof_plane0, &b_reco_daughter_allTrack_Chi2_ndof_plane0);
   fChain->SetBranchAddress("reco_daughter_allTrack_Chi2_ndof_plane1", &reco_daughter_allTrack_Chi2_ndof_plane1, &b_reco_daughter_allTrack_Chi2_ndof_plane1);
   fChain->SetBranchAddress("reco_daughter_allTrack_calibrated_dEdX_SCE_plane0", &reco_daughter_allTrack_calibrated_dEdX_SCE_plane0, &b_reco_daughter_allTrack_calibrated_dEdX_SCE_plane0);
   fChain->SetBranchAddress("reco_daughter_allTrack_calibrated_dEdX_SCE_plane1", &reco_daughter_allTrack_calibrated_dEdX_SCE_plane1, &b_reco_daughter_allTrack_calibrated_dEdX_SCE_plane1);
   fChain->SetBranchAddress("reco_daughter_allTrack_resRange_plane0", &reco_daughter_allTrack_resRange_plane0, &b_reco_daughter_allTrack_resRange_plane0);
   fChain->SetBranchAddress("reco_daughter_allTrack_resRange_plane1", &reco_daughter_allTrack_resRange_plane1, &b_reco_daughter_allTrack_resRange_plane1);
   fChain->SetBranchAddress("reco_daughter_allTrack_Theta", &reco_daughter_allTrack_Theta, &b_reco_daughter_allTrack_Theta);
   fChain->SetBranchAddress("reco_daughter_allTrack_Phi", &reco_daughter_allTrack_Phi, &b_reco_daughter_allTrack_Phi);
   fChain->SetBranchAddress("reco_daughter_allTrack_len", &reco_daughter_allTrack_len, &b_reco_daughter_allTrack_len);
   fChain->SetBranchAddress("reco_daughter_allTrack_alt_len", &reco_daughter_allTrack_alt_len, &b_reco_daughter_allTrack_alt_len);
   fChain->SetBranchAddress("reco_daughter_allTrack_startX", &reco_daughter_allTrack_startX, &b_reco_daughter_allTrack_startX);
   fChain->SetBranchAddress("reco_daughter_allTrack_startY", &reco_daughter_allTrack_startY, &b_reco_daughter_allTrack_startY);
   fChain->SetBranchAddress("reco_daughter_allTrack_startZ", &reco_daughter_allTrack_startZ, &b_reco_daughter_allTrack_startZ);
   fChain->SetBranchAddress("reco_daughter_allTrack_endX", &reco_daughter_allTrack_endX, &b_reco_daughter_allTrack_endX);
   fChain->SetBranchAddress("reco_daughter_allTrack_endY", &reco_daughter_allTrack_endY, &b_reco_daughter_allTrack_endY);
   fChain->SetBranchAddress("reco_daughter_allTrack_endZ", &reco_daughter_allTrack_endZ, &b_reco_daughter_allTrack_endZ);
   fChain->SetBranchAddress("reco_daughter_allTrack_dR", &reco_daughter_allTrack_dR, &b_reco_daughter_allTrack_dR);
   fChain->SetBranchAddress("reco_daughter_allTrack_calo_X", &reco_daughter_allTrack_calo_X, &b_reco_daughter_allTrack_calo_X);
   fChain->SetBranchAddress("reco_daughter_allTrack_calo_Y", &reco_daughter_allTrack_calo_Y, &b_reco_daughter_allTrack_calo_Y);
   fChain->SetBranchAddress("reco_daughter_allTrack_calo_Z", &reco_daughter_allTrack_calo_Z, &b_reco_daughter_allTrack_calo_Z);
   fChain->SetBranchAddress("reco_daughter_allTrack_to_vertex", &reco_daughter_allTrack_to_vertex, &b_reco_daughter_allTrack_to_vertex);
   fChain->SetBranchAddress("reco_daughter_allTrack_vertex_michel_score", &reco_daughter_allTrack_vertex_michel_score, &b_reco_daughter_allTrack_vertex_michel_score);
   fChain->SetBranchAddress("reco_daughter_allTrack_vertex_nHits", &reco_daughter_allTrack_vertex_nHits, &b_reco_daughter_allTrack_vertex_nHits);
   fChain->SetBranchAddress("reco_daughter_allShower_ID", &reco_daughter_allShower_ID, &b_reco_daughter_allShower_ID);
   fChain->SetBranchAddress("reco_daughter_allShower_len", &reco_daughter_allShower_len, &b_reco_daughter_allShower_len);
   fChain->SetBranchAddress("reco_daughter_allShower_startX", &reco_daughter_allShower_startX, &b_reco_daughter_allShower_startX);
   fChain->SetBranchAddress("reco_daughter_allShower_startY", &reco_daughter_allShower_startY, &b_reco_daughter_allShower_startY);
   fChain->SetBranchAddress("reco_daughter_allShower_startZ", &reco_daughter_allShower_startZ, &b_reco_daughter_allShower_startZ);
   fChain->SetBranchAddress("reco_daughter_allShower_dirX", &reco_daughter_allShower_dirX, &b_reco_daughter_allShower_dirX);
   fChain->SetBranchAddress("reco_daughter_allShower_dirY", &reco_daughter_allShower_dirY, &b_reco_daughter_allShower_dirY);
   fChain->SetBranchAddress("reco_daughter_allShower_dirZ", &reco_daughter_allShower_dirZ, &b_reco_daughter_allShower_dirZ);
   fChain->SetBranchAddress("reco_daughter_allShower_energy", &reco_daughter_allShower_energy, &b_reco_daughter_allShower_energy);
   fChain->SetBranchAddress("reco_daughter_PFP_ID", &reco_daughter_PFP_ID, &b_reco_daughter_PFP_ID);
   fChain->SetBranchAddress("reco_daughter_PFP_nHits", &reco_daughter_PFP_nHits, &b_reco_daughter_PFP_nHits);
   fChain->SetBranchAddress("reco_daughter_PFP_nHits_collection", &reco_daughter_PFP_nHits_collection, &b_reco_daughter_PFP_nHits_collection);
   fChain->SetBranchAddress("reco_daughter_PFP_trackScore", &reco_daughter_PFP_trackScore, &b_reco_daughter_PFP_trackScore);
   fChain->SetBranchAddress("reco_daughter_PFP_emScore", &reco_daughter_PFP_emScore, &b_reco_daughter_PFP_emScore);
   fChain->SetBranchAddress("reco_daughter_PFP_michelScore", &reco_daughter_PFP_michelScore, &b_reco_daughter_PFP_michelScore);
   fChain->SetBranchAddress("reco_daughter_PFP_trackScore_collection", &reco_daughter_PFP_trackScore_collection, &b_reco_daughter_PFP_trackScore_collection);
   fChain->SetBranchAddress("reco_daughter_PFP_emScore_collection", &reco_daughter_PFP_emScore_collection, &b_reco_daughter_PFP_emScore_collection);
   fChain->SetBranchAddress("reco_daughter_PFP_michelScore_collection", &reco_daughter_PFP_michelScore_collection, &b_reco_daughter_PFP_michelScore_collection);
   fChain->SetBranchAddress("true_beam_PDG", &true_beam_PDG, &b_true_beam_PDG);
   fChain->SetBranchAddress("true_beam_mass", &true_beam_mass, &b_true_beam_mass);
   fChain->SetBranchAddress("true_beam_ID", &true_beam_ID, &b_true_beam_ID);
   fChain->SetBranchAddress("true_beam_endProcess", &true_beam_endProcess, &b_true_beam_endProcess);
   fChain->SetBranchAddress("true_beam_endX", &true_beam_endX, &b_true_beam_endX);
   fChain->SetBranchAddress("true_beam_endY", &true_beam_endY, &b_true_beam_endY);
   fChain->SetBranchAddress("true_beam_endZ", &true_beam_endZ, &b_true_beam_endZ);
   fChain->SetBranchAddress("true_beam_endX_SCE", &true_beam_endX_SCE, &b_true_beam_endX_SCE);
   fChain->SetBranchAddress("true_beam_endY_SCE", &true_beam_endY_SCE, &b_true_beam_endY_SCE);
   fChain->SetBranchAddress("true_beam_endZ_SCE", &true_beam_endZ_SCE, &b_true_beam_endZ_SCE);
   fChain->SetBranchAddress("true_beam_startX", &true_beam_startX, &b_true_beam_startX);
   fChain->SetBranchAddress("true_beam_startY", &true_beam_startY, &b_true_beam_startY);
   fChain->SetBranchAddress("true_beam_startZ", &true_beam_startZ, &b_true_beam_startZ);
   fChain->SetBranchAddress("true_beam_startPx", &true_beam_startPx, &b_true_beam_startPx);
   fChain->SetBranchAddress("true_beam_startPy", &true_beam_startPy, &b_true_beam_startPy);
   fChain->SetBranchAddress("true_beam_startPz", &true_beam_startPz, &b_true_beam_startPz);
   fChain->SetBranchAddress("true_beam_startP", &true_beam_startP, &b_true_beam_startP);
   fChain->SetBranchAddress("true_beam_endPx", &true_beam_endPx, &b_true_beam_endPx);
   fChain->SetBranchAddress("true_beam_endPy", &true_beam_endPy, &b_true_beam_endPy);
   fChain->SetBranchAddress("true_beam_endPz", &true_beam_endPz, &b_true_beam_endPz);
   fChain->SetBranchAddress("true_beam_endP", &true_beam_endP, &b_true_beam_endP);
   fChain->SetBranchAddress("true_beam_endP2", &true_beam_endP2, &b_true_beam_endP2);
   fChain->SetBranchAddress("true_beam_last_len", &true_beam_last_len, &b_true_beam_last_len);
   fChain->SetBranchAddress("true_beam_startDirX", &true_beam_startDirX, &b_true_beam_startDirX);
   fChain->SetBranchAddress("true_beam_startDirY", &true_beam_startDirY, &b_true_beam_startDirY);
   fChain->SetBranchAddress("true_beam_startDirZ", &true_beam_startDirZ, &b_true_beam_startDirZ);
   fChain->SetBranchAddress("true_beam_nElasticScatters", &true_beam_nElasticScatters, &b_true_beam_nElasticScatters);
   fChain->SetBranchAddress("true_beam_elastic_costheta", &true_beam_elastic_costheta, &b_true_beam_elastic_costheta);
   fChain->SetBranchAddress("true_beam_elastic_X", &true_beam_elastic_X, &b_true_beam_elastic_X);
   fChain->SetBranchAddress("true_beam_elastic_Y", &true_beam_elastic_Y, &b_true_beam_elastic_Y);
   fChain->SetBranchAddress("true_beam_elastic_Z", &true_beam_elastic_Z, &b_true_beam_elastic_Z);
   fChain->SetBranchAddress("true_beam_elastic_deltaE", &true_beam_elastic_deltaE, &b_true_beam_elastic_deltaE);
   fChain->SetBranchAddress("true_beam_elastic_IDE_edep", &true_beam_elastic_IDE_edep, &b_true_beam_elastic_IDE_edep);
   fChain->SetBranchAddress("true_beam_IDE_totalDep", &true_beam_IDE_totalDep, &b_true_beam_IDE_totalDep);
   fChain->SetBranchAddress("true_beam_nHits", &true_beam_nHits, &b_true_beam_nHits);
   fChain->SetBranchAddress("true_beam_reco_byHits_PFP_ID", &true_beam_reco_byHits_PFP_ID, &b_true_beam_reco_byHits_PFP_ID);
   fChain->SetBranchAddress("true_beam_reco_byHits_PFP_nHits", &true_beam_reco_byHits_PFP_nHits, &b_true_beam_reco_byHits_PFP_nHits);
   fChain->SetBranchAddress("true_beam_reco_byHits_allTrack_ID", &true_beam_reco_byHits_allTrack_ID, &b_true_beam_reco_byHits_allTrack_ID);
   fChain->SetBranchAddress("true_daughter_nPi0", &true_daughter_nPi0, &b_true_daughter_nPi0);
   fChain->SetBranchAddress("true_daughter_nPiPlus", &true_daughter_nPiPlus, &b_true_daughter_nPiPlus);
   fChain->SetBranchAddress("true_daughter_nProton", &true_daughter_nProton, &b_true_daughter_nProton);
   fChain->SetBranchAddress("true_daughter_nNeutron", &true_daughter_nNeutron, &b_true_daughter_nNeutron);
   fChain->SetBranchAddress("true_daughter_nPiMinus", &true_daughter_nPiMinus, &b_true_daughter_nPiMinus);
   fChain->SetBranchAddress("true_daughter_nNucleus", &true_daughter_nNucleus, &b_true_daughter_nNucleus);
   fChain->SetBranchAddress("reco_beam_vertex_slice", &reco_beam_vertex_slice, &b_reco_beam_vertex_slice);
   fChain->SetBranchAddress("true_beam_daughter_PDG", &true_beam_daughter_PDG, &b_true_beam_daughter_PDG);
   fChain->SetBranchAddress("true_beam_daughter_ID", &true_beam_daughter_ID, &b_true_beam_daughter_ID);
   fChain->SetBranchAddress("true_beam_daughter_len", &true_beam_daughter_len, &b_true_beam_daughter_len);
   fChain->SetBranchAddress("true_beam_daughter_startX", &true_beam_daughter_startX, &b_true_beam_daughter_startX);
   fChain->SetBranchAddress("true_beam_daughter_startY", &true_beam_daughter_startY, &b_true_beam_daughter_startY);
   fChain->SetBranchAddress("true_beam_daughter_startZ", &true_beam_daughter_startZ, &b_true_beam_daughter_startZ);
   fChain->SetBranchAddress("true_beam_daughter_startPx", &true_beam_daughter_startPx, &b_true_beam_daughter_startPx);
   fChain->SetBranchAddress("true_beam_daughter_startPy", &true_beam_daughter_startPy, &b_true_beam_daughter_startPy);
   fChain->SetBranchAddress("true_beam_daughter_startPz", &true_beam_daughter_startPz, &b_true_beam_daughter_startPz);
   fChain->SetBranchAddress("true_beam_daughter_startP", &true_beam_daughter_startP, &b_true_beam_daughter_startP);
   fChain->SetBranchAddress("true_beam_daughter_endX", &true_beam_daughter_endX, &b_true_beam_daughter_endX);
   fChain->SetBranchAddress("true_beam_daughter_endY", &true_beam_daughter_endY, &b_true_beam_daughter_endY);
   fChain->SetBranchAddress("true_beam_daughter_endZ", &true_beam_daughter_endZ, &b_true_beam_daughter_endZ);
   fChain->SetBranchAddress("true_beam_daughter_Process", &true_beam_daughter_Process, &b_true_beam_daughter_Process);
   fChain->SetBranchAddress("true_beam_daughter_endProcess", &true_beam_daughter_endProcess, &b_true_beam_daughter_endProcess);
   fChain->SetBranchAddress("true_beam_daughter_nHits", &true_beam_daughter_nHits, &b_true_beam_daughter_nHits);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_PFP_ID", &true_beam_daughter_reco_byHits_PFP_ID, &b_true_beam_daughter_reco_byHits_PFP_ID);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_PFP_nHits", &true_beam_daughter_reco_byHits_PFP_nHits, &b_true_beam_daughter_reco_byHits_PFP_nHits);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_PFP_trackScore", &true_beam_daughter_reco_byHits_PFP_trackScore, &b_true_beam_daughter_reco_byHits_PFP_trackScore);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_allTrack_ID", &true_beam_daughter_reco_byHits_allTrack_ID, &b_true_beam_daughter_reco_byHits_allTrack_ID);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_allTrack_startX", &true_beam_daughter_reco_byHits_allTrack_startX, &b_true_beam_daughter_reco_byHits_allTrack_startX);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_allTrack_startY", &true_beam_daughter_reco_byHits_allTrack_startY, &b_true_beam_daughter_reco_byHits_allTrack_startY);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_allTrack_startZ", &true_beam_daughter_reco_byHits_allTrack_startZ, &b_true_beam_daughter_reco_byHits_allTrack_startZ);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_allTrack_endX", &true_beam_daughter_reco_byHits_allTrack_endX, &b_true_beam_daughter_reco_byHits_allTrack_endX);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_allTrack_endY", &true_beam_daughter_reco_byHits_allTrack_endY, &b_true_beam_daughter_reco_byHits_allTrack_endY);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_allTrack_endZ", &true_beam_daughter_reco_byHits_allTrack_endZ, &b_true_beam_daughter_reco_byHits_allTrack_endZ);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_allTrack_len", &true_beam_daughter_reco_byHits_allTrack_len, &b_true_beam_daughter_reco_byHits_allTrack_len);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_allShower_ID", &true_beam_daughter_reco_byHits_allShower_ID, &b_true_beam_daughter_reco_byHits_allShower_ID);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_allShower_startX", &true_beam_daughter_reco_byHits_allShower_startX, &b_true_beam_daughter_reco_byHits_allShower_startX);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_allShower_startY", &true_beam_daughter_reco_byHits_allShower_startY, &b_true_beam_daughter_reco_byHits_allShower_startY);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_allShower_startZ", &true_beam_daughter_reco_byHits_allShower_startZ, &b_true_beam_daughter_reco_byHits_allShower_startZ);
   fChain->SetBranchAddress("true_beam_daughter_reco_byHits_allShower_len", &true_beam_daughter_reco_byHits_allShower_len, &b_true_beam_daughter_reco_byHits_allShower_len);
   fChain->SetBranchAddress("true_beam_Pi0_decay_ID", &true_beam_Pi0_decay_ID, &b_true_beam_Pi0_decay_ID);
   fChain->SetBranchAddress("true_beam_Pi0_decay_parID", &true_beam_Pi0_decay_parID, &b_true_beam_Pi0_decay_parID);
   fChain->SetBranchAddress("true_beam_Pi0_decay_PDG", &true_beam_Pi0_decay_PDG, &b_true_beam_Pi0_decay_PDG);
   fChain->SetBranchAddress("true_beam_Pi0_decay_startP", &true_beam_Pi0_decay_startP, &b_true_beam_Pi0_decay_startP);
   fChain->SetBranchAddress("true_beam_Pi0_decay_startPx", &true_beam_Pi0_decay_startPx, &b_true_beam_Pi0_decay_startPx);
   fChain->SetBranchAddress("true_beam_Pi0_decay_startPy", &true_beam_Pi0_decay_startPy, &b_true_beam_Pi0_decay_startPy);
   fChain->SetBranchAddress("true_beam_Pi0_decay_startPz", &true_beam_Pi0_decay_startPz, &b_true_beam_Pi0_decay_startPz);
   fChain->SetBranchAddress("true_beam_Pi0_decay_startX", &true_beam_Pi0_decay_startX, &b_true_beam_Pi0_decay_startX);
   fChain->SetBranchAddress("true_beam_Pi0_decay_startY", &true_beam_Pi0_decay_startY, &b_true_beam_Pi0_decay_startY);
   fChain->SetBranchAddress("true_beam_Pi0_decay_startZ", &true_beam_Pi0_decay_startZ, &b_true_beam_Pi0_decay_startZ);
   fChain->SetBranchAddress("true_beam_Pi0_decay_len", &true_beam_Pi0_decay_len, &b_true_beam_Pi0_decay_len);
   fChain->SetBranchAddress("true_beam_Pi0_decay_nHits", &true_beam_Pi0_decay_nHits, &b_true_beam_Pi0_decay_nHits);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_PFP_ID", &true_beam_Pi0_decay_reco_byHits_PFP_ID, &b_true_beam_Pi0_decay_reco_byHits_PFP_ID);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_PFP_nHits", &true_beam_Pi0_decay_reco_byHits_PFP_nHits, &b_true_beam_Pi0_decay_reco_byHits_PFP_nHits);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_PFP_trackScore", &true_beam_Pi0_decay_reco_byHits_PFP_trackScore, &b_true_beam_Pi0_decay_reco_byHits_PFP_trackScore);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_allTrack_ID", &true_beam_Pi0_decay_reco_byHits_allTrack_ID, &b_true_beam_Pi0_decay_reco_byHits_allTrack_ID);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_allTrack_startX", &true_beam_Pi0_decay_reco_byHits_allTrack_startX, &b_true_beam_Pi0_decay_reco_byHits_allTrack_startX);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_allTrack_startY", &true_beam_Pi0_decay_reco_byHits_allTrack_startY, &b_true_beam_Pi0_decay_reco_byHits_allTrack_startY);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_allTrack_startZ", &true_beam_Pi0_decay_reco_byHits_allTrack_startZ, &b_true_beam_Pi0_decay_reco_byHits_allTrack_startZ);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_allTrack_endX", &true_beam_Pi0_decay_reco_byHits_allTrack_endX, &b_true_beam_Pi0_decay_reco_byHits_allTrack_endX);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_allTrack_endY", &true_beam_Pi0_decay_reco_byHits_allTrack_endY, &b_true_beam_Pi0_decay_reco_byHits_allTrack_endY);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_allTrack_endZ", &true_beam_Pi0_decay_reco_byHits_allTrack_endZ, &b_true_beam_Pi0_decay_reco_byHits_allTrack_endZ);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_allTrack_len", &true_beam_Pi0_decay_reco_byHits_allTrack_len, &b_true_beam_Pi0_decay_reco_byHits_allTrack_len);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_allShower_ID", &true_beam_Pi0_decay_reco_byHits_allShower_ID, &b_true_beam_Pi0_decay_reco_byHits_allShower_ID);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_allShower_startX", &true_beam_Pi0_decay_reco_byHits_allShower_startX, &b_true_beam_Pi0_decay_reco_byHits_allShower_startX);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_allShower_startY", &true_beam_Pi0_decay_reco_byHits_allShower_startY, &b_true_beam_Pi0_decay_reco_byHits_allShower_startY);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_allShower_startZ", &true_beam_Pi0_decay_reco_byHits_allShower_startZ, &b_true_beam_Pi0_decay_reco_byHits_allShower_startZ);
   fChain->SetBranchAddress("true_beam_Pi0_decay_reco_byHits_allShower_len", &true_beam_Pi0_decay_reco_byHits_allShower_len, &b_true_beam_Pi0_decay_reco_byHits_allShower_len);
   fChain->SetBranchAddress("true_beam_grand_daughter_ID", &true_beam_grand_daughter_ID, &b_true_beam_grand_daughter_ID);
   fChain->SetBranchAddress("true_beam_grand_daughter_parID", &true_beam_grand_daughter_parID, &b_true_beam_grand_daughter_parID);
   fChain->SetBranchAddress("true_beam_grand_daughter_PDG", &true_beam_grand_daughter_PDG, &b_true_beam_grand_daughter_PDG);
   fChain->SetBranchAddress("true_beam_grand_daughter_nHits", &true_beam_grand_daughter_nHits, &b_true_beam_grand_daughter_nHits);
   fChain->SetBranchAddress("true_beam_grand_daughter_Process", &true_beam_grand_daughter_Process, &b_true_beam_grand_daughter_Process);
   fChain->SetBranchAddress("true_beam_grand_daughter_endProcess", &true_beam_grand_daughter_endProcess, &b_true_beam_grand_daughter_endProcess);
   fChain->SetBranchAddress("reco_beam_true_byE_endProcess", &reco_beam_true_byE_endProcess, &b_reco_beam_true_byE_endProcess);
   fChain->SetBranchAddress("reco_beam_true_byE_process", &reco_beam_true_byE_process, &b_reco_beam_true_byE_process);
   fChain->SetBranchAddress("reco_beam_true_byE_origin", &reco_beam_true_byE_origin, &b_reco_beam_true_byE_origin);
   fChain->SetBranchAddress("reco_beam_true_byE_PDG", &reco_beam_true_byE_PDG, &b_reco_beam_true_byE_PDG);
   fChain->SetBranchAddress("reco_beam_true_byE_ID", &reco_beam_true_byE_ID, &b_reco_beam_true_byE_ID);
   fChain->SetBranchAddress("reco_beam_true_byHits_endProcess", &reco_beam_true_byHits_endProcess, &b_reco_beam_true_byHits_endProcess);
   fChain->SetBranchAddress("reco_beam_true_byHits_process", &reco_beam_true_byHits_process, &b_reco_beam_true_byHits_process);
   fChain->SetBranchAddress("reco_beam_true_byHits_origin", &reco_beam_true_byHits_origin, &b_reco_beam_true_byHits_origin);
   fChain->SetBranchAddress("reco_beam_true_byHits_PDG", &reco_beam_true_byHits_PDG, &b_reco_beam_true_byHits_PDG);
   fChain->SetBranchAddress("reco_beam_true_byHits_ID", &reco_beam_true_byHits_ID, &b_reco_beam_true_byHits_ID);
   fChain->SetBranchAddress("reco_beam_true_byE_matched", &reco_beam_true_byE_matched, &b_reco_beam_true_byE_matched);
   fChain->SetBranchAddress("reco_beam_true_byHits_matched", &reco_beam_true_byHits_matched, &b_reco_beam_true_byHits_matched);
   fChain->SetBranchAddress("reco_beam_true_byHits_purity", &reco_beam_true_byHits_purity, &b_reco_beam_true_byHits_purity);
   fChain->SetBranchAddress("true_beam_processes", &true_beam_processes, &b_true_beam_processes);
   fChain->SetBranchAddress("beam_inst_P", &beam_inst_P, &b_beam_inst_P);
   fChain->SetBranchAddress("beam_inst_TOF", &beam_inst_TOF, &b_beam_inst_TOF);
   fChain->SetBranchAddress("beam_inst_TOF_Chan", &beam_inst_TOF_Chan, &b_beam_inst_TOF_Chan);
   fChain->SetBranchAddress("beam_inst_X", &beam_inst_X, &b_beam_inst_X);
   fChain->SetBranchAddress("beam_inst_Y", &beam_inst_Y, &b_beam_inst_Y);
   fChain->SetBranchAddress("beam_inst_Z", &beam_inst_Z, &b_beam_inst_Z);
   fChain->SetBranchAddress("beam_inst_dirX", &beam_inst_dirX, &b_beam_inst_dirX);
   fChain->SetBranchAddress("beam_inst_dirY", &beam_inst_dirY, &b_beam_inst_dirY);
   fChain->SetBranchAddress("beam_inst_dirZ", &beam_inst_dirZ, &b_beam_inst_dirZ);
   fChain->SetBranchAddress("beam_inst_nFibersP1", &beam_inst_nFibersP1, &b_beam_inst_nFibersP1);
   fChain->SetBranchAddress("beam_inst_nFibersP2", &beam_inst_nFibersP2, &b_beam_inst_nFibersP2);
   fChain->SetBranchAddress("beam_inst_nFibersP3", &beam_inst_nFibersP3, &b_beam_inst_nFibersP3);
   fChain->SetBranchAddress("beam_inst_PDG_candidates", &beam_inst_PDG_candidates, &b_beam_inst_PDG_candidates);
   fChain->SetBranchAddress("beam_inst_nTracks", &beam_inst_nTracks, &b_beam_inst_nTracks);
   fChain->SetBranchAddress("beam_inst_nMomenta", &beam_inst_nMomenta, &b_beam_inst_nMomenta);
   fChain->SetBranchAddress("beam_inst_valid", &beam_inst_valid, &b_beam_inst_valid);
   fChain->SetBranchAddress("beam_inst_trigger", &beam_inst_trigger, &b_beam_inst_trigger);
   fChain->SetBranchAddress("reco_beam_Chi2_proton", &reco_beam_Chi2_proton, &b_reco_beam_Chi2_proton);
   fChain->SetBranchAddress("reco_beam_Chi2_ndof", &reco_beam_Chi2_ndof, &b_reco_beam_Chi2_ndof);
   fChain->SetBranchAddress("reco_daughter_allTrack_momByRange_proton", &reco_daughter_allTrack_momByRange_proton, &b_reco_daughter_allTrack_momByRange_proton);
   fChain->SetBranchAddress("reco_daughter_allTrack_momByRange_muon", &reco_daughter_allTrack_momByRange_muon, &b_reco_daughter_allTrack_momByRange_muon);
   fChain->SetBranchAddress("reco_beam_momByRange_proton", &reco_beam_momByRange_proton, &b_reco_beam_momByRange_proton);
   fChain->SetBranchAddress("reco_beam_momByRange_muon", &reco_beam_momByRange_muon, &b_reco_beam_momByRange_muon);
   fChain->SetBranchAddress("reco_daughter_allTrack_momByRange_alt_proton", &reco_daughter_allTrack_momByRange_alt_proton, &b_reco_daughter_allTrack_momByRange_alt_proton);
   fChain->SetBranchAddress("reco_daughter_allTrack_momByRange_alt_muon", &reco_daughter_allTrack_momByRange_alt_muon, &b_reco_daughter_allTrack_momByRange_alt_muon);
   fChain->SetBranchAddress("reco_beam_momByRange_alt_proton", &reco_beam_momByRange_alt_proton, &b_reco_beam_momByRange_alt_proton);
   fChain->SetBranchAddress("reco_beam_momByRange_alt_muon", &reco_beam_momByRange_alt_muon, &b_reco_beam_momByRange_alt_muon);
   fChain->SetBranchAddress("reco_beam_true_byE_endPx", &reco_beam_true_byE_endPx, &b_reco_beam_true_byE_endPx);
   fChain->SetBranchAddress("reco_beam_true_byE_endPy", &reco_beam_true_byE_endPy, &b_reco_beam_true_byE_endPy);
   fChain->SetBranchAddress("reco_beam_true_byE_endPz", &reco_beam_true_byE_endPz, &b_reco_beam_true_byE_endPz);
   fChain->SetBranchAddress("reco_beam_true_byE_endE", &reco_beam_true_byE_endE, &b_reco_beam_true_byE_endE);
   fChain->SetBranchAddress("reco_beam_true_byE_endP", &reco_beam_true_byE_endP, &b_reco_beam_true_byE_endP);
   fChain->SetBranchAddress("reco_beam_true_byE_startPx", &reco_beam_true_byE_startPx, &b_reco_beam_true_byE_startPx);
   fChain->SetBranchAddress("reco_beam_true_byE_startPy", &reco_beam_true_byE_startPy, &b_reco_beam_true_byE_startPy);
   fChain->SetBranchAddress("reco_beam_true_byE_startPz", &reco_beam_true_byE_startPz, &b_reco_beam_true_byE_startPz);
   fChain->SetBranchAddress("reco_beam_true_byE_startE", &reco_beam_true_byE_startE, &b_reco_beam_true_byE_startE);
   fChain->SetBranchAddress("reco_beam_true_byE_startP", &reco_beam_true_byE_startP, &b_reco_beam_true_byE_startP);
   fChain->SetBranchAddress("reco_beam_true_byHits_endPx", &reco_beam_true_byHits_endPx, &b_reco_beam_true_byHits_endPx);
   fChain->SetBranchAddress("reco_beam_true_byHits_endPy", &reco_beam_true_byHits_endPy, &b_reco_beam_true_byHits_endPy);
   fChain->SetBranchAddress("reco_beam_true_byHits_endPz", &reco_beam_true_byHits_endPz, &b_reco_beam_true_byHits_endPz);
   fChain->SetBranchAddress("reco_beam_true_byHits_endE", &reco_beam_true_byHits_endE, &b_reco_beam_true_byHits_endE);
   fChain->SetBranchAddress("reco_beam_true_byHits_endP", &reco_beam_true_byHits_endP, &b_reco_beam_true_byHits_endP);
   fChain->SetBranchAddress("reco_beam_true_byHits_startPx", &reco_beam_true_byHits_startPx, &b_reco_beam_true_byHits_startPx);
   fChain->SetBranchAddress("reco_beam_true_byHits_startPy", &reco_beam_true_byHits_startPy, &b_reco_beam_true_byHits_startPy);
   fChain->SetBranchAddress("reco_beam_true_byHits_startPz", &reco_beam_true_byHits_startPz, &b_reco_beam_true_byHits_startPz);
   fChain->SetBranchAddress("reco_beam_true_byHits_startE", &reco_beam_true_byHits_startE, &b_reco_beam_true_byHits_startE);
   fChain->SetBranchAddress("reco_beam_true_byHits_startP", &reco_beam_true_byHits_startP, &b_reco_beam_true_byHits_startP);
   fChain->SetBranchAddress("reco_beam_incidentEnergies", &reco_beam_incidentEnergies, &b_reco_beam_incidentEnergies);
   fChain->SetBranchAddress("reco_beam_interactingEnergy", &reco_beam_interactingEnergy, &b_reco_beam_interactingEnergy);
   fChain->SetBranchAddress("true_beam_incidentEnergies", &true_beam_incidentEnergies, &b_true_beam_incidentEnergies);
   fChain->SetBranchAddress("true_beam_interactingEnergy", &true_beam_interactingEnergy, &b_true_beam_interactingEnergy);
   fChain->SetBranchAddress("true_beam_slices", &true_beam_slices, &b_true_beam_slices);
   fChain->SetBranchAddress("true_beam_slices_found", &true_beam_slices_found, &b_true_beam_slices_found);
   fChain->SetBranchAddress("true_beam_slices_deltaE", &true_beam_slices_deltaE, &b_true_beam_slices_deltaE);
   fChain->SetBranchAddress("em_energy", &em_energy, &b_em_energy);
   fChain->SetBranchAddress("true_beam_traj_X", &true_beam_traj_X, &b_true_beam_traj_X);
   fChain->SetBranchAddress("true_beam_traj_Y", &true_beam_traj_Y, &b_true_beam_traj_Y);
   fChain->SetBranchAddress("true_beam_traj_Z", &true_beam_traj_Z, &b_true_beam_traj_Z);
   fChain->SetBranchAddress("true_beam_traj_Px", &true_beam_traj_Px, &b_true_beam_traj_Px);
   fChain->SetBranchAddress("true_beam_traj_Py", &true_beam_traj_Py, &b_true_beam_traj_Py);
   fChain->SetBranchAddress("true_beam_traj_Pz", &true_beam_traj_Pz, &b_true_beam_traj_Pz);
   fChain->SetBranchAddress("true_beam_traj_KE", &true_beam_traj_KE, &b_true_beam_traj_KE);
   fChain->SetBranchAddress("true_beam_traj_X_SCE", &true_beam_traj_X_SCE, &b_true_beam_traj_X_SCE);
   fChain->SetBranchAddress("true_beam_traj_Y_SCE", &true_beam_traj_Y_SCE, &b_true_beam_traj_Y_SCE);
   fChain->SetBranchAddress("true_beam_traj_Z_SCE", &true_beam_traj_Z_SCE, &b_true_beam_traj_Z_SCE);
   fChain->SetBranchAddress("g4rw_primary_weights", &g4rw_primary_weights, &b_g4rw_primary_weights);
   fChain->SetBranchAddress("g4rw_primary_plus_sigma_weight", &g4rw_primary_plus_sigma_weight, &b_g4rw_primary_plus_sigma_weight);
   fChain->SetBranchAddress("g4rw_primary_minus_sigma_weight", &g4rw_primary_minus_sigma_weight, &b_g4rw_primary_minus_sigma_weight);
   fChain->SetBranchAddress("g4rw_primary_var", &g4rw_primary_var, &b_g4rw_primary_var);
   fChain->SetBranchAddress("g4rw_alt_primary_plus_sigma_weight", &g4rw_alt_primary_plus_sigma_weight, &b_g4rw_alt_primary_plus_sigma_weight);
   fChain->SetBranchAddress("g4rw_alt_primary_minus_sigma_weight", &g4rw_alt_primary_minus_sigma_weight, &b_g4rw_alt_primary_minus_sigma_weight);
   fChain->SetBranchAddress("g4rw_full_primary_plus_sigma_weight", &g4rw_full_primary_plus_sigma_weight, &b_g4rw_full_primary_plus_sigma_weight);
   fChain->SetBranchAddress("g4rw_full_primary_minus_sigma_weight", &g4rw_full_primary_minus_sigma_weight, &b_g4rw_full_primary_minus_sigma_weight);
   fChain->SetBranchAddress("g4rw_full_grid_weights", &g4rw_full_grid_weights, &b_g4rw_full_grid_weights);
   fChain->SetBranchAddress("g4rw_full_grid_piplus_weights", &g4rw_full_grid_piplus_weights, &b_g4rw_full_grid_piplus_weights);
   fChain->SetBranchAddress("g4rw_full_grid_piplus_weights_fake_data", &g4rw_full_grid_piplus_weights_fake_data, &b_g4rw_full_grid_piplus_weights_fake_data);
   fChain->SetBranchAddress("g4rw_full_grid_piminus_weights", &g4rw_full_grid_piminus_weights, &b_g4rw_full_grid_piminus_weights);
   fChain->SetBranchAddress("g4rw_full_grid_proton_weights", &g4rw_full_grid_proton_weights, &b_g4rw_full_grid_proton_weights);
   fChain->SetBranchAddress("g4rw_primary_grid_weights", &g4rw_primary_grid_weights, &b_g4rw_primary_grid_weights);
   fChain->SetBranchAddress("g4rw_primary_grid_pair_weights", &g4rw_primary_grid_pair_weights, &b_g4rw_primary_grid_pair_weights);
   fChain->SetBranchAddress("reco_beam_spacePts_X", &reco_beam_spacePts_X, &b_reco_beam_spacePts_X);
   fChain->SetBranchAddress("reco_beam_spacePts_Y", &reco_beam_spacePts_Y, &b_reco_beam_spacePts_Y);
   fChain->SetBranchAddress("reco_beam_spacePts_Z", &reco_beam_spacePts_Z, &b_reco_beam_spacePts_Z);
   fChain->SetBranchAddress("reco_daughter_spacePts_X", &reco_daughter_spacePts_X, &b_reco_daughter_spacePts_X);
   fChain->SetBranchAddress("reco_daughter_spacePts_Y", &reco_daughter_spacePts_Y, &b_reco_daughter_spacePts_Y);
   fChain->SetBranchAddress("reco_daughter_spacePts_Z", &reco_daughter_spacePts_Z, &b_reco_daughter_spacePts_Z);
   fChain->SetBranchAddress("reco_daughter_shower_spacePts_X", &reco_daughter_shower_spacePts_X, &b_reco_daughter_shower_spacePts_X);
   fChain->SetBranchAddress("reco_daughter_shower_spacePts_Y", &reco_daughter_shower_spacePts_Y, &b_reco_daughter_shower_spacePts_Y);
   fChain->SetBranchAddress("reco_daughter_shower_spacePts_Z", &reco_daughter_shower_spacePts_Z, &b_reco_daughter_shower_spacePts_Z);
   Notify();
}

Bool_t beamana::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return kTRUE;
}

void beamana::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t beamana::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef beamana_cxx
