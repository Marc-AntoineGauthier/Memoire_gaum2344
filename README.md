# Memoire Marc-Antoine Gauthier
This repository contains the code necessary to reproduce the data from my memoire (https://savoirs.usherbrooke.ca/handle/11143/21591).

The "DOS" folder contains the scripts used to compute de Density of State and the "specific_heat" folder contains the files to do specific heat calculation on a supercomputer.

The data folder is empty for now, but it will contain some data file to be used with various scripts.

Here are some of the files that are not in those folders :
* "BCS_T_old.py" : Computes the temperature dependence of the order parameter with the old method.
* "BCS_T.py" : Computes the temperature dependence of the order parameter with the newer method.
* "chara_table_term.py" : Contains some examples of verification of irrep.
* "gap_con.py" : Contains the H and the SCOPs.
* "gap_pseudo.py" : Contains the H and the SCOPs but in the pseudospin base.
* "group_trans.py" : Definition and functions to verify irrep.
* "handshake.py" : Does the plots of the agreement of the two temperature regimes of the specific heat.
* "k_gen.py" : Contains the generators for the k-grid.
* "pseudo_gap_theta.py" : Computes the gap theta.
* "pseudospin_trans.py" : This script helps to visualize the pseudospin transformation.
* "trs_test.py" : Verifies the time reversal symmetry.
