# PathWeightSampling-Langevin
Path Weight Sampling implementation for Langevin systems in presence of feedback.
Uses Onsager-Machlup formulation for stochastic actions and the Rosenbluth-Rosenbluth method for importance sampling.

Needs Python3, numpy, numba.

To run:
nohup python feedbackpws_rr4.py &

Output gets saved as nohup.out.
Output is an array where columns correspond to chosen trajectory durations, rows correspond alternately to mutual information and time-integrated transfer entropies, for each of a chosen number of trajectories.
