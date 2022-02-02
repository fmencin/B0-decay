# B
Full event reconstruction on B0, anti B0 decays.

## Files
### steering.py
Steering file for B0 decays. It uses FEI to reconstruct B0 from 32 different decay channels, giving us Mbc, deltaE, isSignal (for MC), SigProb, FEIProbRank, cosTBTO, R2 and decaymode for every B0 that could have decayed. It should be used on mixed .root files for signal and charged, ccbar, ssbar, ddbar, uubar .root files for continuum. This was used belle2 MC14a. It returns us a .root file with calculated variables for each .root file it was used on.

### root_to_text.py
Transforms .root files to .txt files for easier processing.

### analysis.py
The main part of our analysis. It's used for calculating optimal cuts on variables from figure of merit (FOM), for plotting distributions of different variables and, ultimately, it's used to calculate ROC (efficiency vs purity for different SigProbs) curves. Based on these ROC curves we can determine which decaymodes are useful and which are not.