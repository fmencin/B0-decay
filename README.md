# B
Full event reconstruction on B0, anti B0 decays.

## Files
### steering.py
Steering file for B0 decays. It uses FEI to reconstruct B0 from 32 different decay channels, giving us Mbc, deltaE, isSignal (for MC), SigProb, FEIProbRank, cosTBTO, R2 and decaymode for every B0 that could have decayed. It should be used on mixed .root files for signal and charged, ccbar, ssbar, ddbar, uubar .root files for continuum. This was used belle2 MC14a. It returns us a .root file with calculated variables for each .root file it was used on.

### analysis.py
The main part of our analysis. 