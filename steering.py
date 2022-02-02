import basf2 as b2
import fei
import modularAnalysis as ma
import sys
from variables import variables as vm

name = sys.argv[1]

main = b2.Path()

ma.inputMdst(filename = name, path = main, environmentType = "default")

b2.conditions.prepend_globaltag(ma.getAnalysisGlobaltag())


particles = fei.get_default_channels(
	B_extra_cut = 'Mbc > 5.22',
	chargedB = False,
	neutralB = True,
	hadronic = True,
	semileptonic = False,
	baryonic = False,
)

configuration = fei.FeiConfiguration(prefix = "FEIv4_2021_MC14_release_05_01_12", monitor = False)
postcutconfig = fei.PostCutConfiguration(value = 0.015)

feistate = fei.get_path(particles, configuration)

main.add_path(feistate.path)

ma.applyCuts(list_name = 'B0:generic', cut = 'extraInfo(SignalProbability) > 0.015', path = main)

ma.matchMCTruth(list_name = 'B0:generic', path = main)

ma.rankByHighest(
        particleList = "B0:generic",
        variable = "extraInfo(SignalProbability)",
        outputVariable = "FEIProbabilityRank",
        path=main,
)

ma.buildRestOfEvent("B0:generic", path = main)
cleanMask = ("cleanMask", "nCDCHits > 0 and useCMSFrame(p)<=3.2", "p>=0.05 and useCMSFrame(p)<=3.2")
ma.appendROEMasks(list_name = "B0:generic", mask_tuples=[cleanMask], path = main)
ma.buildContinuumSuppression(list_name = "B0:generic", roe_mask = "cleanMask", path = main)

vm.addAlias("decayModeID", "extraInfo(decayModeID)")
vm.addAlias("SigProb", "extraInfo(SignalProbability)")
vm.addAlias("FEIProbRank", "extraInfo(FEIProbabilityRank)")

ma.variablesToNtuple(
	'B0:generic',
	[
		'Mbc',
		'deltaE',
		'mcErrors',
		'SigProb',
		'decayModeID',
		'FEIProbRank',
		'isSignal',
		'R2',
		'cosTBTO',
	],
	filename = "variables_"+name[-37:-34]+".root",
	path = main,
)

b2.process(main)
print(b2.statistics)
