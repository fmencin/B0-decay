import os
import numpy as np 
from root_pandas import read_root


variables = os.listdir("/home/belle2/fmencin/B/mixed/sig_and_cont/variabs2/")

text = open("cutted_variables.txt", "w")
text.write("#Mbc\tisSignal\tdeltaE\tSigProb\tFEIProbRank\tcosTBTO\tR2\tdecaymode\tsig_or_cont\n")

#sig_or_cont: 0 = mixed, 1 = charged, 2 = uubar, 3 = ddbar, 4 = ssbar, 5 = ccbar, 6 = mumu, 7 = taupair


for variable in variables:
	if variable[0:1] == 'v':
		print(variable)
		df = read_root("/home/belle2/fmencin/B/mixed/sig_and_cont/variabs2/"+variable)

		Mbc = df["Mbc"][0:].to_numpy()
		isSignal = df["isSignal"][0:].to_numpy()
		deltaE = df["deltaE"][0:].to_numpy()
		SigProb = df["SigProb"][0:].to_numpy()
		FEIProbRank = df["FEIProbRank"][0:].to_numpy()
		cosTBTO = df["cosTBTO"][0:].to_numpy()
		R2 = df["R2"][0:].to_numpy()
		decaymode = df["decayModeID"][0:].to_numpy()
		
		if variable[10:-9] == "charged":
			sig_or_cont = np.ones(len(decaymode))

		elif variable[10:-9] == "uubar":
			sig_or_cont = np.ones(len(decaymode)) * 2

		elif variable[10:-9] == "ddbar":
                        sig_or_cont = np.ones(len(decaymode)) * 3

		elif variable[10:-9] == "ssbar":
                        sig_or_cont = np.ones(len(decaymode)) * 4

		elif variable[10:-9] == "ccbar":
                        sig_or_cont = np.ones(len(decaymode)) * 5

		elif variable[10:-9] == "mumu":
                        sig_or_cont = np.ones(len(decaymode)) * 6

		elif variable[10:-9] == "taupair":
                        sig_or_cont = np.ones(len(decaymode)) * 7

		else:
			sig_or_cont = np.zeros(len(decaymode))


		for i in range(len(Mbc)):
			text.write(str(Mbc[i]) + "\t" + str(isSignal[i]) + "\t" + str(deltaE[i]) + "\t" + str(SigProb[i]) + "\t" + \
				str(FEIProbRank[i]) + "\t" + str(cosTBTO[i]) + "\t" + str(R2[i]) + "\t" + str(decaymode[i]) + "\t" + \
				str(sig_or_cont[i]) + "\n")







