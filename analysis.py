import matplotlib.pyplot as plt
import numpy as np


global mixed_lumin, charged_lumin, uubar_lumin, ddbar_lumin, ssbar_lumin, ccbar_lumin, Nevents

mixed_lumin = 0.0371 
charged_lumin = 0.0350
uubar_lumin = 0.0183
ddbar_lumin = 0.0733
ssbar_lumin = 0.0768
ccbar_lumin = 0.0174
Nevents = 50943396 

def FOM(index, xmin, xmax, nbins, mixed, charged, uubar, ddbar, ssbar, ccbar):

	index = int(index)
	nbins = int(nbins)

	mixed_hist, bins = np.histogram(np.abs(mixed[index]), range = (xmin, xmax), bins = nbins)
	charged_hist, bins = np.histogram(np.abs(charged[index]), range = (xmin, xmax), bins = nbins)
	uubar_hist, bins= np.histogram(np.abs(uubar[index]), range = (xmin, xmax), bins = nbins)
	ddbar_hist, bins= np.histogram(np.abs(ddbar[index]), range = (xmin, xmax), bins = nbins)
	ssbar_hist, bins= np.histogram(np.abs(ssbar[index]), range = (xmin, xmax), bins = nbins)
	ccbar_hist, bins= np.histogram(np.abs(ccbar[index]), range = (xmin, xmax), bins = nbins)
	
	mixed_hist = mixed_hist/mixed_lumin
	charged_hist = charged_hist/charged_lumin
	uubar_hist = uubar_hist/uubar_lumin
	ddbar_hist = ddbar_hist/ddbar_lumin
	ssbar_hist = ssbar_hist/ssbar_lumin
	ccbar_hist = ccbar_hist/ccbar_lumin



	all_hist = mixed_hist + charged_hist + uubar_hist + ddbar_hist + ssbar_hist + ccbar_hist

	


	isSignal_mixed = mixed[1]
	mixed_wisSignal = np.array([])
	
	for i in range(len(isSignal_mixed)):
		if isSignal_mixed[i] == 1:
			mixed_wisSignal = np.append(mixed_wisSignal, np.abs(mixed[index][i]))
	
	mixed_hist_isSignal, bins = np.histogram(mixed_wisSignal, range = (xmin, xmax), bins = nbins)
	mixed_hist_isSignal = mixed_hist_isSignal/mixed_lumin
	
	FOM = np.array([])
	
	#Pazi, ce gledas vecje od nekje dalje, more biti [i:] (kot recimo pri Mbc), ce gledas pa manjse, mora biti pa [:i] (kot recimo pri R2)
	for i in range(len(all_hist)):
		Nsig = np.sum(mixed_hist_isSignal[:i])
		Nb = np.sum(all_hist[:i]) - Nsig
		
		print(Nsig, Nb)

		y = Nsig / np.sqrt(Nsig + Nb)
		FOM = np.append(FOM, y)		
	
	
	if index == 0:
		variable = "Mbc"

	elif index == 1:
		variable = "isSignal"

	elif index == 2:
		variable = "abs(deltaE)"
	
	elif index == 3:
		variable = "SigProb"

	elif index == 4:
		variable = "FEIProbRank"

	elif index == 5:
		variable = "cosTBTO"

	elif index == 6:
		variable = "R2"

	elif index == 7:
		variable = "decaymode"
	
	i = 0
	while np.isnan(FOM[i]) == True:
		i+= 1

	maksimum = np.max(FOM[i:])
	maxbin = bins[:-1][np.where(FOM == maksimum)]
	print(maksimum, maxbin)
	print(FOM)
	
	plt.plot([maxbin for i in np.linspace(min(FOM[i:]), max(FOM[i:]), 5)], np.linspace(min(FOM[i:]), max(FOM[i:]), 5), color = 'gray', alpha = 0.75, ls = 'dashed', label = variable+" = %.2f" % maxbin)	
	plt.plot(bins[:-1], FOM)
	plt.title("FOM "+variable)
	plt.xlabel(variable)
	plt.ylabel("sig/sqrt(sig+bkg)")
	plt.legend()
	plt.savefig("FOM_"+variable+"_drugi.png")
	plt.close()
	
	return()





def separated(Mbc, isSignal, deltaE, SigProb, FEIProbRank, cosTBTO, R2, decaymode, sig_or_cont):
	Mbc_charged = np.array([])
	Mbc_mixed = np.array([])
	Mbc_uubar = np.array([])
	Mbc_ddbar = np.array([])
	Mbc_ssbar = np.array([])
	Mbc_ccbar = np.array([])

	isSignal_charged = np.array([])
	isSignal_mixed = np.array([])
	isSignal_uubar = np.array([])
	isSignal_ddbar = np.array([])
	isSignal_ssbar = np.array([])
	isSignal_ccbar = np.array([])

	deltaE_charged = np.array([])
	deltaE_mixed = np.array([])
	deltaE_uubar = np.array([])
	deltaE_ddbar = np.array([])
	deltaE_ssbar = np.array([])
	deltaE_ccbar = np.array([])

	SigProb_charged = np.array([])
	SigProb_mixed = np.array([])
	SigProb_uubar = np.array([])
	SigProb_ddbar = np.array([])
	SigProb_ssbar = np.array([])
	SigProb_ccbar = np.array([])

	FEIProbRank_charged = np.array([])
	FEIProbRank_mixed = np.array([])
	FEIProbRank_uubar = np.array([])
	FEIProbRank_ddbar = np.array([])
	FEIProbRank_ssbar = np.array([])
	FEIProbRank_ccbar = np.array([])

	cosTBTO_charged = np.array([])
	cosTBTO_mixed = np.array([])
	cosTBTO_uubar = np.array([])
	cosTBTO_ddbar = np.array([])
	cosTBTO_ssbar = np.array([])
	cosTBTO_ccbar = np.array([])

	R2_charged = np.array([])
	R2_mixed = np.array([])
	R2_uubar = np.array([])
	R2_ddbar = np.array([])
	R2_ssbar = np.array([])
	R2_ccbar = np.array([])

	decaymode_charged = np.array([])
	decaymode_mixed = np.array([])
	decaymode_uubar = np.array([])
	decaymode_ddbar = np.array([])
	decaymode_ssbar = np.array([])
	decaymode_ccbar = np.array([])



	for i in range(len(Mbc)):
		if sig_or_cont[i] == 2:
			Mbc_uubar = np.append(Mbc_uubar, Mbc[i])
			isSignal_uubar = np.append(isSignal_uubar, isSignal[i])
			deltaE_uubar = np.append(deltaE_uubar, deltaE[i])
			SigProb_uubar = np.append(SigProb_uubar, SigProb[i])
			FEIProbRank_uubar = np.append(FEIProbRank_uubar, FEIProbRank[i])
			cosTBTO_uubar = np.append(cosTBTO_uubar, cosTBTO[i])
			R2_uubar = np.append(R2_uubar, R2[i])
			decaymode_uubar = np.append(decaymode_uubar, decaymode[i])


		elif sig_or_cont[i] == 3:
			Mbc_ddbar = np.append(Mbc_ddbar, Mbc[i])
			isSignal_ddbar = np.append(isSignal_ddbar, isSignal[i])
			deltaE_ddbar = np.append(deltaE_ddbar, deltaE[i])	
			SigProb_ddbar = np.append(SigProb_ddbar, SigProb[i])
			FEIProbRank_ddbar = np.append(FEIProbRank_ddbar, FEIProbRank[i])
			cosTBTO_ddbar = np.append(cosTBTO_ddbar, cosTBTO[i])
			R2_ddbar = np.append(R2_ddbar, R2[i])
			decaymode_ddbar = np.append(decaymode_ddbar, decaymode[i])		

		elif sig_or_cont[i] == 4:
			Mbc_ssbar = np.append(Mbc_ssbar, Mbc[i])
			isSignal_ssbar = np.append(isSignal_ssbar, isSignal[i])
			deltaE_ssbar = np.append(deltaE_ssbar, deltaE[i])
			SigProb_ssbar = np.append(SigProb_ssbar, SigProb[i])
			FEIProbRank_ssbar = np.append(FEIProbRank_ssbar, FEIProbRank[i])
			cosTBTO_ssbar = np.append(cosTBTO_ssbar, cosTBTO[i])
			R2_ssbar = np.append(R2_ssbar, R2[i])
			decaymode_ssbar = np.append(decaymode_ssbar, decaymode[i])

		elif sig_or_cont[i] == 5:
			Mbc_ccbar = np.append(Mbc_ccbar, Mbc[i])
			isSignal_ccbar = np.append(isSignal_ccbar, isSignal[i])
			deltaE_ccbar = np.append(deltaE_ccbar, deltaE[i])
			SigProb_ccbar = np.append(SigProb_ccbar, SigProb[i])
			FEIProbRank_ccbar = np.append(FEIProbRank_ccbar, FEIProbRank[i])
			cosTBTO_ccbar = np.append(cosTBTO_ccbar, cosTBTO[i])
			R2_ccbar = np.append(R2_ccbar, R2[i])
			decaymode_ccbar = np.append(decaymode_ccbar, decaymode[i])

		elif sig_or_cont[i] == 1:
			Mbc_charged = np.append(Mbc_charged, Mbc[i])
			isSignal_charged = np.append(isSignal_charged, isSignal[i])
			deltaE_charged = np.append(deltaE_charged, deltaE[i])
			SigProb_charged = np.append(SigProb_charged, SigProb[i])
			FEIProbRank_charged = np.append(FEIProbRank_charged, FEIProbRank[i])
			cosTBTO_charged = np.append(cosTBTO_charged, cosTBTO[i])
			R2_charged = np.append(R2_charged, R2[i])
			decaymode_charged = np.append(decaymode_charged, decaymode[i])


		elif sig_or_cont[i] == 0:
			Mbc_mixed = np.append(Mbc_mixed, Mbc[i])
			isSignal_mixed = np.append(isSignal_mixed, isSignal[i])
			deltaE_mixed = np.append(deltaE_mixed, deltaE[i])			
			SigProb_mixed = np.append(SigProb_mixed, SigProb[i])
			FEIProbRank_mixed = np.append(FEIProbRank_mixed, FEIProbRank[i])
			cosTBTO_mixed = np.append(cosTBTO_mixed, cosTBTO[i])
			R2_mixed = np.append(R2_mixed, R2[i])
			decaymode_mixed = np.append(decaymode_mixed, decaymode[i])
	
	uubar = [Mbc_uubar, isSignal_uubar, deltaE_uubar, SigProb_uubar, FEIProbRank_uubar, cosTBTO_uubar, R2_uubar, decaymode_uubar]
	ddbar = [Mbc_ddbar, isSignal_ddbar, deltaE_ddbar, SigProb_ddbar, FEIProbRank_ddbar, cosTBTO_ddbar, R2_ddbar, decaymode_ddbar]
	ssbar = [Mbc_ssbar, isSignal_ssbar, deltaE_ssbar, SigProb_ssbar, FEIProbRank_ssbar, cosTBTO_ssbar, R2_ssbar, decaymode_ssbar]
	ccbar = [Mbc_ccbar, isSignal_ccbar, deltaE_ccbar, SigProb_ccbar, FEIProbRank_ccbar, cosTBTO_ccbar, R2_ccbar, decaymode_ccbar]
	charged = [Mbc_charged, isSignal_charged, deltaE_charged, SigProb_charged, FEIProbRank_charged, cosTBTO_charged, R2_charged, decaymode_charged]
	mixed = [Mbc_mixed, isSignal_mixed, deltaE_mixed, SigProb_mixed, FEIProbRank_mixed, cosTBTO_mixed, R2_mixed, decaymode_mixed]



	return(mixed, charged, uubar, ddbar, ssbar, ccbar)





def analiza(Mbc, isSignal, deltaE, SigProb, FEIProbRank, cosTBTO, R2, decaymode, sig_or_cont, SigProb_limit = 0, R2_limit = 1, absdeltaE_limit = 2, cosTBTO_limit = 1, Mbc_limit = 0,  use_FEIProbRank = True):
	
	Mbc_analysis = np.array([])
	isSignal_analysis = np.array([])
	deltaE_analysis = np.array([])
	SigProb_analysis = np.array([])
	FEIProbRank_analysis = np.array([])
	cosTBTO_analysis = np.array([])
	R2_analysis = np.array([])
	decaymode_analysis = np.array([])
	sig_or_cont_analysis = np.array([])


	if use_FEIProbRank == True:
		for i in range(len(Mbc)):
			if (SigProb[i] > SigProb_limit) and (FEIProbRank[i] == 1) and (R2[i] < R2_limit) and (np.abs(deltaE[i]) < absdeltaE_limit) and (cosTBTO[i] < cosTBTO_limit) and (Mbc[i] > Mbc_limit):
				Mbc_analysis = np.append(Mbc_analysis, Mbc[i])
				isSignal_analysis = np.append(isSignal_analysis, isSignal[i])
				deltaE_analysis = np.append(deltaE_analysis, deltaE[i])
				SigProb_analysis = np.append(SigProb_analysis, SigProb[i])
				FEIProbRank_analysis = np.append(FEIProbRank_analysis, FEIProbRank[i])
				cosTBTO_analysis = np.append(cosTBTO_analysis, cosTBTO[i])
				R2_analysis = np.append(R2_analysis, R2[i])
				decaymode_analysis = np.append(decaymode_analysis, decaymode[i])
				sig_or_cont_analysis = np.append(sig_or_cont_analysis, sig_or_cont[i])




	else:
		for i in range(len(Mbc)):
			if (SigProb[i] > SigProb_limit) and (R2[i] < R2_limit) and (np.abs(deltaE[i]) < absdeltaE_limit) and (cosTBTO[i] < cosTBRO_limit) and (Mbc[i] > Mbc_limit):
				Mbc_analysis = np.append(Mbc_analysis, Mbc[i])
				isSignal_analysis = np.append(isSignal_analysis, isSignal[i])
				deltaE_analysis = np.append(deltaE_analysis, deltaE[i])
				SigProb_analysis = np.append(SigProb_analysis, SigProb[i])
				FEIProbRank_analysis = np.append(FEIProbRank_analysis, FEIProbRank[i])
				cosTBTO_analysis = np.append(cosTBTO_analysis, cosTBTO[i])
				R2_analysis = np.append(R2_analysis, R2[i])
				decaymode_analysis = np.append(decaymode_analysis, decaymode[i])
				sig_or_cont_analysis = np.append(sig_or_cont_analysis, sig_or_cont[i])


	return(Mbc_analysis, isSignal_analysis, deltaE_analysis, SigProb_analysis, FEIProbRank_analysis, cosTBTO_analysis, R2_analysis, decaymode_analysis, sig_or_cont_analysis)








def plot(index, xmin, xmax, nbins, mixed, charged, uubar, ddbar, ssbar, ccbar, isSignal = True):
	
	index = int(index)
	nbins = int(nbins)

	mixed_hist, bins = np.histogram(mixed[index], range = (xmin, xmax), bins = nbins)
	charged_hist, bins = np.histogram(charged[index], range = (xmin, xmax), bins = nbins)
	uubar_hist, bins= np.histogram(uubar[index], range = (xmin, xmax), bins = nbins)
	ddbar_hist, bins= np.histogram(ddbar[index], range = (xmin, xmax), bins = nbins)
	ssbar_hist, bins= np.histogram(ssbar[index], range = (xmin, xmax), bins = nbins)
	ccbar_hist, bins= np.histogram(ccbar[index], range = (xmin, xmax), bins = nbins)

	mixed_hist = mixed_hist/mixed_lumin
	charged_hist = charged_hist/charged_lumin
	uubar_hist = uubar_hist/uubar_lumin
	ddbar_hist = ddbar_hist/ddbar_lumin
	ssbar_hist = ssbar_hist/ssbar_lumin
	ccbar_hist = ccbar_hist/ccbar_lumin


	plt.hist(bins[:-1], bins, weights = mixed_hist + charged_hist + ccbar_hist + ssbar_hist + ddbar_hist + uubar_hist, label = 'all')
	#plt.hist(bins[:-1], bins, weights = charged_hist + ccbar_hist + ssbar_hist + ddbar_hist + uubar_hist, label = '+charged')
	#plt.hist(bins[:-1], bins, weights = ccbar_hist + ssbar_hist + ddbar_hist + uubar_hist, label = '+ccbar')
	#plt.hist(bins[:-1], bins, weights = ssbar_hist + ddbar_hist + uubar_hist, label = '+ssbar')
	#plt.hist(bins[:-1], bins, weights = ddbar_hist + uubar_hist, label = 'uubar+ddbar')
	#plt.hist(bins[:-1], bins, weights = uubar_hist, label = 'uubar')

	if isSignal == True:
		isSignal_mixed = mixed[1]
		mixed_wisSignal = np.array([])
	
		for i in range(len(isSignal_mixed)):
			if isSignal_mixed[i] == 1:
				mixed_wisSignal = np.append(mixed_wisSignal, mixed[index][i])

		mixed_hist_isSignal, bins = np.histogram(mixed_wisSignal, range = (xmin, xmax), bins = nbins)
		mixed_hist_isSignal = mixed_hist_isSignal/mixed_lumin
		plt.hist(bins[:-1], bins, weights = mixed_hist_isSignal, label = 'isSignal = 1', color = 'black', alpha = 0.75)



	if index == 0:
		variable = "Mbc"

	elif index == 1:
		variable = "isSignal"

	elif index == 2:
		variable = "deltaE"

	elif index == 3:
		variable = "SigProb"

	elif index == 4:
		variable = "FEIProbRank"

	elif index == 5:
		variable = "cosTBTO"

	elif index == 6:
		variable = "R2"

	elif index == 7:
		variable = "decaymode"	

	title = "Different "+variable+", scaled to luminosity,\nR2 < 0.25"
	#title = "Different "+variable+", scaled to luminosity, FOM = %.2f" % (sum(mixed_hist_isSignal) / np.sqrt(sum(mixed_hist + charged_hist + ccbar_hist + ssbar_hist + ddbar_hist + uubar_hist)))
	plt.title(title)
	plt.xlabel(variable)
	plt.ylabel("N")
	plt.legend()
	plt.savefig(variable+"_separated.png")
	plt.close()


	return()





def eff_vs_purity(mixed, charged, uubar, ddbar, ssbar, ccbar, deletemode = False, delete = 0):

	sig_probs = np.append(np.linspace(0, 0.1, 100), np.linspace(0.1, 0.8, 100, endpoint = False))
	sig_probs = np.append(sig_probs, np.linspace(0.8, 1, 100, endpoint = False))
	#sig_probs = np.linspace(0, 1, 100, endpoint = False)
	
	mixed_isSignal = mixed[1]

	mixed_SigProb = mixed[3]
	charged_SigProb = charged[3]
	uubar_SigProb = uubar[3]
	ddbar_SigProb = ddbar[3]
	ssbar_SigProb = ssbar[3]
	ccbar_SigProb = ccbar[3]


	if deletemode == True:
		mixed_decaymode = mixed[7]
		charged_decaymode = charged[7]
		uubar_decaymode = uubar[7]
		ddbar_decaymode = ddbar[7]
		ssbar_decaymode = ssbar[7]
		ccbar_decaymode = ccbar[7]


	eff = np.array([])
	pure = np.array([])
	
	processed_mixed = len(mixed_SigProb)/mixed_lumin

	'''
	cnt_isSignal_all = 0
	for i in mixed_isSignal:
		if i == 1:
			cnt_isSignal_all += 1

	cnt_isSignal_all = cnt_isSignal_all / mixed_lumin
	'''

	for i in sig_probs:
		if deletemode == True:
			mixed_isSignal = np.delete(mixed_isSignal, np.where(mixed_decaymode == delete))

		mixed_isSignal = np.delete(mixed_isSignal, np.where(mixed_SigProb < i))
		mixed_isSignal = mixed_isSignal[np.logical_not(np.isnan(mixed_isSignal))]

		cnt_isSignal = np.sum(mixed_isSignal) / mixed_lumin
		#print(cnt_isSignal)
	
		mixed_SigProb = np.delete(mixed_SigProb, np.where(mixed_SigProb < i))
		charged_SigProb = np.delete(charged_SigProb, np.where(charged_SigProb < i))
		uubar_SigProb = np.delete(uubar_SigProb, np.where(uubar_SigProb < i))
		ddbar_SigProb = np.delete(ddbar_SigProb, np.where(ddbar_SigProb < i))
		ssbar_SigProb = np.delete(ssbar_SigProb, np.where(ssbar_SigProb < i))
		ccbar_SigProb = np.delete(ccbar_SigProb, np.where(ccbar_SigProb < i))


		if deletemode == True:
			mixed_SigProb = np.delete(mixed_SigProb, np.where(mixed_decaymode == delete))
			charged_SigProb = np.delete(charged_SigProb, np.where(charged_decaymode == delete))
			uubar_SigProb = np.delete(uubar_SigProb, np.where(uubar_decaymode == delete))
			ddbar_SigProb = np.delete(ddbar_SigProb, np.where(ddbar_decaymode == delete))
			ssbar_SigProb = np.delete(ssbar_SigProb, np.where(ssbar_decaymode == delete))
			ccbar_SigProb = np.delete(ccbar_SigProb, np.where(ccbar_decaymode == delete))


		mixed_hist, bins = np.histogram(mixed_SigProb, range = (0, 1), bins = 100)
		charged_hist, bins = np.histogram(charged_SigProb, range = (0, 1), bins = 100)
		uubar_hist, bins = np.histogram(uubar_SigProb, range = (0, 1), bins = 100)
		ddbar_hist, bins = np.histogram(ddbar_SigProb, range = (0, 1), bins = 100)
		ssbar_hist, bins = np.histogram(ssbar_SigProb, range = (0, 1), bins = 100)
		ccbar_hist, bins = np.histogram(ccbar_SigProb, range = (0, 1), bins = 100)

		mixed_hist = mixed_hist / mixed_lumin
		charged_hist = charged_hist / charged_lumin
		uubar_hist = uubar_hist / uubar_lumin
		ddbar_hist = ddbar_hist / ddbar_lumin
		ssbar_hist = ssbar_hist / ssbar_lumin
		ccbar_hist = ccbar_hist / ccbar_lumin


		eff = np.append(eff, cnt_isSignal/Nevents)
		pure = np.append(pure, cnt_isSignal/sum(mixed_hist + charged_hist + uubar_hist + ddbar_hist + ssbar_hist + ccbar_hist))


	return(eff, pure)




Mbc, isSignal, deltaE, SigProb, FEIProbRank, cosTBTO, R2, decaymode, sig_or_cont = np.loadtxt("/home/belle2/fmencin/B/mixed/sig_and_cont/variabs2/cutted_variables.txt", unpack = True)


Mbc_anal, isSignal_anal, deltaE_anal, SigProb_anal, FEIProbRank_anal, cosTBTO_anal, R2_anal, decaymode_anal, sig_or_cont_anal = \
	analiza(Mbc, isSignal, deltaE, SigProb, FEIProbRank, cosTBTO, R2, decaymode, sig_or_cont, R2_limit = 1, cosTBTO_limit = 1, Mbc_limit = 5.27, use_FEIProbRank = True)


mixed, charged, uubar, ddbar, ssbar, ccbar = separated(Mbc_anal, isSignal_anal, deltaE_anal, SigProb_anal, FEIProbRank_anal, cosTBTO_anal, R2_anal, decaymode_anal, sig_or_cont_anal)

plot(7, -0.5, 32.5, 33, mixed, charged, uubar, ddbar, ssbar, ccbar)

"""
eff_all, pure_all = eff_vs_purity(mixed, charged, uubar, ddbar, ssbar, ccbar)


for i in range(32):
	eff_0, pure_0 = eff_vs_purity(mixed, charged, uubar, ddbar, ssbar, ccbar, deletemode = True, delete = i)

	plt.title('pure vs eff')
	plt.plot(pure_all, eff_all/2, label = 'all decaymodes') #dejanskih mixed je 2x vec, ker sta B in antiB
	plt.plot(pure_0, eff_0/2, label = 'without decaymode %i' % i, ls = 'dashed')
	plt.ylabel("eff")
	plt.xlabel("purity")
	plt.legend()
	plt.savefig("eff_vs_purity_without%i.png" % i)
	plt.close()
"""

"""
arr = [0, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1]
eff_arr = np.array([])
purity_arr = np.array([])
effvspurity_arr = np.array([])
print("effpurity")


for j in arr:
	cnt = 0
	Mbc_sigprob_FEIProbRank = np.array([])
	for i in range(len(SigProb)):
		if SigProb[i] > j:
			#Mbc_sigprob = np.append(Mbc_sigprob, Mbc[i])
			#decaymode_sigprob = np.append(decaymode_sigprob, decaymode[i])
			#SigProb_sigprob = np.append(SigProb_sigprob, SigProb[i])

			if FEIProbRank[i] == 1:
				Mbc_sigprob_FEIProbRank = np.append(Mbc_sigprob_FEIProbRank, Mbc[i])
				#decaymode_sigprob_FEIProbRank = np.append(decaymode_sigprob_FEIProbRank, decaymode[i])
				#SigProb_sigprob_FEIProbRank = np.append(SigProb_sigprob_FEIProbRank, SigProb[i])

				if isSignal[i] == 1:
					cnt += 1

	print("Stevilo pravilno identificiranih B: %i" % cnt)
	print("Stevilo B (isSignal = 1): %i" % len(Mbc_correct))
	print("Stevilo identificiranih B (z SigProb in FEIProbRank): %i" % len(Mbc_sigprob_FEIProbRank))
	print("Stevilo B (samo s SigProb): %i" % len(Mbc_sigprob))
	print("Stevilo B (samo s FEIProbRank): %i" % len(Mbc_FEIProbRank))
	print("Stevilo B (brez kakrsnegakoli cuta): %i" % len(Mbc))
	
	
	eff = cnt/len(Mbc_correct)
	purity = cnt / len(Mbc_sigprob_FEIProbRank)

	eff_arr = np.append(eff_arr, eff)
	purity_arr = np.append(purity_arr, purity)
	effvspurity_arr = np.append(effvspurity_arr, eff/purity)
	print(j, eff, purity)


#print(len(Mbc_correct), len(Mbc_sigprob_FEIProbRank), len(Mbc_sigprob), len(Mbc_FEIProbRank), len(Mbc))	


plt.errorbar(arr, eff_arr, label = 'eff', ls = 'dashed', marker = 'x')
plt.errorbar(arr, purity_arr, label = 'purity', ls = 'dashed', marker = 'x')
plt.xlabel("SigProb limit")
plt.legend()
plt.savefig("effpurity2.png")
plt.close()


plt.title("eff vs purity")
plt.errorbar(purity_arr, eff_arr, ls = 'dashed', marker = 'x')
plt.ylabel("eff")
plt.xlabel("purity")
plt.savefig("effvspurity.png")
plt.close()
"""




"""
plt.title("Mbc")
plt.hist(Mbc, bins = 100, range = (5.15, 5.3))
plt.savefig("Mbc_all.png")
plt.close()

plt.title("Mbc FEIProbRank = 1")
plt.hist(Mbc_FEIProbRank, bins = 100, range = (5.15, 5.3))
plt.savefig("Mbc_FEIProbRank_all.png")
plt.close()

plt.title("Mbc isSignal = 1")
plt.hist(Mbc_correct, bins = 100, range = (5.15, 5.3))
plt.savefig("Mbc_isSignal_all.png")
plt.close()

plt.title("Mbc SigProb > 0.35")
plt.hist(Mbc_sigprob, bins = 100, range = (5.15, 5.3))
plt.savefig("Mbc_sigprob_all.png")
plt.close()

plt.title("Mbc SigProb > 0.35 & FEIProbRank = 1")
plt.hist(Mbc_sigprob_FEIProbRank, bins = 100, range = (5.15, 5.3))
plt.savefig("Mbc_sigprob_feiprobrank_all.png")
plt.close()




plt.title("SigProb isSignal = 1")
plt.hist(SigProb_correct, bins = 100, range = (0, 1))
plt.yscale("log")
plt.savefig("SigProb_isSignal_all.png")
plt.close()


plt.title("SigProb SigProb > 0.35 & FEIProbRank = 1")
plt.hist(SigProb_sigprob_FEIProbRank, bins = 100, range = (0, 1))
plt.yscale("log")
plt.savefig("SigProb_sigprob_feiprobrank_all.png")
plt.close()





plt.title("deltaE")
plt.hist(deltaE, bins = 100)
plt.savefig("deltaE_all.png")
plt.close()

plt.title("deltaE FEIProbRank = 1")
plt.hist(deltaE_FEIProbRank, bins = 100)
plt.savefig("deltaE_FEIProbRank_all.png")
plt.close()

plt.title("deltaE isSignal = 1")
plt.hist(deltaE_correct, bins = 100)
plt.savefig("deltaE_isSignal_all.png")
plt.close()

plt.title("deltaE SigProb > 0.35")
plt.hist(deltaE_sigprob, bins = 100 )
plt.savefig("deltaE_sigprob_all.png")
plt.close()

plt.title("deltaE SigProb > 0.35 & FEIProbRank = 1")
plt.hist(deltaE_sigprob_FEIProbRank, bins = 100)
plt.savefig("deltaE_sigprob_feiprobrank_all.png")
plt.close()





plt.title("decaymodeid isSignal = 1")
plt.hist(decaymode_correct, bins = 32, range = (0, 31), alpha = 0.75, fill = False, hatch = '/', edgecolor = 'red')
plt.savefig("decaymode_isSignal_all.png")
plt.close()


plt.title("decaymodeid SigProb > 0.35 & FEIProbRank = 1")
plt.hist(decaymode_sigprob_FEIProbRank, bins = 32, range = (0, 31), alpha = 0.75)
plt.savefig("decaymode_sigprob_feiprobrank_all.png")
plt.close()




plt.title("decaymodeid isSignal = 1 and\nSigProb > 0.35 & FEIProbRank = 1")
plt.hist(decaymode_sigprob_FEIProbRank, bins = 32, range = (0, 31), alpha = 0.75, edgecolor = 'black', label = 'SigProb > 0.05 & FEIProbRank = 1')
plt.hist(decaymode_correct, bins = 32, range = (0, 31), alpha = 0.75, color = 'red', fill = False, hatch = '/',  edgecolor = 'red', label = 'isSignal = 1')
plt.legend()
plt.savefig("decaymode_all.png")
plt.close()


plt.title("Mbc isSignal = 1 and\nSigProb > 0.35 & FEIProbRank = 1")
plt.hist(Mbc_sigprob_FEIProbRank, bins = 100, range = (5.15, 5.3), alpha = 0.75, label = 'SigProb > 0.05 & FEIProbRank = 1')
plt.hist(Mbc_correct, bins = 100, range = (5.15, 5.3), alpha = 0.75, fill = False, hatch = '/', edgecolor = 'red', label = 'isSignal = 1', histtype = 'step')
plt.legend()
plt.savefig("Mbc2_all.png")
plt.close()


plt.title("deltaE isSignal = 1 and\nSigProb > 0.35 & FEIProbRank = 1")
plt.hist(deltaE_sigprob_FEIProbRank, bins = 100, range = (-0.5, 0.5), alpha = 0.75, label = 'SigProb > 0.05 & FEIProbRank = 1')
plt.hist(deltaE_correct, bins = 100, range = (-0.5, 0.5), alpha = 0.75, fill = False, hatch = '/', edgecolor = 'red', label = 'isSignal = 1', histtype = 'step')
plt.legend()
plt.savefig("deltaE_all.png")
plt.close()




plt.title("SigProb isSignal = 1 and\nSigProb > 0.35 & FEIProbRank = 1")
plt.hist(SigProb_sigprob_FEIProbRank, bins = 100, range = (0, 1), alpha = 0.75, label = 'SigProb > 0.05 & FEIProbRank = 1')
plt.hist(SigProb_correct, bins = 100, range = (0, 1), alpha = 0.75, fill = False, hatch = '/', edgecolor = 'red', label = 'isSignal = 1', histtype = 'step')
plt.legend()
plt.yscale("log")
plt.savefig("SigProb_all.png")
plt.close()





#limits = np.array([0, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.007, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.125, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185, 0.19, 0.195, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

#arr, limits = FOM(SigProb, isSignal, limits)

limits = np.array([5.2, 5.205, 5.21, 5.215, 5.22, 5.225, 5.23, 5.235, 5.24, 5.245, 5.25, 5.255, 5.26, 5.265, 5.27, 5.275, 5.28, 5.285, 5.29])
arr, limits = FOM(Mbc, isSignal, limits)

print(max(arr), limits[np.where(arr == max(arr))])
plt.plot(limits, arr)
plt.savefig("FOM_Mbc.png")
plt.close()
"""
