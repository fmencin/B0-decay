import matplotlib.pyplot as plt
import numpy as np

class SingleDecay:
    def __init__(self, Mbc, isSignal, deltaE, SigProb, FEIProbRank, cosTBTO, R2, decaymode, sig_or_cont):
        self.Mbc = Mbc
        self.isSignal = isSignal    
        self.deltaE = deltaE
        self.SigProb = SigProb
        self.FEIProbRank = FEIProbRank
        self.cosTBTO = cosTBTO
        self.R2 = R2
        self.decaymode = decaymode
        self.sig_or_cont = sig_or_cont


class TypeOfDecay:
	def __init__(self):
		self.decays = []
		self.Mbc = np.array([])
		self.isSignal = np.array([])
		self.deltaE = np.array([])
		self.SigProb = np.array([])
		self.FEIProbRank = np.array([])
		self.cosTBTO = np.array([])
		self.R2 = np.array([])
		self.decaymode = np.array([])

	def add_decays(self, SingleDecay, sig_or_cont, Mbc_limit = 0, deltaE_limit = 2, SigProb_limit = 0, cosTBTO_limit = 1, R2_limit = 1):
		if (SingleDecay.sig_or_cont ==  sig_or_cont) and (SingleDecay.FEIProbRank == 1)\
			and (SingleDecay.Mbc > Mbc_limit) and (np.abs(SingleDecay.deltaE) < deltaE_limit)\
			and (SingleDecay.SigProb > SigProb_limit) and (SingleDecay.cosTBTO < cosTBTO_limit) and (SingleDecay.R2 < R2_limit):

			self.decays.append(SingleDecay)
			self.Mbc = np.append(self.Mbc, SingleDecay.Mbc)
			self.isSignal = np.append(self.isSignal, SingleDecay.isSignal)
			self.deltaE = np.append(self.deltaE, SingleDecay.deltaE)
			self.SigProb = np.append(self.SigProb, SingleDecay.SigProb)
			self.FEIProbRank = np.append(self.FEIProbRank, SingleDecay.FEIProbRank)
			self.cosTBTO = np.append(self.cosTBTO, SingleDecay.cosTBTO)
			self.R2 = np.append(self.R2, SingleDecay.R2)
			self.decaymode = np.append(self.decaymode, SingleDecay.decaymode)
		pass
		

#these global parameters are determined from scale_to_lumin.py and checkNevents1.py and checkNevents2.py
global mixed_lumin, charged_lumin, uubar_lumin, ddbar_lumin, ssbar_lumin, ccbar_lumin, Nevents

mixed_lumin = 0.0371 
charged_lumin = 0.0350
uubar_lumin = 0.0183
ddbar_lumin = 0.0733
ssbar_lumin = 0.0768
ccbar_lumin = 0.0174
Nevents = 50943396 

def FOM(variable, xmin, xmax, nbins, mixed, charged, uubar, ddbar, ssbar, ccbar):
	#calculates the optimal cut for input variable (to get the best FOM)

	if variable == "Mbc":
		mixed_var = mixed.Mbc
		charged_var = charged.Mbc
		uubar_var = uubar.Mbc
		ddbar_var = ddbar.Mbc
		ssbar_var = ssbar.Mbc
		ccbar_var = ccbar.Mbc

	elif variable == "isSignal":
		mixed_var = mixed.isSignal
		charged_var = charged.isSignal
		uubar_var = uubar.isSignal
		ddbar_var = ddbar.isSignal
		ssbar_var = ssbar.isSignal
		ccbar_var = ccbar.isSignal

	elif variable == "deltaE":
		mixed_var = mixed.deltaE
		charged_var = charged.deltaE
		uubar_var = uubar.deltaE
		ddbar_var = ddbar.deltaE
		ssbar_var = ssbar.deltaE
		ccbar_var = ccbar.deltaE

	elif variable == "SigProb":
		mixed_var = mixed.SigProb
		charged_var = charged.SigProb
		uubar_var = uubar.SigProb
		ddbar_var = ddbar.SigProb
		ssbar_var = ssbar.SigProb
		ccbar_var = ccbar.Sigprob
		
	elif variable == "FEIProbRank":
		mixed_var = mixed.FEIProbRank
		charged_var = charged.FEIProbRank
		uubar_var = uubar.FEIProbRank
		ddbar_var = ddbar.FEIProbRank
		ssbar_var = ssbar.FEIProbRank
		ccbar_var = ccbar.FEIProbRank

	elif variable == "cosTBTO":
		mixed_var = mixed.cosTBTO
		charged_var = charged.cosTBTO
		uubar_var = uubar.cosTBTO
		ddbar_var = ddbar.cosTBTO
		ssbar_var = ssbar.cosTBTO
		ccbar_var = ccbar.cosTBTO

	elif variable == "R2":
		mixed_var = mixed.R2
		charged_var = charged.R2
		uubar_var = uubar.R2
		ddbar_var = ddbar.R2
		ssbar_var = ssbar.R2
		ccbar_var = ccbar.R2

	elif variable == "decaymode":
		mixed_var = mixed.decaymode
		charged_var = charged.decaymode
		uubar_var = uubar.decaymode
		ddbar_var = ddbar.decaymode
		ssbar_var = ssbar.decaymode
		ccbar_var = ccbar.decaymode


	mixed_hist, bins = np.histogram(mixed_var, range = (xmin, xmax), bins = nbins)
	charged_hist, bins = np.histogram(charged_var, range = (xmin, xmax), bins = nbins)
	uubar_hist, bins= np.histogram(uubar_var, range = (xmin, xmax), bins = nbins)
	ddbar_hist, bins= np.histogram(ddbar_var, range = (xmin, xmax), bins = nbins)
	ssbar_hist, bins= np.histogram(ssbar_var, range = (xmin, xmax), bins = nbins)
	ccbar_hist, bins= np.histogram(ccbar_var, range = (xmin, xmax), bins = nbins)

	mixed_hist = mixed_hist/mixed_lumin
	charged_hist = charged_hist/charged_lumin
	uubar_hist = uubar_hist/uubar_lumin
	ddbar_hist = ddbar_hist/ddbar_lumin
	ssbar_hist = ssbar_hist/ssbar_lumin
	ccbar_hist = ccbar_hist/ccbar_lumin


	all_hist = mixed_hist + charged_hist + uubar_hist + ddbar_hist + ssbar_hist + ccbar_hist

	isSignal_mixed = mixed.isSignal
	mixed_wisSignal = np.array([])
	
	for i in range(len(isSignal_mixed)):
		if isSignal_mixed[i] == 1:
			mixed_wisSignal = np.append(mixed_wisSignal, mixed_var[i])
	
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
	plt.savefig("FOM_"+variable+"_class.png")
	plt.close()
	
	return()


def plot(variable, xmin, xmax, nbins, mixed, charged, uubar, ddbar, ssbar, ccbar, isSignal = True):
	#plots the distribution of variable from xmin to xmax with bins = nbins, with separation between diferent types

	if variable == "Mbc":
		mixed_var = mixed.Mbc
		charged_var = charged.Mbc
		uubar_var = uubar.Mbc
		ddbar_var = ddbar.Mbc
		ssbar_var = ssbar.Mbc
		ccbar_var = ccbar.Mbc

	elif variable == "isSignal":
		mixed_var = mixed.isSignal
		charged_var = charged.isSignal
		uubar_var = uubar.isSignal
		ddbar_var = ddbar.isSignal
		ssbar_var = ssbar.isSignal
		ccbar_var = ccbar.isSignal

	elif variable == "deltaE":
		mixed_var = mixed.deltaE
		charged_var = charged.deltaE
		uubar_var = uubar.deltaE
		ddbar_var = ddbar.deltaE
		ssbar_var = ssbar.deltaE
		ccbar_var = ccbar.deltaE

	elif variable == "SigProb":
		mixed_var = mixed.SigProb
		charged_var = charged.SigProb
		uubar_var = uubar.SigProb
		ddbar_var = ddbar.SigProb
		ssbar_var = ssbar.SigProb
		ccbar_var = ccbar.Sigprob
		
	elif variable == "FEIProbRank":
		mixed_var = mixed.FEIProbRank
		charged_var = charged.FEIProbRank
		uubar_var = uubar.FEIProbRank
		ddbar_var = ddbar.FEIProbRank
		ssbar_var = ssbar.FEIProbRank
		ccbar_var = ccbar.FEIProbRank

	elif variable == "cosTBTO":
		mixed_var = mixed.cosTBTO
		charged_var = charged.cosTBTO
		uubar_var = uubar.cosTBTO
		ddbar_var = ddbar.cosTBTO
		ssbar_var = ssbar.cosTBTO
		ccbar_var = ccbar.cosTBTO

	elif variable == "R2":
		mixed_var = mixed.R2
		charged_var = charged.R2
		uubar_var = uubar.R2
		ddbar_var = ddbar.R2
		ssbar_var = ssbar.R2
		ccbar_var = ccbar.R2

	elif variable == "decaymode":
		mixed_var = mixed.decaymode
		charged_var = charged.decaymode
		uubar_var = uubar.decaymode
		ddbar_var = ddbar.decaymode
		ssbar_var = ssbar.decaymode
		ccbar_var = ccbar.decaymode


	mixed_hist, bins = np.histogram(mixed_var, range = (xmin, xmax), bins = nbins)
	charged_hist, bins = np.histogram(charged_var, range = (xmin, xmax), bins = nbins)
	uubar_hist, bins= np.histogram(uubar_var, range = (xmin, xmax), bins = nbins)
	ddbar_hist, bins= np.histogram(ddbar_var, range = (xmin, xmax), bins = nbins)
	ssbar_hist, bins= np.histogram(ssbar_var, range = (xmin, xmax), bins = nbins)
	ccbar_hist, bins= np.histogram(ccbar_var, range = (xmin, xmax), bins = nbins)

	mixed_hist = mixed_hist/mixed_lumin
	charged_hist = charged_hist/charged_lumin
	uubar_hist = uubar_hist/uubar_lumin
	ddbar_hist = ddbar_hist/ddbar_lumin
	ssbar_hist = ssbar_hist/ssbar_lumin
	ccbar_hist = ccbar_hist/ccbar_lumin


	plt.hist(bins[:-1], bins, weights = mixed_hist + charged_hist + ccbar_hist + ssbar_hist + ddbar_hist + uubar_hist, label = 'all')
	plt.hist(bins[:-1], bins, weights = charged_hist + ccbar_hist + ssbar_hist + ddbar_hist + uubar_hist, label = '+charged')
	plt.hist(bins[:-1], bins, weights = ccbar_hist + ssbar_hist + ddbar_hist + uubar_hist, label = '+ccbar')
	plt.hist(bins[:-1], bins, weights = ssbar_hist + ddbar_hist + uubar_hist, label = '+ssbar')
	plt.hist(bins[:-1], bins, weights = ddbar_hist + uubar_hist, label = 'uubar+ddbar')
	plt.hist(bins[:-1], bins, weights = uubar_hist, label = 'uubar')

	if isSignal == True:
		mixed_isSignal = mixed.isSignal
		mixed_wisSignal = np.array([])
	
		for i in range(len(mixed_isSignal)):
			if mixed_isSignal[i] == 1:
				mixed_wisSignal = np.append(mixed_wisSignal, mixed_var[i])

		mixed_hist_isSignal, bins = np.histogram(mixed_wisSignal, range = (xmin, xmax), bins = nbins)
		mixed_hist_isSignal = mixed_hist_isSignal/mixed_lumin
		plt.hist(bins[:-1], bins, weights = mixed_hist_isSignal, label = 'isSignal = 1', color = 'black', alpha = 0.75)


	title = "Different "+variable+", scaled to luminosity"
	#title = "Different "+variable+", scaled to luminosity, FOM = %.2f" % (sum(mixed_hist_isSignal) / np.sqrt(sum(mixed_hist + charged_hist + ccbar_hist + ssbar_hist + ddbar_hist + uubar_hist)))
	plt.title(title)
	plt.xlabel(variable)
	plt.ylabel("N")
	plt.legend()
	plt.savefig(variable+"_separated_class.png")
	plt.close()


	return()





def eff_vs_purity(mixed, charged, uubar, ddbar, ssbar, ccbar, deletemode = False, delete = 0):
	#calculates eff and purity
	sig_probs = np.append(np.linspace(0, 0.1, 100), np.linspace(0.1, 0.8, 100, endpoint = False))
	sig_probs = np.append(sig_probs, np.linspace(0.8, 1, 100, endpoint = False))

	
	mixed_isSignal = mixed.isSignal
	mixed_SigProb = mixed.SigProb
	charged_SigProb = charged.SigProb
	uubar_SigProb = uubar.SigProb
	ddbar_SigProb = ddbar.SigProb
	ssbar_SigProb = ssbar.SigProb
	ccbar_SigProb = ccbar.SigProb


	if deletemode == True:
		mixed_decaymode = mixed.decaymode
		charged_decaymode = charged.decaymode
		uubar_decaymode = uubar.decaymode
		ddbar_decaymode = ddbar.decaymode
		ssbar_decaymode = ssbar.decaymode
		ccbar_decaymode = ccbar.decaymode


	eff = np.array([])
	pure = np.array([])
	

	for i in sig_probs:
		if deletemode == True:
			mixed_isSignal = np.delete(mixed_isSignal, np.where(mixed_decaymode == delete))

		mixed_isSignal = np.delete(mixed_isSignal, np.where(mixed_SigProb < i))
		mixed_isSignal = mixed_isSignal[np.logical_not(np.isnan(mixed_isSignal))]

		cnt_isSignal = np.sum(mixed_isSignal) / mixed_lumin
	
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



data = np.loadtxt("/home/belle2/fmencin/B/mixed/sig_and_cont/variabs2/cutted_variables.txt")

decays = [SingleDecay(*i) for i in data]


mixed = TypeOfDecay()
charged = TypeOfDecay()
uubar = TypeOfDecay()
ddbar = TypeOfDecay()
ssbar = TypeOfDecay()
ccbar = TypeOfDecay()

for i in decays:
    mixed.add_decays(i, 0)
    charged.add_decays(i, 1)
    uubar.add_decays(i, 2)
    ddbar.add_decays(i, 3)
    ssbar.add_decays(i, 4)
    ccbar.add_decays(i, 5)


#FOM("R2", 0, 1, 100, mixed, charged, uubar, ddbar, ssbar, ccbar)


for i in range(32):
	eff_0, pure_0 = eff_vs_purity(mixed, charged, uubar, ddbar, ssbar, ccbar, deletemode = True, delete = i)
	eff_all, pure_all = eff_vs_purity(mixed, charged, uubar, ddbar, ssbar, ccbar)

	plt.title('pure vs eff')
	plt.plot(pure_all, eff_all/2, label = 'all decaymodes') #dejanskih mixed je 2x vec, ker sta B in antiB
	plt.plot(pure_0, eff_0/2, label = 'without decaymode %i' % i, ls = 'dashed')
	plt.ylabel("eff")
	plt.xlabel("purity")
	plt.legend()
	plt.savefig("eff_vs_purity_without%i.png" % i)
	plt.close()