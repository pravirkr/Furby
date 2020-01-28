import numpy as np
import matplotlib.pyplot as plt
import argparse
import os, sys
import glob
import time

from collections import namedtuple

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "params.yaml")

def parse_config(fname):
    with open(fname, 'r') as fobj:
        config = yaml.load(fobj)
    return config

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

params = AttrDict(parse_config(CONFIG_FILE))

consts = {
    'tfactor': int( params.tsamp / 0.00001024 ),               #40.96 microseconds
    'ffactor': int( ((params.ftop-params.fbottom)/params.nch)/0.01),      #We dont want dmsmear to be approximated beyond 5 kHz chw. So ffactor = chw/ 0.005
    }

tmp = namedtuple("co", consts.keys())
C = tmp(*consts.values())


def tscrunch(data, tx):
    if tx==1:
        return data
    
    if len(data.shape)==1:
        endpoint = int(len(data) / tx) * tx
        return data[:endpoint].reshape(-1, tx).sum(axis=-1)

    if len(data.shape)==2:
        nr=data.shape[0]
        nc=data.shape[1]
    
        endpoint=int(nc/tx) * tx
        tmp=data[:,:endpoint].reshape(nr, nc/tx, tx)
        tsdata=tmp.sum(axis=-1)

        return tsdata
    else:
        raise RuntimeError("Can only scrunch 1D/2D arrays")

def fscrunch(data, fx):
    if fx==1:
        return data
    if fx==0:
        raise ValueError("Cannot fscrunch by a factor of 0")
    nr = data.shape[0]
    nc = data.shape[1]

    if nr%fx!=0:
        raise RuntimeError(f'Cannot scrunch at factors which do not exactly '
                           f'divide the no. of channels')
    fsdata = np.mean(data.reshape(nr/fx, -1, nc), axis=1)
    return fsdata

def get_clean_noise_rms():
    #noise rms per channel
    return (params.noise_per_channel)

def gauss(x, a, x0, sigma):
    return a/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-x0*1.)**2 / (2.*sigma**2))

def gauss2(x, a, x0, FWHM):
    # FWHM = 2 * sqrt( 2 * ln(2) ) * sigma
    sigma = FWHM/2. /(2*np.log(2))**0.5			
    return a/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-x0*1.)**2 / (2.*sigma**2))


def get_pure_frb(snr, width, nch, nsamps):
    # this is ideal noise rms per channel
    clean_noise_rms = get_clean_noise_rms()
    # Dividing snr equally among all channels for the pure case
    snr_per_channel = snr*1./np.sqrt(nch)        

    # width is supposed to be FWHM
    tmp_sigma = width/2. /(2*np.log(2))**0.5	
    W_tophat_gauss = np.sqrt(2*np.pi) * tmp_sigma

    desired_signal = snr_per_channel * clean_noise_rms * np.sqrt(W_tophat_gauss)

    x = np.arange(int(nsamps * C.tfactor) )
    width = width * C.tfactor
    pure_frb_single_channel = gauss2(x, desired_signal, int(len(x)/2), width)

    if np.abs(np.sum(pure_frb_single_channel) - desired_signal) > desired_signal/50.:
      raise RuntimeError(f'The generated signal is off by more than 2% of the '
                         f'desired value, desired_signal = {desired_signal}, '
                         f'generated_signal = {np.sum(pure_frb_single_channel)}. '
                         f'Diff: {((np.sum(pure_frb_single_channel) - desired_signal)/desired_signal * 100)}%')

    # Copying single channel nch times as a 2D array
    pure_frb = np.array([pure_frb_single_channel] * nch)     

    assert (pure_frb.shape[0] == nch), f'Could not copy 1D array {nch} times'
    
    return pure_frb

def get_bandpass(nch):
    bp = np.loadtxt("/home/vgupta/resources/BANDPASS_normalized_320chan.cfg")
    if nch == 320:
        pass
    elif nch == 40:
        bp = tscrunch(bp, 8) / 8.
    else:
        raise ValueError(f'NCHAN expected: [40 or 320]. Got: {str(nch)}')
    return bp*1./bp.max()

def apply_bandpass(frb):
    nch = frb.shape[0]
    bp  = get_bandpass(nch)
    bp  = bp/bp.max()
    #bp = bp - bp.mean() +1

    frb = frb * bp.reshape(-1,1)
    return frb

def create_freq_structure(frb, kind):
    nch = frb.shape[0]
    x = np.arange(nch)
    # kind of scintillation
    
    if kind == 'slope':
        #Slope will be a random number between -0.5 and 0.5
        slope = np.random.uniform(-0.5*nch, 0.5*nch, 1)  
        f = x * slope
    if kind == 'smooth_envelope':
        #Location of Maxima of the smooth envelope can be on any channel
        center = np.random.uniform(0, nch, 1)            
        z1 = center - nch/2
        z2 = center + nch/2
        f = -1 * (x - z1) * (x - z2)
    if kind == 'two_peaks':
        z1 = 0
        z2 = np.random.uniform(0 + 1, nch/2, 1)
        z3 = np.random.uniform(nch/2, nch-1 , 1)
        z4 = nch
        f = -1 * (x-z1) * (x-z2) * (x-z3) * (x-z4)
    if kind == 'three_peaks':
        z1 = 0
        z2 = np.random.uniform(0 +1, nch/4, 1)
        z3 = np.random.uniform(1*nch/4, 2*nch/4, 1)
        z4 = np.random.uniform(2*nch/4, 3*nch/4, 1)
        z5 = np.random.uniform(3*nch/4, nch-1, 1)
        z6 = nch
        f = -1 * (x-z1) * (x-z2) * (x-z3) * (x-z4) * (x-z5) * (x-z6)
    if kind == 'ASKAP':
        n_blobs = np.floor(np.random.exponential(scale = 3, size=1)) + 1
        f = np.zeros(nch)
        for i in range(n_blobs):
            center_of_blob = np.random.uniform(0, nch, 1)
            # We want roughly 4 +- 1 MHz blobs. 4 MHz = 4/chw 
            # chans = 4./((params.ftop - params.bottom)/nch) chans
            NCHAN_PER_MHz = np.abs(1./( (params.ftop-params.fbottom)/nch ))
            width_of_blob = np.random.normal(loc = 4.*NCHAN_PER_MHz, 
                                             scale = NCHAN_PER_MHz, size = 1)
            # For just one blob (n_blobs=1), this does not matter because we 
            # rescale the maxima to 1 evetually. For more than one blobs, 
            # this random amp will set the relative power in different blobs. 
            # So, the power in weakest blob can be as low as 1/3rd of the 
            # strongest blob
            
            amp_of_blob = np.random.uniform(1, 3, 1)             
            f += gauss(x, amp_of_blob, center_of_blob, width_of_blob)
            
    f = f - f.min()                                # Bringing the minima to 0
    f = f * 1./f.max()                             # Bringing the maxima to 1
    f = f - f.mean() + 1                           # Shifting the mean to 1

    frb = frb * f.reshape(-1, 1)
    return frb, f

def disperse(frb, dm, pre_shift, dmsmear):
    tsum = 0
    if not dmsmear:
        ffactor = 1
    else:
        ffactor = C.ffactor

    if args.v:
      print("Ffactor:", ffactor)
    nch  = frb.shape[0] * ffactor
    # ms. Effectively 10.24 micro-seconds, just framing it in terms of 
    # tres of Hires filterbanks
    tres = params.tsamp / C.tfactor *1e3  

    chw = (params.ftop-params.fbottom)/nch
    f_ch = np.linspace(params.ftop - chw/2, params.fbottom + chw/2, nch)
    # Only works if freq in MHz and D in ms. Output delays in ms
    delays = params.D * f_ch**(-2) * dm    
    delays -= delays[int(nch/2)]
    # here we will have slight approximations due to quantization, 
    # but at 10.24 usec resolution they should be minimal
    delays_in_samples = np.rint(delays / tres).astype('int') 

    #nsamps = delays_in_samples[-1] - delays_in_samples[0] + 2*frb.shape[1]
    #nsamps = delays_in_samples[-1]*2 + 2*frb.shape[1]
    nsamps = 9000 * C.tfactor
    start = nsamps/2 - int(pre_shift*C.tfactor)
    end = start + frb.shape[1]

    dispersed_frb = np.zeros(nch * nsamps).reshape(nch, nsamps)
    undispersed_time_series = np.zeros(nsamps)
    idxs = np.arange(nsamps)

    if args.v:
        print(f'Initial frb shape {frb.shape} nsamps: {nsamps} \n'
              f'start, end: {start}{end}'
              f'Tfactor, Ffactor, pre_shift: {C.tfactor}{ffactor}{int(pre_shift)}')

    for i in range(nch):
        delay = delays_in_samples[i]
        mid_channel_coarse = int(i/ffactor) *ffactor + int(ffactor/2.0)
        dispersed_frb[i, start+delay: end+delay] += frb[int(i/ffactor)]
        undispersed_time_series += np.take(dispersed_frb[i], 
                                    idxs + delays_in_samples[mid_channel_coarse], 
                                    mode='wrap')
        tsum = undispersed_time_series.sum()
        if args.v:
            sys.stdout.write(f'nch : {i}/{nch}  tsum = {tsum}\r')
    final_dispersed_frb = fscrunch(dispersed_frb, ffactor)
    return final_dispersed_frb, undispersed_time_series/ffactor, nsamps/C.tfactor


def scatter(frb, tau0, nsamps, desired_snr): 
    nch = frb.shape[0]
    ftop = params.ftop        #MHz
    fbottom = params.fbottom     #MHz
    chw = (ftop-fbottom)/(1.0*nch)
    f_ch = np.linspace(ftop - chw/2, fbottom + chw/2, nch)
    nsamps = nsamps * C.tfactor
    tau0 = tau0 * C.tfactor

    k = tau0 * (f_ch[0])**params.scattering_index     # proportionality constant
    taus = k / f_ch**params.scattering_index          # Calculating tau for each channel
    exps=[]
    scattered_frb=[]
    for i,t in enumerate(taus):
        # making the exponential with which to convolve each channel
        exps.append( np.exp(-1 * np.arange(nsamps) / t) )       
        # convolving each channel with the corresponding exponential 
        # (np.convolve gives the output with length = len(frb) + len(exp) )
        result = np.convolve(frb[i], exps[-1])                 
        #result *= 1./result.max() * frb[i].max()
        result *= 1./result.sum() * frb[i].sum()
        scattered_frb.append(result) 

    scattered_frb = np.array(scattered_frb)
    scattered_tseries = scattered_frb.sum(axis=0)
    scattered_width = scattered_tseries.sum() / np.max(scattered_tseries) / C.tfactor
    new_snr = scattered_tseries.sum() / (np.sqrt(nch) * get_clean_noise_rms()) / np.sqrt(scattered_width)
    normalizing_factor = new_snr / desired_snr
    scattered_frb /= normalizing_factor
    return scattered_frb

def make_psrdada_header(hdr_len, params):
    header=""
    for i in params:
        header += i
        tabs = 3 - int(len(i)/8)
        header += "\t"*tabs
        header += str(params[i])+"\n"
    leftover_space = hdr_len - len(header)
    header += '\0' * leftover_space
    return header

def get_FWHM(frb_tseries):
    maxx = np.argmax(frb_tseries)
    hp = frb_tseries[maxx] / 2.
    #Finding the half-power points
    hpp1 = (np.abs(frb_tseries[:maxx] - hp)).argmin()
    hpp2 = (np.abs(frb_tseries[maxx:] - hp)).argmin() + maxx

    FWHM = (1.*hpp2-hpp1)/C.tfactor
    assert FWHM>0, (f'FWHM calculation went wrong somewhere. HPP points, '
                    f'maxx point and FWHM are {hpp1} {hpp2} {maxx} {FWHM}')
    return FWHM, np.max(tscrunch(frb_tseries, C.tfactor))

def start_logging(ctl, db_d):
    if os.path.exists(ctl):
	logger = open(ctl, 'a')
    else:
	logger = open(ctl, 'a')
	logger.write("#This is the furby catalogue for {0} directory\n".format(db_d))
	logger.write("#Created on : {0}\n\n".format(time.asctime()))
	existing_furbies = glob.glob(db_d+"furby_*")
	if len(existing_furbies) > 0:
	    logger.write("#The following furbies were found to be present in the directory before the creation of this catalogue:\n")
	    for furby in existing_furbies:
		logger.write("#"+furby+"\n")
	    logger.write("\n")
	logger.write("#FURBY_ID\tDM\tFWHM\tTAU0\tSNR\tSIGNAL\n")
    return logger

def check_for_permissions(db_d):
    if not os.path.exists(db_d):
	try:
	    print "The database directory: {0} does not exist. Attempting to create it now.".format(db_d)
	    os.makedirs(db_d)
	except OSError as E:
	    print "Attempt to create the database directory failed because:\n{0}".format(E.strerror)
	    print "Exiting...."
	    sys.exit(1)

    if os.access(db_d, os.W_OK) and os.access(db_d, os.X_OK):
        return 
    else:
	print "Do not have permissions to write/create in the database directory: {0}".format(db_d)
	print "Exiting..."
	sys.exit(1)

def main(args):
    database_directory = os.path.join(args.D, '')
    
    order = "TF"
    
    if args.plot and args.Num > 1:
      raise IOError("Sorry cannot plot more than one furby at a time")

    if not args.plot:
	   check_for_permissions(database_directory)
	   catalogue = database_directory+"furbies.cat"
	   logger = start_logging(catalogue, database_directory)

    ID_series = params.ID_series

    if args.v:
        print("Starting FRB Generator...")
    tsamp = params.tsamp                              #seconds
    nch = params.nch
    supported_kinds = ["slope", "smooth_envelope", "two_peaks", "three_peaks", "ASKAP"]
    
    if isinstance(args.snr, float):
        snrs = args.snr * np.ones(args.Num)
    elif isinstance(args.snr, list) and len(args.snr) ==1:
        snrs = args.snr[0] * np.ones(args.Num)
    elif isinstance(args.snr, list) and len(args.snr) ==2:
        snrs = np.random.uniform(args.snr[0], args.snr[1], args.Num)
    else:
        raise IOError("Invalid input for SNR")
    #snr = 15

    if isinstance(args.width, float):
        #widths = (args.width *1e-3/ params.tsamp) * np.ones(args.Num)
        widths = args.width *1e-3/ params.tsamp * np.ones(args.Num)
    elif isinstance(args.width, list) and len(args.width) ==1:
        #widths = int(args.width[0]*1e-3/params.tsamp) * np.ones(args.Num)
        widths = args.width[0]*1e-3/params.tsamp * np.ones(args.Num)
    elif isinstance(args.width, list) and len(args.width) ==2:
        #widths = np.random.randint(args.width[0]*1e-3/params.tsamp,args.width[1]*1e-3/params.tsamp, args.Num)   #samples
        widths = np.random.uniform(args.width[0]*1e-3/params.tsamp,args.width[1]*1e-3/params.tsamp, args.Num)   #samples
    else:
        raise IOError("Invalid input for Width")
    #width =3
    
    if isinstance(args.dm, float):
        dms = args.dm * np.ones(args.Num)
    elif isinstance(args.dm, list) and len(args.dm) ==1:
        dms = args.dm[0] * np.ones(args.Num)
    elif isinstance(args.dm, list) and len(args.dm) ==2:
        dms = np.random.uniform(args.dm[0], args.dm[1], args.Num)
    else:
        raise IOError("Invalid input for DM")
    #dm = 900

    # --------------------------------------------------------------------------
    for num in range(args.Num):
        dm = dms[num]
        snr = snrs[num]
        width = widths[num]

        if args.kind:
            kind = args.kind
            assert kind in supported_kinds
        else:
            kind = supported_kinds[np.random.randint(0, len(supported_kinds), 1)[0]]

        while(True):
            ID = np.random.randint((ID_series*params.N_per_IDseries + 1), 
                                   (ID_series+1)*params.N_per_IDseries, 1)[0] 
            ID = str(ID).zfill(int(np.log10(params.N_per_IDseries)))
            name = "furby_"+ID
            if os.path.exists(database_directory+name):
                continue
            else:
                break

        tau0 = np.abs(np.random.normal(loc = dm / 1000., scale = 2, size=1))[0] 
        #tau0 = 10.1/C.tfactor
      
        # = half of nsamps required for the gaussian. i.e. 
        # The total nsamps will be 10 * sigma.  
        nsamps_for_gaussian = 5 * width            
        if nsamps_for_gaussian < 1:
            nsamps_for_gaussian = 1
        nsamps_for_exponential = int(6 * tau0 * ((params.ftop+params.fbottom)/2 / params.fbottom)**params.scattering_index)

        if args.v:
            print("Randomly generated Parameters:")
            print(f'ID= {ID}, SNR= {snr}, Width= {width*tsamp*1e3}ms, DM= {dm}, '
                  f'tau0= {tau0*tsamp*1e3}ms, kind= {kind}')
      
        if args.v:
            print("Getting pure FRB")
        try:
            frb = get_pure_frb(snr=snr, width = width, nch=nch, nsamps=nsamps_for_gaussian)
        except RuntimeError as R:
            print(R)
            continue
      
        pure_signal = np.sum(frb.flatten())
        if args.v:
            print("Creating frequency structure")
        frb,f = create_freq_structure(frb, kind=kind)

        pure_width = pure_signal / np.max( frb.sum(axis=0) )/C.tfactor
        pure_snr = pure_signal / (np.sqrt(nch) * get_clean_noise_rms() * np.sqrt(pure_width))
      
        if args.v:
            print(f'Pure signal (input) = {pure_signal}, '
                  f'signal after freq_struct = {np.sum(frb.flatten())}, '
                  f'pure_snr = {pure_snr}, pure_width = {pure_width*params.tsamp*1e3}ms')

        #if args.v:
        #  print "Applying Bandpass"
        #frb = apply_bandpass(frb)
        #if args.v:
        #  print "Signal after bandpass calib = {0}".format(np.sum(frb.flatten()))
      
        if args.v:
            print("Scattering the FRB")
        if nsamps_for_exponential==0:
            print("Tau0 = 0, no scattering applied")
            pass
        else:
            frb = scatter(frb, tau0, nsamps_for_exponential, pure_snr)
        sky_signal = np.sum(frb.flatten())
        sky_frb_tseries = np.sum(frb, axis=0)
        #sky_frb_peak = np.max( tscrunch(sky_frb_tseries, C.tfactor)   )
        sky_frb_top_hat_width = sky_signal / np.max(sky_frb_tseries) / C.tfactor
        #sky_frb_top_hat_width = sky_signal / sky_frb_peak
        sky_snr = sky_signal / ( get_clean_noise_rms() * np.sqrt(nch) * np.sqrt(sky_frb_top_hat_width) )

        if args.v:
            print(f'Sky_signal = {sky_signal}, '
                  f'sky_width = {sky_frb_top_hat_width*params.tsamp*1e3} ms, '
                  f'sky_snr = {sky_snr}')
      
        frb_b_d = frb.copy()      #FRB before dispersing
      
        if args.v:
            print("Dispersing")
        # Remember, nsamps_for_gaussian is already half the number of samples
        frb, undispersed_tseries, NSAMPS = disperse(frb, dm, 
                                                pre_shift=nsamps_for_gaussian,
                                                dmsmear = args.dmsmear) 

        signal_after_disp = np.sum(frb.flatten())

        FWHM,maxima = get_FWHM(undispersed_tseries)

        if args.v:
            print("Scrunching")

        scrunched_frb = tscrunch(frb, C.tfactor)
        signal_after_scrunching = np.sum(scrunched_frb.flatten())

        #final_top_hat_width = signal_after_scrunching / maxima		
        #comes out in samples
        final_top_hat_width = signal_after_scrunching / np.max(undispersed_tseries) / C.tfactor
        output_snr = signal_after_scrunching / (get_clean_noise_rms() * np.sqrt(nch) *  np.sqrt(final_top_hat_width) )

        if args.v:
            print("Input signal, Sky_signal, Output signal, Input SNR, Sky SNR, \
                   Output SNR, Final_top_hat_width",  pure_signal, sky_signal, \
                   signal_after_scrunching, snr, sky_snr, output_snr, \
                   final_top_hat_width * tsamp * 1e3, "ms\n")
      
        final_frb = scrunched_frb.astype('float32')

        if args.v:
            print("Generating Header")
        header_size = 16384                 #bytes
        params = {
              "HDR_VERSION":  1.0,
              "HDR_SIZE": header_size,
              "TELESCOPE":    "MOST",
              "ID":   ID,
              "SOURCE":	name,
              "FREQ": (params.ftop + params.fbottom)/2,         #MHz
              "BW":   params.fbottom - params.ftop,              #MHz. Negative to indicate that the first channel has highest frequency
              "NPOL": 1,
              "NBIT": 32,
              "TSAMP":    tsamp * 1e6,      	#usec	--Dont change this, dspsr needs tsamp in microseconds to work properly
              "NSAMPS":	NSAMPS,
              "UTC_START":    "2018-02-12-00:00:00",       #Never happened :P
              "STATE":    "Intensity",
              "OBS_OFFSET":   0,           #samples
              "NCHAN":    nch,
              "ORDER":    order,            #This is ensured later in the code while writing data with flatten(order=O)
              "FTOP": params.ftop,        #MHz
              "FBOTTOM": params.fbottom,     #MHz
              "TRACKING": "false",
              "FB_ENABLED":   "true",
              "INSTRUMENT":   "MOPSR",
              "SNR":  output_snr,
              "SKY_SNR":  sky_snr,
      	      "WIDTH":	final_top_hat_width*tsamp*1e3,	#milliseconds
              "SIGNAL":   signal_after_scrunching,
              "SKY_SIGNAL":   sky_signal,
              "WIDTH_GAUSS":  width*tsamp*1e3,             #milliseconds
              "FWHM": FWHM*tsamp*1e3,                      #milliseconds
              "DM":   dm,                              #pc / cm^3
              "DMSMEAR":  str(args.dmsmear).upper(),
              "TAU0": tau0*tsamp*1e3,                      #milliseconds
              "KIND": kind,                            #kind of frequency structure put into the FRB
              "PEAK": maxima,                          #peak signal of the FRB
              }

        header_string = make_psrdada_header(header_size, params)
        if not args.plot:
            if args.v:
                print("Saving the FRB in '"+database_directory+"' directory")

            out = open(database_directory+name, 'wb')
            out.write(header_string)
            if order == "TF":
                #Order = 'F' means column-major, since we are writing data in TF format.
                O = 'F'				
            if order == "FT":
                #Order = 'C' means row-major, since we are writing data in FT format.
                O = 'C'				
          final_frb.flatten(order=O).tofile(out)          
          out.close()
          logger.write(ID+"\t"+str(dm)+"\t"+str(FWHM*tsamp)+"\t" + str(tau0*tsamp)+"\t" + str(output_snr)+"\t"+str(signal_after_scrunching)+"\n")
          
          print("Name : ", params["SOURCE"])

        #-----------------------------------------------------------------------
    if not args.plot:
        logger.close()

    if args.plot:
        if args.v:
            print "Plotting"
        plt.figure()
        plt.imshow(frb_b_d, aspect='auto', cmap='afmhot', interpolation='None')
        plt.title("FRB before dispersion")

        plt.figure()
        plt.plot(np.sum(frb_b_d, axis=0))
        plt.title("FRB tseries before dispersing")

        plt.figure()
        plt.imshow(scrunched_frb, aspect = 'auto', cmap = 'afmhot', interpolation='nearest')
        plt.title("FRB")

        plt.figure()
        plt.plot(tscrunch(undispersed_tseries, C.tfactor))
        plt.title("Un-dispersed FRB time series. We should be aiming to recover this profile")
        plt.show()

if __name__ == '__main__':
    a=argparse.ArgumentParser()
    a.add_argument("Num", type=int, help="Number of furbies to generate")
    a.add_argument("-kind", type=str, help="Kind of frequency structure wanted. \
                    Options:[slope, smooth_envelope, two_peaks, three_peaks, ASKAP]")
    a.add_argument("-plot", action='store_true', default = False, 
                    help = "Plot the FRB instead of saving it?")
    a.add_argument("-dm", nargs='+', type=float, default = 1000.0,
                    help="DM or DM range endpoints")
    a.add_argument("-snr", nargs='+', type=float, default = 20.0, 
                    help="SNR or SNR range endpoints")
    a.add_argument("-width", nargs='+', type=float, default = 2.0,
                    help="Width or width range endpoints (in ms)")
    a.add_argument("-dmsmear", action='store_true', default=False,
                    help = "Enable smearing within individual channels (def=False)")
    a.add_argument("-v", action='store_true', default = False, 
                    help="Verbose output")
    a.add_argument("-D", type=str, default = "./", 
                    help="Path to the database to which the furby should be \
                          added (def = ./)")
    args = a.parse_args()
    main(args)
