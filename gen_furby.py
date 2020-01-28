import numpy as np
import matplotlib.pyplot as plt
import argparse
import os, sys
import glob
import time
import yaml

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "params.yaml")

def parse_config(fname):
    with open(fname, 'r') as fobj:
        config = yaml.load(fobj)
    return config

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


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
        tmp=data[:,:endpoint].reshape(nr, nc//tx, tx)
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

def get_clean_noise_rms(params):
    #noise rms per channel
    return (params.noise_per_channel)

def gauss(x, a, x0, sigma):
    return a/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-x0*1.)**2 / (2.*sigma**2))

def gauss2(x, a, x0, FWHM):
    # FWHM = 2 * sqrt( 2 * ln(2) ) * sigma
    sigma = FWHM/2. /(2*np.log(2))**0.5			
    return a/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-x0*1.)**2 / (2.*sigma**2))


def get_pure_frb(snr, width, nchan, nsamps, params):
    # this is ideal noise rms per channel
    clean_noise_rms = get_clean_noise_rms(params)
    # Dividing snr equally among all channels for the pure case
    snr_per_channel = snr*1./np.sqrt(nchan)        

    # width is supposed to be FWHM
    tmp_sigma = width/2. /(2*np.log(2))**0.5	
    W_tophat_gauss = np.sqrt(2*np.pi) * tmp_sigma

    desired_signal = snr_per_channel * clean_noise_rms * np.sqrt(W_tophat_gauss)

    x = np.arange(int(nsamps * params.tfactor) )
    width = width * params.tfactor
    pure_frb_single_channel = gauss2(x, desired_signal, int(len(x)/2), width)

    if np.abs(np.sum(pure_frb_single_channel) - desired_signal) \
              > desired_signal/50:
        diff = (np.sum(pure_frb_single_channel) \
                     - desired_signal)/desired_signal * 100
        raise RuntimeError(f'The generated signal is off by more than 2% of the '
                        f'desired value, desired_signal = {desired_signal}, '
                        f'generated_signal = {np.sum(pure_frb_single_channel)}. '
                        f'Diff: {diff}%')

    # Copying single channel nchan times as a 2D array
    pure_frb = np.array([pure_frb_single_channel] * nchan)     

    assert (pure_frb.shape[0] == nchan), f'Could not copy 1D array {nchan} times'
    
    return pure_frb

def get_bandpass(nchan):
    bp = np.loadtxt("/home/vgupta/resources/BANDPASS_normalized_320chan.cfg")
    if nchan == 320:
        pass
    elif nchan == 40:
        bp = tscrunch(bp, 8) / 8.
    else:
        raise ValueError(f'NCHAN expected: [40 or 320]. Got: {str(nchan)}')
    return bp*1./bp.max()

def apply_bandpass(frb):
    nchan = frb.shape[0]
    bp    = get_bandpass(nchan)
    bp    = bp/bp.max()
    #bp = bp - bp.mean() +1

    frb = frb * bp.reshape(-1,1)
    return frb

def create_freq_structure(frb, kind, params):
    nchan = frb.shape[0]
    x = np.arange(nchan)
    # kind of scintillation
    
    if kind == 'slope':
        #Slope will be a random number between -0.5 and 0.5
        slope = np.random.uniform(-0.5*nchan, 0.5*nchan, 1)  
        f = x * slope
    if kind == 'smooth_envelope':
        #Location of Maxima of the smooth envelope can be on any channel
        center = np.random.uniform(0, nchan, 1)            
        z1 = center - nchan/2
        z2 = center + nchan/2
        f = -1 * (x - z1) * (x - z2)
    if kind == 'two_peaks':
        z1 = 0
        z2 = np.random.uniform(0 + 1, nchan/2, 1)
        z3 = np.random.uniform(nchan/2, nchan-1 , 1)
        z4 = nchan
        f = -1 * (x-z1) * (x-z2) * (x-z3) * (x-z4)
    if kind == 'three_peaks':
        z1 = 0
        z2 = np.random.uniform(0 +1, nchan/4, 1)
        z3 = np.random.uniform(1*nchan/4, 2*nchan/4, 1)
        z4 = np.random.uniform(2*nchan/4, 3*nchan/4, 1)
        z5 = np.random.uniform(3*nchan/4, nchan-1, 1)
        z6 = nchan
        f = -1 * (x-z1) * (x-z2) * (x-z3) * (x-z4) * (x-z5) * (x-z6)
    if kind == 'ASKAP':
        n_blobs = int(np.floor(np.random.exponential(scale = 3, size=1)[0]) + 1)
        f = np.zeros(nchan)
        for i in range(n_blobs):
            center_of_blob = np.random.uniform(0, nchan, 1)
            # We want roughly 4 +- 1 MHz blobs. 4 MHz = 4/chw 
            # chans = 4./((params.ftop - params.bottom)/nchan) chans
            NCHAN_PER_MHz = np.abs(1./( (params.ftop-params.fbottom)/nchan ))
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

def disperse(frb, dm, pre_shift, dmsmear, params):
    tsum = 0
    if not dmsmear:
        ffactor = 1
    else:
        ffactor = params.ffactor

    if args.v:
        print("Ffactor:", ffactor)
    nchan  = frb.shape[0] * ffactor
    # ms. Effectively 10.24 micro-seconds, just framing it in terms of 
    # tres of Hires filterbanks
    tres = params.tsamp / params.tfactor *1e3  

    chw = (params.ftop-params.fbottom)/nchan
    f_ch = np.linspace(params.ftop - chw/2, params.fbottom + chw/2, nchan)
    # Only works if freq in MHz and D in ms. Output delays in ms
    delays = params.D * f_ch**(-2) * dm    
    delays -= delays[int(nchan/2)]
    # here we will have slight approximations due to quantization, 
    # but at 10.24 usec resolution they should be minimal
    delays_in_samples = np.rint(delays / tres).astype('int') 

    #nsamps = delays_in_samples[-1] - delays_in_samples[0] + 2*frb.shape[1]
    #nsamps = delays_in_samples[-1]*2 + 2*frb.shape[1]
    nsamps = 9000 * params.tfactor
    start = nsamps//2 - int(pre_shift*params.tfactor)
    end = start + frb.shape[1]

    dispersed_frb = np.zeros(nchan * nsamps).reshape(nchan, nsamps)
    undispersed_time_series = np.zeros(nsamps)
    idxs = np.arange(nsamps)

    if args.v:
        print(f'Initial frb shape {frb.shape} nsamps: {nsamps} \n'
              f'start, end: {start}, {end}'
              f'Tfactor, Ffactor, pre_shift: {params.tfactor}, {ffactor}, {int(pre_shift)}')

    for i in range(nchan):
        delay = delays_in_samples[i]
        mid_channel_coarse = int(i/ffactor) *ffactor + int(ffactor/2.0)
        dispersed_frb[i, start+delay: end+delay] += frb[int(i/ffactor)]
        undispersed_time_series += np.take(dispersed_frb[i], 
                                    idxs + delays_in_samples[mid_channel_coarse], 
                                    mode='wrap')
        tsum = undispersed_time_series.sum()
        if args.v:
            sys.stdout.write(f'nchan : {i}/{nchan}  tsum = {tsum}\r')
    final_dispersed_frb = fscrunch(dispersed_frb, ffactor)
    return final_dispersed_frb, undispersed_time_series/ffactor, nsamps/params.tfactor


def scatter(frb, tau0, nsamps, desired_snr, params): 
    nch = frb.shape[0]
    ftop = params.ftop        #MHz
    fbottom = params.fbottom     #MHz
    chw = (ftop-fbottom)/(1.0*nchan)
    f_ch = np.linspace(ftop - chw/2, fbottom + chw/2, nchan)
    nsamps = nsamps * params.tfactor
    tau0 = tau0 * params.tfactor

    k = tau0 * (f_ch[0])**params.scattering_index     # proportionality constant
    taus = k / f_ch**params.scattering_index          # Calculating tau for each channel
    exps = []
    scattered_frb = []
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
    scattered_width = scattered_tseries.sum() / np.max(scattered_tseries) / params.tfactor
    new_snr = scattered_tseries.sum() / (np.sqrt(nchan) * get_clean_noise_rms(params)) \
                                      / np.sqrt(scattered_width)
    normalizing_factor = new_snr / desired_snr
    scattered_frb /= normalizing_factor
    return scattered_frb

def make_psrdada_header(hdr_len, hdr_params):
    header=""
    for i in hdr_params:
        header += i
        tabs = 3 - int(len(i)/8)
        header += "\t"*tabs
        header += str(hdr_params[i])+"\n"
    leftover_space = hdr_len - len(header)
    header += '\0' * leftover_space
    return header

def get_FWHM(frb_tseries, params):
    maxx = np.argmax(frb_tseries)
    hp = frb_tseries[maxx] / 2.
    #Finding the half-power points
    hpp1 = (np.abs(frb_tseries[:maxx] - hp)).argmin()
    hpp2 = (np.abs(frb_tseries[maxx:] - hp)).argmin() + maxx

    FWHM = (1.*hpp2-hpp1)/params.tfactor
    assert FWHM>0, (f'FWHM calculation went wrong somewhere. HPP points, '
                    f'maxx point and FWHM are {hpp1} {hpp2} {maxx} {FWHM}')
    return FWHM, np.max(tscrunch(frb_tseries, params.tfactor))

def start_logging(ctl, db_d):
    if os.path.exists(ctl):
        logger = open(ctl, 'a')
    else:
        logger = open(ctl, 'a')
        logger.write(f'#This is the furby catalogue for {db_d} directory\n')
        logger.write(f'#Created on : {time.asctime()}\n\n')
        existing_furbies = glob.glob(db_d + "furby_*")
        if len(existing_furbies) > 0:
            logger.write(f'#The following furbies were found to be present in '
                         f'the directory before the creation of this catalogue:\n')
            for furby in existing_furbies:
                logger.write(f'#{furby}\n')
            logger.write("\n")
        logger.write("#ID\tDM\tFWHM\t\tTAU0\t\tSNR\tSIGNAL\n")
    return logger

def check_for_permissions(db_d):
    if not os.path.exists(db_d):
        try:
            print(f'The database directory: {db_d} does not exist. '
                  f'Attempting to create it now.')
            os.makedirs(db_d)
        except OSError as error:
            print("Attempt to create the database directory failed")
            raise error

    if os.access(db_d, os.W_OK) and os.access(db_d, os.X_OK):
        return 
    else:
        print(f'Do not have permissions to write/create in the '
              f'database directory: {db_d}')
        print("Exiting...")
        sys.exit(1)


def get_values(input_value, num_furby, msg):
    """
    Get params values for each Furby
    """
    if isinstance(input_value, float):
        new_value = input_value * np.ones(num_furby)
    elif isinstance(input_value, list) and len(input_value) ==1:
        new_value = input_value[0] * np.ones(num_furby)
    elif isinstance(input_value, list) and len(input_value) ==2:
        new_value = np.random.uniform(input_value[0], input_value[1], num_furby)
    else:
        raise IOError(f'Invalid input for {msg}')

    return new_value

def furby_id(params):
    ID_series = params.ID_series
    N_per_IDseries = params.N_per_IDseries

    ID = np.random.randint(
                (ID_series*N_per_IDseries + 1),
                (ID_series+1)*params.N_per_IDseries, 1)[0] 
    ID = str(ID).zfill(int(np.log10(N_per_IDseries)))

    name = f'furby_{ID}'
    return name


def main(args):
    params = AttrDict(parse_config(CONFIG_FILE))

    # tfactor: 40.96 microseconds
    # We dont want dmsmear to be approximated beyond 5 kHz chw. 
    # So ffactor = chw/ 0.005
    params.update({"tfactor":int(params.tsamp / 0.00001024), 
                   "ffactor":int(((params.ftop-params.fbottom)/params.nch)/0.01)})

    database_directory = os.path.join(args.D, '')
    

    if args.plot and args.Num > 1:
        raise IOError("Sorry cannot plot more than one furby at a time")

    if not args.plot:
        check_for_permissions(database_directory)
        catalogue = database_directory+"furbies.cat"
        logger = start_logging(catalogue, database_directory)

    if args.v:
        print("Starting FRB Generator...")
    tsamp = params.tsamp    # seconds
    nchan = params.nchan
    supported_kinds = ["slope", "smooth_envelope", "two_peaks", "three_peaks", 
                       "ASKAP"]
    
    snrs   = get_values(args.snr, args.Num, "SNR")
    dms    = get_values(args.dm, args.Num, "DM")
    widths = get_values(args.width, args.Num, "Width")*1e-3/params.tsamp
    
    # --------------------------------------------------------------------------
    for num in range(args.Num):
        dm = dms[num]
        snr = snrs[num]
        width = widths[num]

        kind = args.kind if args.kind else np.random.choice(supported_kinds) 
        assert kind in supported_kinds

        while True:
            name = furby_id(params)
            if os.path.exists(database_directory + name):
                continue
            else:
                break

        tau0 = np.abs(np.random.normal(loc = dm/1000, scale=2, size=1))[0]
      
        # = half of nsamps required for the gaussian. i.e. 
        # The total nsamps will be 10 * sigma.  
        nsamps_for_gaussian    = max(5*width, 1)            
        nsamps_for_exponential = int(6*tau0*(\
                                    (params.ftop+params.fbottom)/2/\
                                     params.fbottom)**params.scattering_index)

        if args.v:
            print("Randomly generated Parameters:")
            print(f'ID= {ID}, SNR= {snr}, Width= {width*tsamp*1e3} ms, '
                  f'DM= {dm}, tau0= {tau0*tsamp*1e3:.10f}ms, kind= {kind}')

            print("Getting pure FRB")
        try:
            frb = get_pure_frb(snr=snr, width = width, nchan=nchan, 
                               nsamps=nsamps_for_gaussian, params=params)
        except RuntimeError as error:
            print(error)
            continue
      
        pure_signal = np.sum(frb.flatten())
        if args.v:
            print("Creating frequency structure")
        frb, f = create_freq_structure(frb, kind=kind, params=params)

        pure_width = pure_signal / np.max(frb.sum(axis=0))/params.tfactor
        pure_snr   = pure_signal / (np.sqrt(nchan) \
                        * get_clean_noise_rms(params) * np.sqrt(pure_width))
      
        if args.v:
            print(f'Pure signal (input) = {pure_signal}, '
                  f'signal after freq_struct = {np.sum(frb.flatten())}')
            print(f'Pure_snr = {pure_snr:.2f}, '
                  f'pure_width = {pure_width*params.tsamp*1e3:.3f} ms')

        #if args.v:
        #  print "Applying Bandpass"
        #frb = apply_bandpass(frb)
        #if args.v:
        #  print "Signal after bandpass calib = {0}".format(np.sum(frb.flatten()))
      
        if args.v:
            print("Scattering the FRB")
        if nsamps_for_exponential==0:
            print("Tau0 = 0, no scattering applied")
        else:
            frb = scatter(frb, tau0, nsamps_for_exponential, pure_snr, params)
        sky_signal = np.sum(frb.flatten())
        sky_frb_tseries = np.sum(frb, axis=0)
        #sky_frb_peak = np.max( tscrunch(sky_frb_tseries, params.tfactor)   )
        sky_frb_top_hat_width = sky_signal / np.max(sky_frb_tseries) / params.tfactor
        #sky_frb_top_hat_width = sky_signal / sky_frb_peak
        sky_snr = sky_signal / (get_clean_noise_rms(params) \
                            * np.sqrt(nchan) * np.sqrt(sky_frb_top_hat_width))

        if args.v:
            print(f'Sky_signal = {sky_signal}, '
                  f'sky_width = {sky_frb_top_hat_width*params.tsamp*1e3} ms, '
                  f'sky_snr = {sky_snr:.2f}')
      
        frb_b_d = frb.copy()      #FRB before dispersing
      
        if args.v:
            print("Dispersing")
        # Remember, nsamps_for_gaussian is already half the number of samples
        frb, undispersed_tseries, NSAMPS = disperse(frb, dm, 
                                                pre_shift=nsamps_for_gaussian,
                                                dmsmear = args.dmsmear,
                                                params=params) 

        signal_after_disp = np.sum(frb.flatten())

        FWHM, maxima = get_FWHM(undispersed_tseries, params)

        if args.v:
            print("Scrunching")
            print("")
        scrunched_frb = tscrunch(frb, params.tfactor)
        signal_after_scrunching = np.sum(scrunched_frb.flatten())

        #final_top_hat_width = signal_after_scrunching / maxima		
        #comes out in samples
        final_top_hat_width = signal_after_scrunching \
                            / np.max(undispersed_tseries) / params.tfactor
        output_snr = signal_after_scrunching / (get_clean_noise_rms(params) \
                            * np.sqrt(nchan) *  np.sqrt(final_top_hat_width))

        if args.v:
            print(f'Input signal: {pure_signal}, Sky_signal: {sky_signal}, '
                  f'Output signal: {signal_after_scrunching}')
            print(f'Input SNR: {snr}, Sky SNR: {sky_snr}, '
                  f'Output SNR: {output_snr}')
            print(f'Final_top_hat_width: {final_top_hat_width*tsamp*1e3} ms\n')
      
        final_frb = scrunched_frb.astype('float32')

        if args.v:
            print("Generating Header")
        header_size = 16384                            # bytes
        hdr_params = {
              "HDR_VERSION":  1.0,
              "HDR_SIZE": header_size,
              "TELESCOPE":    "MOST",
              "ID":   ID,
              "SOURCE":	name,
              "FREQ": (params.ftop + params.fbottom)/2,# MHz
              "BW":   params.fbottom - params.ftop,    # MHz. Negative to indicate that the first channel has highest frequency
              "NPOL": 1,
              "NBIT": 32,
              "TSAMP":    tsamp * 1e6,      	       # usec  --Dont change this, dspsr needs tsamp in microseconds to work properly
              "NSAMPS":	NSAMPS,
              "UTC_START":    "2018-02-12-00:00:00",   # Never happened :P
              "STATE":    "Intensity",
              "OBS_OFFSET":   0,                       # samples
              "NCHAN":    nchan,
              "ORDER":    order,                       # This is ensured later in the code while writing data with flatten(order=O)
              "FTOP": params.ftop,                     # MHz
              "FBOTTOM": params.fbottom,               # MHz
              "TRACKING": "false",
              "FB_ENABLED":   "true",
              "INSTRUMENT":   "MOPSR",
              "SNR":  output_snr,
              "SKY_SNR":  sky_snr,
      	      "WIDTH":	final_top_hat_width*tsamp*1e3, # milliseconds
              "SIGNAL":   signal_after_scrunching,
              "SKY_SIGNAL":   sky_signal,
              "WIDTH_GAUSS":  width*tsamp*1e3,         # milliseconds
              "FWHM": FWHM*tsamp*1e3,                  # milliseconds
              "DM":   dm,                              # pc / cm^3
              "DMSMEAR":  str(args.dmsmear).upper(),
              "TAU0": tau0*tsamp*1e3,                  # milliseconds
              "KIND": kind,                            # kind of frequency structure put into the FRB
              "PEAK": maxima,                          # peak signal of the FRB
              }

        header_string = make_psrdada_header(header_size, hdr_params)
        if not args.plot:
            if args.v:
                print(f'Saving the FRB in {database_directory}')

            with open(database_directory+name, 'wb') as out:
                out.write(header_string.encode())
                if params.order == "TF":
                    # column-major, since we are writing data in TF format.
                    order = 'F'				
                if params.order == "FT":
                    # row-major, since we are writing data in FT format.
                    order = 'C'				
                final_frb.flatten(order=order).tofile(out)
            logger.write(f'{ID}\t{dm:.2f}\t{FWHM*tsamp:.8f}\t{tau0*tsamp:.10f}'
                         f'\t{output_snr:.2f}\t{signal_after_scrunching:.2f}\n')
          
            print("Name : ", hdr_params["SOURCE"])

        #-----------------------------------------------------------------------
    if not args.plot:
        logger.close()

    if args.plot:
        if args.v:
            print("Plotting")
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
        plt.plot(tscrunch(undispersed_tseries, params.tfactor))
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
