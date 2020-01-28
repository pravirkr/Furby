import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

from Furby_reader import Furby_reader
from Furby_reader import Furby_Error

def tscrunch(fsdata, tx, avg=False):
    if tx==1:
        return fsdata

    if len(fsdata.shape)==1:
        endp = int(fsdata.shape[0]/tx) * tx
        tfsdata = fsdata[:endp].reshape(-1, tx).sum(axis=-1)

    elif len(fsdata.shape)==2:
        nr = fsdata.shape[0]
        nc = fsdata.shape[1]

        endpoint = int(nc/tx) * tx

        # P.S.: We lose a few time samples in the end if the tx factor does not 
        # exactly divide the initial number of time samples in the 
        # filterbank file
        tmp = fsdata[:,:endpoint].reshape(nr, nc/tx, tx)  
        tfsdata = tmp.sum(axis=-1)
    else:
        raise(ValueError("Currently only 1D and 2D arrays can be tscrunched"))

    if avg == True:
        tfsdata /= (1.*tx)
    return tfsdata

def fscrunch(data, fx, avg=False):
    if fx == 1:
        return data

    if len(data.shape) != 2:
        raise ValueError("Only 2D arrays with frq along axis=0 can be fscrunched")
    if data.shape[0]%fx != 0:
        print("!!!No of freq channels must be an integer multiple of the Freq scrunch factor!!!")
        sys.exit(1)

    fsdata = np.zeros((data.shape[0]//fx, data.shape[1]))

    for i in range(data.shape[0]):
        fsdata[i//fx] += data[i]
    
    if avg==True:
        fsdata /= fx
    return fsdata


def main(args):
    print(f'Plotting {len(args.furbies)} furby(s)')
    for k, fil in enumerate(args.furbies):
        try:
            f = Furby_reader(fil)
        except Furby_Error as fe:
            if fe.id == 0:
                print(fe.message)
                continue

        # dada format requires tsamp to be in usec
        tres = f.header.TSAMP/1e6		
        print(f'Filename: {f.filename}\nID:{f.header.ID}, SNR:{f.header.SNR}, '
              f'DM:{f.header.DM}, Width(top hat):{f.header.WIDTH} ms, '
              f'Width(FWHM):{f.header.FWHM} ms, Kind:{f.header.KIND}')

        data = f.read_data()
        dm = 0
        if args.dedisp:
            ddata = f.read_data(dd=True)
            dm = f.header.DM
        else:
            ddata = data

        fsdata = fscrunch(ddata,args.freq_sc)
        tfsdata = tscrunch(fsdata, args.t_sc)
        
        tseries = tfsdata.sum(axis=0)*1.0
        fseries = tfsdata.sum(axis=1)*1.0

        toff = 0.5*tres*args.t_sc
        x = np.arange(0,len(tseries))*tres*args.t_sc + toff

        f0 = f.header.FBOTTOM
        fn = f.header.FTOP
        chw = f.header.BW/f.header.NCHAN

        if f.header.BW<0:
            (f0, fn) = (fn, f0)

        fa = f0 + chw/2*args.freq_sc
        fb = fn - chw/2*args.freq_sc
        y  = np.arange(fa, fb+chw*args.freq_sc, chw*args.freq_sc)
	
        extent = [x[0]-toff, x[-1]+toff, fn, f0]

        fig=plt.figure(k, figsize=(6.5,5))

        ax1=plt.subplot2grid((6,8), (0,0), rowspan=5, colspan=6)
        ax1.imshow(tfsdata, interpolation='none', aspect='auto', cmap='afmhot', 
                   extent=extent)
        ax1.set_title(fil+" De-DM: "+str(dm), fontsize=8)
        ax1.set_xlim(0,tfsdata.shape[1])
        ax1.set_ylabel("Freq (MHz)")
	
        ax2=plt.subplot2grid((6,8), (5,0), rowspan=1, colspan = 6, sharex=ax1)
        ax2.plot(x, tseries)
        ax2.set_xlim(x[0]-toff, x[-1]+toff)
        ax2.set_xlabel("Time (s)")

        ax3=plt.subplot2grid((6,8), (0,6),rowspan = 5, colspan = 2, sharey=ax1)
        ax3.plot(fseries, y)
        ax3.set_xlabel("Power")
        ax3.set_ylim(fn, f0)

        plt.subplots_adjust(hspace=0,wspace=0, bottom=0.1)
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax3.get_yticklabels(), visible=False)
        #plt.setp(ax2.get_yticklabels(), visible=False)
        #if not args.one:     
        #    mgr=plt.get_current_fig_manager()
        #    mgr.window.move((k%3)*640, int(k/3)*600)

        if args.pngs:
            print("saving",fil)
            plt.savefig((str(fil))+".png", dpi=200)
            plt.close('all')
            continue

        if(k < len(args.furbies)-1):
            plt.show(block=False)
        input("<Press Enter to see next plot>\n")
        plt.close('all')
    if not args.pngs:
        plt.show()

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("furbies", type=str, nargs='+', help="Furby files to plot")
    a.add_argument("-dd", "--dedisp", action='store_true', 
                   help="Dedisperse the furby? (def=False)", default=False)
    a.add_argument("-fs","--freq_sc", type=int, default=1, 
                   help="Freq scrunch factor (def=1)")
    a.add_argument("-ts","--t_sc", type=int, default=1, 
                   help="Time scrunch factor (def=1)")

    a.add_argument("-pngs", action='store_true', 
                   help="Save pngs instead of plotting")
    args = a.parse_args()
    main(args)




