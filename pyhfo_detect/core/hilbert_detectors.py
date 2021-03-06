#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:23:07 2018

Hilbert detector used in Kucewicz et al. 2015 and its newer versions.

Ing.,Mgr. (MSc.) Jan Cimbálník, PhD.
Biomedical engineering
International Clinical Research Center
St. Anne's University Hospital in Brno
Czech Republic
&
Mayo systems electrophysiology lab
Mayo Clinic
200 1st St SW
Rochester, MN
United States
"""

# Std imports
from multiprocessing import Pool

# Third pary imports
import numpy as np
from scipy.signal import butter, hilbert, filtfilt

# Local imports
from ..io.data_operations import create_output_df, correct_boundary_dtype

def hilbert_detector_1_0(data,fs,low_fc, high_fc, threshold = 3,
                         window = 10, window_overlap = 0,
                         band_spacing = 'linear', num_bands = 300, 
                         cyc_th = 1, gap_th = 1, mp = 1):

    """
    Slightly modified algorithm which uses the 2D HFO hilbert detection.\n
    Avoids using image processing libraries.\n
    
    KUCEWICZ, Michal T., Jan CIMBÁLNÍK, Joseph Y MATSUMOTO, \n
    Benjamin H BRINKMANN, Mark BOWER, Vincent VASOLI, Vlastimil SULC, \n
    Fred MEYER, W R MARSH, S M STEAD and Gregory a WORRELL. \n
    High frequency oscillations are associated with cognitive processing \n
    in human recognition memory. Brain : a journal of neurology [online]. \n
    2014, pp. 1–14. \n

    Parameters:
    ----------
        data - name (and path) of the file (string)\n
        fs - sampling frequency (int)\n
        low_fc(float) - low cut-off frequency\n
        high_fc(float) - high cut-off frequency\n
        window(float) - statstical window in secs\n
        window_overlap(float) - stat. window overlap (range 0-1)\n
        band_spacing(string) - options: 'linear', 'log' ('linear')\n
        num_bands(int) - number of bands if band_spacing = log (300)\n
        threshold(float) - threshold for detection (3)\n
        cyc_th(float) - minimum number of cycles to detect (1)\n
        gap_th(float) - number of cycles for gaps (1)\n
        mp(int) - number of cores to use (def = 1)\n

    Returns:
    ---------
        df_out(pandas.DataFrame) - output dataframe with detections\n
    """

    # Create output dataframe 
    
    df_out = create_output_df(fields=['freq_min','freq_max','freq_at_max',
                                      'max_amplitude'])

    # Initial values
    win_start = 0
    win_size = int(window * fs)
    
    

    df_idx = 0
    
    

    # Construct filter cut offs
    
    if band_spacing == 'log':
        low_fc = float(low_fc)
        high_fc = float(high_fc)
        coffs = np.logspace(0,np.log10(high_fc),num_bands)
        coffs = coffs[(coffs>low_fc) & (coffs<high_fc)]
        freq_span = len(coffs) - 1
    elif band_spacing == 'linear':
        coffs = np.arange(low_fc, high_fc)
        freq_span = (high_fc - low_fc) - 1

    #Create a pool of workers
    if mp > 1: work_pool = Pool(mp)

    #Start the looping
    while win_start+win_size <= len(data):
        
        win_end = win_start + win_size

        x = data[win_start:win_end]

        tdetects_concat = []
        if mp > 1:
        
            
            # Run the filters in their threads and return the result
            iter_mat =  [[x, fs, coffs[i], coffs[i+1],
                          cyc_th, gap_th, threshold] for i in range(freq_span)]
            tdetects_concat = work_pool.map(band_z_score_detect,iter_mat)

            work_pool.join
        else:
            for i in range(freq_span):
                bot = coffs[i]
                top = coffs[i+1]
                
                args = [x, fs, bot, top, cyc_th, gap_th, threshold]
                
                tdetects_concat.append(band_z_score_detect(args))

         # Process detects
        detects = np.array([det for band in tdetects_concat for det in band])

        outlines = []
        if len(detects):
            while sum(detects[:,0] != 0):
                det_idx = np.where(detects[:,0] !=0)[0][0]
                HFO_outline = []
                outlines.append(np.array(run_detect_branch(detects,
                                                           det_idx,
                                                           HFO_outline)))

        # Get the detections
        for ouline in outlines[:10]:
            start = min(ouline[:,1])
            stop = max(ouline[:,2])
            freq_min = int(ouline[0,0])
            freq_max =  int(ouline[-1,0])
            frequency_at_max = int(ouline[np.argmax(ouline[:,3]),0])
            max_amplitude = max(ouline[:,3])

            event_start = int(start + win_start)
            event_stop = int(stop + win_start)
            df_out.loc[df_idx] = [event_start, event_stop,
                                  freq_min, freq_max, frequency_at_max,
                                  max_amplitude]
            df_idx += 1
            
        win_start += win_size

        #Plot the image
#        if plot_flag:
#            f, axarr = plt.subplots(2, sharex = True)
#            axarr[0].plot(data)
#            plt.ion()
#            axarr[1].imshow(hfa, aspect='auto',origin='lower')
#            labels = [(i*100)+int(low_mat_fc) for i in range(int(np.ceil((high_mat_fc-low_mat_fc)/100.0)))]
#            x_pos = [i*100 for i in range(int(np.ceil((high_mat_fc-low_mat_fc)/100)))]
#            plt.yticks(x_pos,labels)
#            plt.show(block=False)
#            plt.waitforbuttonpress(1)
        
    if mp > 1: work_pool.close()
        
    correct_boundary_dtype(df_out)
    return df_out

# =============================================================================
# Subfunctions
# =============================================================================

def band_z_score_detect(args):
    
    x_cond = args[0]
    fs = args[1]
    bot = args[2]
    top = args[3]
    cyc_th = args[4]
    gap_th = args[5]
    threshold = args[6]
    
    tdetects = []
    thresh_sig = np.zeros(len(x_cond), dtype='int8')
    
    b,a = butter (3,bot/(fs/2),'highpass')
    fx = filtfilt (b,a,x_cond)

    b,a = butter (3,top/(fs/2),'lowpass')
    fx = filtfilt (b,a,fx)

    #Compute the z-scores

    ms = np.mean(fx)
    sd = np.std(fx)
    fx = [(x-ms)/sd for x in fx]

    hfx = np.abs(hilbert(fx))

     # Create dot product and threshold the signal
    thresh_sig[:] = 0
    thresh_sig[hfx>threshold] = 1

    #Now get the lengths
    idx = 0
    th_idcs = np.where(thresh_sig == 1)[0]
    gap_samp = round(gap_th * fs / bot)
    while idx < len(th_idcs)-1:
        if (th_idcs[idx+1] - th_idcs[idx]) == 1:
            start_idx = th_idcs[idx]
            while idx < len(th_idcs)-1:
                if (th_idcs[idx+1] - th_idcs[idx]) == 1:
                    idx += 1  # Move to the end of the detection
                    if idx == len(th_idcs)-1:
                        stop_idx = th_idcs[idx]
                        #Check for number of cycles
                        dur = (stop_idx-start_idx) / fs
                        cycs = bot * dur
                        if cycs > cyc_th:
                            # Carry the amplitude and frequency info
                            tdetects.append([bot,start_idx,stop_idx,
                                             max(hfx[start_idx:stop_idx])])
                else:  # Check for gap
                    if (th_idcs[idx+1] - th_idcs[idx]) < gap_samp:
                        idx +=1
                    else:
                        stop_idx = th_idcs[idx]
                        #Check for number of cycles
                        dur = (stop_idx-start_idx) / fs
                        cycs = bot * dur
                        if cycs > cyc_th:
                            tdetects.append([bot,start_idx,stop_idx,
                                             max(hfx[start_idx:stop_idx])])
                        idx += 1
                        break
        else:
            idx += 1
            
    return tdetects

def detection_overlap_check(gs,dd):
    """
    Evaluates if two detections overlap

    Paramters:
    ----------
    gs - gold standard detection [start,stop] (list)\n
    dd - detector detection [start,stop] (list)\n

    Returns:
    --------
    overlap - boolean\n

    """

    overlap = False

    if (dd[1] >= gs[0]) and (dd[1] <= gs[1]): # dd stop in gs
        overlap = True

    if (dd[0] >= gs[0]) and (dd[0] <= gs[1]): # dd start in gs
        overlap = True

    if (dd[0] <= gs[0]) and (dd[1] >= gs[1]): # gs inside dd
        overlap = True

    return overlap

def run_detect_branch(detects,det_idx,HFO_outline):
    """
    Function to process detections from new hilbert detector.
    
    HFO_outline structure:
    [0] - bands in which the detection happened
    [1] - starts for each band
    [2] - stop for each band
    [3] - 
    """

    HFO_outline.append(np.copy(detects[det_idx,:]))

    # Create a subset for next band
    next_band_idcs = np.where(detects[:,0]==detects[det_idx,0]+1)
    if not len((next_band_idcs)[0]):
        # No detects in band - finish the branch
        detects[det_idx,0] = 0 #Set the processed detect to zero
        return HFO_outline
    else:
        # Get overllaping detects
        for next_det_idx in next_band_idcs[0]:
            #detection = detects[0]
            if detection_overlap_check([detects[det_idx,1],detects[det_idx,2]],
                                                      [detects[next_det_idx,1],
                                                       detects[next_det_idx,
                                                               2]]):
                #Go up the tree
                run_detect_branch(detects,next_det_idx,HFO_outline)

        detects[det_idx,0] = 0
        return HFO_outline