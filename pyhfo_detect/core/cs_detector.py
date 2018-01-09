#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:27:15 2017

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

# Third pary imports
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.special import gammaincinv

# Local imports
from ..io.data_operations import create_output_df

# %% CS detector

#def cs_detect(data, fs, low_fc, high_fc,
#              threshold, band_detections = True,
#              stat_window_size = 10, cycs_per_detect = 4):
#    """
#    CS detection algorithm.
#    
#    CIMBÁLNÍK, Jan, Angela HEWITT, Greg WORRELL and Matt STEAD. \n
#    The CS Algorithm: A Novel Method for High Frequency Oscillation \n
#    Detection in EEG. Journal of Neuroscience Methods [online]. \n
#    2017, vol. 293, pp. 6–16. ISSN 01650270.\n
#    Available at: doi:10.1016/j.jneumeth.2017.08.023
#    
#    
#    
#    Parameters:
#    -----------
#    data(1-d numpy array) - raw data\n
#    fs(int) - sampling frequency\n
#    low_fc(float) - low cut-off frequency\n
#    high_fc(float) - high cut-off frequency\n
#    stat_window_size(float) - statistical window size in secs (default = 10)\n
#    det_window_size(float) - number of cycles in secs (default = 5)\n
#    
#    Returns:
#    --------
#    df_out(pandas.DataFrame) - output dataframe with detections\n
#    """    
#
#    # Create output dataframe 
#    
#    df_out = create_output_df()
#    
#    return
    
def cs_detect_beta(data, fs, low_fc, high_fc,
                   threshold, band_detections = True,
                   stat_window_size = 10, cycs_per_detect = 4):
    
    """
    Beta version of CS detection algorithm. Which was used to develop \n
    CS detection algorithm.
    
    CIMBÁLNÍK, Jan, Angela HEWITT, Greg WORRELL and Matt STEAD. \n
    The CS Algorithm: A Novel Method for High Frequency Oscillation \n
    Detection in EEG. Journal of Neuroscience Methods [online]. \n
    2017, vol. 293, pp. 6–16. ISSN 01650270.\n
    Available at: doi:10.1016/j.jneumeth.2017.08.023
    
    
    
    Parameters:
    -----------
    data(1-d numpy array) - raw data\n
    fs(int) - sampling frequency\n
    low_fc(float) - low cut-off frequency\n
    high_fc(float) - high cut-off frequency\n
    band_detections - output includes bandwise detections (default=True)\n
    stat_window_size(float) - statistical window size in secs (default = 10)\n
    det_window_size(float) - number of cycles in secs (default = 4)\n
    
    Returns:
    --------
    df_out(pandas.DataFrame) - output dataframe with detections\n
    """    
    
    # Create output dataframe 
    
    df_out = create_output_df(fields=['low_fc','high_fc',
                                      'amp','fhom','dur','prod',
                                      'type'])
    
    # TODO - move the settings to a data file
    
    BAND_STARTS = [44,52,62,73,86,102,121,143,169,199,237,280,332,392,464,549,650]
    BAND_CENTERS = [52,62,73,86,102,121,143,169,199,237,280,332,392,464,549,650,769]
    BAND_STOPS = [62,73,86,102,121,143,169,199,237,280,332,392,464,549,650,769,909]
    nyquist = (fs / 2) - 1
    N_BANDS = len([x for x in BAND_STOPS if x <= nyquist])

    AMP_KS = [1.13970939, 0.90183703, 1.26436011, 1.03769074, 0.85849874, 0.94987266, 0.80845992, 1.67940963, 1.04080418, 1.24382275, 1.60240884, 1.10695014, 1.17010383, 0.88196648, 1.04245538, 0.70917389, 2.21536184]
    AMP_THETAS = [1.65277574, 3.48530721, 2.98961385, 11.54210813, 18.93869204, 10.11982852, 10.53609476, 5.91562993, 11.09205920, 8.84505258, 6.92641365, 18.89938640, 23.76501855, 30.42839963, 27.30653900, 22.48544327, 0.08329301]
    AMP_OFFSETS = [6.41469207, 6.39345582, 6.40000914, 7.32380252, 8.32055181, 8.58559154, 8.27742490, 9.97358643, 10.49550234, 12.41888242, 15.86698463, 21.34769474, 21.89082728, 17.18456284, 18.93825748, 16.30660646, 7.69330283]

    FHOM_KS = [1.66197234, 1.00540463, 1.79692941, 1.15586041, 1.02455216, 1.21727010, 1.12610054, 0.70076969, 0.98379084, 1.54577304, 1.51861533, 1.23976157, 1.43199934, 1.17238163, 0.58636256, 1.12205645, 0.09508500]
    FHOM_THETAS = [4.71109440, 6.05698300, 3.84238418, 6.23370380, 7.89603172, 7.87712768, 8.45272550, 10.00101086, 6.58376596, 3.53488296, 5.27183305, 6.36805821, 7.56839088, 8.24757240, 14.90634368, 18.85016717, 260.59793175]
    FHOM_OFFSETS = [8.16878678, 10.55275451, 8.07166998, 8.07086829, 8.94105317, 7.75703706, 7.89853517, 7.14019430, 8.17322770, 8.55596745, 6.90226263, 7.17550663, 7.77665423, 9.07663424, 14.82474643, 20.20094041, 17.71110000]

    PROD_KS = [0.84905609, 1.01954096, 1.58872304, 1.88690171, 1.27908635, 1.06280570, 0.92824868, 1.49057163, 1.38457279, 2.14489528, 1.35910370, 1.44452982, 1.89318549, 0.92291990, 0.97845756, 1.42279817, 0.09633877]
    PROD_THETAS = [5.84241875, 2.72996718, 3.68246691, 6.69128325, 10.43308700, 11.90997028, 13.04316866, 6.93301203, 8.31241387, 4.62399907, 7.32859575, 11.79756235, 12.32143937, 26.04107818, 17.76146131, 18.81871472, 195.40205368]
    PROD_OFFSETS = [16.32704840, 19.47650057, 16.18710622, 16.34553372, 19.25022797, 18.30852676, 18.15222002, 18.98117587, 19.84269749, 21.64225522, 24.19732683, 25.65335524, 26.52948797, 24.05945634, 38.10559556, 34.94781992, 20.41020467]

    DUR_KS = [0.94831016, 1.20644724, 1.19723676, 1.24834990, 1.72876216, 1.88991915, 1.45709687, 1.76097598, 1.42626762, 1.81104799, 2.09379726, 2.28979796, 1.92883462, 2.15155894, 1.14187099, 1.42071107, 0.38495461]
    DUR_THETAS = [0.04543605, 0.04113687, 0.03842913, 0.03390445, 0.02099894, 0.01687568, 0.01622539, 0.00794505, 0.00857187, 0.00499798, 0.00489236, 0.00462047, 0.00532479, 0.00263985, 0.00623849, 0.01249162, 0.00115305]
    DUR_OFFSETS = [0.10320000, 0.09316255, 0.06500000, 0.05480000, 0.04420000, 0.03220000, 0.02820000, 0.02580000, 0.02291436, 0.01940000, 0.01760000, 0.01500000, 0.01180000, 0.01000000, 0.01180000, 0.01500000, 0.00844698]

    edge_thresh = 0.1

    df_i = 0

    stat_win_samp = int(fs * stat_window_size)

    start_samp = 0
    stop_samp = start_samp + stat_win_samp
    
    conglom_arr = np.zeros([N_BANDS,stat_win_samp],'bool')
    
    while stop_samp <= len(data):

        x = data[start_samp:stop_samp]

        event_cnt = 0
        for band_idx in range(N_BANDS):

            wind_secs = cycs_per_detect / BAND_CENTERS[band_idx]

            b,a = butter(3,[(BAND_CENTERS[band_idx] / 4)/(fs/2),BAND_STOPS[band_idx]/(fs/2)],'bandpass')
            bp_x = filtfilt(b,a,x)
            b,a = butter(3,[BAND_STARTS[band_idx]/(fs/2),BAND_STOPS[band_idx]/(fs/2)],'bandpass')
            np_x = filtfilt(b,a,x)

            h = hilbert(np_x)
            x_amps = np.abs(h)

            np_x_f = np.cos(np.angle(h))

            h = hilbert(bp_x)
            bp_x_f = np.cos(np.angle(h))
            x_fhoms = sliding_snr(np_x_f,bp_x_f,fs,wind_secs)

            # Normalisation
            p1 = round(stat_win_samp / 3)
            p2 = round((2 * stat_win_samp) / 3)
            sort_arr = np.sort(x_amps)
            amp_dev = (sort_arr[p2] - sort_arr[p1]) / 2
            x_amps = (x_amps - sort_arr[p2]) / amp_dev
            sort_arr = np.sort(x_fhoms)
            fhom_dev = (sort_arr[p2] - sort_arr[p1]) / 2
            x_fhoms = (x_fhoms - sort_arr[p2]) / fhom_dev

            x_prods = x_amps * x_fhoms
            for i in range(len(x_prods)):
                if (x_fhoms[i] < 0) and (x_amps[i] < 0):
                    x_prods[i] = -x_prods[i]
                if x_prods[i] < 0:
                    x_prods[i] = -np.sqrt(-x_prods[i])
                else:
                    x_prods[i] = np.sqrt(x_prods[i])

            sort_arr = np.sort(x_prods)
            prod_dev = (sort_arr[p2] - sort_arr[p1]) / 2
            x_prods = (x_prods - sort_arr[p2]) / prod_dev

            # Threshold calculation
            amp_min = inverse_gamma_cdf(threshold,
                                        AMP_KS[band_idx],
                                        AMP_THETAS[band_idx],
                                        AMP_OFFSETS[band_idx])
            amp_max = 5 * inverse_gamma_cdf(.99,
                                            AMP_KS[band_idx],
                                            AMP_THETAS[band_idx],
                                            AMP_OFFSETS[band_idx])
            fhom_min = inverse_gamma_cdf(threshold,
                                         FHOM_KS[band_idx],
                                         FHOM_THETAS[band_idx],
                                         FHOM_OFFSETS[band_idx])
            fhom_max = 5 * inverse_gamma_cdf(.99,
                                             FHOM_KS[band_idx],
                                             FHOM_THETAS[band_idx],
                                             FHOM_OFFSETS[band_idx])
            prod_min = inverse_gamma_cdf(threshold,
                                         PROD_KS[band_idx],
                                         PROD_THETAS[band_idx],
                                         PROD_OFFSETS[band_idx])
            prod_max = 5 * inverse_gamma_cdf(.99,
                                             PROD_KS[band_idx],
                                             PROD_THETAS[band_idx],
                                             PROD_OFFSETS[band_idx])
            dur_min = inverse_gamma_cdf(threshold,
                                        DUR_KS[band_idx],
                                        DUR_THETAS[band_idx],
                                        DUR_OFFSETS[band_idx])
            dur_max = 5 * inverse_gamma_cdf(.99,
                                            DUR_KS[band_idx],
                                            DUR_THETAS[band_idx],
                                            DUR_OFFSETS[band_idx])

            # Detect
            j = 0
            
            while j < len(x):
                if x_prods[j] > edge_thresh:
                    event_start = j
                    j += 1
                    while j < len(x) and x_prods[j] > edge_thresh:
                        j += 1

                    event_stop = j

                    #Calculate duration
                    dur = float(event_stop - event_start + 1) / fs
                    if (dur < dur_min) or (dur > dur_max):
                        j += 1
                        continue
                    dur_scale = np.sqrt(dur / wind_secs)

                    #Calculate amplitude
                    amp = np.mean(x_amps[event_start:event_stop])
                    amp = amp * dur_scale
                    if (amp < amp_min) or (amp > amp_max):
                        j += 1
                        continue

                    #Calculate fhom
                    fhom = np.mean(x_fhoms[event_start:event_stop])
                    fhom = fhom * dur_scale
                    if (fhom < fhom_min) or (fhom > fhom_max):
                        j += 1
                        continue

                    #Calculate product
                    prod = np.mean(x_prods[event_start:event_stop])
                    prod = prod * dur_scale
                    if (prod < prod_min) or (prod > prod_max):
                        j += 1
                        continue

                    event_cnt += 1
                    
                    conglom_arr[band_idx,event_start:event_stop] = 1

                    #Put in output-df
                    df_out.loc[df_i] = [event_start, event_stop,
                                        BAND_STARTS[band_idx],
                                        BAND_STOPS[band_idx],
                                        amp, fhom, dur, prod,
                                        'band']
                    
                    df_i += 1


                else:

                    j += 1

        # Create congloms
        conglom_1d = np.sum(conglom_arr,0)
        new_det_idx = len(df_out)
        if any(conglom_1d):
            det_locs = np.where(conglom_1d)[0]
            starts_det_locs = np.where(np.diff(det_locs)>1)[0]+1
            stops_det_locs = np.where(np.diff(det_locs)>1)[0]
            if len(starts_det_locs):
                det_starts = np.concatenate([[det_locs[0]],
                                             det_locs[starts_det_locs]])
                det_stops = np.concatenate([det_locs[stops_det_locs],
                                            [det_locs[-1]]])
            else:
                det_starts = np.array([det_locs[0]])
                det_stops = np.array([det_locs[-1]])
                
            det_stops += 1
            
            new_det_idx -= event_cnt
            
            sub_df = df_out.loc[new_det_idx:]
            
            # Insert congloms
            for event_start,event_stop in zip(det_starts,det_stops):
                det_df = sub_df.loc[(sub_df.event_start >= event_start)
                                    & (sub_df.event_stop <= event_stop)]
                low_fc = det_df.loc[:,'low_fc'].min()
                high_fc = det_df.loc[:,'high_fc'].max()
                amp = det_df.loc[:,'amp'].max()
                fhom = det_df.loc[:,'fhom'].max()
                prod = det_df.loc[:,'amp'].max()
                dur = float(event_stop - event_start) / fs
                
                df_out.loc[df_i] = [event_start, event_stop,
                                    low_fc, high_fc,
                                    amp, fhom, dur, prod,
                                    'conglom']
            
                df_i += 1

        # Reset conglom array
        conglom_arr[:,:] = 0
        
        # Adjust starts / stops
        if len(df_out):
            df_out.loc[new_det_idx:,'event_start'] += start_samp
            df_out.loc[new_det_idx:,'event_stop'] += start_samp

        start_samp += stat_win_samp
        stop_samp += stat_win_samp
        
    if not band_detections:
        df_out = df_out[~(df_out.type == 'band')]
        df_out.reset_index(drop=True,inplace=True)
        
    return df_out

# =============================================================================
# Subfunctions    
# =============================================================================

def inverse_gamma_cdf(p,k,theta,offset):
    """
    Inverse gamma cumulative distribution function.
    """

    x = gammaincinv(k,p)
    x = (x * theta) + offset

    return x

def sliding_snr(np_x,bp_x,Fs,wind_secs):
    """
    "Signal-to-noise ratio" like metric that compares narrow band and broad
    band signals to eliminate increased power generated by sharp transients.
    
    Parameters:
    -----------
    np_x - narrow band signal\n
    bp_x - broad band signal\n
    fs - sampling frequency\n
    wind_secs - sliding window size (seconds)\n
    
    Returns:
    --------
    snr - "Signal-to-noise ratio" like metric

    """

    #Define starting values
    wind = Fs*wind_secs
    half_wind = int(round(wind/2))
    wind = int(round(wind))

    N = min([len(np_x),len(bp_x)])

    snr=np.zeros([N])

    npxx = 0
    bpxx = 0

    #Fill in the beginning and initial window values
    for i in range(int(wind)):
        t1 = np_x[i]
        npxx = npxx + (t1 * t1)
        t2 = bp_x[i] - t1
        bpxx = bpxx + (t2 * t2)

    np_rms = np.sqrt(float(npxx) / wind)
    bp_rms = np.sqrt(float(bpxx) / wind)

    snr[:half_wind] = (np_rms / bp_rms)

    #Slide the window
    i = 1
    for k in range(int(N-wind+1)):
        p = k + wind - 1

        #Beginning of the window
        t1 = np_x[i]
        npxx = npxx - (t1 * t1)
        t2 = bp_x[i] - t1
        bpxx = bpxx - (t2 * t2)

        #End of the window
        t1 = np_x[p]
        npxx = npxx + (t1 * t1)
        t2 = bp_x[p] - t1
        bpxx = bpxx + (t2 * t2)

        np_rms = np.sqrt(float(npxx) / wind) # Unnecessary to divide by wind
        bp_rms = np.sqrt(float(bpxx) / wind)

        snr[k+half_wind] = (np_rms/bp_rms)

        i += 1

    #Fill in the end
    snr[-half_wind:] = snr[-(half_wind+1)]

    return snr
    
    