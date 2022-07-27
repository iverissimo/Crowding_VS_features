
import numpy as np
import os, sys
import os.path as op
import pandas as pd
import yaml

def get_dva_per_pix(height_cm = 30, distance_cm = 70, vert_res_pix = 1080):

    """ calculate degrees of visual angle per pixel, 
    to use for screen boundaries when plotting/masking
    Parameters
    ----------
    height_cm : int
        screen height
    distance_cm: float
        screen distance (same unit as height)
    vert_res_pix : int
        vertical resolution of screen
    
    Outputs
    -------
    deg_per_px : float
        degree (dva) per pixel
    
    """

    # screen size in degrees / vertical resolution
    deg_per_px = (2.0 * np.degrees(np.arctan(height_cm /(2.0*distance_cm))))/vert_res_pix

    return deg_per_px 


def get_peak_vel_threshold(vel, samples, init_thresh = 150, std_vel_margin = 6, thresh_diff = 1):
    
    """ calculate peak velocity threshold
    Parameters
    ----------
    vel : arr
        velocity values array (deg/sec)
    samples: arr
        samples array (time)
    init_thresh : int/float
        initial threshold value to use
    std_vel_margin: int/float
        standard deviation margin
    thresh_diff: int/float
        minimum difference between thresholds to stop searching
        
    """
    
    # saccades samples with peak velocity above thresh
    curr_sacc_samples = np.zeros(samples.shape)
    curr_sacc_samples[vel > init_thresh] = 1

    # list to append thresholds
    thresholds = [init_thresh, 2*init_thresh]

    # Peak velocity detection threshold
    while np.abs(np.diff(thresholds)[-1]) > thresh_diff: # while threshold diff bigger than 1 deg

        # calculate new thresholds
        thresholds.append((np.mean(vel[curr_sacc_samples<1]) + np.std(vel[curr_sacc_samples<1]) * std_vel_margin))
        # update current sample where we detect saccades
        curr_sacc_samples[vel > thresholds[-1]] = 1
        
    return thresholds[-1]


def get_initial_saccade_onoffset(vel, samples, peak_thresh, min_sacc_dur = 10, sampl_freq = 1000,
                                max_sacc_vel = 1000):

    """ calculate peak velocity threshold
    
    Parameters
    ----------
    vel : arr
        velocity values array (deg/sec)
    samples: arr
        samples array (time)
    peak_thresh : int/float
        peak velocity threshold value to use
    min_sacc_dur: int/float
        minimum saccade duration in ms
    sampl_freq: int
        sampling frequency
    max_sacc_vel: int/float
        max saccadic velocity in degrees/sec

    """
    
    # saccades samples with peak velocity above thresh
    curr_sacc_samples = np.zeros(samples.shape)
    curr_sacc_samples[vel > peak_thresh] = 1
    
    # current saccade onset
    curr_sacc_onset = samples[:-1][np.diff(curr_sacc_samples) == 1]
    # current saccade offset
    curr_sacc_offset = samples[:-1][np.diff(curr_sacc_samples) == -1]

    if len(curr_sacc_onset) > len(curr_sacc_offset):
        curr_sacc_onset = curr_sacc_onset[:-1] # if trial ends before landing point registered, remove that saccade

    ## make list of [on, off] for each detected saccade
    # if saccade duration < X ms, not a saccade
    # if saccade peak velocity > X deg/s, not a saccade
    curr_sacc = [[val, curr_sacc_offset[i]] for i,val in enumerate(curr_sacc_onset) if (curr_sacc_offset[i] - val) >= (min_sacc_dur * 1000)/sampl_freq and \
                np.max(vel[np.where((samples >= val) & (samples <= curr_sacc_offset[i]))[0]]) <= max_sacc_vel]

    return curr_sacc

