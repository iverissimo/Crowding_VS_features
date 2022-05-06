
import numpy as np
import os, sys
import os.path as op
import math
import random
import pandas as pd
import yaml

from psychopy import visual, tools, colors, event
import psychopy.tools.colorspacetools as ct
import itertools

import time
import colorsys
import seaborn as sns

def dva_per_pix(height_cm = 30, distance_cm = 70, vert_res_pix = 1080):

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


def circle_points(radius, n_points):

    """ define positions in circle 

    Parameters
    ----------
    radius : list/arr
        list of radius
    n_points: list/arr
        number of points per radius
    
    Outputs
    -------
    circles : list
        list of [x,y] positions per radius
    
    """
    circles = []
    
    for r, n in zip(radius, n_points):
        t = np.arange(0,2*np.pi,2*np.pi/float(n)) #np.linspace(0, 2 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    
    return circles

def get_grid_array(positions, ecc_range, convert2pix = True, screen = [1920, 1080], 
               height_cm = 30, distance_cm = 70, 
               constraint_type = 'ellipse', constraint_bounds_pix = [500,700]):
    
    """ get position array
    needs postion list with positions per ecc
    and ecc range

    Parameters
    ----------
    positions : list/arr
        list of [x,y] positions per ecc
    ecc_range: list/arr
        list with eccs in position
    convert2pix: bool
        if outputted list in pixels or not
    constrain_type: str
        type of position contraint to use eg: 'ellipse', 'square', 'rectangle'
    constraint_bounds_pix: list/arr
        bounds to constraint positions to
    
    Outputs
    -------
    pos_list : arr
        list of [x,y] positions (pix if convert2pix == True)
    ecc_list: arr
        list of ecc per position pair (dva)
    
    """
    
    pos_list = []
    ecc_list = []

    # if converting to pix, then need to convert the bounds to deg
    if convert2pix: 
        constraint_bounds = constraint_bounds_pix * dva_per_pix(height_cm = height_cm, 
                                                            distance_cm = distance_cm, 
                                                            vert_res_pix = screen[-1])
    else:
       constraint_bounds = constraint_bounds_pix 

    for ind, e in enumerate(positions):

        # append list of positions
        for pos in e:
            # check if within bounds
            if constraint_type == 'ellipse' and \
                (((pos[0]**2)/(max(constraint_bounds)**2) + (pos[1]**2)/(min(constraint_bounds)**2)) <= 1):
            
                pos_list.append(list(pos))

                # append eccentricity of these positions
                ecc_list.append(ecc_range[ind])


    if convert2pix:
        pos_list = pos_list/dva_per_pix(height_cm = height_cm, 
                              distance_cm = distance_cm, 
                              vert_res_pix = screen[-1])
    else:
        pos_list = np.array(pos_list)


    return pos_list, np.array(ecc_list) 


def draw_instructions(win, instructions, keys = ['b'], visual_obj = [], 
                      color = (1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), height = 40, #.65,
                        italic = True, alignHoriz = 'center', alignVert = 'center'):
    
    """ draw instructions on screen
    
    Parameters
    ----------
    win : object
        window object to draw on
    instructions : str
        instruction string to draw 
    key: list
        list of keys to skip instructions
    visual_obj: list
        if not empty, should have psychopy visual objects (to add to the display ex: side rectangles to limit display)
        
    """
    
    text = visual.TextStim(win = win,
                        text = instructions,
                        color = color, 
                        font = font, 
                        pos = pos, 
                        height = height,
                        italic = italic, 
                        alignHoriz = alignHoriz, 
                        alignVert = alignVert
                        )
    
    # draw text again
    text.draw()

    if len(visual_obj)>0:
        for w in range(len(visual_obj)):
            visual_obj[w].draw()
            
    win.flip()

    key_pressed = event.waitKeys(keyList = keys)

    return(key_pressed)


def rgb255_2_hsv(arr):
    
    """ convert RGB 255 to HSV
    
    Parameters
    ----------
    arr: list/array
        1D list of rgb values
        
    """
    
    rgb_norm = np.array(arr)/255
    
    hsv_color = np.array(colorsys.rgb_to_hsv(rgb_norm[0],rgb_norm[1],rgb_norm[2]))
    hsv_color[0] = hsv_color[0] * 360
    
    return hsv_color


def near_power_of_2(x,near='previous'):
    """ Get nearest power of 2
    
    Parameters
    ----------
    x : int/float
        value for which we want to find the nearest power of 2
    near : str
        'previous' or 'next' to indicate if floor or ceiling power of 2        
    """
    if x == 0:
        val = 1
    else:
        if near == 'previous':
            val = 2**math.floor(math.log2(x))
        elif near == 'next':
            val = 2**math.ceil(math.log2(x))

    return val


def update_elements(ElementArrayStim, elem_positions = [], grid_pos = [], 
                    elem_color = [204, 204, 204], elem_ori = [353, 7],
                    elem_sf = 4, elem_names = ['bR', 'bL', 'pL', 'pR'], 
                    key_name = ['bR', 'bL']):
    
    """ update element array settings
    
    Parameters
    ----------
    ElementArrayStim: Psychopy object
    	ElementArrayStim to be updated 
    condition_settings: dict
        dictionary with all condition settings
    this_phase: str
        string with name of condition to be displayed
    elem_positions: arr
         numpy array with element positions to be updated and shown (N,2) -> (number of positions, [x,y])
         to be used for opacity update
    grid_pos: arr
        numpy array with element positions (N,2) of whole grid -> (number of positions, [x,y])
    monitor: object
        monitor object (to get monitor references for deg2pix transformation)
    screen: arr
        array with display resolution
    luminance: float or None
        luminance increment to alter color (used for flicker task)
    update_settings: bool
        choose if we want to update settings or not (mainly for color changes)
    new_color: array
        if we are changing color to be one not represented in settings (ca also be False if no new color used)
        
    """
    
    # set number of elements
    nElements = grid_pos.shape[0]

    ## to make colored gabor, need to do it a bit differently (psychopy forces colors to be opposite)
    # get rgb color and convert to hsv
    hsv_color = rgb255_2_hsv(elem_color)
    grat_res = near_power_of_2(ElementArrayStim.sizes[0][0], near = 'previous') # use power of 2 as grating res, to avoid error
    
    # initialise grating
    grating = visual.filters.makeGrating(res = grat_res)
    grating_norm = (grating - np.min(grating))/(np.max(grating) - np.min(grating)) # normalize between 0 and 1
    
    # initialise a base texture 
    colored_grating = np.ones((grat_res, grat_res, 3)) 

    # replace the base texture red/green channel with the element color value, and the value channel with the grating
    colored_grating[..., 0] = hsv_color[0]
    colored_grating[..., 1] = hsv_color[1]
    colored_grating[..., 2] = grating_norm * hsv_color[2]

    elementTex = ct.hsv2rgb(colored_grating) # convert back to rgb

    # update element colors to color of the patch 
    element_color = np.ones((int(np.round(nElements)),3)) 
    
    # update element spatial frequency
    element_sfs = np.ones((nElements)) * elem_sf # in cycles/gabor width

    # get left and right indices from keys names
    L_indices = [ind for ind, k in enumerate(elem_names) if k in key_name and 'L' in k]
    R_indices = [ind for ind, k in enumerate(elem_names) if k in key_name and 'R' in k]

    # make grid and element position lists of lists
    list_grid_pos = [list(val) for _,val in enumerate(grid_pos)]
    list_Lelem_pos = [list(val) for _,val in enumerate(elem_positions[L_indices])]
    list_Relem_pos = [list(val) for _,val in enumerate(elem_positions[R_indices])]

    # get left and right global indices (global, because indices given grid pos)
    L_glob_indices = [list_grid_pos.index(list_Lelem_pos[i]) for i in range(len(list_Lelem_pos))]
    R_glob_indices = [list_grid_pos.index(list_Relem_pos[i]) for i in range(len(list_Relem_pos))]
        
    # update element orientation
    element_ori = np.ones((nElements))
    if len(L_indices)>0:
        element_ori[L_glob_indices] = np.array(elem_ori)[L_indices][0] 
    if len(R_indices)>0:
        element_ori[R_glob_indices] = np.array(elem_ori)[R_indices][0] 

    # combine left and right global indices
    glob_indices = L_glob_indices + R_glob_indices 

    # set element contrasts
    element_contrast = np.zeros(len(grid_pos))
    element_contrast[glob_indices] = 1
    
    # set element opacities
    element_opacities = np.zeros(len(grid_pos))
    element_opacities[glob_indices] = 1

    # set all of the above settings
    ElementArrayStim.setTex(elementTex)
    ElementArrayStim.setSfs(element_sfs)
    ElementArrayStim.setOpacities(element_opacities)
    ElementArrayStim.setOris(element_ori)
    ElementArrayStim.setColors(element_color, 'rgb')
    ElementArrayStim.setContrs(element_contrast)

    return(ElementArrayStim)


def update_grating(GratingStim,
                    elem_color = [204, 204, 204], elem_ori = 7,
                    elem_sf = 4, 
                    elem_pos = (0, 0)):
    
    """ update grating stim settings
    
    Parameters
    ----------
    GratingStim: Psychopy object
    	GratingStim to be updated 
      
    """

    ## to make colored gabor, need to do it a bit differently (psychopy forces colors to be opposite)
    # get rgb color and convert to hsv
    hsv_color = rgb255_2_hsv(elem_color)
    grat_res = near_power_of_2(GratingStim.size[0], near = 'previous') # use power of 2 as grating res, to avoid error
    
    # initialise grating
    grating = visual.filters.makeGrating(res = grat_res)
    grating_norm = (grating - np.min(grating))/(np.max(grating) - np.min(grating)) # normalize between 0 and 1
    
    # initialise a base texture 
    colored_grating = np.ones((grat_res, grat_res, 3)) 

    # replace the base texture red/green channel with the element color value, and the value channel with the grating
    colored_grating[..., 0] = hsv_color[0]
    colored_grating[..., 1] = hsv_color[1]
    colored_grating[..., 2] = grating_norm * hsv_color[2]

    elementTex = ct.hsv2rgb(colored_grating) # convert back to rgb

    # update element colors to color of the patch 
    element_color = np.ones((1,3))

    # set all of the above settings
    GratingStim.tex = elementTex
    GratingStim.pos = elem_pos
    GratingStim.sf = elem_sf
    GratingStim.ori = elem_ori
    GratingStim.setColor(element_color, 'rgb')
    GratingStim.mask = 'gauss'

    return(GratingStim)