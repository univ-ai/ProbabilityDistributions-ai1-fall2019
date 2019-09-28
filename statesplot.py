import numpy as np # imports a fast numerical programming library
import scipy as sp # imports a statistical programming library
import matplotlib as mpl # imports the standard plotting library
import matplotlib.cm as cm # imports plotting colormaps
import matplotlib.pyplot as plt # imports the MATLAB compatible plotting API
import pandas as pd # imports a library to handle data as dataframes
import seaborn.apionly as sns
import json
from collections import defaultdict 





#this mapping between states and abbreviations will come in handy later
states_abbrev = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

def load_states_geom(statesfile):
    with open(statesfile) as fd: # makes sure file is closed.
        data = json.load(fd)
    #adapted from  https://github.com/dataiap/dataiap/blob/master/resources/util/map_util.py
    #load in state geometry
    state2poly = defaultdict(list) # a dictionary where every value is an empty list


    for f in data['features']:
        state = states_abbrev[f['id']] #use our old dictionary to map to a state name
        geo = f['geometry']
        if geo['type'] == 'Polygon':
            for coords in geo['coordinates']:
                state2poly[state].append(coords)
        elif geo['type'] == 'MultiPolygon':
            for polygon in geo['coordinates']:
                state2poly[state].extend(polygon)
    return state2poly

def draw_state(ax, stategeom, **kwargs):
    """
    draw_state(ax, stategeom, color=..., **kwargs)
    
    Automatically draws a filled shape representing the state in
    subplot.
    The color keyword argument specifies the fill color.  It accepts keyword
    arguments that plot() accepts
    """
    for polygon in stategeom:
        xs, ys = zip(*polygon)
        ax.plot(xs, ys, 'k', lw=1)
        ax.fill(xs, ys, **kwargs)

        
def make_map(state2poly, states, label, figsize=(12, 9)):
    """
    Draw a cloropleth map, that maps data onto the United States
    
    Inputs
    -------
    state2poly: state geometry dictionary
    states : Column of a DataFrame
        The value for each state, to display on a map
    label : str
        Label of the color bar

    Returns
    --------
    The map
    """
    fig = plt.figure(figsize=figsize) # create a figure
    ax = plt.gca() # get axes from the figure
    
    if states.max() < 2: # colormap for election probabilities   
        cmap = cm.RdBu
        vmin, vmax = 0, 1
    else:  # colormap for electoral votes, or other values
        cmap = cm.binary
        vmin, vmax = 0, states.max()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    skip = set(['National', 'District of Columbia', 'Guam', 'Puerto Rico',
                'Virgin Islands', 'American Samoa', 'Northern Mariana Islands'])
    for state in states_abbrev.values():
        if state in skip:
            continue
        color = cmap(norm(states.loc[state]))
        draw_state(ax, state2poly[state], color = color)

    #add an inset colorbar
    ax1 = fig.add_axes([0.45, 0.70, 0.4, 0.02])    
    cb1=mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                  norm=norm,
                                  orientation='horizontal')
    ax1.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-180, -60)
    ax.set_ylim(15, 75)
    sns.despine(left=True, bottom=True)
    return ax