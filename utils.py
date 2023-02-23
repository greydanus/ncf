# Nature's Cost Function: Simulating Physics by Minimizing the Action
# Tim Strang, Isabella Caruso, and Sam Greydanus | 2023 | MIT License

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, os
from celluloid import Camera


############################# PLOTTING UTILITIES #############################

def make_video(xs, path, interval=60, ms=10, **kwargs): # xs: [time, N, 2]
    fig = plt.gcf() ; fig.set_dpi(100) ; fig.set_size_inches(3, 3)
    camera = Camera(fig)
    for i in range(xs.shape[0]):
        plt.plot(xs[i][...,0], xs[i][...,1], 'k.', markersize=ms)
        plt.axis('equal') ; plt.xlim(0,1) ; plt.ylim(0,1)
        plt.xticks([], []); plt.yticks([], [])
        camera.snap()
    anim = camera.animate(blit=True, interval=interval, **kwargs)
    anim.save(path) ; plt.close()


############################# EPHEMERIS UTILITIES #############################

''' Before running this code, you'll need to download the raw ephemeris data for the 
    planets (or other celestial bodies) that you want to plot. To do this, go to 
    ttps://ssd.jpl.nasa.gov/horizons/app.html#/. For 'Ephemeris Type' select 'Vector Table'. 
    For 'Target Body' select the body you want (eg. 'Earth'). For Coordinate Center try 
    using 'Solar System Barycenter (SSB) '. For time specification, I manually selected a 
    timespan of five years; the default data interval is 1440 minutes (1 day). Once you've 
    chosen your desired settings, click 'Generate Ephemeris' and then click 'Download Results' 
    when the results load. This will let you download a .txt file to a local directory. 
    Name the file after the planet, eg `earth.txt`. Repeat this process for all the planets 
    you want, saving each of them to a different text file in the same folder. Once you have 
    all the .txt files saved to that folder, you will be ready to run this code.'''

def get_planet_colors():
    return {'sun':'yellow','venus':'orange','mercury':'pink','earth':'blue','mars':'red'}

def plot_planets(df, planets, fig=None):
    colors = get_planet_colors()
    fig = plt.figure(figsize=[5,5], dpi=100) if fig is None else fig
    
    for i, name in enumerate(planets):
        x, y, z = df[name + '_x'], df[name + '_y'], df[name + '_z']
        plt.plot(x, y, '.', alpha=0.33, color=colors[name], label=name + ' (data)', markersize=2)
        #plt.plot(x.iloc[0], y.iloc[0], 'x', alpha=0.33, color=colors[name])
        plt.plot(x.iloc[-1], y.iloc[-1], '.', alpha=0.33, color=colors[name], markersize=9)
    plt.title("Ephemeris data from JPL's Horizon System")
    plt.tight_layout() ; plt.axis('equal')
    return fig

def load_planet(planet_name, data_dir):
    '''Reads a file named, eg.,"earth.txt" with ephemeris data in a vector table format
       downloaded from https://ssd.jpl.nasa.gov/horizons/app.html#/'''
    with open(data_dir + '{}.txt'.format(planet_name), 'r') as f:
        text = f.read()

    main_data = text[text.find('$$SOE')+5:text.find('$$EOE')].split('\n')
    s_xyz = main_data[2::4]

    f_xyz = []
    for l in s_xyz:
        splits = [s.strip(' ').split(' ')[0] for s in l.split('=')[1:]]
        f_xyz.append([float(s)*1e3 for s in splits]) # convert from km to meters
    return np.asarray(f_xyz)

def get_colnames(names):
    '''Generates DataFrame column names for each x, y, z coordinate dimension'''
    colnames = []
    for n in names:
        colnames += [n + '_x', n + '_y', n + '_z']
    return colnames

def get_colformat(coords):
    '''Reshape from [planets, time, xyz] to [time, planets*xyz]'''
    N = coords.shape[0]
    return coords.transpose(1,0,2).reshape(-1,N*3)

def process_raw_ephemeris(planets, data_dir, last_n_days=None):
    '''Loads raw ephemeris files for a list of planet names, organizes the data in a DataFrame,
       and then saves the DataFrame as a csv in the same directory as the raw files.'''
    
    if os.path.exists(data_dir + 'ephemeris_ablate.csv'):
        print('Loading {}...'.format(data_dir + 'ephemeris_ablate.csv'))
        return pd.read_csv(data_dir + 'ephemeris_ablate.csv')
    
    coords = np.stack([load_planet(p, data_dir) for p in planets])
    if last_n_days is not None:
        coords = coords[:,-last_n_days:]
    assert coords.shape[1] > 300, 'length should be over 300'
    coords = np.concatenate([coords[:,:150], coords[:,-150:]], axis=1)
    df = pd.DataFrame(data=get_colformat(coords), columns=get_colnames(planets))
    df.to_csv(data_dir + 'ephemeris_ablate.csv')
    return df