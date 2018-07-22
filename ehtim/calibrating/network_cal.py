# network_cal.py
# functions for network-calibration
#
#    Copyright (C) 2018 Andrew Chael
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.special as spec
import scipy.optimize as opt
import sys
import itertools as it
import time
import copy

import ehtim.obsdata
from ehtim.observing.obs_helpers import *

from multiprocessing import cpu_count
from multiprocessing import Pool

from ehtim.calibrating.cal_helpers import *



ZBLCUTOFF = 1.e7;

###################################################################################################################################
#Network-Calibration
###################################################################################################################################
def network_cal(obs, zbl, sites=[], zbl_uvdist_max=ZBLCUTOFF, method="both", pad_amp=0.,gain_tol=.2, 
                caltable=False, processes=-1,show_solution=False, debias=True, msgtype='bar'):

    """Network-calibrate a dataset with zero baseline constraints.

       Args:
           obs (Obsdata): The observation to be calibrated
           zbl (float): zero baseline flux in Jy
           sites (list): list of sites to include in the network calibration. empty list calibrates all sites

           zbl_uvdist_max (float): considers all baselines of lambda-length less than this as zero baselines
           method (str): chooses what to calibrate, 'amp', 'phase', or 'both' 
           pad_amp (float): adds fractional uncertainty to amplitude sigmas in quadrature
           gain_tol (float): gains that exceed this value will be disfavored by the prior

           debias (bool): If True, debias the amplitudes
           caltable (bool): if True, returns a Caltable instead of an Obsdata 
           processes (int): number of cores to use in multiprocessing
           show_solution (bool): if True, display the solution as it is calculated
           msgtype (str): type of progress message to be printed, default is 'bar'           

       Returns:
           (Obsdata): the calibrated observation, if caltable==False
           (Caltable): the derived calibration table, if caltable==True
    """

    # V = model visibility, V' = measured visibility, G_i = site gain
    # G_i * conj(G_j) * V_ij = V'_ij
    if len(sites) == 0:
        print("No stations specified in network cal: defaulting to calibrating all stations!")
        sites = obs.tarr['site']

    # find colocated sites and put into list allclusters
    cluster_data = make_cluster_data(obs, zbl_uvdist_max)

    # get scans
    scans     = obs.tlist()
    scans_cal = copy.copy(scans)

    # Make the pool for parallel processing
    if processes > 0:
        counter = Counter(initval=0, maxval=len(scans))
        if processes > len(scans):
            processes = len(scans)
        print("Using Multiprocessing with %d Processes" % processes)
        pool = Pool(processes=processes, initializer=init, initargs=(counter,))
    elif processes == 0:
        counter = Counter(initval=0, maxval=len(scans))
        processes = int(cpu_count())
        if processes > len(scans):
            processes = len(scans)
        print("Using Multiprocessing with %d Processes" % processes)
        pool = Pool(processes=processes, initializer=init, initargs=(counter,))
    else:
        print("Not Using Multiprocessing")

    # loop over scans and calibrate
    tstart = time.time()
    if processes > 0: # with multiprocessing
        scans_cal = pool.map(get_network_scan_cal,
                                             [[i, len(scans), scans[i],
                                               zbl, sites, cluster_data, method, pad_amp, gain_tol,
                                               caltable, show_solution,debias,msgtype]
                                              for i in range(len(scans))
                                             ])
    else: # without multiprocessing
        for i in range(len(scans)):
            prog_msg(i, len(scans), msgtype=msgtype, nscan_last=i-1)
            scans_cal[i] = network_cal_scan(scans[i], zbl, sites, cluster_data,
                                            method=method, show_solution=show_solution, caltable=caltable,
                                            pad_amp=pad_amp, gain_tol=gain_tol,debias=debias)

    tstop = time.time()
    print("\nnetwork_cal time: %f s" % (tstop - tstart))

    if caltable: # create and return  a caltable
        allsites = obs.tarr['site']
        caldict = {k:v.reshape(1) for k,v in scans_cal[0].items()}
        for i in range(1,len(scans_cal)):
            row = scans_cal[i]
            if len(row) == 0:
                continue

            for site in allsites:
                try: dat = row[site]
                except KeyError: continue

                try: caldict[site] = np.append(caldict[site], row[site])
                except KeyError: caldict[site] = dat

        caltable = ehtim.caltable.Caltable(obs.ra, obs.dec, obs.rf, obs.bw, caldict, obs.tarr,
                                           source = obs.source, mjd=obs.mjd, timetype=obs.timetype)
        out = caltable

    else: # return the calibrated observation
        obs_cal = ehtim.obsdata.Obsdata(obs.ra, obs.dec, obs.rf, obs.bw,
                                        np.concatenate(scans_cal), obs.tarr, source=obs.source, mjd=obs.mjd,
                                        ampcal=obs.ampcal, phasecal=obs.phasecal, dcal=obs.dcal, frcal=obs.frcal,
                                        timetype=obs.timetype, scantable=obs.scans)
        out = obs_cal

    # close multiprocessing jobs
    if processes != -1:
        pool.close()

    return out

def network_cal_scan(scan, zbl, sites, clustered_sites, zbl_uvidst_max=ZBLCUTOFF, method="both", 
                     show_solution=False, pad_amp=0., gain_tol=.2, caltable=False, debias=True):
    """Network-calibrate a scan with zero baseline constraints.

       Args:
           obs (Obsdata): The observation to be calibrated
           zbl (float): zero baseline flux in Jy
           sites (list): list of sites to include in the network calibration. empty list calibrates all sites
           clustered_sites (tuple): information  on clustered sites, returned by make_cluster_data function

           zbl_uvdist_max (float): considers all baselines of lambda-length less than this as zero baselines
           method (str): chooses what to calibrate, 'amp', 'phase', or 'both' 
           pad_amp (float): adds fractional uncertainty to amplitude sigmas in quadrature
           gain_tol (float): gains that exceed this value will be disfavored by the prior

           debias (bool): If True, debias the amplitudes
           caltable (bool): if True, returns a Caltable instead of an Obsdata 
           show_solution (bool): if True, display the solution as it is calculated
           

       Returns:
           (Obsdata): the calibrated scan, if caltable==False
           (Caltable): the derived calibration table, if caltable==True
    """


    if len(sites) < 2:
        print("less than 2 stations specified in network cal: defaulting to calibrating all !")
        sites = list(set(np.hstack((scan['t1'], scan['t2']))))

    # clustered site information
    allclusters = clustered_sites[0]
    clusterdict = clustered_sites[1]
    clusterbls = clustered_sites[2]

    # create a dictionary to keep track of gains
    tkey = {b:a for a,b in enumerate(sites)}
    clusterkey = clusterdict

    # make two lists of gain keys that relates scan bl gains to solved site ones
    # -1 means that this station does not have a gain that is being solved for

    # and make a list of scan keys that relates scan bl visibilities to solved cluster ones
    # -1 means it's a zero baseline!

    g1_keys = []
    g2_keys = []
    scan_keys = []
    for row in scan:
        try:
            g1_keys.append(tkey[row['t1']])
        except KeyError:
            g1_keys.append(-1)
        try:
            g2_keys.append(tkey[row['t2']])
        except KeyError:
            g2_keys.append(-1)

        clusternum1 = clusterkey[row['t1']]
        clusternum2 = clusterkey[row['t2']]
        if clusternum1 == clusternum2: # sites are in the same cluster
            scan_keys.append(-1)
        else: #sites are not in the same cluster
            bl_index = clusterbls.index(set((clusternum1, clusternum2)))
            scan_keys.append(bl_index)


    # no sites to calibrate on this scan!
    if np.all(g1_keys == -1):
        return scan

    # scan visibilities and sigmas with extra padding
    if method=='amp':
        if debias:
            vis = amp_debias(np.abs(scan['vis']), np.abs(scan['sigma']))
        else:
            vis = np.abs(vis)
    else:
        vis = scan['vis']

    sigma_inv = 1.0/np.sqrt(scan['sigma']**2+ (pad_amp*np.abs(scan['vis']))**2)

    # initial guesses for parameters
    n_gains = len(sites)
    n_clusterbls = len(clusterbls)
    gpar_guess = np.ones(n_gains, dtype=np.complex128).view(dtype=np.float64)
    vpar_guess = np.ones(n_clusterbls, dtype=np.complex128)
    for i in range(len(scan_keys)):
        if scan_keys[i] < 0: continue
        vpar_guess[scan_keys[i]] = vis[i]
    vpar_guess = vpar_guess.view(dtype=np.float64)
    gvpar_guess = np.hstack((gpar_guess, vpar_guess))

    # error function
    def errfunc(gvpar):

        # all the forward site gains (complex)
        g = gvpar[0:2*n_gains].astype(np.float64).view(dtype=np.complex128)
        # all the intercluster visibilities (complex)
        v = gvpar[2*n_gains:].astype(np.float64).view(dtype=np.complex128)

        # choose to only scale ampliltudes or phases
        if method=="phase":
            g = g/np.abs(g) # TODO: use exp(i*np.arg())?
        if method=="amp":
             g = np.abs(np.real(g))
            #g = np.abs(g)

        # append the default values to g for missing points
        # and to v for the zero baseline points
        g = np.append(g, 1.)
        v = np.append(v, zbl)

        # scan visibilities are either an intercluster visibility or the fixed zbl

        v_scan = v[scan_keys]
        g1 = g[g1_keys]
        g2 = g[g2_keys]

        if method=='amp':
            verr = np.abs(vis) - g1*g2.conj() * np.abs(v_scan)
        else:
            verr = vis - g1*g2.conj() * v_scan

        chisq = np.sum((verr.real * sigma_inv)**2) + np.sum((verr.imag * sigma_inv)**2)

        # prior on the gains
        g_fracerr = gain_tol 
        chisq_g = np.sum((np.log(np.abs(g))**2 / g_fracerr**2))
        chisq_v = np.sum((np.abs(v)/zbl)**4)

        return chisq + chisq_g + chisq_v

    # run the minimizer to get a solution
    optdict = {'maxiter' : 500} # minimizer params
    res = opt.minimize(errfunc, gvpar_guess, method='CG', options=optdict)

    # get solution
    g_fit = res.x[0:2*n_gains].view(np.complex128)
    v_fit = res.x[2*n_gains:].view(np.complex128)

    if method=="phase":
        g_fit = g_fit / np.abs(g_fit)
    if method=="amp":
        g_fit = np.abs(np.real(g_fit))

    if show_solution == True:
        print (np.abs(g_fit))
        print (np.abs(v_fit))


    g_fit = np.append(g_fit, 1.)
    v_fit = np.append(v_fit, zbl)

   
    if caltable:  # derive caltable
        allsites = list(set(scan['t1']).union(set(scan['t2']))) if g_fit[-2] != 1.0 else []

        caldict = {}
        for site in allsites:
            if site in sites:
                site_key = tkey[site]
            else:
                site_key = -1
            caldict[site] = np.array((scan['time'][0], g_fit[site_key]**-1, g_fit[site_key]**-1), dtype=DTCAL)
        out = caldict

    else: # apply calibration solution
        g1_fit = g_fit[g1_keys]
        g2_fit = g_fit[g2_keys]
        gij_inv = (g1_fit * g2_fit.conj())**(-1)
        scan['vis']  = gij_inv * scan['vis']
        scan['qvis'] = gij_inv * scan['qvis']
        scan['uvis'] = gij_inv * scan['uvis']
        scan['vvis'] = gij_inv * scan['vvis']
        scan['sigma']  = np.abs(gij_inv) * scan['sigma']
        scan['qsigma'] = np.abs(gij_inv) * scan['qsigma']
        scan['usigma'] = np.abs(gij_inv) * scan['usigma']
        scan['vsigma'] = np.abs(gij_inv) * scan['vsigma']
        out = scan

    return out

def init(x):
    global counter
    counter = x

def get_network_scan_cal(args):
    return get_network_scan_cal2(*args)

def get_network_scan_cal2(i, n, scan, zbl, sites, cluster_data, method, pad_amp,gain_tol,caltable, show_solution,debias,msgtype):
    if n > 1:
        global counter
        counter.increment()
        prog_msg(counter.value(), counter.maxval,msgtype,counter.value()-1)

    return network_cal_scan(scan, zbl, sites, cluster_data, zbl_uvidst_max=ZBLCUTOFF, 
                            method=method,caltable=caltable, show_solution=show_solution, 
                            pad_amp=pad_amp, gain_tol=gain_tol,debias=debias)

