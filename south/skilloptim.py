# -*- coding: utf-8 -*-
"""
Code expanded from Willam Gregory and further developped by Daniel Charrier
Created on Mon Jul 25 11:55:19 2022

For a given month of training data, this file computes the forecast skills
for 12 months in the future. For each month, 17x17 different hyperparameters 
are tried. 

Possible training data consist of:
    - SIC data
    - OHC data
    - various atmospheric indices
"""

import numpy as np
import datetime
import shutil
import urllib.request as request
from contextlib import closing
from netCDF4 import Dataset
from scipy.interpolate import griddata
from scipy.stats import pearsonr
from scipy.linalg import expm
import glob
import struct
import pandas as pd
from scipy.stats import linregress
import os
import warnings
import itertools
import zipfile
import pickle

warnings.filterwarnings('ignore')

def make_npstere_grid(boundinglat,lon_0,grid_res=25e3):
    import pyproj as proj
    p = proj.Proj('+proj=stere +R=6370997.0 +units=m +lon_0='+str(float(lon_0))+' +lat_ts=-90.0 +lat_0=-90.0',\
                  preserve_units=True)
    llcrnrlon = lon_0 + 45
    urcrnrlon = lon_0 - 135
    y_ = p(lon_0,boundinglat)[1]
    llcrnrlat = p(np.sqrt(2.)*y_,0.,inverse=True)[1]
    urcrnrlat = llcrnrlat
    llcrnrx,llcrnry = p(llcrnrlon,llcrnrlat)
    p = proj.Proj('+proj=stere +R=6370997.0 +units=m +lon_0='+str(float(lon_0))+' +lat_ts=-90.0 +lat_0=-90.0 +x_0='\
                  +str(-llcrnrx)+' +y_0='+str(-llcrnry), preserve_units=True)
    urcrnrx,urcrnry = p(urcrnrlon,urcrnrlat)

    nx = -int(urcrnrx/grid_res)+1
    ny = -int(urcrnry/grid_res)+1
    dx = urcrnrx/(nx-1)
    dy = urcrnry/(ny-1)

    x = dx*np.indices((ny,nx),np.float32)[1,:,:]
    y = dy*np.indices((ny,nx),np.float32)[0,:,:]
    lon,lat = p(x,y,inverse=True)
    return lon,lat,x,y,p

def read_SIE(fmin,fmax,month): 
    SIEs = {}
    SIEs_dt = {}
    SIEs_trend = {}
    
    month_num = Month_dict[month] 

    xls = pd.ExcelFile(home+'/DATA/S_Sea_Ice_Index_Regional_Monthly_Data_G02135_v3.0.xlsx',engine='openpyxl')
    SIEs['Pan-Antarctic'] = (np.genfromtxt(home+'/DATA/S_'+month_num+'_extent_v3.0.csv',delimiter=',').T[4][1:])[:fmax-1979+1]
    SIEs['Ross'] = (np.array(np.array(pd.read_excel(xls, 'Ross-Extent-km^2')[month])[3:-1]/1e6,dtype='float32')[:fmax-1979+1]).round(3)
    SIEs['Weddell'] = (np.array(np.array(pd.read_excel(xls, 'Weddell-Extent-km^2')[month])[3:-1]/1e6,dtype='float32')[:fmax-1979+1]).round(3)
    SIEs['Bell-Amundsen'] = (np.array(np.array(pd.read_excel(xls, 'Bell-Amundsen-Extent-km^2')[month])[3:-1]/1e6,dtype='float32')[:fmax-1979+1]).round(3)
    SIEs['Indian'] = (np.array(np.array(pd.read_excel(xls, 'Indian-Extent-km^2')[month])[3:-1]/1e6,dtype='float32')[:fmax-1979+1]).round(3)
    SIEs['Pacific'] = (np.array(np.array(pd.read_excel(xls, 'Pacific-Extent-km^2')[month])[3:-1]/1e6,dtype='float32')[:fmax-1979+1]).round(3)
    
    for tag in SIEs:
        trend = np.zeros((fmax-(fmin-1)+1,2)) #tableau qui a une ligne de plus 
        dt = np.zeros((fmax-(fmin-1)+1,fmax-1979+1)) #Valeurs detrended. Il faut un vecteur colonne pour chaque année de prédiction. 
        for year in range(fmin-1,fmax+1):
            n = year-1979+1
            reg = linregress(np.arange(n),SIEs[tag][range(n)])
            lineT = (reg[0]*np.arange(n)) + reg[1]
            trend[year-(fmin-1),0] = reg[0]
            trend[year-(fmin-1),1] = reg[1]
            dt[year-(fmin-1),range(n)] = SIEs[tag][range(n)]-lineT
        SIEs_trend[tag] = trend
        SIEs_dt[tag] = dt.round(3)
    return SIEs,SIEs_dt,SIEs_trend

def readNSIDC(fmin,fmax,month):
    month_num = Month_dict[month] 
    if month_num in ['01','03','05','07','08','10','12']:
        max_days = 31
    elif month_num == '02':
        max_days = 28
    else:
        max_days = 30
    #dimX et dimY donnent les dimensions de la grille des satellites.     
    dimX = 332
    dimY = 316
    SIC = {}
    SIC['lat'] = (np.fromfile(home+"/misc/pss25lats_v3.dat",dtype='<i4').reshape(dimX,dimY))/100000
    SIC['lon'] = (np.fromfile(home+"/misc/pss25lons_v3.dat",dtype='<i4').reshape(dimX,dimY))/100000
    SIC['psa'] = (np.fromfile(home+"/misc/pss25area_v3.dat",dtype='<i4').reshape(dimX,dimY))/1000
    SIC['lonr'],SIC['latr'],SIC['xr'],SIC['yr'],p = make_npstere_grid(-55,180,1e5)
    SIC['x'],SIC['y'] = p(SIC['lon'],SIC['lat'])
    dXR,dYR = SIC['xr'].shape
    SIC['psar'] = 16*griddata((SIC['x'].ravel(),SIC['y'].ravel()),SIC['psa'].ravel(),(SIC['xr'],SIC['yr']),'linear')
    data_regrid = np.zeros((dXR,dYR,fmax-1979+1))*np.nan
    k = 0
    for year in range(1979,fmax+1):
        if (year == ymax) or (year == ymax-1):
            if len(glob.glob(home+'/DATA/nt_'+str(year)+month_num+'*nrt_s.bin'))==0:
                for day in range(1,max_days+1):
                    with closing(request.urlopen(sic_ftp1+'/nt_'+str(year)+month_num+str('%02d'%day)+'_f18_nrt_s.bin')) as r:
                        with open(home+'/DATA/nt_'+str(year)+month_num+str('%02d'%day)+'_f18_nrt_s.bin', 'wb') as f:
                            shutil.copyfileobj(r, f)
                    
            files = sorted(glob.glob(home+'/DATA/nt_'+str(year)+month_num+'*nrt_s.bin'))
            daily = np.zeros((dimX,dimY,len(files)))*np.nan
            f = 0
            for file in files:
                icefile = open(file,'rb')
                contents = icefile.read()
                icefile.close()
                s="%dB" % (int(dimX*dimY),)
                z=struct.unpack_from(s, contents, offset = 300)
                daily[:,:,f] = (np.array(z).reshape((dimX,dimY)))/250
                f += 1
            monthly = np.nanmean(daily,2) #moyenne sur le mois
        else:
            if year < 1987 or (year == 1987 and int(month_num) < 9):
                sat = 'n07'
            elif ((year > 1987) or (year == 1987 and int(month_num) > 8)) & (year < 1992):
                sat = 'f08'
            elif (year > 1991) & ((year < 1995) or (year == 1995 and int(month_num) < 10)):
                sat = 'f11'
            elif ((year > 1995) or (year == 1995 and int(month_num) > 9)) & (year < 2008):
                sat = 'f13'
            elif year > 2007:
                sat = 'f17'
            files = glob.glob(home+'/DATA/nt_'+str(year)+month_num+'*.1_s.bin')
            if len(files) == 0:
                with closing(request.urlopen(sic_ftp2+'/nt_'+str(year)+month_num+'_'+sat+'_v1.1_s.bin')) as r:
                    with open(home+'/DATA/nt_'+str(year)+month_num+'_'+sat+'_v1.1_s.bin', 'wb') as f:
                            shutil.copyfileobj(r, f)
            icefile = open(glob.glob(home+'/DATA/nt_'+str(year)+month_num+'*.1_s.bin')[0], 'rb')
            contents = icefile.read()
            icefile.close()
            s="%dB" % (int(dimX*dimY),)
            z=struct.unpack_from(s, contents, offset = 300)
            monthly = (np.array(z).reshape((dimX,dimY)))/250
        monthly[monthly>1] = np.nan
        data_regrid[:,:,k] = griddata((SIC['x'].ravel(),SIC['y'].ravel()),monthly.ravel(),\
                                             (SIC['xr'],SIC['yr']),'linear')
        k += 1
    SIC['data'] = data_regrid 
    
    return SIC

def readORAS5(fmin,fmax,month):
    """ 
This program downloads the OHC in the first 300 meters from the ORAS5 
reanalysis and return a grid of OHC values. Latitudes have been arbitrarly 
chosen to be limited to -60°S 
"""

    month_num = Month_dict[month] 
    if os.path.exists(home+'/DATA/ORAS5_'+month+'_OHC_'+str(fmax)+'.zip')==False:
        import cdsapi

        c = cdsapi.Client()
        dates = np.arange(1979,fmax+1)
        c.retrieve(
            'reanalysis-oras5',
            {
                'format': 'zip',
                'product_type': ['consolidated','operational'],
                'vertical_resolution': 'single_level',
                'variable': 'ocean_heat_content_for_the_upper_300m',
                'year': [str(date) for date in dates],
                'month': month_num,
            },
            home+'/DATA/ORAS5_'+month+'_OHC_'+str(fmax)+'.zip')
        with zipfile.ZipFile(home+'/DATA/ORAS5_'+month+'_OHC_'+str(fmax)+'.zip', 'r') as zip_ref:
            zip_ref.extractall(home+'/DATA/')
        
    OHC = {}
    file = Dataset(home+'/DATA/sohtc300_control_monthly_highres_2D_'+str(fmin)+month_num+'_CONS_v0.1.nc')
    #To get data for latitudes <-60°, you should stop at index 200. 
    #For latitudes, I take a step = 5 
    datalat = np.array(file['nav_lat'])[np.arange(0,200,5),:]
    datalon = np.array(file['nav_lon'])[np.arange(0,200,5),:]
    #For longitudes, I take a step = 10 
    OHC['lat'] = datalat[:,np.arange(0,1440,10)]
    OHC['lon'] = datalon[:,np.arange(0,1440,10)]
    dimX = 40 #200/5 = 40
    dimY =144 #1440/10 = 144
    n = fmax-1979+1
    data = np.zeros((dimX,dimY,n))
    for year in range(1979,fmax+1):
        k = year - 1979
        if year < 2015:
            file = Dataset(home+'/DATA/sohtc300_control_monthly_highres_2D_'+str(year)+month_num+'_CONS_v0.1.nc')
        else:
            file = Dataset(home+'/DATA/sohtc300_control_monthly_highres_2D_'+str(year)+month_num+'_OPER_v0.1.nc')
        South_data = np.array(file['sohtc300'][0,np.arange(0,200,5),:])
        South_data[South_data>9.9e+36] = np.nan
        
        data[:,:,k] = South_data[:,np.arange(0,1440,10)]
    OHC['data'] = data/2e9 # "normalize" data
    return OHC

def readindices(month,fmin,fmax,SAM_in,ASL_in,ENSO_in):
    """ 
This program downloads various atmospheric indices from the National Center 
for Atmopsheric Research (NCAR):
    - the southern annular mode index (SAM). Please find information at:
https://climatedataguide.ucar.edu/climate-data/marshall-southern-annular-mode-sam-index-station-based
    - various Amundsen Low Index (ASL): the georgraphic position (lat and lon) 
of the low and its actual central pressure. Please find information at:
https://climatedataguide.ucar.edu/climate-data/amundsen-sea-low-indices
    - the ENSO3.4 index. Please find information at:
https://psl.noaa.gov/gcos_wgsp/Timeseries/
"""
    
    month_num = int(Month_dict[month])
    indices = {}
    
    if SAM_in == 'y':
        
        url = 'http://www.nerc-bas.ac.uk/public/icd/gjma/newsam.1957.2007.txt'
        open(home+'/DATA/newsam.txt', 'wb').write(request.urlopen(url).read())

        with open(home+'/DATA/newsam.txt', 'r') as f:
            f.readline()
            f.readline()
            SAM = []
            for line in f.readlines():
                L = line.split()
                year = int(L[0])
                if year >= 1979 and year <= fmax:
                    SAM.append(float(L[month_num]))
                    
            for year in range(fmin,fmax+1):
                k = year-1979+1
                mean = np.mean(np.asarray(SAM[0:k]),0)
                indices['SAM_'+str(year)] = np.asarray(SAM[0:k]) - mean
                    
    if ASL_in == 'y':
        url = 'https://raw.githubusercontent.com/scotthosking/amundsen-sea-low-index/master/asli_era5_v3-latest.csv'
        open(home+'/DATA/asl.csv','wb').write(request.urlopen(url).read())

        ASL_list = pd.read_csv(home+'/DATA/asl.csv',header=25)
        latASL = []
        lonASL = []
        PresASL = []
        for year in range(1979,fmax+1):
            k = year-1979 
            i = month_num-1+k*12 
            latASL.append(ASL_list['lat'][i])
            lonASL.append(ASL_list['lon'][i])
            PresASL.append(ASL_list['ActCenPres'][i])           
            
        for year in range(fmin,fmax+1):
            k = year-1979+1
            latmean = np.mean(np.asarray(latASL[0:k]),0)
            lonmean = np.mean(np.asarray(lonASL[0:k]),0)
            Presmean = np.mean(np.asarray(PresASL[0:k]),0)
            
            indices['ASL_lat'+str(year)] = np.asarray(latASL[0:k]) - latmean
            indices['ASL_lon'+str(year)] = np.asarray(lonASL[0:k]) - lonmean
            indices['ASL_Pres'+str(year)] = np.asarray(PresASL[0:k]) - Presmean
            
    if ENSO_in == 'y':
        url = 'https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.anom.data'
        open(home+'/DATA/nino3.4.csv','wb').write(request.urlopen(url).read())
        
        with open(home+'/DATA/nino3.4.csv', 'r') as f:
            lbegin = 1979 - 1869
            lend = fmax - 1869
            ENSO = []
            for line in f.readlines()[lbegin:lend+1]: #ENSO indices for 1985-2021
                L = line.split()
                year = int(L[0])
                ENSO.append(float(L[month_num]))

                
            for year in range(fmin,fmax+1):
                k = year-1979+1
                mean = np.mean(np.asarray(ENSO[0:k]),0)
                indices['ENSO_'+str(year)] = np.asarray(ENSO[0:k]) - mean
        
    return indices


def detrend(dataset,fmin,fmax):
    
    for year in range(fmin,fmax+1):
        n = year-1979+1
        data = dataset['data'][:,:,range(n)]
        X = data.shape[0] ; Y = data.shape[1] ; T = data.shape[2]
        detrended = np.zeros(data.shape)*np.nan
        trend = np.zeros((X,Y,2))*np.nan 
        for i,j in itertools.product(range(X),range(Y)):
            if ~np.isnan(data[i,j,range(T)]).all(): 
                reg = linregress(np.arange(T),data[i,j,range(T)])
                lineT = (reg[0]*np.arange(T)) + reg[1]
                trend[i,j,0] = reg[0]
                trend[i,j,1] = reg[1]
                detrended[i,j,range(T)]=data[i,j,range(T)]-lineT

        dataset['dt_'+str(year)] = detrended 
        dataset['trend_'+str(year)] = trend 


def networks(dataset,fmin,fmax,latlon=True):
    import ComplexNetworks as CN
    for year in range(fmin,fmax+1):
        network = CN.Network(data=dataset['dt_'+str(year)])
        CN.Network.tau(network, 0.01)
        CN.Network.area_level(network,latlon_grid=latlon)
        if latlon:
            CN.Network.intra_links(network, lat=dataset['lat'])
        else:
            CN.Network.intra_links(network, area=dataset['psar'])
        dataset['nodes_'+str(year)] = network.V
        dataset['anoms_'+str(year)] = network.anomaly

def buildXmat(fmin,fmax,NY,SIC_in,OHC_in,SAM_in,ASL_in,ENSO_in):
    """
Given the nature of the data, this programm returns:
    - the matrix of training inputs: Xmat
    - the vector of test input: Xsvect 
    - the convariance matric of training input: Mmat
for each year and regions """
 
    Xmat = {}
    Xsvect = {}
    Mmat = {}
    
    for k in range(len(regions)):
        fmean = np.zeros(fmax-fmin+1) 
        fvar = np.zeros(fmax-fmin+1) 
        fmean_rt = np.zeros(fmax-fmin+1) 
        for year in range(fmin,fmax+1):
            if NY:
                y = np.asarray([SIEs_dt[regions[k]][year-(fmin-1)-1,range(1,year-1979)]]).T #n x 1 
                
                X = []
                if SIC_in == 'y':
                    for area in SIC['anoms_'+str(year-1)]:
                        if len(SIC['nodes_'+str(year-1)][area]) > 3:
                            r,p = pearsonr(y[:,0],SIC['anoms_'+str(year-1)][area][:-1]) 
                            if r> 0: 
                                X.append(SIC['anoms_'+str((year-1))][area])
                if OHC_in == 'y':
                    for area in OHC['anoms_'+str(year-1)]:
                        if len(OHC['nodes_'+str(year-1)][area]) > 3:
                            r,p = pearsonr(y[:,0],OHC['anoms_'+str(year-1)][area][:-1])
                            if r < 0: 
                                X.append(-OHC['anoms_'+str(year-1)][area])  
                if SAM_in == 'y':
                    X.append(index_tab['SAM_'+str(year-1)])
                if ASL_in == 'y':
                    X.append(index_tab['ASL_lat'+str(year-1)])
                    X.append(index_tab['ASL_lon'+str(year-1)])
                    X.append(index_tab['ASL_Pres'+str(year-1)])
                if ENSO_in == 'y':
                    X.append(index_tab['ENSO_'+str(year-1)])
            else:
                y = np.asarray([SIEs_dt[regions[k]][year-(fmin-1)-1,range(year-1979)]]).T

                X = []
                if SIC_in == 'y':
                    for area in SIC['anoms_'+str(year)]:
                        if len(SIC['nodes_'+str(year)][area])> 3:
                            r,p = pearsonr(y[:,0],SIC['anoms_'+str(year)][area][:-1])
                            if r > 0:
                                X.append(SIC['anoms_'+str(year)][area])
                if OHC_in == 'y':
                    for area in OHC['anoms_'+str(year)]:
                        if len(OHC['nodes_'+str(year)][area]) > 3:
                            r,p = pearsonr(y[:,0],OHC['anoms_'+str(year)][area][:-1])
                            if r < 0:
                                X.append(-OHC['anoms_'+str(year)][area])  
                if SAM_in == 'y':
                    X.append(index_tab['SAM_'+str(year)])
                if ASL_in == 'y':
                    X.append(index_tab['ASL_lat'+str(year)])
                    X.append(index_tab['ASL_lon'+str(year)])
                    X.append(index_tab['ASL_Pres'+str(year)])
                if ENSO_in == 'y':
                    X.append(index_tab['ENSO_'+str(year)])

            X = np.asarray(X).T #training inputs.  
            Xs = np.asarray([X[-1,:]]) #test input
            X = X[:-1,:]
            
            
            M = np.abs(np.cov(X, rowvar=False, bias=True)) #Size NxN
            np.fill_diagonal(M,0)
            np.fill_diagonal(M,-np.sum(M,axis=0))

            Xmat[regions[k]+str(year)] = X
            Xsvect[regions[k]+str(year)] = Xs
            Mmat[regions[k]+str(year)] = M

    return Xmat,Xsvect,Mmat

def forecast(Xmat,Xsvect,Mmat,fmin,fmax,NY,i,j):
    """

    Parameters
    ----------
    Xmat : numpy array NxN
        Matrix of training input
    Xsvect : numpy array
        Vector of test input
    Mmat : numpy array NxN
        Covariance matrix
    fmin : int
        first year forecasted
    fmax : int
        last year forecasted
    NY : bool
        tell if data month and forecasted month are the same year
    i : int
        first index for hyperparameters
    j : int
        second index for hyperparameters

    Returns
    -------
    GPR : dictionary
        contains the mean value,the uncertainty and the retrended mean value of SIE 
        for different regions and months.

    """
    
    GPR = {}

    for k in range(len(regions)):
        fmean = np.zeros(fmax-fmin+1) 
        fvar = np.zeros(fmax-fmin+1) 
        fmean_rt = np.zeros(fmax-fmin+1) 

        for year in range(fmin,fmax+1):

            X = Xmat[regions[k]+str(year)]
            Xs = Xsvect[regions[k]+str(year)]
            M = Mmat[regions[k]+str(year)]
            if NY:
                y = np.asarray([SIEs_dt[regions[k]][year-(fmin-1)-1,range(1,year-1979)]]).T 
            else:
                y = np.asarray([SIEs_dt[regions[k]][year-(fmin-1)-1,range(year-1979)]]).T
  
            n = len(y)  

            ℓ = np.logspace(-7,2,20)[i] ; σn_tilde = np.logspace(-3,9,20)[j]
            #sigma_f optimization 
            Σ_tilde = expm(ℓ*M)
            B = np.linalg.multi_dot([X,Σ_tilde,X.T]) + np.eye(n)*σn_tilde
            L_tilde = np.linalg.cholesky(B)
            A_tilde = np.linalg.solve(L_tilde.T,np.linalg.solve(L_tilde,y))
            σf = (np.dot(y.T,A_tilde)/n)[0][0]
            σn = σf*σn_tilde
            #Gaussian process:
            Σ = σf * expm(ℓ*M)
            L = np.linalg.cholesky(np.linalg.multi_dot([X,Σ,X.T]) + np.eye(n)*σn)
            α = np.linalg.solve(L.T,np.linalg.solve(L,y))
            KXXs = np.linalg.multi_dot([X,Σ,Xs.T])
            KXsXs = np.linalg.multi_dot([Xs,Σ,Xs.T]) + σn
            v = np.linalg.solve(L,KXXs)

            fmean[year-fmin] = (np.dot(KXXs.T,α)[0][0]).round(3)
            fvar[year-fmin] = ((KXsXs - np.dot(v.T,v))[0][0]).round(3)
            lineT = (np.arange(year-1979+1)*SIEs_trend[regions[k]][year-(fmin-1)-1,0]) + SIEs_trend[regions[k]][year-(fmin-1)-1,1]
            fmean_rt[year-fmin] = (fmean[year-fmin] + lineT[-1]).round(3)
        
        GPR[regions[k]+'_fmean'] = fmean
        GPR[regions[k]+'_fvar'] = fvar
        GPR[regions[k]+'_fmean_rt'] = fmean_rt
        
    return GPR


def skill(fmin,fmax):
    skill_dt = []
    dt_obs = []
    for k in range(len(regions)):
        dt = []
        for t in range(fmin,fmax+1):
            n = t - 1979
            dt.append(SIEs_dt[regions[k]][t-(fmin-1),n])
        dt_obs.append(dt)
        
        forecast_dt = GPR[regions[k]+'_fmean']
        c = np.mean((dt-forecast_dt)**2)
        d = np.mean((dt-np.nanmean(dt))**2)
        skill_dt.append((1 - (c/d)).round(3))
    return skill_dt,dt_obs

home = os.getcwd()
if os.path.exists(home+'/DATA')==False:
    os.mkdir(home+'/DATA')
    os.chmod(home+'/DATA',0o0777)
if os.path.exists(home+'/Clusters')==False:
    os.mkdir(home+'/Clusters')
    os.chmod(home+'/Clusters',0o0777)

sie_ftp = 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02135'
sic_ftp1 = 'ftp://sidads.colorado.edu/DATASETS/nsidc0081_nrt_nasateam_seaice/south'
sic_ftp2 = 'ftp://sidads.colorado.edu/DATASETS/nsidc0051_gsfc_nasateam_seaice/final-gsfc/south/monthly'

ymax = int(datetime.date.today().year)
fmin = int(input('Please specify first year you would like to forecast (must be > 1980):\n'))
fmax = int(input('Please specify last year you would like to forecast (must be < '+str(ymax)+'):\n'))
if fmin < 1981:
    fmin = 1981
if fmax > ymax-1:
    fmax = ymax-1

Month_dict = {'January':'01','February':'02','March':'03','April':'04','May':'05','June':'06','July':'07','August':'08','September':'09','Oc\
tober':'10','November':'11','December':'12'}
month_data = input('Please enter the month you would like to pick the data from\n:')
if month_data not in Month_dict.keys():
    assert('Month entered does not exist')
month_num_data = int(Month_dict[month_data])

SIC_in = input('Do you want SIC data?\n')
OHC_in = input('Do you want OHC data?\n')

if SIC_in == 'y':
    if os.path.exists(home+'/Clusters/SIC'+month_data+'1984-2021') == True:
        SIC = pickle.load(open(home+'/Clusters/SIC'+month_data+'1984-2021','rb'))
    else:
        print('Downloading and reading SIC data')
        SIC = readNSIDC(fmin-1,fmax,month_data)
        print('Processing SIC data...')
        detrend(SIC,fmin-1,fmax)
        networks(SIC,fmin-1,fmax,latlon=False)
        pickle.dump(SIC,open(home+'/Clusters/SIC'+month_data+str(fmin-1)+'-'+str(fmax),'wb'))
if OHC_in == 'y':
    if os.path.exists(home+'/Clusters/OHC'+month_data+'1984-2021') == True:
        OHC = pickle.load(open(home+'/Clusters/OHC'+month_data+'1984-2021','rb'))
    else:
        print('Downloading and reading OHC data')
        OHC = readORAS5(fmin-1,fmax,month_data)
        print('Processing OHC data...')
        detrend(OHC,fmin-1,fmax)
        networks(OHC,fmin-1,fmax,latlon=True) 
        pickle.dump(OHC,open(home+'/Clusters/OHC'+month_data+str(fmin-1)+'-'+str(fmax),'wb'))

SAM_in = input('Do you want the SAM index?\n')
ASL_in = input('Do you want the ASL indices?\n')
ENSO_in = input('Do you want the ENSO index?\n')

if SAM_in or ASL_in or ENSO_in:
    index_tab = readindices(month_data,fmin-1,fmax,SAM_in,ASL_in,ENSO_in)

regions = ['Pan-Antarctic','Ross','Weddell','Bell-Amundsen','Indian','Pacific']
for i in range(0,12):
    
    month_num_fc = (month_num_data+i)%12
    month_fc = list(Month_dict.keys())[month_num_fc] #month of forecast

    print('forecasting SIE in '+month_fc+' from '+month_data+ 'data')
    next_year =  (int(Month_dict[month_fc]) <= int(Month_dict[month_data]))

    print('Downloading and reading SIE data')
    SIEs,SIEs_dt,SIEs_trend = read_SIE(fmin,fmax,month_fc)
    
    lines = []
    skill_tab_dt = np.zeros((289,len(regions)))

    m = 0
    print('Get rid of small and uncorrelated nodes')
    Xmat,Xsvect,Mmat = buildXmat(fmin,fmax,next_year,SIC_in,OHC_in,SAM_in,ASL_in,ENSO_in)

    print('Running forecast...')
    for i,j in itertools.product(range(17),range(17)):
        GPR = forecast(Xmat,Xsvect,Mmat,fmin,fmax,next_year,i,j)
        skill_dt,dt_obs = skill(fmin,fmax)

        lines.append(str(i)+','+str(j))
        for k in range(len(skill_dt)):
            skill_tab_dt[m,k] = skill_dt[k]
        m +=1

    if os.path.exists(home+'/skills')==False:
        os.mkdir(home+'/skills')
        os.chmod(home+'/skills',0o0777)

    file_skill_dt = pd.DataFrame(skill_tab_dt, index=lines, columns=regions)
    file_skill_dt.to_csv(home+'/skills/'+month_data+'to'+month_fc+'detrended_skills'+str(fmin)+'-'+str(fmax)+'OHC'+OHC_in+'SIC'+SIC_in+'.csv')
