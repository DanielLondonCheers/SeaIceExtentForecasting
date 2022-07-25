import numpy as np
import datetime
import shutil
import urllib.request as request
from contextlib import closing
from netCDF4 import Dataset
from scipy.interpolate import griddata
from scipy.optimize import minimize
from scipy.stats import pearsonr
from scipy.linalg import expm
from scipy.linalg import sqrtm
from scipy.linalg import cholesky
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



def forecast(fmin,fmax,NY,SIC_in,OHC_in,HP):
    regions = ['Pan-Antarctic','Ross','Weddell']#,'Bell-Amundsen','Indian','Pacific']
    GPR = {}

    for k in range(len(regions)):
        
        fmean = np.zeros(fmax-fmin+1) 
        fvar = np.zeros(fmax-fmin+1) 
        fmean_rt = np.zeros(fmax-fmin+1) 
        
        l_init = np.logspace(-7,2,20)[HP[0][k]]
        sigma_init = np.logspace(-3,9,20)[HP[1][k]]
        
        for year in range(fmin,fmax+1):

            if NY:
                y = np.asarray([SIEs_dt[regions[k]][year-(fmin-1)-1,range(1,year-1979)]]).T #n x 1

                X = []
                if SIC_in == 'y':
                    SIC['weights_'+regions[k]+str(year-1)] = {}
                    for area in SIC['anoms_'+str(year-1)]:
                        r,p = pearsonr(y[:,0],SIC['anoms_'+str(year-1)][area][:-1]) #ce test est un peu étrange. Pourquoi? Il ne prend en compte que certains noeuds. Pourquoi??
                        if r>0:
                            X.append(SIC['anoms_'+str((year-1))][area])
                            SIC['weights_'+regions[k]+str(year-1)][area] = 0


                if OHC_in == 'y':
                    OHC['weights_'+regions[k]+str(year-1)] = {}
                    for area in OHC['anoms_'+str(year-1)]:
                        r,p = pearsonr(y[:,0],OHC['anoms_'+str(year-1)][area][:-1])
                        if r<0:
                            X.append(-OHC['anoms_'+str(year-1)][area])
                            OHC['weights_'+regions[k]+str(year-1)][area] = 0
            else:
                y = np.asarray([SIEs_dt[regions[k]][year-(fmin-1)-1,range(year-1979)]]).T

                X = []
                if SIC_in == 'y':
                    SIC['weights_'+regions[k]+str(year)] = {}
                    for area in SIC['anoms_'+str(year)]:
                        r,p = pearsonr(y[:,0],SIC['anoms_'+str(year)][area][:-1])
                        if r > 0:
                            X.append(SIC['anoms_'+str(year)][area])
                            SIC['weights_'+regions[k]+str(year)][area] = 0


                if OHC_in == 'y':
                    OHC['weights_'+regions[k]+str(year)] = {}
                    for area in OHC['anoms_'+str(year)]:
                        r,p = pearsonr(y[:,0],OHC['anoms_'+str(year)][area][:-1])
                        if r<0:
                            X.append(-OHC['anoms_'+str(year)][area])
                            OHC['weights_'+regions[k]+str(year)][area] = 0


            n = len(y) 
            #######################################################


            X = np.asarray(X).T 
            Xs = np.asarray([X[-1,:]]) 
            X = X[:-1,:]
            ##########################################

            M = np.abs(np.cov(X, rowvar=False, bias=True)) 
            np.fill_diagonal(M,0)
            np.fill_diagonal(M,-np.sum(M,axis=0))

            ℓ = l_init ; σn_tilde = sigma_init
            
            Σ_tilde = expm(ℓ*M)
            L_tilde = np.linalg.cholesky(np.linalg.multi_dot([X,Σ_tilde,X.T]) + np.eye(n)*σn_tilde)
            A_tilde = np.linalg.solve(L_tilde.T,np.linalg.solve(L_tilde,y))
            σf = (np.dot(y.T,A_tilde)/n)[0][0]
            σn = σf*σn_tilde
            
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
            
            #Compute weight vector
            Σinv = np.linalg.inv(Σ)
            B = np.dot(X.T,X)+σn*Σinv
            Binv = np.linalg.inv(B)
            weights = np.linalg.multi_dot([Binv,X.T,y])
            
            m = 0
            if NY:
                if SIC_in == 'y':
                    for area in SIC['weights_'+regions[k]+str(year-1)]:
                        SIC['weights_'+regions[k]+str(year-1)][area] = weights[m]*Xs[0,m]
                        m += 1
                if OHC_in == 'y':
                    for area in OHC['weights_'+regions[k]+str(year-1)]:
                        OHC['weights_'+regions[k]+str(year-1)][area] = weights[m]*Xs[0,m]
                        m += 1
            else:
                if SIC_in == 'y':
                    for area in SIC['weights_'+regions[k]+str(year)]:
                        SIC['weights_'+regions[k]+str(year)][area] = weights[m]*Xs[0,m]
                        m += 1
                if OHC_in == 'y':
                    for area in OHC['weights_'+regions[k]+str(year)]:
                        OHC['weights_'+regions[k]+str(year)][area] = weights[m]*Xs[0,m]
                        m += 1

        GPR[regions[k]+'_fmean'] = fmean
        GPR[regions[k]+'_fvar'] = fvar
        GPR[regions[k]+'_fmean_rt'] = fmean_rt
    return GPR


def skill(fmin,fmax):
    regions = ['Pan-Antarctic','Ross','Weddell']#,'Bell-Amundsen','Indian','Pacific']
    skill_rt = []
    skill_dt = []
    dt_obs = []
    for k in range(len(regions)):
        dt = []
        for t in range(fmin,fmax+1):
            n = t - 1979
            dt.append(SIEs_dt[regions[k]][t-(fmin-1),n])
        dt_obs.append(dt)
        forecast_rt = GPR[regions[k]+'_fmean_rt']
        obs_rt = SIEs[regions[k]][fmin-1979:]
        a = np.mean((obs_rt-forecast_rt)**2)
        b = np.mean((obs_rt-np.nanmean(obs_rt))**2)
        skill_rt.append((1 - (a/b)).round(3))
        
        forecast_dt = GPR[regions[k]+'_fmean']
        c = np.mean((dt-forecast_dt)**2)
        d = np.mean((dt-np.nanmean(dt))**2)
        skill_dt.append((1 - (c/d)).round(3))
    return skill_rt,skill_dt,dt_obs

home = os.getcwd()
if os.path.exists(home+'/DATA')==False:
    os.mkdir(home+'/DATA')
    os.chmod(home+'/DATA',0o0777)
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
month_fc = input('Please enter the month you would like to forecast\n:')
if (month_data or month_fc) not in Month_dict.keys():
    assert('Month entered does not exist')
month_num_data = int(Month_dict[month_data])
month_num_fc = int(Month_dict[month_fc])

SIC_in = input('Do you want SIC data?\n')
OHC_in = input('Do you want OHC data?\n')

print('forecasting '+month_fc+' from '+month_data)
next_year =  (int(Month_dict[month_fc]) <= int(Month_dict[month_data]))

print('Downloading and reading data...')
print('Downloading and reading SIE data')
SIEs,SIEs_dt,SIEs_trend = read_SIE(fmin,fmax,month_fc)

if SIC_in == 'y':
    SIC = pickle.load(open(home+'/Clusters/SIC'+month_data+'1984-2021','rb'))

if OHC_in == 'y':
    OHC = pickle.load(open(home+'/Clusters/OHC'+month_data+'1984-2021','rb'))

skills_list = pd.read_csv(home+'/SICskills/'+month_data+'to'+month_fc+'detrended_skills_'+
                          str(fmin)+'-'+str(fmax)+'.csv',usecols=['Pan-Antarctic','Ross','Weddell']) 

if skills_list.shape[0] == 289:
    HP = divmod(np.array((skills_list.idxmax())),17)
elif skills_list.shape[0] == 400:
    HP = divmod(np.array((skills_list.idxmax())),20)

print('here are optimized hyperparameters:\n')
regions = ['Pan-Antarctic','Ross','Weddell']
for i in range(len(regions)):
    print(regions[i],HP[0][i],HP[1][i],'\n')
    
dumb = input('take your time if you want to check if this is correct!\n')
    
print('Running forecast...')
GPR = forecast(fmin,fmax,next_year,SIC_in,OHC_in,HP)
skill_rt,skill_dt,dt_obs = skill(fmin,fmax)

if os.path.exists(home+'/Results5')==False:
    os.mkdir(home+'/Results5')
    os.chmod(home+'/Results5',0o0777)

years = np.arange(fmin,fmax+1).tolist()
years.append('Skill')

def prep(data,skill=None):
    if type(data)!=list:
        data = data.tolist()
    if skill is not None:
        data.append(skill)
    else:
        data.append('')
    return data

columns1 = ['Pan-Antarctic$_o$','Pan-Antarctic$_f$','Pan-Antarctic$_f$ unc','Ross$_o$','Ross$_f$','Ross$_f$ unc','Weddell$_o$','Weddell$_f$','Weddell$_f$ unc']#,'Bell-Amundsen$_o$','Bell-Amundsen$_f$','Bell-Amundsen$_f$ unc','Indian$_o$','Indian$_f$','Indian$_f$ unc','Pacific$_o$','Pacific$_f$','Pacific$_f$ unc']
columns2 = ['Pan-Antarctic$_o$','Pan-Antarctic$_f$','Ross$_o$','Ross$_f$','Weddell$_o$','Weddell$_f$']#,'Bell-Amundsen$_o$','Bell-Amundsen$_f$','Indian$_o$','Indian$_f$','Pacific$_o$','Pacific$_f$']

data_dt = list(zip(prep(dt_obs[0]),prep(GPR['Pan-Antarctic_fmean'],skill_dt[0]),prep(np.sqrt(GPR['Pan-Antarctic_fvar']).round(3)),prep(dt_obs[1]),prep(GPR['Ross_fmean'],skill_dt[1]),prep(np.sqrt(GPR['Ross_fvar']).round(3)),\
                   prep(dt_obs[2]),prep(GPR['Weddell_fmean'],skill_dt[2]),prep(np.sqrt(GPR['Weddell_fvar']).round(3)))) #),\
 #                  prep(dt_obs[3]),prep(GPR['Bell-Amundsen_fmean'],skill_dt[3]),prep(np.sqrt(GPR['Bell-Amundsen_fvar']).round(3)),\
 #                  prep(dt_obs[4]),prep(GPR['Indian_fmean'],skill_dt[4]),prep(np.sqrt(GPR['Indian_fvar']).round(3)),\
 #                  prep(dt_obs[5]),prep(GPR['Pacific_fmean'],skill_dt[5]),prep(np.sqrt(GPR['Pacific_fvar']).round(3))))
df_dt = pd.DataFrame(data_dt, index=years, columns=columns1)

data_rt = list(zip(prep(SIEs['Pan-Antarctic'][fmin-1979:]),prep(GPR['Pan-Antarctic_fmean_rt'],skill_rt[0]),\
                   prep(SIEs['Ross'][fmin-1979:]),prep(GPR['Ross_fmean_rt'],skill_rt[1]),\
                   prep(SIEs['Weddell'][fmin-1979:]),prep(GPR['Weddell_fmean_rt'],skill_rt[2])))#,\
 #                  prep(SIEs['Bell-Amundsen'][fmin-1979:]),prep(GPR['Bell-Amundsen_fmean_rt'],skill_rt[3]),\
 #                  prep(SIEs['Indian'][fmin-1979:]),prep(GPR['Indian_fmean_rt'],skill_rt[4]),\
 #                  prep(SIEs['Pacific'][fmin-1979:]),prep(GPR['Pacific_fmean_rt'],skill_rt[5])))
df_rt = pd.DataFrame(data_rt, index=years, columns=columns2)

df_dt.to_csv(home+'/Results5/'+month_data+'to'+month_fc+'_detrended_forecasts_'+str(fmin)+'-'+str(fmax)+'.csv')
df_rt.to_csv(home+'/Results5/'+month_data+'to'+month_fc+'_with_trend_'+str(fmin)+'-'+str(fmax)+'.csv')

if SIC_in == 'y':
    pickle.dump(SIC,open(home+'/Clusters/SIC'+month_data+'to'+month_fc+str(fmin)+'-'+str(fmax),'wb'))

if OHC_in == 'y':
    pickle.dump(OHC,open(home+'/Clusters/OHC'+month_data+str(fmin)+'-'+str(fmax),'wb'))
