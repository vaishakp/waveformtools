#########################################################################################################
#Import libraries
import os,sys
import math
import numpy as np
import h5py
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cmath
import scipy
import scipy.optimize as opt
from scipy.optimize import curve_fit
#from mpi4py import MPI
import time
import pycbc
from pycbc.waveform import get_td_waveform
import seaborn as sns
import datetime
import waveformtools
import config
##########################################################################################################

#########################################################################################################
def bintp2(x,fx,width,order,plot=0):
        #Function to bin the data at width and interpolate them at order
        #Interpolation orders 
        kind = [0,'linear','quadratic','cubic']
        #Parse width
        width = int(width)
        #Number of bins
        n = int(len(x)/width)
        #Location of the bins
        binloc = [np.mean(x[width*i:width*i+width]) for i in range(0,n+1)]
        #print(binloc)
        #Assigning y values to the bins
        yvals = [np.mean(fx[width*i:width*i+width]) for i in range(0,n+1)]
        #Assigning x values to the smoothened data
        xf=x[width:-(width)/2]
        yf = yvals
        #Interpolate if specified order is more than 0
        if order!=0:
                y2 = scipy.interpolate.interp1d(binloc, yvals, kind=kind[order])
        #Reassign yf
        yf = y2(xf)
        #Set xf to binloc if order=0
        if order==0:
                xf=binloc
        #y = signal.savgol_filter(fx,win,order)
        if plot:
                #Plot the filtered data
                plt.plot(x,fx,label='data')
                plt.plot(xf,yf,label='smoothened data')
                plt.title("Smoothened data by binning and interpolation")
                plt.grid(which='both',axis='both')
                plt.legend()
                plt.show()
        #Returns a list consisting of bin loacations and the correspnding y values
        return [binloc,yvals]

def xtract_semiperiod(data,dt=0):
        #Function to extract the time gap between two zero crossings in data
        #If dt is not specified, assume input data is pycbc timeseries
        if not dt:
                dt = data.delta_t
        #Convert input data to numpy array
        data=np.array(data)
        #list for holding turning points
        zero_crossings = []
        for i in range(0,len(data)-1):
                #If the function crosses zero, bookkeep the index
                if (data[i+1]>0 and data[i]<0) or (data[i+1]<0 and data[i]>0):
                        zero_crossings.append(i)
        print("Number of zero crossings:%d"%len(zero_crossings))
        #Return the location and the duration between the crossings
        return [np.array(zero_crossings),np.diff(zero_crossings)*dt]


def static_average(data,window,dt=0,epoch=0):
        #Function to compute the average of the data over the given window
        #If dt is not specified, assume input data is a pycbc timeseries
        if not dt:
                dt = data.delta_t
        #Convert input data into numpy array
        data = np.array(data)
        n_bins=int(len(data)/window)
        element_data_averaged=np.zeros([n_bins])
        for i in range(0,n_bins):
                element_data_averaged[i]=np.mean(data[epoch+window*i:epoch+window*(i+1)])
        data_averaged.append(element_data_averaged)
        return data_averaged

def static_semiorbital_avg(waveform,data_list,dt=0):
        #Function to average the input data over semiorbital period of the waveform
        #If dt is not specified, assume input data is a pycbc timeseries
        if not dt:
                dt = data.delta_t
        #Convert input data into n
        #List to hold all averaged data
        avgd_data_list=[]
        #Find the location of zero crossings and the semiorbital period
        zero_crossings,semiorbital_period=xtract_semiperiod(waveform,dt)
        #print("Zero crossings:",zero_crossings,"\n","Semiorbital_period",semiorbital_period)
        for data in data_list:
                #List to hold averaged data
                avgd_data=[]
                #Convert input data into numpy array
                data = np.array(data)
                #Static average over the semiorbital_periods
                for i in range(0,len(zero_crossings)-1):
                        #Append averaged data segment to data_averaged
                        avgd_data.append(np.mean(data[zero_crossings[i]:zero_crossings[i+1]]))
                #Append averaged data to alldata_averaged
                avgd_data_list.append(avgd_data)
        return [zero_crossings,semiorbital_period,avgd_data_list]
#######################################################<Make templates for multipole project>############################################
def multipole_model(mass1,mass2,distance,alpha210,dt=0):
        #If dt is not specified, assume input data is a pycbc timeseries
        if not dt:
                dt = distance.delta_t
        #Convert input data into numpy array
        distance = np.array(distance)
        multipole_second=alpha210*(mass1**(5)*mass2)/distance**3
        return pycbc.types.timeseries.TimeSeries(multipole_second,dt)   
################################################################<Interpolate the waveform>#########################################################

def interpolate_wfs(ts_data,interp_func,**kwargs):
        #Function to interpolate a list of timeseries data using the given interp_func function and the keyword arguments.
        #List for storing the interpolated data function
        interp_data=[]
        #Loop over items in input
        for wfs in ts_data:
                #Find sampling time_step
                dt = wfs.delta_t
                #Interpolate using the supplied function. The keyword arguments supplied to the fuction are the keyword arguments to be supplied to the interpolating function)
                #Append to the list of interpolated data function
                interp_data.append(interp_func(wfs.sample_times,np.array(wfs),**kwargs))
        #Return the interpolated data function list
        return interp_data



def resample(interp_data,epoch,dt,length):
        #Function to generate timeseries out of the given interpolated data function, epoch,sampling frequency, length(duration)
        data=[]
        #Loop over objects in interp_data
        for i in range(len(interp_data)):
                #Prepare timeaxis
                timeaxis=np.linspace(epoch,epoch+length,int(length/dt))
                #Append the timeseries to the data list
                ydata=interp_data[i](timeaxis)
                data.append(pycbc.types.timeseries.TimeSeries(ydata, dt, epoch=epoch))
        #Return the list of samples timeseries          
        return data
        
def interpolate_resample_wfs(ts_data, interp_func, epoch, dt, length, **kwargs):
        #Wrapper function for interpolation+resampling.
        #Interpolate
        interp_data=interpolate_wfs(ts_data, interp_func, **kwargs)
        #Resample
        return resample(interp_data,epoch,dt,length)
##################################################<Waveform match>#####################################################################################
def match_wfs(waveforms,delt=None):
        #Match given waveforms
        #Input: List of pairs [waveform A, waveform B]
        #Assumes: delt is same for each pair.
        #Returns: [match score (float), shift (number), start_index, end_index]
        match=[]
        #Iterate over (signal,template) pairs in waveforms
        for waveformdat in waveforms:
                #Carryout the match
                if not delt:
                        try:
                            delt=waveformdat[0].delta_t
                        except: 
                            waveformtools.message('Waveform is not a pycbc TimeSeries. Please provide the gridspacing delt')
                #Match procedure
                signaldat=waveformtools.lengtheq(waveformdat[0],waveformdat[1],delt)

                waveform1 = signaldat[0]
                waveform2 = signaldat[1]

                #alignedwvs = pycbc.waveform.utils.coalign_waveforms(signaldat,hpa)
                #Compute the match to calculate match and shift. 
                #Note: The match function from pycbc returns the match of the normalized templates
                (match_score, shift) = pycbc.filter.matchedfilter.match(waveform1,waveform2)
                #Shift the matched data against the template using the shift obtained above
                waveform1 = waveformtools.shiftmatched(np.array(waveform1),int(shift),delt)
                #Compute the start and end of the non-zero signal
                #Note: The criterion that waveformtools.startend() uses is that the signal exists in non-zero portion of the data. 
                starti,endi = waveformtools.startend(waveform1)
                #Convert the non-zero portion of the signal and template to time-series
                signal = pycbc.types.timeseries.TimeSeries(np.array(waveform1)[starti:endi]/np.linalg.norm(np.array(waveform1)[starti:endi]),delt)
                template = pycbc.types.timeseries.TimeSeries(np.array(waveform2)[starti:endi]/np.linalg.norm(np.array(waveform2)[starti:endi]),delt)
                #Sanity check: The template and the signal must be of the same length at this point in execution
                if len(signal)!=len(template):
                        print("Error\n")
                        print("Length of data, template after truncation are %d,%d"%(len(signal),len(template)))
                        sys.exit(0)
                #Compute the match, shift again on the truncated data
                #print("length of data %d, aligned data %d, template %d"%(len(signaldat),len(alignedwvs[0]),len(alignedwvs[1])))    
                (match_score, shift) = pycbc.filter.matchedfilter.match(signal,template)
                match.append([match_score,shift,starti,endi])
        return match
##############################################################<End>####################################################################################
#The print function with verbosity levels and logging facility

def message(*args,message_verbosity=2,print_verbosity=config.print_verbosity,log_verbosity=config.log_verbosity,**kwargs):
        #message_verbosity:Each message carries with it a verbosity level. More the verbosity more the priority. Default value is 2
        #print_verbosity: prints all messages above this level of verbosity
        #log_verbosity: logs all messages above this level of verbosity

        #Verbosity  levels:
        #0: Errors
        #1: Warnings
        #2: Information

        #If message verbosity matches the global verbosity level, then print
        if message_verbosity<=print_verbosity:
                print(*args,**kwargs)
        if log_verbosity<=message_verbosity:
                now = str(datetime.datetime.now())
                tstamp = now[:10]+'_'+now[11:16]
                with open('log_'+ tstamp + ".txt","a") as f:
                        f.write(*args)
                        f.write('\n')
        return 1

#def cprint(*args,**kwargs):
#       #Verbosity  levels: 0: Dont't print anything, 1: Print only serious issues, 2: Print error messages, 3: Print informational messages, 4: Print everything
#       print(*args,**kwargs)
