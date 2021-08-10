import config
import datetime
from inspect import getframeinfo, stack
import traceback
import os,sys
import math
import numpy as np
import h5py
import cmath
import scipy
from scipy import signal
import scipy.optimize as opt
import pycbc
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import statistics
from termcolor import colored
import pickle
###########################################################################################################
''' Basic functions '''
###########################################################################################################

def message(*args,message_verbosity=2,print_verbosity=config.print_verbosity,log_verbosity=config.log_verbosity,**kwargs):
		''' **The print function with verbosity levels and logging facility.**
			
			**Verbosity levels**

			**message_verbosity**:Each message carries with it a verbosity level. More the verbosity more the priority. Default value is 2
			
			**print_verbosity**: prints all messages above this level of verbosity.
			
			**log_verbosity**: logs all messages above this level of verbosity. 

					Verbosity  levels:
					0: Errors
					1: Warnings
					2: Information
			
			-----------
			**Input**: 
				``*args`` : same as to that of print functions,
				message_verbosity=2, 
				print_verbosity, 
				log_verbosity, 
				``**kwargs`` to print function.
			
			------------
			**Returns**: 
				messages to stdout and logging of messages. Function returns 1.'''

		
		#If message verbosity matches the global verbosity level, then print
		if message_verbosity<=print_verbosity:
				print(*args,**kwargs)
		if log_verbosity<=message_verbosity:
				now = str(datetime.datetime.now())
				tstamp = now[:10]+'_'+now[11:16]
				caller = getframeinfo(stack()[1][0])
				#frameinfo = getframeinfo(currentframe())
				if not os.path.isdir('logs'):
					os.mkdir('logs')

				with open('logs/'+tstamp + ".log","a") as f:
						if message_verbosity==-1:
							for line in traceback.format_stack():
								f.write(line.strip())
						f.write('\n')	 
						f.write('{}:{}\t{}'.format(caller.filename,caller.lineno,*args))
						f.write('\n')
		return 1
############################################################################################################
############################################################################################################
''' Data handling functions '''
############################################################################################################
''' This library consists of a bunch of functions that were frequently used for handling NR data. '''

############################################################################################################
''' Notes
1. Some functions here display plots. If you intend to use these on the cluster and if xterm is not adequately setup, you may have to either comment the plot 
codes or use the 'Agg' mode for plotting and save the figures instead of showing by changing the appropriate lines in the code.

2. These is a module consisting of functions, not completely optimized for speed. This will happen in future.
3. These functions are not defined in classes as they mostly use and operate on the objects of pycbc's builtin classes.
4. Any suggestions, comments, critisism invited to vaishak@iucaa.in!'''



############################################< Data I/O functions >######################################### 

def save_obj(obj, name, dir = './', protocol = pickle.HIGHEST_PROTOCOL ):
	''' A function to save python objects to disk using pickle.

	-----------
	**Input:**

	1. obj		 (object) : The python object to be saved.
	2. name		 (string) : The filename.
	3. dir		 (string) : The path to directory to be saved in. Defaults to PWD
	4. protocol		(int) : The protocol to be used to save. Default is binary.

							Protocols:
							-----

							0 : Text
							5 : Binary

							See man page of pickle for more details.
	------------
	**Returns:**

	Nothing.
	'''

	#Create the directory dir if it doesn't exist.
	if not os.path.isdir(dir):
		os.mkdir(dir)

	#Pickle the file to disk.
	with open( dir+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, protocol)

def load_obj(name, dir='./' ):
	''' A function to load python objects from the disk using pickle.
	
	-----------
	**Input:**
	
	1. name		 (string) : The filename.
	2. dir		 (string) : The path to directory in which file exists. Defaults to PWD.

	------------
	**Returns:**

	1. obj		 (object) : A python object with the contents of the file.
	'''

	#Load the pickled data
	with open( dir + name + '.pkl', 'rb') as f:
		return pickle.load(f)

############################################<Calculus on a TimeSeries>######################################
#Differentiate a time series
#Todo: Add higher order methods

def removeNans(x,y):
	''' Remove Nans from (x,y) data pair. Removes Nans in x and y and the corresponding y and x entries.

	-----------
	**Input:**

	1. x (1d)		:	The x axis of the data.
	2. y (1d)		:	The y axis of the data.

	------------
	**Returns:**

	x,y (1d, 1d)	:	The data pair with Nans removed.

	'''

	#Find the location of x Nans to be removed.
	nan_locs=np.where(np.isnan(x))[0]
	#Remove the x and the corresponding y entries.
	x=np.delete(x,nan_locs)
	y=np.delete(y,nan_locs)
	#Find the location of the y Nans to be removed.
	nan_locs=np.where(np.isnan(y))[0]
	#Remove the y and the correspoinding x entries.
	x=np.delete(x,nan_locs)
	y=np.delete(y,nan_locs)
	#Find the location of the inifinities to be removed along x.
	inf_locs=np.where(np.isinf(x))[0]
	#Remove the x and the correspoiding y entries.
	x=np.delete(x,inf_locs)
	y=np.delete(y,inf_locs)
	#Find the location of the infinities to be removed along y.
	inf_locs=np.where(np.isinf(y))[0]
	#Remove the y and the corresponding x locations.
	x=np.delete(x,inf_locs)
	y=np.delete(y,inf_locs)

	#Return the reconditioned x and y data.
	return x,y

def differentiate(dat,dt=None):
		''' **Differentiate a timeseries (Simple Euler) **
		
		-----------
		**Input:** 
			a pycbc TimeSeries or a numpy array,
			grid spacing dt. Supplying dt overrides the delta_t attribute of TimeSeries.

		------------
		**Returns:** 
			the differentiated 1d data as pycbc TimeSeries'''

		if not dt:
			try:
				dt = dat.delta_t
			except:
				message('Input is not a TimeSeries. Please supply gridspacing as dt',message_verbosity=0)
		return pycbc.types.timeseries.TimeSeries(np.diff((np.array(dat))/dt),dt)

#Integrate a timeseries
def integrate(data, t_start=None, t_end=None, dt=None, taper='no'):
		''' **Integrate a timeseries using first order method.**

		** Capabilities**: Simple Euler integrator, option to taper the ends.
		
		-----------
		**Input**: 
			pycbc TimeSeries or numpy nd array or list, 
			grid spacing dt, 
			taper flag (string 'yes' or 'no' )
		
		-----------
		**Returns**: 
			TimeSeries of the time integrated data'''
		
		#Check if object is pycbc timeseries. Recover dt, t_start, t_end if yes.
		if not dt:
			try:
				#Find sampling time_step
				dt = data.delta_t
				data_time = data.sample_times
				t_start_dat, t_end_dat = data_time[0], data_time[-1]

			except:
				message('Input is not a TimeSeries. Please input a pycbc TimeSeries or supply gridspacing as dt',message_verbosity=0)
		else:
			t_start_dat = 0
			t_end_dat	= len(data)*dt

		if not t_start:
			t_start = t_start_dat

		if not t_end:
			t_end	= t_end_dat

		data = np.array(data)

		#Revert t_start and t_end to t_start_dat and t_end_dat i.e. to the starting time and  duration of the data respectively if user specified t_start is shorter and t_end is longer than t_start + the duration of the data.
		t_start =  max(t_start, t_start_dat)
		t_end = min(t_end, t_end_dat)

		start_index = int((t_start- t_start_dat)/dt)
		end_index	= int((t_end-t_start_dat)/dt)


		if taper=='yes':
			data = taper(data,dt)
		
		integdat = np.zeros([end_index-start_index])

		integdat[0] = 0.0#dat[0]*dt
		totinteg = np.sum(np.array(data))*dt
		
		#mean = np.mean(dat)
		
		data = np.array(data)
		
		for i in range(1,end_index-start_index):
				integdat[i] = integdat[i-1] + (data[start_index + i-1])*dt
		
		#return (integdat- totinteg)
		return pycbc.types.timeseries.TimeSeries(integdat,dt, epoch=t_start)
##############################################<Parameter conversions>###################################################

def freq(tc,t,Mc):
		''' **Compute the Newtonian instantaneous frequency of strain waveform from coalescence time and chirp mass.**
		
		----------
		**Input**: 
			tc: coalescence time (float) , 
			t: time (float or numpy 1d array) , 
			Mc: chirpmass (float).
		
		-----------
		**Returns**: 
			The instantaneous frequency of the strain waveform (float)'''
		
		return (1./(np.pi * Mc))*(5./256)**(3./8) * (Mc/(tc-t))**(3./8)

def totalmass(q,mchirp):
		''' **Find total mass from mass ratio and chirpmass.**
		
		-----------
		**Input**: 
			mass ratio (float) and 
			chirp mass (float).
		
		-----------
		**Returns**: 
			Total mass (float)'''
		
		return ((mchirp * (1.+q)**(6./5))/q**(3./5))

def massratio(Mchirp):
		''' **Find mass ratio from chirpmass. Assumes total mass to be 1. **
		
		-----------
		**Input:** 
			Mchirp: Chirp mass of the system (float).
		
		-----------
		**Returns**: 
			Mass ratio of the system (float) '''
		return ((Mchirp**(1./3) - 2. *Mchirp**2. - np.sqrt(Mchirp**(2./3) - 4.* Mchirp**(7./3)))/(2.* Mchirp**2.))


#Defining function for calculating Chirpmass from a2
def Mchirpc(a2):
		''' Compute the chirpmass from a2, the coefficient of time of the Finn-Chernoff waform model.
		
		-----------
		**Input:** 
			a2 (float).
		
		-----------
		**Returns:** 
			ChirpMass (float)'''
		return 2**(8./5) / (5* np.array(a2)**(8./5))

###############################################<Functions for handling data>#############################################
# Reconciling the length of strain

def lengtheq(data_a,data_b,dt=None, ts = 1):
		''' Equalize the length of two timeseries/array by appending zeros at the end of the array. No tapering.

		-----------
		**Procedure**

		1. Check if input data are timeseries. If not then construct timeseries using dt.
		2. Check data length. If they are already equal then skip and return the inputs.
		3. Check if data_a is smaller then data_b ( or vice versa). Then augment zeroes at either ends of array to data_a ( data_b) and return.
		-----------
		**Input:** 
			waveforms a and b as timeseries or arrays/lists.

		**Recommended usage**: change length of waveform 'a' to match with waveform 'b'.
		
		-----------
		**Returns:** 
			List containing the length equalized waveforms	as pycbc TimeSeries and a flag that conveys which input array has been modified.'''
		
		#Check if input data vectors are pycbc TimeSeries. If yes, create a copy of data and extract dt.
		if ts:
			if not dt:
				signala = data_a
				signalb = data_b
			
				try:
					dt=signala.delta_t
			
				except AttributeError:
						try:
							dt=signalb.delta_t
						except:
							message('Input is not a TimeSeries. Please supply a pycbc TimeSeries object or the gridspacing as dt',message_verbosity=0)
		else:
			dt = 1
		#If not TimeSeries, then construct TimeSeries using dt.
		data_a = np.array(data_a)
		data_b = np.array(data_b)
		signala = pycbc.types.timeseries.TimeSeries(data_a,dt)
		signalb = pycbc.types.timeseries.TimeSeries(data_b,dt)
		
		#If the length is already equal, then skip.
		if len(data_a)==len(data_b):
				lflag='ab'
		
		#If data_a < data_b
		elif len(data_a)<len(data_b):
				#add zeros to data_a when a is smaller
				lflag = 'a' 
				zers = len(data_b)-len(data_a)
				signala=np.transpose(np.concatenate((np.transpose(data_a),np.transpose(np.zeros([zers])))))
				signala=pycbc.types.timeseries.TimeSeries(signala,dt)
				#return pycbc.types.timeseries.TimeSeries(signalb,dt),lflag
		#If data_b < data_a
		else:
				#print("Error!")
				#add zeros to b when b is smaller
				lflag='b'
				zers = len(data_a)-len(data_b)
				signalb=np.transpose(np.concatenate((np.transpose(data_b),np.transpose(np.zeros([zers])))))
				signalb=pycbc.types.timeseries.TimeSeries(signalb,dt)
				#return pycbc.types.timeseries.TimeSeries(signala,dt),lflag
		
		#Returns a list containing the length equalized arrays and the flag.
		if not ts:
			signala = np.array(signala)
			signalb = np.array(signalb)

		return [signala, signalb, lflag]

def taperlengtheq(data_a, data_b,dt = None):
		''' Taper and equalize the lengths of two arrays
			-----------
			**Input:** 
				waveforms data_a, data_b as pycbc timeseries or list/array, 
				dt: grid spacing (float).
			
			------------
			**Returns:** 
				Tapered, length equalized waveforms data_b and
				A flag that determines which waveform's length was altered.'''
		
		#Check if input data is pycbc TimeSeries. If yes, then ectract dt.
		if not dt:
			signala = data_a
			signalb = data_b

			try:
				dt=signala.delta_t

			except AttributeError:
					try:
						dt=signalb.delta_t
					except:
						message('Input is not a TimeSeries. Please supply a pycbc TimeSeries object or the gridspacing as dt',message_verbosity=0)

		#Ensure data is numpy array.
		signalb = np.array(data_b)
		#Taper waveform A.
		signala = np.array(taper(data_a,dt))
		
		#equalize the length of a to match with b and return the length equalized arrays.
		return lengtheq(signala, signalb, dt)

#Check data for discontinuity
def iscontinuous_old(timeaxis,dt=0):
		''' Check if the data has discontinuities. This checks for repetitive time rows and jumps. 
		Types of discontunuities:
		0: Continuous.
		1: Repetitive rows.
		2: Jumps in timeaxis.
	
		-----------
		**Input:** 
			timeaxis: 1d array of timeaxis or a multi dimensional array of which the first axis is time.
			dt:		  The timestep dt.	
		
		------------
		**Returns:** 
			details of discontinuity (list) : actual location of discontinuity in time, value of time location of original array and the type of discontinuity.'''
		
		#If data array is supplied, assign first column as timeaxis
		if np.array(timeaxis).ndim>1:
			timeaxis=timeaxis[:,0]
		#Check data for continuity. 
		#If not timeseries
		if dt==0:
				dt=timeaxis[1]-timeaxis[0]
		#Set epoch to first element of timeaxis
		epoch=timeaxis[0]
		#List to hold discintinuity details
		discontinuity_details=[]
		#List to hold indices
		indices=[]
		#Set start index,epoch_index to 0
		index=0
		epoch_index=0
		#List to hold discontinuity type
		discontinuity_type=[]

		for timestamp in timeaxis:
		#Iterate over every timestamp in timeaxis
				#Check for discontinuity
				if timestamp!=timeaxis[0]+epoch_index*dt:
						#Check for type of discontinuity
						#Repetitive rows
						if timestamp<timeaxis[0]+epoch_index*dt:
								discontinuity_type=1
						#Missing rows
						elif epoch>timeaxis[0] + epoch_index+dt:
								discontinuity_type=2
						#Append [index location,correct timestamp 
						discontinuity_details.append([epoch_index,timeaxis[epoch_index],discontinuity_type])
				epoch_index+=1
				#message("Progress: %f%%\r"%(epoch_index*100./len(timeaxis)))
				
		#Print the result
		if len(discontinuity_timestamps)!=0:
				message("The data is discontinuous!")
		else:
				message("The data is continuous!")
		#Return the details of discontinuity
		return discontinuity_details

#Check data for discontinuity
def iscontinuous(data,dt=0,toldt=1e-3):
		
		''' Check if the data has discontinuities. This checks for repetitive time rows and jumps. 
		Types of discontunuities:
		0: Continuous.
		1: Repetitive rows.
		2: Jumps in timeaxis.

		-----------
		**Input:** 
			data : 1d array of timeaxis or a multi dimensional array of which the first axis is time.
			dt	 : timespacing.
			toldt: tolerance for error in repetition checking.
		
		------------
		**Returns:** 
			details of discontinuity: actual location of discontinuity in time,
			value of time location of original array and
			the type of discontinuity and the global discontinuity type.'''

		#Check the data (time,datar,datai) for locations of repetition and discontinuities.
		message('Checking continuity of data',message_verbosity=1)
		#Ensure data as numpy array
		data=np.array(data)
		message('Data shape:',(data.shape),message_verbosity=3)
		#Find if data contanins more than timeaxis
		axis=data.ndim-1
		message('Axis:%d'%axis,message_verbosity=3)
		#Associate data[0] as timeaxis
		if axis>0:
			shapes=data.shape
			if shapes[0]>shapes[1]:
				data = np.transpose(data)
			timeaxis=data[0,:]
			message('The time axis is :%s'%timeaxis,message_verbosity=3)
		else:
			timeaxis=data
		message('shape of data:',(data.shape),message_verbosity=3)

		#If data array is supplied, assign first column as timeaxis
		#if np.array(timeaxis).ndim>1:
		#		 timeaxis=np.array([item[0] for item in timeaxis])
		#message('Timeaxis',timeaxis,message_verbosity=3)
		#Check data for continuity. 
		#If dt is not supplied
		if dt==0:
				dt=statistics.mode(np.diff(timeaxis))
		#Set epoch to first element of timeaxis
		epoch=timeaxis[0]
		#List to hold discintinuity details
		discontinuity_details=[]
		#List to hold indices
		indices=[]
		#Initialize start index,epoch_index to 0
		index=0
		#Initialize epoch to start of timeaxis
		epoch=timeaxis[0]
		#Initialize epoch_index to 0
		epoch_index=0
		#Repetition flag
		repetition=0
		#Discountinuity flag
		discont = 0
		#Discontinuity type
		discont_type=0
		#List to hold discontinuity type
		discont_type_details=[]
		#Counters
		repetition_counter=0
		discont_counter=0
		#Iterate over every timestamp in timeaxis. 
		#Note the type of discontinuity. 
		#Ignore repetition and check for actual discontinuity i.e. missing rows.
		
		
		while index<len(timeaxis):
				#Read original timestamp at the location of operation number
				original_timestamp=timeaxis[index]
				#Calculate the recentered timestamp starting at epoch where previous discontinuity was found.
				recentered_timestamp=epoch+(index-epoch_index)*dt
				#message(original_timestamp,recentered_timestamp)
				#Check for next discontinuity
				if abs(original_timestamp-recentered_timestamp)>=toldt*dt:
						#Reset epoch, epoch_index
						epoch=original_timestamp
						epoch_index=index
						#Check for type of discontinuity
						#Repetitive rows
						if (original_timestamp==recentered_timestamp) or (recentered_timestamp-original_timestamp)>=toldt*dt:
								#if recentered_timestamp-original_timestamp<=0.01*dt:
								#		 message("Error!!",message_verbosity=0)
								repetition=1
								discont_type=repetition
								message('Repetitive rows found at index: %d,timestamp: %f'%(index,original_timestamp),message_verbosity=3)
								message("Repetition at timestamp original: %f correct %f\n"%(original_timestamp,recentered_timestamp),message_verbosity=3)
								repetition_counter+=1
						#Missing rows
						if ((original_timestamp-recentered_timestamp)>=(1.+toldt)*dt):
								discont=2
								discont_type=discont
								message('Jump discountinuity in data found at index:%d,timestamp:%f'%(index,original_timestamp),message_verbosity=2)
								message('dt=%f'%dt,message_verbosity=1)
								#message("Jump at timestamp original: %f correct %f\n Dt = %f"%(timeaxis[index-1],timeaxis[index],(timeaxis[index]-timeaxis[index-1])/dt),message_verbosity=1)
								message("Jump at timestamp original: %f correct %f\n Dt = %f"%(original_timestamp,recentered_timestamp,(original_timestamp-recentered_timestamp)/dt),message_verbosity=1)
								discont_counter+=1
						#message("timei: %f timef %f\n"%(timeaxis[index-1],timeaxis[index]),message_verbosity=3)

						#discont_type=repetition+discont
						discont_type_details.append([index,discont_type])
						#Append [index location,recentered timestamp 
						#discontinuity_details.append([index,recentered_timestamp,discontinuity_type])
				#Increment the no. of total operations
				index+=1
				#message("Progress: %f%%\r"%(epoch_index*100./len(timeaxis)))
		#discont_type=(repetition+discont)
		#Print the result
		if discont_type:
				message("The data is not clean!",message_verbosity=1)
				global_discont_type=repetition+discont
				message('Discontinuity type:',global_discont_type,message_verbosity=1)
				if global_discont_type==1 or global_discont_type==3:
						message('The data has repetitive rows at %d locations'%repetition_counter,message_verbosity=1)
				elif global_discont_type==2 or global_dinscont_type==3:
						message('The data has %d discontinuities'%discont_counter,message_verbosity=1)
				elif global_discont_type==3:
						message('The data has repetitive rows and is discontinious',message_verbosity=1)

				return [discont_type_details,global_discont_type]
		else:
				message('Data is continuous',message_verbosity=2)
				return [[0,0],0]
		#Return the details of discontinuity

#cleaning data for repeatative values
def cleandata_old(data,verbose='no'):
		''' Old version. Check the data (time,datar,datai) for repetetive rows and remove them.
			
			-----------
			**Input:** 
				data   : Numpy array or list and
				verbose: verbosity flag.
			
			-------------
			**Returns:** 
				Data array with repetitive rows removed'''

		#Ensure data as numpy array
		data=np.array(data)
		if verbose=='yes':
			message('Data shape:',data.shape)
		#Set axis along which to remove
		axis=data.ndim-1
		if verbose=='yes':
			message('Axis:',axis)
		#Associate data[0] as timeaxis
		if axis>0:
			shapes=data.shape
			if shapes[0]>shapes[1]:
				data = np.transpose(data)
			time=data[0,:]
			if verbose=='yes':
				message('The time array:',time)
		else:
			time=data
		#Assign dt
		dt=statistics.mode(np.diff(time))
		if verbose=='yes':
			message('length,shape of data',len(data),data.shape)
		#Reassign data without time_array
		#data=data[:,1:]
		#message("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
		#message("Checking data for repetative rows...\n")
		#Index of ros to delete
		dind=[]
		#Initial data length
		ki= len(time)
		#message("length of old array is %d\n" %ki)
		#Row iteration variable
		ii=0
		#Flag to identify if any repetetive rows were found
		pi=0
		#ci=0
		#rowcounter=np.zeros([])
		#Iterate over rows
		while (ii<ki-1):
		#Repetition condition :if the successive time stamp is less than or equal to the present, delete the row.
				if (time[ii+1]-time[ii])<=-0.1*dt:
					#Set flag to 1 if repetition condition is met.
					pi=1
					if verbose=='yes':
						message ("found a repeating row at %d, time %f\n" %(ii+1,time[ii+1]))
						message("timei: %f timef %f\n"%(time[ii],time[ii+1])) 
					#ci = ci+1
					#Delete the entire row
					time = np.delete(time,ii+1)
					#data = np.delete(data,ii,1)
					#data = [np.delete(item,ii+1) for item in data]
					data = np.delete(data,ii+1,axis)
					#Append the deleted index to the bookkeeping array
					dind.append(ii+1)
					#If a row is deleted, step back the iter variable by one step
					ii = ii-1
				#Advance the iter variable
				ii=ii+1
				#Recalculate the array length
				ki = len(time)
		if verbose=='yes':
				if pi==1:
						message("No. of points removed = %d\n" %len(dind))
						message("length of new array is %d\n" %len(time))
				else:
						message("No points removed\n")
				#message("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
		#Return the "cleaned" data matrix
		cleaned_data=data
		#cleaned_data=[time]
		#for item in data:
		#	 cleaned_data.append(item)
		#cleaned_data.append(item for item in data)
		#Transpose the data back to original shape
		if shapes[0]>shapes[1]:
			cleaned_data=np.transpose(cleaned_data)
		return cleaned_data

#cleaning data for repeatative values
def cleandata(data, toldt=1e-3, bridge='no', verbose=0):
		''' Check the data (time,datar,datai) for repetetive rows and remove them.
			
			-----------
			**Input:** 
				data  : Numpy array or list,
				toldt : Tolerance for error in checking. defaluts to toldt=1e-3, 
				bridge: ('yes' or 'no') Bridge flag to interpolate and resample to fill in jump discontinuities.
			
			-----------
			**Returns:** 
				data array with repetitive rows and gaps (if bridge='yes') removed.

		'''

		#Check the data (time,datar,datai) for repetetive rows and remove them.
		#Ensure data as numpy array
		data=np.array(data)
		message('Data shape:',(data.shape),message_verbosity=3)
		#Set axis along which to remove
		axis=data.ndim-1
		message('Axis:%d'%axis,message_verbosity=3)
		#Associate data[0] as timeaxis
		if axis>0:
			shapes=data.shape
			if shapes[0]>shapes[1]:
				data = np.transpose(data)
			time=data[0,:]
			message('The time array:%s'%time,message_verbosity=3)
		else:
			time=data
		dt=statistics.mode(np.diff(time))
		message('shape of data:',(data.shape),message_verbosity=3)
		#Reassign data without time_array
		#data=data[:,1:]
		#message("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
		#message("Checking data for repetative rows...\n")
		#Index of ros to delete
		dind=[]
		#Initial data length
		ki= len(time)
		#message("length of old array is %d\n" %ki)
		#Row iteration variable
		ii=0
		#Flag to identify if any repetetive rows were found
		pi=0
		#ci=0
		#rowcounter
		counter=0
		#Iterate over rows
		while (ii<ki-1):
		#Repetition condition :if the successive time stamp is less than or equal to the present, delete the row.
				if (time[ii+1]==time[ii]) or (time[ii]-time[ii+1])>=toldt*dt:
					#if time[ii]-time[ii+1]<=0.01*dt:
					#		 message("Error!!",message_verbosity=0)
					#Set flag to 1 if repetition condition is met.
					pi=1
					counter+=1
					message("found a repeating row at %d, time %f\n" %(ii+1,time[ii+1]),message_verbosity=3)
					message("timei: %f timef %f\n"%(time[ii],time[ii+1]),message_verbosity=3)
					#ci = ci+1
					#Delete the entire row
					time = np.delete(time,ii+1,0)
					#data = np.delete(data,ii,1)
					#data = [np.delete(item,ii+1) for item in data]
					data = np.delete(data,ii+1,axis)
					#Append the deleted index to the bookkeeping array
					dind.append(ii+1)
					#If a row is deleted, step back the iter variable by one step
					ii = ii-1
				#Advance the iter variable
				ii=ii+1
				#Recalculate the array length
				ki = len(time)
		if pi==1:
				message("Repetitive rows were removed",message_verbosity=2)
				message("No. of rows removed = %d\n" %counter,message_verbosity=2)
				message("Length of new array is %d\n" %len(time),message_verbosity=3)
		else:
				message("No points removed\n",message_verbosity=2)
				#message("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
		#Return the "cleaned" data matrix
		cleaned_data=data
		if bridge=='yes' and iscontinuous(cleaned_data)[-1]>=2:
				message('The data will be interpolated to bridge the gaps',message_verbosity=2)
				#Interpolate the data to fill in the discontinuities
				tf=time[-1]
				ti=time[0]
				#Find dt
				dt=time[1]-time[0]
				index=1
				#If second row is repetitive (If its discontinuous, help!)
				while dt<=0:
						dt=np.diff(time)[index]
						index+=1
				proper_timeaxis = np.arange(ti,tf,dt)
				interp_data=[]
				interp_data.append(proper_timeaxis)
				if axis>0:
						for index in range(1, min(shapes[0],shapes[1])):
							interp_datai = scipy.interpolate.interp1d(time,data[index,:])
							interp_data.append(interp_datai(proper_timeaxis))
					
				cleaned_data=np.array(interp_data)
				message('The data has been interpolated',message_verbosity=2)
		message('Cleaned!',message_verbosity=2)
		#cleaned_data=[time]
		#for item in data:
		#	 cleaned_data.append(item)
		#cleaned_data.append(item for item in data)
		#Transpose the data back to original shape
		if axis>0:
				if shapes[0]>shapes[1]:
						cleaned_data=np.transpose(cleaned_data)
		return cleaned_data

#Time shift an array
def shiftmatched(hdat,ind,dt=None):
		''' Timeshift an array. IMP: After timeshifting, the original length of the array is retained by clipping last(first) when ind > 0(ind <0) 'ind' number of points!!.
		Make sure the input array already has number of zeros z > ind (z<ind) initially at the end. 
		
		-----------
		**Input:**	
			hdat: an array or a pycbc timeseries, 
			ind : the number of timesteps to shift and
			dt	: the grid spacing in time 'dt'.
		
		-------------
		**Returns:** 
			a pycbc timeseries	array of same length timeshifted by 'ind' units by prepending zeros. '''
		
		if not dt:
			try:
				dt = hdat.delta_t
			except:
				message('Input is not a TimeSeries. Please supply gridspacing as dt',message_verbosity=0)

		if ind==0:
				#Do nothing if ind is 0
				return pycbc.types.timeseries.TimeSeries(np.array(hdat),dt)
		elif ind>0:
				#ind>0 case for shifting array to the right
				#message hdat
				#Length of the data
				l = len(hdat)
				#Array holding zeroes to be appended
				z = np.zeros([ind])
				#The shifted array
				msig= np.transpose(np.concatenate((np.transpose(z),np.transpose(hdat))))
				#message msig
				#message msig[:-ind]
				#Return the clipped, shifted timeseries
				return pycbc.types.timeseries.TimeSeries(msig[:-ind],dt)
		elif ind<0:
				#ind <0 case for shifting array to the left
								#Length of the data
								l = len(hdat)
								#Array holding zeroes to be appended
								z = np.zeros([ind])
								#The shifted array
								msig= np.transpose(np.concatenate((np.transpose(hdat),np.transpose(z))))
								#message msig
								#message msig[:-ind]
								#Return a timeseries
								return pycbc.types.timeseries.TimeSeries(msig[ind:],dt)


#Complex Phase-Amplitude representation of data
def xtract_cphase(tsdata_p,tsdata_x,dt=None, plot = 'no'):
		''' Given real and imaginary parts of a complex timeseries, extract the phase of the waveform :arctan_(Img(data)/Re(data))
		
		-----------
		**Input:** 
			tsdata_p, tsdata_x : plus and cross polarized components of the waveforms tsdata_p and tsdata_x as pycbc TimeSeries or 1d arrays and 
			dt				   : gridspacing dt.
		
		-------------
		**Returns:** 
			nd array of extracted phase. '''
		
		#Assign the timestep. Real and imaginary parts are assumed to have same timestep.
		if not dt:
			try:
				dt=tsdata_p.delta_t
			except AttributeError:
				try:
					dt=tsdata_x.delta_t
				except:
					message('Input is not a TimeSeries. Please supply gridspacing as dt',message_verbosity=0)


		#Assign the timestep. Real and imaginary parts are assumed to have same timestep.
		#Convert the data in numpy arrays
		datap = np.array(tsdata_p)
		datax = np.array(tsdata_x)
		
				#Calculate the wrapped phase (phi0 -> (0,2Pi))
		phi0 = np.pi + np.arctan2(datax,datap)
		
		#Unwrapping the phase : find turning points

		#Unwrap the phase by finding turning points in phi0: 
		#Finding turning points for unwrapping arctan2 function
		#Bookkeeping for upper (tpu) and lower (tpd) turning points
		tpu = []
		tpd = []
		#j = 0 
		#k = 0
		#Upper turning point criterion
		for i in range(0,len(phi0)-2):
				if phi0[i]>5 and phi0[i+1]<1:
						tpu.append(i)
						#j = j+1
		#Lower turning point
				if phi0[i]< 1 and phi0[i+1]>5:
						tpd.append(i)
								#k = k+1 
				
		#Trim any zeros in the array( Note: Unecessary)
				#tpu = np.trim_zeros(tpu)
				#tpd = np.trim_zeros(tpd)
				#Calculate the timestamp of the turning points
		tput = dt * np.array(tpu)
		tpdt = dt * np.array(tpd)
		
				#Unwrapping the phase: Unwrap using the turning points
		
		#Variable for unwrapped phase
		phic = phi0
		#Iteration varible for turning points
		j=0
		
		#Unwrap the upper turning points by adding 2pi for every 2p
		for i in range(0,len(tpu)):
				for j in range(int(tpu[i])+1,len(phic)):
						phic[j] = phic[j]+2.*np.pi
		#Unwrap lower turning points by subtracting 2*pi for every tp
		for i in range(0,len(tpd)):    
				for j in range(int(tpd[i])+1,len(phic)):
						phic[j] = phic[j]-2.*np.pi
				
		#Plots.
		#Phase vs time.
		if plot=='yes':
			timeaxis = np.linspace(0,len(phi0)*dt,len(phi0))
			message(len(timeaxis), len(phi0))
			plt.scatter(timeaxis,phi0,s=1)
			plt.title("Phase")
			plt.xlabel("cctk_time")
			plt.ylabel("Phase in radians")
			plt.grid()
			#plt.savefig('../graphs/waveform_phase_{}_q1a0.pdf'.format(name)) 
			plt.show() 
			
			timeaxis = np.linspace(0,len(phic)*dt,len(phic))
			message(len(timeaxis), len(phic))
			#Unwrapped phase vs time
			plt.scatter(timeaxis,phic,s=1)
			plt.title("Phase (unwrpped)")
			plt.xlabel("cctk_time")
			plt.ylabel("Phase in radians")
			plt.grid()
			#plt.savefig('../graphs/waveform_phase_complete_{}_q1a0.pdf'.format(name)) 
			plt.show()
			#Return a 1d list containing the unwrapped phase
		return phic

def xtract_camp(tsdata_p,tsdata_x,dt=None):
		''' Given real and imaginary parts of a complex timeseries, extract the amplitude of the complex data vector : (tsdata_p + i * tsdata_x)
		
		-----------
		**Input:** 
			tsdata_p, tsdata_x : plus and cross polarized components of the waveforms tsdata_p and tsdata_x as pycbc TimeSeries or 1d arrays and 
			dt				   : gridspacing dt.
		
		-------------
		**Returns:** 
			nd array of extracted amplitude. '''
		
		# Assign the timestep. Real and imaginary parts are assumed to have same timestep.
		if not dt:
			try:
				dt=tsdata_p.delta_t
			except AttributeError:
				try:
					dt=tsdata_x.delta_t
				except:
					message('Input is not a TimeSeries. Please supply gridspacing as dt',message_verbosity=0)

		# Complex modulous of the data
		camp = np.sqrt(np.array(tsdata_p)**2 + np.array(tsdata_x)**2)
		
		if plot=='yes':
			# Plot amplitude vs time
			plt.scatter(tsdata_p.sample_times,camp,s=1)
			plt.title("Amplitude vs time")
			plt.xlabel("cctk_time")
			plt.ylabel("Amplitude")
			plt.grid()
			#plt.savefig('../graphs/waveform_phase_complete_{}_q1a0.pdf'.format(name)) 
			plt.show()
		#Returns the 1d numpy array amplitude 
		return camp

def xtract_cphaseamp(tsdata_1,tsdata_2,dt=None):
		''' Wrapper for extracting the amplitude and the phase of the complex vector.
		
		-----------
		**Input:** 
			tsdata_1, tsdata_2: Two timeseries vectors, the plus and cross polarized components.
			dt				  : timestepping dt.
		
		-------------
		**Returns:** 
			list containing complex amplitude (list) and phase (list). '''

		return xtract_camp(tsdata_1,tsdata_2,dt),xtract_cphase(tsdata_1,tsdata_2,dt)



#Simple overlap
def olap(data1,data2,psd=1.):
		'''  Calcuate the overlap between two data vectors weighted by the psd=1.
		
		-----------
		**Input:** 
			data vectors data1, data2: ndarray or list or pycbc TimeSeries.
			psd						 : the power spectral density to weight.
		
		-------------
		**Returns:** 
			(float) overlap divided by the psd. '''

		data1 = np.array(data1)
		data2 = np.array(data2)
		overlap = np.sum(data1*data2)/psd
		#Returns the overlap weighted by the psd if any
		return overlap

def norm(hdat,psd=1.):
		''' Calculate the norm of a vector.
		
		-----------
		**Input:** 
			hdat		   : data. numpy 1d array/ list/ pycbc TimeSeries.
			psd (1d array) : The noise power spectral density of the inner product.
		
		-------------
		**Returns:** 
			The norm with weighting by the psd.'''

		hdat = np.array(hdat)

		return np.sqrt(np.sum(hdat*hdat)/np.array(psd))
		#message("Norm is %f"%normfa)
		#return normf

def flatten(nflist):
		''' Flatten a list of list of lists. i.e. [[[],[]], [[],[]] ---> []
		
		-----------
		**Input:** 
			nflist : a list of list of lists ( a list of depth three).
		
		-----------
		**Returns:** 
			the flattened list'''
		
		flattened_list = []
		
		for x in nflist:
				for item in x:
						flattened_list.append(item)
		
		message("list length: (%d)"%(len(flattened_list)))
		#Return the 1d flattened list
		return flattened_list

#Start and end points of a waveform
#def startend(data):
#		data=np.array(data)
#		return np.where(data!=0)[0][0],np.where(data!=0)[0][-1]+1

def startend(data):
		''' Identify the start and endpoints of the data.
		**Procedure**

		The starting and ending index of the non-zero part of the data is the identification criterion. 
		Requires the data to be exactly zero outside a certain domain.
		
		-----------
		**Input:** 
			data : 1d data as list or numpy nd array or pycbc TimeSeries
		
		-------------
		**Returns:** 
			(a pair) the pair of indices denoting the start and end points of an array '''
		
		try:
			start_index = np.where(np.array(data)!=0)[0][0]
		except:
			message(colored('Warning! Start index not found!!','red'), message_verbosity=1)
			start_index=0

		try:
			end_index = np.where(np.array(data)!=0)[0][-1]+1
		except:
			message(colored('Warning! End index not found!!','red'), message_verbosity=1)
			end_index=0
		return start_index, end_index


def apxstartend(data,tol=1e-5):
		''' Identify the Approximate start and endpoints of the data.

		**Procedure**

		The starting and ending index of the peak tol (default 1e-5)  part of the data is the identification criterion.
		Requires the data to fall off to tol*peak absolute value outside a certain range.
		
		-----------
		**Input:** 
			data : 1d data as 1d list or numpy nd array or pycbc TimeSeries.
		
		-------------
		**Returns:** 
			the pair of indices denoting the start and end points of an array '''

		data = np.array(data)
		locs = np.where(data>np.amax(data)*tol)[0]
		#return the pair of indices
		return locs[0],locs[-1]+1

#Add z Zeros
def addzeros(data,z):
		''' Append zeros to an array without tapering.
		
		----------
		**Input:** 

			data as 1d list or numpy 1d array or pycbc Timeseries.
		
		------------
		**Returns:** 
			data with 'z' zeros concatenated at the end as numpy 1d array
			'''
		
		return np.transpose(np.concatenate((np.transpose(np.array(data)),np.transpose(np.zeros([z])))))

#Remove Zeros
def removezeros(data,dt):
			''' Remove zeros from the input waveform from either sides. Similar to startend but return the truncated array.
			
			-----------
			**Input:** 
				waveform as 1d list or numpy array or pycbc TimeSeries.
			
			-------------
			**Returns:** 
				(list) A list containing  waveforms with zeros removed on either sides, the start and end indices'''
		
			#Assign the timestep. Real and imaginary parts are assumed to have same timestep.
			if not dt:
				try:
					dt=data.delta_t
				except:
					message('Input is not a TimeSeries. Please supply gridspacing as dt',message_verbosity=0)
		
			starti,endi=startend(data)
			ret_data = pycbc.types.timeseries.TimeSeries(np.array(data)[starti:endi],dt)
			return [ret_data,[starti,endi]]

def shorten(tsdata,start,end,dt=None):
			''' Shorten an array given the start and end points.
			
			-----------
			**Input:** 
				1d list or numpy array or pycbc TimeSeries.
			
			-------------
			**Returns:** 
				(pycbc TimeSeries object) The shortened data as pycbc TimeSeries.
			'''
			#Assign the timestep. Real and imaginary parts are assumed to have same timestep.
			if not dt:
				try:
					dt=tsdata.delta_t
				except:
					message('Input is not a TimeSeries. Please supply gridspacing as dt',message_verbosity=0)
		
			return pycbc.types.timeseries.TimeSeries(np.array(tsdata)[start:end],dt)
		
def taper(data,dt=1,z=150):
		''' Function to taper and append additional zeros at either ends.
			
			-----------
			**Input:** 
				data : 1d data as list or numpy array or pycbc timeseries.
				dt	 : the timestepping.
				z	 : the number of zeros to be added.
			
			-------------
			**Returns:** 
				(pycbc TimeSeries object) 1d data tapered and zero padded as pycbc timeseries. '''

		#Check if data is pycbc timeseries:
		if type(data)!=pycbc.types.timeseries.TimeSeries:
				flag=1
				#Convert to numpy array		   
				data=np.array(data)
				#First taper both sides of the data i.e. the start and end of the data.
				#Convert to pycbc TimeSeries
				data = pycbc.types.timeseries.TimeSeries(data,dt)
		elif type(data)==pycbc.types.timeseries.TimeSeries:
				flag=0
				dt=data.delta_t

		#Taper the timeseries
		data = pycbc.waveform.utils.taper_timeseries(data,tapermethod='TAPER_STARTEND')
		
		#Append the zeros
		data = np.array(data)

		#Pad ends with extra zeros
		z = np.zeros([z])
		#Prepend with z zeros
		data=np.transpose(np.concatenate((np.transpose(z),np.transpose(data))))
		#Append with extra zeros
		data=np.transpose(np.concatenate((np.transpose(data),np.transpose(z))))
		#Convert back to timeseries
		data = pycbc.types.timeseries.TimeSeries(data,dt)
		#Return the timeseries
		return data


def center(wvp,wvc=None ,dt=None):
		''' Center a waveform (wvp, wvc) at the peak of the complex modulous sqrt(wvp**2 + wvc**2).
			--------------
			**Procedure**
			1. Findout if both polarizations are supplied. If not assume cross pol is plus.
			2. Find the absolute magnitde location in the array.
			3. Construct the time limits using this information.
			3. Construct timeseries with epoch as first index of the array.



			
			-----------
			**Input:** 
				wvp, wvc:  The one/two components of the waveforms as 1d list or numpy arrays or pycbc timeseries.
				dt		:  The timestepping dt.
			
			------------
			**Returns:** 
				(a pair of pycbc TimeSeries) The two 1d centered waveform(s) as individual pycbc timeseries. '''
		#Flag to find out if both polarizations are supplied or not.
		flag = 0	   
		#If only one waveform is provided, assume cross pol = plus pol.
		if not wvc:
			flag = 1
			wvc = wvp
		#Assign the timestep. Real and imaginary parts are assumed to have same timestep.
		if not dt:
			try:
				dt=wvp.delta_t
			except AttributeError:
				try:
					dt=wvc.delta_t
				except:
					message('Input is not a TimeSeries. Please supply gridspacing as dt',message_verbosity=0)

		datap = np.array(wvp)
		datac = np.array(wvc)
		
		#Assign the complex amplitude
		amp = np.power(datap,2) + np.power(datac,2)
		#Find the location of the max amplitude
		ind = np.where(amp==np.max(amp))[0][0]
		#Calculate the epoch
		tlim = [-ind*dt,(len(datap)-ind)*dt]
		#Returns the centered wvp and wvc if wvc was provided else returns just the former.
		
		if flag==1:
			return pycbc.types.timeseries.TimeSeries(datap, dt, epoch=tlim[0])
		else: 
			return pycbc.types.timeseries.TimeSeries(datap, dt, epoch=tlim[0]), pycbc.types.timeseries.TimeSeries(datac, dt, epoch=tlim[0]) 

def plot(x,fx, save = 'no'):

		''' Basic plotting function.
		
		-----------    
		**Input:** 
			x	 : x axis of the function, 
			fx	 : y axis of the function. Can be supplied as 1d lists or arrays,
			save : 'yes' or 'no'. Whether the plot should be saved or not.

		------------
		**Returns:**
			1.
			Displays the plot,
			Saves with filename provided.
		'''
		
		plt.plot(np.array(x),np.array(fx))
		plt.title('f(x) vs x')
		plt.grid(which='both',axis='both')
		plt.xlabel('x')
		plt.ylabel('f(x)')
		if save!='no':
			plt.savefig(save+'.pdf')
		plt.show()
		return 1

#Custom coalign function

def coalignwfs(tsdata1,tsdata2,dt=None):
		''' Coalign two timeseries. Wrapper and modification around pycbc functions.
		
		---------------
		**Procedure:**
		
		1. Adjust length of either waveforms if needed
		
		2. Compute The complex SNR.
		
		3. Shift and roll the first.
		
		-----------
		**Input:** 
			tsdata1, tsdata2 : two data vectors as 1d lists or numpy arrays or pycbc timeseries,
			dt				 : time stepping.
			
		------------
		**Returns:** 
			(a pair of pycbc TimeSeries objects) The aligned waveforms.
		'''

		#Lengths of the two input timeseries
		l1 = len(tsdata1)
		l2 = len(tsdata2)

		#Add zeros at the end of waveform 1 without tapering if l2>l1
		#if l2>l1:
		#				 tsdata1,lflag = lengtheq(tsdata2,tsdata1,tsdata1.delta_t)
		#Add zeros at the end of waveform 2 without tapering if l1>l2
		#elif l1>l2:
		#				 tsdata2,lflag = lengtheq(tsdata2,tsdata1,tsdata1.delta_t)
		

		#Assign the timestep. Real and imaginary parts are assumed to have same timestep.
		if not dt:
			try:
				dt=tsdata1.delta_t
			except AttributeError:
				try:
					dt=tsdata2.delta_t
				except:
					message('Input is not a TimeSeries. Please supply gridspacing as dt',message_verbosity=0)

		tsdata1,tsdata2,lflag = lengtheq(tsdata1,tsdata2,tsdata1.delta_t)

		#Calculate complex SNR using pycbc function. Note: This complex SNR is actually the complex SNR * norm of the timeseries.
		csnr = pycbc.filter.matchedfilter.matched_filter(tsdata1,tsdata2)
		#Find the absolute value of the complex SNR timeseries
		acsnr = np.array(np.abs(csnr))

		#message(acsnr,np.max(acsnr))
		#Find the location of the maximum element in acsnr
		maxloc = (np.where(acsnr == np.max(acsnr)))[0][0]
		message("Max location is %s, match is %s"%(maxloc,np.max(acsnr)))

		#Shift the waveform 1 in time using maxloc
		tsdata1 = shiftmatched(tsdata1,maxloc,tsdata1.delta_t)
		#Phase shift ( rotate) the waveform 1 by multipying the frequency series of waveform 1 with the phase of the max element in acsnr
		#Calculate the rotation (as phase of the max complex modulous element in acsnr
		rotation = csnr[maxloc]/np.absolute(csnr[maxloc])
		#Rotate and take the inverse Fourier transform
		ctsdata1 = (rotation*tsdata1.to_frequencyseries()).to_timeseries()
		#Return the time and phase shifted waveform 1 to coalign with 2 and waveform 2. Note that the max modulous element of acsnr is only used to compute the time shift and is not used to normalize the 
		#waveforms. This therefore returns waveforms with their original normalization.
		return ctsdata1,tsdata2


#def coalignwfs(tsdata1,tsdata2):
#				 #Coalign two timeseries. 
#				 #Adjust length of either waveforms if needed
#				 #Compute The complex SNR
#				 #Shift and roll the first
#				 #Returns normalized waveforms
#
#		l1 = len(tsdata1)
#		l2 = len(tsdata2)
#		if l2>l1:
#				tsdata1,lflag = lengtheq(tsdata2,tsdata1,tsdata1.delta_t)
#		elif l1>l2:
#				tsdata2,lflag = lengtheq(tsdata2,tsdata1,tsdata1.delta_t)
#		csnr = pycbc.filter.matchedfilter.matched_filter(tsdata1,tsdata2)
#		acsnr = np.array(np.abs(csnr))
#		#message(acsnr,np.max(acsnr))
#		maxloc = (np.where(acsnr == np.max(acsnr)))[0][0]
#		message("Max location is %s, match is %s"%(maxloc,np.max(acsnr)))
#		tsdata1 = shiftmatched(tsdata1,maxloc,tsdata1.delta_t)
#		rotation = csnr[maxloc]/np.absolute(csnr[maxloc])
#		ctsdata1 = (rotation*tsdata1.to_frequencyseries()).to_timeseries()
#		return [ctsdata1,tsdata2]

def coalignwfs2(tsdata1,tsdata2,dt = None):
		''' Coalign two waveforms function 2. 
		
		**Procedure:**
		
		1. Adjust length of either waveforms if needed
		
		2. Normalize the two timeseries
		
		3. Compute The complex SNR
		
		4. Shift and roll the first
		
		5. Returns normalized waveforms
		
		See coalignwfs for description. This algorithm does not use the builtin coalign function from pycbc. 
		This involves normalization of the data vectors explicitly and identifiies the timeshift by computing
		the complex SNR and finding the maximum element.

		-----------
		**Input:** 
			tsdata1, tsdata2 : two data vectors as 1d lists or numpy arrays or pycbc timeseries.
		
		------------
		**Returns:** 
			a list: [the aligned waveforms, [norm1, norm2, location of maximum]].
		'''
		# Lengths of the two input timeseries
		l1						= len(tsdata1)
		l2						= len(tsdata2)
		
		tsdata1,tsdata2,lflag	= lengtheq(tsdata2,tsdata1,tsdata1.delta_t)
		
		# Find startend.
		if l1==l2:
				try:
					start, end		= startend(np.array(tsdata1))
				except:
					start, end		= apxstartend(np.array(tsdata1))
		
		# Add zeros at the end of waveform 1 without tapering if l2>l1
		if l2>l1:
		#		tsdata1,lflag = lengtheq(tsdata2,tsdata1,tsdata1.delta_t)
				try:
					start,end		= startend(np.array(tsdata1))
				except:
					start, end		= apxstartend(np.array(tsdata1))
		
		# Add zeros at the end of waveform 2 without tapering if l1>l2
		elif l1>l2:
				#tsdata2,lflag = lengtheq(tsdata2,tsdata1,tsdata1.delta_t)
				try:
					start,end		= startend(np.array(tsdata2))
				except:
					start,end		= apxstartend(np.array(tsdata2))
		
		# match,shift=pycbc.filter.matchedfilter.match(tsdata1,tsdata2)
		
		# Normalize the waveforms
		norm1					= norm(np.array(tsdata1[start:end]))
		norm2					= norm(np.array(tsdata2[start:end]))
		tsdata1					= pycbc.types.timeseries.TimeSeries(np.array(tsdata1)/norm1,tsdata1.delta_t)
		tsdata2					= pycbc.types.timeseries.TimeSeries(np.array(tsdata2)/norm2,tsdata2.delta_t)
		
		# Calculate complex SNR using pycbc function. Note: This complex SNR is actually the complex SNR * norm of the timeseries.
		csnr					= pycbc.filter.matchedfilter.matched_filter(tsdata1,tsdata2)
		
		# Find the absolute value of the complex SNR timeseries
		acsnr					= np.array(np.abs(csnr))
		#message(acsnr,np.max(acsnr))
		
		# Find the location of the maximum element in acsnr
		maxloc					= (np.where(acsnr == np.max(acsnr)))[0][0]
		message("Max location is %s, match is %s"%(maxloc,np.max(acsnr)))
		
		# Shift the waveform 1 in time using maxloc
		tsdata1					= shiftmatched(tsdata1,maxloc,tsdata1.delta_t)
		
		# Phase shift ( rotate) the waveform 1 by multipying the frequency series of waveform 1 with the phase of the max element in acsnr
		# Calculate the rotation (as phase of the max complex modulous element in acsnr
		rotation				= csnr[maxloc]/np.absolute(csnr[maxloc])
		
		#Rotate and take the inverse Fourier transform
		ctsdata1				= (rotation*tsdata1.to_frequencyseries()).to_timeseries()

		#Recenter waveform 0 and assign the timeaxis of waveform 0 to waveform1
		ctsdata1, dummy			= center(ctsdata1,ctsdata1)
		tsdata2					= pycbc.types.timeseries.TimeSeries(np.array(tsdata2), tsdata2.delta_t, epoch = ctsdata1.sample_times[0])

		#Return the normalized, time and phase shifted waveform 1 to coalign with 2 and waveform 2.
		return [ctsdata1,tsdata2,[norm1,norm2,maxloc]]


def simplematch_wfs(waveforms, delt=None):
		''' Simple match the given waveforms. Does not clip the waveforms at either ends.
		--------------
		**Procedure:**

		1. For each pair of waveforms as a list:
			a. Findout if delt has been specified.
			b. Findout if the object has attribute delta_t to discern whether it is a pycbc timeseries ( not exactly. ). If not then exit.
			c. Equalize the lengths.
			d. Compute the match score and shift using the pycbc shift function.
			e. Findout the start and the end of the waveform using handles.startend.
			f. Reconstruct normalized and clipped pycbc.timeseries of the waveforms.
			g. Confirm the equalization of the lengths of the waveoforms.
			h. Append the match details to an array [ waveform list, [ match score, shift, start_index, end_index]]
		2. Retun the match details for all the waveforms.

		
		-----------
		**Input:** 
			List of pairs [waveform A, waveform B].
		

		**Assumes:** delt is same for each pair.
		
		------------
		**Returns:** 
			(a list of dicts) [{ Aligned waveforms} , {match score (float), shift (number)}]
		'''

		match=[]
		# Iterate over (signal,template) pairs in waveforms
		for waveformdat in waveforms:
				# Carryout the match
				if not delt:
						try:
							delt		= waveformdat[0].delta_t
						except:
							message('Waveform is not a pycbc TimeSeries. Please provide the gridspacing delt')
							sys.exit(0)
				# Match procedure
				signaldat				= lengtheq(waveformdat[0], waveformdat[1], delt)

				waveform1				= signaldat[0]
				waveform2				= signaldat[1]

				# alignedwvs = pycbc.waveform.utils.coalign_waveforms(signaldat,hpa)
				# Compute the match to calculate match and shift.
				# Note: The match function from pycbc returns the match of the normalized templates
				(match_score, shift)	= pycbc.filter.matchedfilter.match(waveform1, waveform2)

				# Coalign the waveforms using pycbc coalign.
				waveform1, waveform2	= pycbc.waveform.utils.coalign_waveforms(waveform1, waveform2)

				#Normalize the waveforms

				waveform1				= waveform1/norm(waveform1)
				waveform2				= waveform2/norm(waveform2)

				try:
					(match_score, shift)	= pycbc.filter.matchedfilter.match(waveform1, waveform2)
				except:
					message('Final match couldn\'t be found!')
					match_score				= None
					shift					= None

				match.append({'Waveforms' : [waveform1, waveform2], 'Match score' : match_score, 'Shift' : shift})
		return match

def pmmatch_wfs(waveforms, offset=25, crop=None):

	'''
	Match function for post merger waveforms.


	------------
	**Procedure**

	1. Equalize the lengths
	2. Crop the waveforms if necessary.
	3. Normalize to their respective maximum amplitudes.
	4. Align the waveforms in phase.
	5. Compute the match score.

	------------
	**Inputs**
	1. waveforms		(list of pairs) : pairs of waveforms to match.
	2. offset			(int)			: Number of indices to shift the data.
	3. crop				(string)		: Options 1. signal 2. template 3. both.

	-----------
	**Returns**
	matchdet			(list of dicts) : A list of dictionaries. Each contains 1. waveform pair, 2. match score, 3. shift.

	'''
	matchdet = []
	for waveformdat in waveforms:

		signal, template						= waveformdat

		#message(type(signal), type(template))
		
		#message(type(signal), type(template), len(signal), len(template))

		signal, template, lflag					= lengtheq(signal, template)

		#message(type(signal), type(template))

		# Crop the template
		if crop=='both':
			signal								= signal[np.argmax(np.array(signal))+offset:]
			template							= template[np.argmax(np.array(template))+offset:]

		if crop=='signal':
			signal								= signal[np.argmax(np.array(signal))+offset:]

		if crop=='template':
			template							= template[np.argmax(np.array(template))+offset:]


		#message(type(signal), type(template))
		#message(np.amax(np.array(template_plus)), np.amax(np.array(signal_plus)))

		# Normalize the waveforms to their peak amplitudes.
		signal								= signal/norm(np.array(signal))
		template							= template/norm(np.array(template))
		
		try:
			delt = signal.delta_t
		except:
			signal = pycbc.types.timeseries.TimeSeries(signal, delta_t=1)

		
		try:
			delt = template.delta_t
		except:
			template = pycbc.types.timeseries.TimeSeries(template, delta_t=1)


		#message(type(signal), type(template))
		# Align the waveforms in phase
		signal_al, template_al				= pycbc.waveform.utils.coalign_waveforms(signal,  template)

		#message(np.where(np.array(signalp_al)!=0))

		# Compute the match score
		(matchscore, finalshift)	 = pycbc.filter.matchedfilter.match(signal_al, template_al)

		#message('+ The match score, shift are %f, %d'%(matchscore, finalshift))
		matchdet.append({'Waveforms': [signal_al, template_al], 'Match score' : matchscore, 'Shift': finalshift})

	return matchdet

def match_wfs(waveforms,delt=None):
		''' Match given waveforms.
		----
		----------
		**Procedure:**

		1. For each pair of waveforms as a list:
			a. Findout if delt has been specified.
			b. Findout if the object has attribute delta_t to discern whether it is a pycbc timeseries ( not exactly. ). If not then exit.
			c. Equalize the lengths.
			d. Compute the match score and shift using the pycbc shift function.
			e. Findout the start and the end of the waveform using handles.startend.
			f. Reconstruct normalized and clipped pycbc.timeseries of the waveforms.
			g. Confirm the equalization of the lengths of the waveoforms.
			h. Append the match details to an array [ waveform list, [ match score, shift, start_index, end_index]]
		2. Retun the match details for all the waveforms.

		
		-----------
		**Input:** 
			List of pairs [waveform A, waveform B].
		

		**Assumes:** delt is same for each pair.
		
		------------
		**Returns:** 
			(a list of dicts) {match score (float), shift (number), start_index, end_index}
		'''
		match=[]
		# Iterate over (signal,template) pairs in waveforms
		for waveformdat in waveforms:
				# Carryout the match
				if not delt:
						try:
							delt		= waveformdat[0].delta_t
						except:
							message('Waveform is not a pycbc TimeSeries. Please provide the gridspacing delt')
							sys.exit(0)
				# Match procedure
				signaldat				= lengtheq(waveformdat[0], waveformdat[1], delt)

				waveform1				= signaldat[0]
				waveform2				= signaldat[1]

				# alignedwvs = pycbc.waveform.utils.coalign_waveforms(signaldat,hpa)
				# Compute the match to calculate match and shift.
				# Note: The match function from pycbc returns the match of the normalized templates
				(match_score, shift)	= pycbc.filter.matchedfilter.match(waveform1, waveform2)
				# Shift the matched data against the template using the shift obtained above
				waveform1				= shiftmatched(np.array(waveform1), int(shift), delt)
				# Compute the start and end of the non-zero signal
				# First try with absolute startend. Then with approximate startend.
				#Note: The criterion that handles.startend() uses is that the signal exists in non-zero portion of the data.
				try:
					starti, endi		= startend(waveform1)
				except:
					message('Absolute startend not found. Fixing approximate startend')
					starti, endi		= apxstartend(waveform1)
					message('starti, endi')

				# Convert the non-zero portion of the signal and template to time-series
				signal					= pycbc.types.timeseries.TimeSeries(np.array(waveform1)[starti:endi]/np.linalg.norm(np.array(waveform1)[starti:endi]),delt)
				template				= pycbc.types.timeseries.TimeSeries(np.array(waveform2)[starti:endi]/np.linalg.norm(np.array(waveform2)[starti:endi]),delt)
				# Sanity check: The template and the signal must be of the same length at this point in execution
				if len(signal)!= len(template):
						message("Error\n")
						message("Length of data, template after truncation are %d,%d"%(len(signal),len(template)))
						sys.exit(0)
				# Compute the match, shift again on the truncated data
				# message("length of data %d, aligned data %d, template %d"%(len(signaldat),len(alignedwvs[0]),len(alignedwvs[1])))
				
				try:
					(match_score, shift)	= pycbc.filter.matchedfilter.match(signal,template)
				except:
					message('Final match couldn\'t be found!')
					match_score				= None
					shift					= None

				match.append({'Match score' : match_score, 'Shift' : shift, 'Start index' : starti, 'End index' : endi})
		return match

def roll(tsdata,i):
		''' Roll the data circularly. Circular counterpart of shiftmatched function. 
		
		-----------
		**Input:** 
			tsdata : 1D data vector in the form of a list/ numpy array or timeseries.
			i	   : The number of indices to roll the array. 
		
		------------
		**Returns:** 
			(pycbc TimeSeries object) The rolled wavefrom.
		'''

		#Assign the time step.
		dt = tsdata.delta_t
		#Assign the data array
		tsdata = np.array(tsdata)
		#Break the array into two parts as last i + first i entries.
		arr1 = tsdata[-i:]
		arr2 = tsdata[:-i]
		#Join the two arrays and return them
		return pycbc.types.timeseries.TimeSeries(np.transpose(np.concatenate((np.transpose(arr1),np.transpose(arr2)))),dt)

####################################################<Data smoothening functions>##########################################
''' Data soothening functions '''
def smoothen(fx,win,order, x=None, plot='no'):
		''' Use the Savitzky-Golay Filter to smoothen the data. Show the plots if plot='yes'.

		-----------
		**Input:** 
			fx(1d)			: the y axis, 
			win (int)		: Window for smoothening. Must be odd, 
			order(int)		: Order of the polynomial used for interpolation,
			x(1d)			: Optional. 1D list or numpy array, to plot the smoothened function. Only required if plot='yes'.
			plot(string)	: 'yes' or 'no'. Whether or not to display the plot.
		
		------------
		**Returns:** 
			y		   : (1d) The Savgol filtered list.
		
		'''

		#Apply the filter
		y = scipy.signal.savgol_filter(fx,win,order)
		#Show plots
		if plot=='yes':
				plt.plot(x,fx,label='data')
				plt.plot(x,y,label='smoothened data')
				plt.title("Smoothened data using Savitzky-Golay Filter")
				plt.grid(which='both',axis='both')
				plt.legend()
				plt.show()
		#Returns the filtered data
		return y

def bintp(x,fx,width,order,plot=0):
		''' Function to bin the data and interpolate it at specified width and order.
		
		-----------
		**Input:** 
			x(1d)		: 1D list or numpy array, 
			fx(1d)		: the y axis, 
			width (int) : Window size for smoothening, 
			order(int)	: Order of the polynomial used for interpolation.
		
		------------
		**Returns:** 
			(a list). [binloc, yvals]: The location of the bins and the y values associated with the bins. 
		
		'''
		
		#Interpolation orders 
		kind = [0,'linear','quadratic','cubic']
		#Parse width
		width = int(width)
		#Number of bins
		n = int(len(x)/width)
		#Location of the bins
		binloc = [np.mean(x[width*i:width*i+width]) for i in range(0,n+1)]
				#message(binloc)
		#Assigning y values to the bins
		yvals = [np.mean(fx[width*i:width*i+width]) for i in range(0,n+1)]
		#Assigning x values to the smoothened data
		#xf=x[width:-(width)/2]
		yf = yvals
				#Interpolate if specified order is more than 0
		if order!=0:
				y2 = scipy.interpolate.interp1d(binloc, yvals, kind=kind[order])
		#Reassign yf
		yf = y2(binloc)
		#Set xf to binloc if order=0
		#if order==0:
		#		xf=binloc
				#y = signal.savgol_filter(fx,win,order)
		if plot:
				#Plot the filtered data
						plt.plot(x,fx,label='data')
						plt.plot(binloc,yf,label='smoothened data')
						plt.title("Smoothened data by binning and interpolation")
						plt.grid(which='both',axis='both')
						plt.legend()
						plt.show()
		#Returns a list consisting of bin loacations and the correspnding y values
		return [binloc,yvals]


def mavg(fx,width):
		''' Function to smoothen data. Moving average over the window width.
			
			-----------
			**Input:** 
				fx (1D)		:  A list or numpy array of y axis. 
				Width (int) :  The width of the moving average window.
			
			------------
			**Returns:** 
				fxavgd(1D)	:  1D array of moving averaged y axis. 
		
		'''
		
		#message(len(fx))
		#List to store smoothened data
		fxavgd=[]
		#Calculate the moving-average upto last but width num of points
		for j in range(0,len(fx)-width):
				fxavgd.append(np.mean(fx[j:width+j]))
		#Calculate the moving-averaged values for the last width num of points
		for j in range(len(fx)-width,len(fx)):
				fxavgd.append(np.mean(fx[j:]))
		#Returns the list containing the moving-averaged values
		return fxavgd

################################################################<Interpolating the waveforms>#########################################################

def interpolate_wfs(ts_data,interp_func,dt=None,**kwargs):
		''' Function to interpolate a list of timeseries data using the user specified interp_func function and the keyword arguments.
		
		-----------
		**Input:** 
			ts_data (List)		  : The 1d data. A list of waveforms as list or numpy array or pycbc TimeSeries, 
			interp_fun (function) : An interpolating function, 
			dt (float)			  : Timestep, 
			``**kwargs``		  : additional arguments to the user specified interp_func.
		
		------------
		**Returns:** 
			interp_data (list).   : A list containing interpolated data. 
		
		'''
				

		#List for storing the interpolated data function
		interp_data=[]
		#Loop over items in input
		for wfs in ts_data:
				if not dt:
					try:
						#Find sampling time_step
						dt = wfs.delta_t
						timeaxis = wfs.sample_times

					except:
						message('Input is not a TimeSeries. Please supply gridspacing as dt',message_verbosity=0)
				else:
					timeaxis=np.arange(0,len(wfs)*dt,dt)

				#Interpolate using the supplied function. The keyword arguments supplied to the fuction are the keyword arguments to be supplied to the interpolating function)
				#Append to the list of interpolated data function
				interp_data.append(interp_func(timeaxis,np.array(wfs),**kwargs))
		#Return the interpolated data function list
		return interp_data



def resample(interp_data, new_dt, epoch,length,old_dt=None):
		''' Function to generate timeseries out of the given interpolated data function, epoch,sampling frequency, length(duration). 
		
		-----------
		**Input:** 
			interp_data (1D) :	The yaxis to be interpolated, 
			epoch (float)	 :	The starting point in time., 
			dt(float)		 :	New grid spacing to be sampled at. 
			length(int)		 :	The duration of x axis.
		
		------------
		**Returns:** 
			data(list)		 :	A list containing resampled data as pycbc TimeSeries.'''
		

		data=[]
		#Loop over objects in interp_data
		for i in range(len(interp_data)):
				if not old_dt:
					try:
						old_dt = interp_data[i].delta_t
					
					except:
						message('Input is not a TimeSeries. Please supply gridspacing as dt',message_verbosity=0)
				else:
					interp_data[i] = pycbc.types.timeseries.TimeSeries(interp_data[i],dt_old)

				#Prepare timeaxis
				timeaxis=np.linspace(epoch,epoch+length,int(length/new_dt))
				#Append the timeseries to the data list
				ydata=interp_data[i](timeaxis)
				data.append(pycbc.types.timeseries.TimeSeries(ydata,new_dt,epoch=epoch))
		#Return the list of samples timeseries					
		return data

def interpolate_resample_wfs(ts_data, interp_func, new_dt, epoch, length, old_dt=None, **kwargs):
				''' Wrapper function for interpolation and resampling.
					
					-----------
					**Input:** 
						interp_data (1D)			 : The yaxis to be interpolated, 
						epoch (float)				 : The starting point in time., 
						old_dt(float), new_dt(float) : Old and New grid spacing to be sampled at. length(int). The duration of x axis.
					
					------------
					**Returns:** 
						(1D)						 : Interpolated and resampled data.
				'''

				#Interpolate
				interp_data=interpolate_wfs(ts_data, interp_func, old_dt, **kwargs)
				#Resample
				return resample(interp_data, old_dt, new_dt, epoch, length)



def wavextractinf(data, r, t_start = None , t_end=None, dt=None, M=1.):
	''' Extracts a given waveform at a particular co-ordinate radius to infinity.

	-----------
	** Assumes **

	That the background is a Schwarzschild spacetime.

	-----------
	**Input**

	data(1d)		: The 1d waveform data.
	r (float)		: The (current) extraction radius of the data.
	t_start (float) : The start time of the data in t/M.
	t_end (float)	: The end time of the data in t/M.
	dt (float)		: The time step in t/M.
	M  (float)		: The total ADM mass of the spacetime.

	-----------
	**Returns**

	The extracted waveform (1d).

	'''

	#Check if object is pycbc timeseries. Recover dt, t_start, t_end if yes.
	if not dt:
		try:
			#Find sampling time_step
			dt = data.delta_t
			data_time = data.sample_times
			t_start_dat, t_end_dat = data_time[0], data_time[-1]
		
		except:
			message('Input is not a TimeSeries. Please input a pycbc TimeSeries or supply gridspacing as dt',message_verbosity=0)
	else:
		t_start_dat = 0
		t_end_dat	= len(data)*dt

	if not t_start:
		t_start = t_start_dat

	if not t_end:
		t_end	= t_end_dat 

	data = np.array(data)

	#Revert t_start and t_end to t_start_dat and t_end_dat i.e. to the starting time and  duration of the data respectively if user specified t_start is shorter and t_end is longer than t_start + the duration of the data.
	t_start =  max(t_start, t_start_dat)
	t_end = min(t_end, t_end_dat)
	

	start_index = int((t_start- t_start_dat)/dt)
	end_index	= int((t_end-t_start_dat)/dt)

	#Extract the data
	ext_data = (1./r)*(1.-2.*M/r)*(r*data[start_index:end_index]-(2.)* integrate(data, t_start, t_end, dt))
	
	return ext_data



def progress():
		''' Function to track the progress of an MPI code execution. Incomplete.
		
		-----------
		**Input:** 
			Nothing.
		
		------------
		**Returns:** 
			1'''
		
		count = count + 1
		message("%f"%(count*100./n))
		return 1
def progressbar(present_count,total_counts, normalize = 'yes'):
		''' Display the progress bar to std out from present_count and total_count.
		
		-----------
		**Input:** 
			present_count (int) : The present count state. 
			total_counts(int)	: The final state.
		
		
		------------
		**Returns:** 
			1
			The progress bar is messageed to stdout.
		
		'''
		
			
		if normalize=='yes':
			final_progress=98
			normalized_total_counts=final_progress *10
			present_count = int(normalized_total_counts*present_count/total_counts)
			total_counts = normalized_total_counts
			
		#present_count = comm.gather(count,root=rank)
		else:
			 final_progress=(int(total_counts/10))
		
		present_stage=int(present_count/10)
		#message(present_stage)
		#message(total_counts)
		if present_stage!=0:
			if present_count==total_counts:
				present_semi_progress='#'
			else:
				present_semi_progress=present_count%10
		else:
			present_semi_progress=present_count
		offset=final_progress-present_stage
		pc = 100*present_count/total_counts
		
		sys.stdout.write('\r' + "Progress:|" + '#'*present_stage +'%s'%present_semi_progress+' '*(offset) +'|'+'%.f%%'%pc)
		sys.stdout.flush()
		return 1

##########################################################################################################################################################
#def progress():
#		count = count + 1
#		message("%f"%(count*100./n))
#		return 1
#######################################END####################################################################
''' Data container for Numerical Relativity data '''

class sim:

	''' A data container for simulation data.
	
	-----------
	Arrtibutes:
			
			--------
			Primary:

			0.1 ROOTDIR				(string)			: Root directory as a string containing the simulation folders.
			0.2 WAVDIR				(string)			: Root directory as a string containing the simulation directies containing the wavefom data.
			0.3 datadir				(string)			: The path of the folder containing data relative to the simulation direcory.
			0.4 strain_dir			(string)			: The path of the folder containing the waveform data relative to the strain directory.
			1.	aliases				(List of strings)	: The names/aliases for the simulations.
			2.	multipoles			(dict of lists)		: The multipole moments of the simulation as a dictionary. Each entry is a list of width 4 with axis 0 the timeaxis of multipoles.
			3.	mass1				(dict of floats)	: The BH1 horizon mass.
			4.	mass2				(dict of floats)	: The BH2 horizon mass.
			5.	mass3				(dict of floats)	: The BH3 horizon mass.
			6.	delta_t				(dict of floats)	: The time stepping in simulation units (dt/M).
			7.	timeaxis			(dict of 1d)		: The timeaxis of the simulations.
			8.	distance			(dict of 1d)		: The distances of simulations.
			9.	merger_ind			(dict of ints)		: The merger index/ common horizon formation index of simulations.
			10. dinit				(dict of floats)	: The initial distances.
			11. multipoles			(dict of lists)		: The (2) mass multipoles of the three horizons. Axis 0 is usuall the time array.
			12. mass_multipoles		(dict of lists)		: The mass multipoles upto (l=8).
			13. spin_multipoles		(dict of lists)		: The spin multipoles upto (l=8). 
			14. data_length			(dict of float)		: The data length of the multipole simulation data loaded. 
			15. dist_data_length	(dict of ints)		: The data length of distances of simulations.
		   
			-------
			Derived

			1. merger_distance	   (dict of floats)    : The distance between the blackholes at the merger index.
			2. true_merger_dist    (dict of floats)    : The true merger distance (non normalized).
			3. sampling_f		   (dict of floats)    : The sampling frequency of simulations (1./dt).
			4. merger_time		   (dict of floats)    : The cctk_time stamp at merger.
			5. massratio		   (dict of floats)    : The massratio of the simulations.
			6. chirpmass		   (dict of floats)    : The chirpmass of the simulations.
			7. totalmass		   (dict of floats)    : The total mass of the simulations.
			8. log_multipoles	   (dict of lists)	   : The natural logarithm of the negative of the multipole moments as list [time, multipole1, multipole2, multipole3(if exists)].
			9. data_duration	   (dict of floats)    : The total cctk_time units of simulations present.
		
		----------
		Functions:

			1. calc_ref_multipoles						: Calculate the reference multipoles as time average of first few timesclices of (2) multipole moment data.
														  
														  Assignes/Updates:
														  -----
														  1. ref_multipoles.
														 

			2. calc_log_multipoles						: Calculate the natural logarithm of the negative of the (2) multipole moment.
														  
														  Assignes/ Updates:
														  -----
														  1. log_multipoles.

			3. calc_delta_multipoles					: Calculate the delta (2) multipoles - ref multipoles.
														  
														  Assignes/ Updates:
														  -----
														  1. delta_multipoles.

			4. calc_amp_phase							: Extract the amplitudes and phases of the strain waveforms.
														  
														  Assignes/Updates:
														  -----
														  1. strain_amp.
														  2. strain_phase.

			5. load_data								: Load the multipole and distance data of the simulations.
														
														  Assignes/Updates:
														  -----
														  1. distance.
														  2. dinit.
														  3. mass1.
														  4. mass2.
														  5. merger_index.
														  6. merger_time.
														  7. merger_distance.
														  8. true_merger_distance.
														  9. data_length.
														  10. data_duration.
														  11. multipoles.
														  12. mass_multipoles.
														  13. spin_multipoles.

			6. load_strain								: Load the strain data of the simulations from waveform directories.
												
														  Assignes/Updates
														  -----
														  1. strain.
														  2. strain_amp.
														  3. strain_phase.
														  4. strain_shift.

			7. load_shears								: Load the shear data at a pole of respective horizons of the simulations from waveform directories.
												
														  Assignes/Updates
														  -----
														  1. shear.
														  2. shear_amp.
														  3. shear_phase.
														  4. shear_shift.

			8. ret_horizon_radii						: A method to retrieve the areal radii of the horizons.

														  Assigns/Updates
														  -----
														  1. areal_radii

			9. _resize_multipoles						: Private method to resize the sim.multipoles after retrieving the length of distance.
			10. _ifreversal								 : Provate method to reverse the data of BH 1 and 2 if mass2 > mass1.


	'''
	def __init__(self,
				 #Variables for initialization.
				 aliases			= None,
				 multipoles			= None,
				 mass_multipoles	= None,
				 spin_multipoles	= None,
				 mass1				= None,
				 mass2				= None,
				 mass3				= None,
				 delta_t			= None,
				 merger_ind			= None,
				 actmerger_time		= None,
				 timeaxis			= None,
				 dinit				= None,
				 distance			= None,
				 ROOTDIR			= None,
				 WAVDIR				= None,
				 simdir				= None,
				 datadir			= 'data/primary/',
				 data_length		= None,
				 dist_data_length	= None,
				 comm_data_length	= None,
				 strain_dir			= 'output/',
				 strain				= None,
				 strain_phase		= None,
				 strain_frequency	= None,
				 strain_amplitude	= None,
				 strain_indexshifts = None,
				 indexjn			= None,
				 distjn				= None):

		#Load the variables at initialization.	  
		self.aliases				= aliases			 or {}					   
		self.multipoles				= multipoles		 or {}
		self.spin_multipoles		= spin_multipoles	 or {}
		self.mass_multipoles		= mass_multipoles	 or {}
		self.mass1					= mass1				 or {}
		self.mass2					= mass2				 or {}
		self.mass3					= mass3				 or {}
		self.delta_t				= delta_t			 or {}
		self.timeaxis				= timeaxis			 or {}
		self.distance				= distance			 or {}
		self.merger_ind				= merger_ind		 or {}
		self.actmerger_time			= actmerger_time	 or {}
		self.dinit					= dinit				 or {}
		self.data_length			= data_length		 or {}
		self.dist_data_length		= dist_data_length	 or {}
		self.comm_data_length		= comm_data_length	 or {}
		self.strain					= strain			 or {}
		self.strain_phase			= strain_phase		 or {}
		self.strain_frequency		= strain_frequency	 or {}
		self.strain_amplitude		= strain_amplitude	 or {}
		self.strain_indexshifts		= strain_indexshifts or {}
		self.indexjn				= indexjn			 or {}
		self.distjn					= distjn			 or {}
		self.ROOTDIR				= ROOTDIR
		self.WAVDIR					= WAVDIR
		self.simdir					= simdir
		self.strain_dir				= strain_dir
		self.datadir				= datadir
	
	@property
	def data_duration(self):
		'''Compute and return the data duration of the simulations.'''
		return {alias : self.data_length[alias]*self.delta_t[alias] for alias in self.aliases}
	
	@property
	def comm_data_duration(self):
		''' Compute and return the duration of the common data (multipole and Bhdiag/distance) of the simulations. '''
		return {alias : self.comm_data_length[alias]*self.delta_t[alias] for alias in self.aliases}
	
	@property
	def pm_data_duration(self):
		''' Compute and return the duration of post-merger data present in the simulations.'''
		return {alias : self.data_duration[alias] - self.merger_time[alias] for alias in self.aliases}

	@property
	def pm_data_length(self):
		''' Compute and return the post merger data length avaialable for all simulations. '''
		return {alias : self.data_length[alias] - self.merger_ind[alias] for alias in self.aliases}
	
	@property
	def merger_distance(self):
		''' Compute and return the distance at the merger index of the simulations. '''
		merger_dist={}
		for alias in self.aliases:
			try:
				merger_dist.update({alias:self.distance[alias][self.merger_ind[alias]]})
			except IndexError:
				if (self.merger_ind[alias]-len(self.distance[alias]))==1:
					merger_dist.update({alias : self.distance[alias][self.merger_ind[alias]-1]})
				else:
					message("%s :Distance upto merger not defined. Setting final value"%alias)
					merger_dist.update({alias : self.distance[alias][-1]})
		return merger_dist

	@property
	def true_merger_distance(self):
		''' Compute and return the true (i.e. non-normalized) distance at merger for the simulations.'''
		return {alias : self.dinit[alias]*self.merger_distance[alias] for alias in self.aliases}

	@property
	def sampling_f(self):
		''' Compute and return the sampling frequency of the simulations. '''
		return {alias : 1./self.delta_t[alias] for alias in self.aliases}

	@property
	def merger_time(self):
		''' Compute and return the cctk_time stamp at the merger for the simulations. '''
		return {alias : self.merger_ind[alias]*self.delta_t[alias] for alias in self.aliases}

	@property
	def massratio(self):
		''' Compute and return the massratio of the simulations. '''
		return {alias : self.mass2[alias]/self.mass1[alias] for alias in self.aliases}

	@property
	def chirpmass(self):
		''' Compute and return the chirp mass of the simulations. '''
		return {alias : ((self.mass1[alias]*self.mass2[alias])**(3./5))/(self.mass1[alias]+self.mass2[alias])**(1./5) for alias in self.aliases}

	@property
	def totalmass(self):
		''' Compute and return the total mass of the simulations. '''
		return {alias : self.mass1[alias]+self.mass2[alias] for alias in self.aliases}
	

	def calc_junkend(self, tjn = 200.0):
		
		''' Compute the indices and the starting distances of the system at timestamp t = 200.
		
		-------
		Inputs:
		1. tjn			  (float)  : The definition of time end of junk radiation. Default is 200.

		---------
		Computes:

		1. self.indjn	   (dict)  : A dictionary containing the index location corresponding to timestamp tjn.
		2. self.distjn	   (dict)  : A dictionary containing the normalized co-ordinate distance between the two BHs at tjn.

		'''

		#Find the starting distance at t = 200M.
		#Initialize the directory.
		self.indjn = {} 
		for alias in self.aliases:
			#Iterate over simulations.
			#Find the index corresponding to the timestamp tjn.
			indjn = np.argmin(np.absolute(self.timeaxis[alias]-tjn))
			#Update the sjn dictionary.
			self.indjn.update({alias : indjn})
			#message(len(sim_A.distance[alias]), ind)
			#Update th djn dictionary.
			self.distjn.update({alias : self.distance[alias][1][indjn]})
		#Return 1
		return 1

	def calc_ref_multipoles(self):
		''' Compute and assign the reference (l=2) multuipoles to sim.ref_multipoles of the simulations. '''
		refmult={}
		index=0
		for alias in self.aliases:
			message(alias)
			item=np.transpose(self.multipoles[alias])
			ml_length=len(item[:,0])
			if ml_length<self.comm_data_length[alias]:
				self.comm_data_length[alias]=ml_length
				self.distance[alias]=self.distance[alias][:ml_length]
			message(item.shape)
			'''Unpack data as 1d arrays'''
			time , Ml12, Ml22, Ml32 = item[:self.comm_data_length[alias],0],item[:self.comm_data_length[alias],1],item[:self.comm_data_length[alias],2], item[self.merger_ind[alias]:,3]
			'''Compute delta multipoles'''
			Ml12_ref = np.mean(Ml12[30:100])
			Ml22_ref = np.mean(Ml22[30:100])
			Ml32_ref = np.mean(Ml32[-100:])

			refmult.update({alias : [Ml12_ref, Ml22_ref, Ml32_ref]})
			self.ref_multipoles=refmult

	def calc_log_multipoles(self):
		''' Compute and assign the natural logarithm of the (l=2) multipoles to sim.log_multipoles of the simulations. '''
		log_mult={}
		for alias in self.aliases:
			message(alias)
			item=np.transpose(self.multipoles[alias])
			ml_length=len(item[:,0])
			if ml_length<self.comm_data_length[alias]:
				self.comm_data_length[alias]=ml_length
				self.distance[alias]=self.distance[alias][:ml_length]
			message(item.shape)
			log_mult.update({alias: [item[:self.comm_data_length[alias],0],np.log(np.absolute(-item[:self.comm_data_length[alias],1])),np.log(np.absolute(-item[:self.comm_data_length[alias],2])), np.log(np.absolute(-item[self.merger_ind[alias]:,3]))]})
		self.log_multipoles=log_mult

	def calc_delta_multipoles(self):
		''' Compute and return the delta multipoles (w.r.t. reference (l=2) multipoles) '''
		log_deltmult={}
		for alias in self.aliases:
			message(alias)
			#Multipole data
			item=np.transpose(self.multipoles[alias])
			'''Length of multipoles'''
			ml_length=len(item[:,0])

			'''Check the lengths of multipole and distance data'''
			if ml_length<self.comm_data_length[alias]:
				'''Reset the datalengths'''
				self.comm_data_length[alias]=ml_length
				'''Resize the distance array'''
				self.distance[alias]=self.distance[alias][:ml_length]
			message(item.shape)
			'''Unpack data as 1d arrays'''
			time , Ml12, Ml22, Ml32 = item[:self.comm_data_length[alias],0],item[:self.comm_data_length[alias],1],item[:self.comm_data_length[alias],2], item[self.merger_ind[alias]:,3]
			'''Compute delta multipoles'''
			Ml12_ref = np.mean(Ml12[30:100])
			Ml22_ref = np.mean(Ml22[30:100])
			Ml32_ref = np.mean(Ml32[-100:])
			dMl12 = Ml12 - Ml12_ref
			dMl22 = Ml22 - Ml22_ref
			dMl32 = Ml32 - Ml32_ref

			log_deltmult.update({alias: [time,np.log(np.absolute(-dMl12)),np.log(np.absolute(-dMl22)), np.log(np.absolute(-dMl32))]})

		self.log_deltamultipoles=log_deltmult
	
	def calc_log_multipoles2(self):
		''' Compute and assign the natural logarithm of the (l=2) multipoles to sim.log_multipoles of the simulations. '''
		log_mult={}
		for alias in self.aliases:
			message(alias)
			item=np.transpose(self.multipoles[alias])
			ml_length=len(item[:,0])
			if ml_length<self.comm_data_length[alias]:
				self.comm_data_length[alias]=ml_length
				self.distance[alias]=self.distance[alias][:ml_length]
			message(item.shape)
			log_mult.update({alias: [item[:self.comm_data_length[alias],0],np.log(-item[:self.comm_data_length[alias],1]),np.log(-item[:self.comm_data_length[alias],2]), np.log(-item[self.merger_ind[alias]:,3])]})
		self.log_multipoles2=log_mult

	def calc_delta_multipoles2(self):
		''' Compute and return the delta multipoles (w.r.t. reference (l=2) multipoles) '''
		log_deltmult={}
		for alias in self.aliases:
			message(alias)
			#Multipole data
			item=np.transpose(self.multipoles[alias])
			'''Length of multipoles'''
			ml_length=len(item[:,0])

			'''Check the lengths of multipole and distance data'''
			if ml_length<self.comm_data_length[alias]:
				'''Reset the datalengths'''
				self.comm_data_length[alias]=ml_length
				'''Resize the distance array'''
				self.distance[alias]=self.distance[alias][:ml_length]
			message(item.shape)
			'''Unpack data as 1d arrays'''
			time , Ml12, Ml22, Ml32 = item[:self.comm_data_length[alias],0],item[:self.comm_data_length[alias],1],item[:self.comm_data_length[alias],2], item[self.merger_ind[alias]:,3]
			'''Compute delta multipoles'''
			Ml12_ref = np.mean(Ml12[30:100])
			Ml22_ref = np.mean(Ml22[30:100])
			Ml32_ref = np.mean(Ml32[-100:])
			dMl12 = Ml12 - Ml12_ref
			dMl22 = Ml22 - Ml22_ref
			dMl32 = Ml32 - Ml32_ref

			log_deltmult.update({alias: [time,np.log(-dMl12),np.log(-dMl22), np.log(dMl32)]})

		self.log_deltamultipoles2=log_deltmult

	def load_data(self):
		''' Load data of the simulations. 
			
			--------------------
			Data is assigned to:

				1. multipoles.
				2. mass_multipoles.
				3. spin_multipoles.
				4. timeaxis.
				5. mass1.
				6. mass2.
				7. mass3.
				8. delta_t.
				9. distance.
				10. merger_ind.
				11. actmerger_time.
				12. dinit.
				13. data_length
				14. dist_data_length

				'''
		''' Common variables '''
		multipoles_one=[]
		masses_one=[]
		masses_one_all=[]
		M1_one=[]
		M2_one=[]
		M3_one=[]
		dt_one=[]
		timeaxis_one=[]
		#Multipole variables

		all_mass_multipoles_one = []
		all_spin_multipoles_one = []


		#Load Distances
		d_one=[]
		timeaxis_one=[]
		act_shape_one=[]
		d0_one=[]
		merger_time_one=[]


		ml_data_length=[]

		dist_data_length=[]
		all_mass_multipoles_1 = []
		all_mass_multipoles_2 = []
		all_mass_multipoles_3 = []

		all_spin_multipoles_1 = []
		all_spin_multipoles_2 = []
		all_spin_multipoles_3 = []


 
		#simulation directories.
		sim1 = [item +'/' for item in self.aliases]
		#Array for merger index.
		ind_merger_one=np.zeros(len(self.aliases),dtype=np.int)-1

		for sim_index in range(0,len(self.aliases)):
			#Loop over simulations.

			message(self.aliases[sim_index])
			message('------------------')
			
			#Load the qlm_multipole moment data.
			f0 = np.genfromtxt(self.ROOTDIR+sim1[sim_index]+self.data_dir+'quasilocalmeasures-qlm_multipole_moments..asc')

			#Assign the timeaxis.
			cctk_time = f0[:,8]
			
			#Assign the multipole data lengths to a list.
			ml_data_length.append(len(cctk_time))
			
			#Assign the timeaxis to a list.
			timeaxis_one.append(cctk_time)
			#Assign the time stepping to a list.
			dt_one.append(cctk_time[1]-cctk_time[0])
			#Load the BH horizon masses.
			M1_one_all=f0[:,12]
			M2_one_all=f0[:,13]
			M3_one_all=f0[:,14]
			
			#################################################################################
			#I.  
			#	1. Load multipole data. 
			#	2. Mass data.
			#	3. Find merger index.
			#################################################################################
			
			#Lists to hold mass multipole data of the three horizons of a simulation.
			amm1=[]
			amm2=[]
			amm3=[]

			#Lists to hold spin multipole data of the three horizons of a simulation.
			asm1=[]
			asm2=[]
			asm3=[]

			
			for col in range(0,9):
				#Extract the mass multipole data.
				amm1.append(f0[:,12+ col*3])
				amm2.append(f0[:,13+ col*3])
				amm3.append(f0[:,14+ col*3])

				#Extract the spin multipole data.
				asm1.append(f0[:,39+ col*3])
				asm2.append(f0[:,40+ col*3])
				asm3.append(f0[:,41+ col*3])




			#check for data continuity
			#Load a set of data to chek into a list.
			data0 = [cctk_time, M1_one_all, M2_one_all, M3_one_all]

			#Prepend the timeaxis to the mass multipoles.
			data1 = [cctk_time] + amm1
			data2 = [cctk_time] + amm2
			data3 = [cctk_time] + amm3

			#Prepend the timeaxis to the spin multipoles
			#message(len(data1), len(amm1))
			
			data4 = [cctk_time] + asm1
			data5 = [cctk_time] + asm2
			data6 = [cctk_time] + asm3

			#Clean the data using handles.cleandata.
			#Collect the data set 1.
			cctk_time, M1_one_all, M2_one_all, M3_one_all = cleandata(data0)

			#Collect the cleaned data set 2 and 3.
			cctk_time, *amm1 = cleandata(data1)
			cctk_time, *amm2 = cleandata(data2)
			cctk_time, *amm3 = cleandata(data3)

			cctk_time, *asm1 = cleandata(data4)
			cctk_time, *asm2 = cleandata(data5)
			cctk_time, *asm3 = cleandata(data6)


			#Find the merger index.
			#Note: This may not work if no merger has happened. This captures the length of data in that case.
			
			#Use the first non-zero index of mass3 array common horizon as the merger index.
			
			#Append to merger_one_a.
			try:
				#Try to find the non-zero index of horizon mass3.
				merger_one_a=np.where(M3_one_all!=0)[0]
			
			except:
				#Else set merger_one_a to length of mass3.
				merger_one_a=len(M3_one_all)
				message('Mass3 not non-zero anywhere in the data. Merger has probably not happened yet.  Reverting to data length.')
			#When merger happens, the BH horizon masses mass1 and mass2 acquire the same mass as the common horizon. Find the index where mass1 jumps by more than 0.1.
			#Assign to merger_one.

			try:
				#Try to find the jump location of the horizon mass1 data.
				merger_one=np.where(np.diff(M1_one_all)>0.1)[0]
			
			except:
				merger_one=len(M1_one_all)
				message('Mass1 has no jumps. Merger has probably not happened. Reverting to data length.')

			message('Merger index (through jump in mass1):', merger_one)

			if merger_one_a.any() and merger_one.any():
				#If both indices have been found, select the least of the two as merger index.
				merger_one = min(merger_one_a[0],merger_one[0])
				

				if merger_one==merger_one_a[0]:
					message('Merger data set from Mass3.')

				else:
					message('Merger data set from Mass1 jump.')

			elif not merger_one_a.any() and not merger_one.any():
				#If index of non-zero mass3 and mass1 jump was not found, declare no merger data exists.
				message('No merger data exists')
				merger_one = -1

			else:
				#If both indices are not equal, declare inconsistency but accept the result from either.
				
				message('Merger index inconsistency found. Mergertime may not be correct')
				
				try:
					# First try with non-zero mass3 index.
					merger_one = merger_one[0]
					message('Merger time set from Mass3')
				except:
					# Else assign from mass1 jump.
					merger_one = merger_one_a[0]
					message('Merger time set from Mass1 jump')

			#Declare merger time.
			message('Merger time:',dt_one[sim_index]*merger_one)
			message('Merger index:',merger_one)
			
			#Update the merger index of the simulation.
			ind_merger_one[sim_index]=merger_one

			#Load the masses.

			M1_one.append(np.mean(M1_one_all[200:300]))
			M2_one.append(np.mean(M2_one_all[200:300]))
			M3_one.append(np.mean(M3_one_all[-100:]))

			#Load the (l=2) multipoles.
			Ml12_one = f0[:,18]
			Ml22_one = f0[:,19]
			Ml32_one = f0[:,20]

			
			#message('Ml32_one', len(Ml32_one))

			
			#Update the mass multipoles for all l.
			all_mass_multipoles_1.append(amm1)
			all_mass_multipoles_2.append(amm2)
			all_mass_multipoles_3.append(amm3)

			#Update the spin multipoles for all l.
			all_spin_multipoles_1.append(asm1)
			all_spin_multipoles_2.append(asm2)
			all_spin_multipoles_3.append(asm3)

			#Gather the mass and spin multipoles together.
			all_mass_multipoles_one.append([amm1, amm2, amm3])
			all_spin_multipoles_one.append([asm1, asm2, asm3])

			
			#message(len(cctk_time))
			
			#Gather the l=2 mass multipoles.
			multipoles_one.append(np.array([cctk_time,Ml12_one,Ml22_one, Ml32_one]))

			#Gather the masses.
			masses_one.append([M1_one,M2_one,M3_one])
			masses_one_all.append([M1_one_all,M2_one_all,M3_one_all])
			
			###########################################################################
			# II . Load the distance data.
			# Compute the distance using the coordinate positions in BHdiagnostics files.
			###########################################################################
			temp0		=	np.genfromtxt(self.ROOTDIR+sim1[sim_index]+self.data_dir+'BH_diagnostics.ah1.gp')
			temp1		=	np.genfromtxt(self.ROOTDIR+sim1[sim_index]+self.data_dir+'BH_diagnostics.ah2.gp')

			t0			=	temp0[:, 1]
			x0_locs		=	temp0[:, 2]
			y0_locs		=	temp0[:, 3]
			
			message('Coord len', len(t0))
			t1			=	temp1[:, 1]
			x1_locs		=	temp1[:, 2]
			y1_locs		=	temp1[:, 3]
   
			temp0		=	np.transpose(np.array([t0, x0_locs, y0_locs]))
			temp1		=	np.transpose(np.array([t1, x1_locs, y1_locs]))
			
			message('Dist shapes', temp0.shape, temp1.shape)
			#check for data continuity.
			#Load data.
			data0		=	temp0
			data1		=	temp1
			
			#Clean the data.
			temp0		=	cleandata(data0)
			temp1		=	cleandata(data1)
			
			message('Dist shapes after cleaning', temp0.shape, temp1.shape)

			#Retrieve shape.
			shape0		=	temp0.shape
			shape1		=	temp1.shape

			#Retrieve timestepping.
			dt			=	np.diff(temp0[:,0])[0]

			#message('Time step',dt)

			#Retrieve merger index.
			mergerind	=	ind_merger_one[sim_index]
			
			#Update the merger time list.
			merger_time_one.append(ind_merger_one[sim_index]*dt_one[sim_index])


			#Find the shorter data. BHdiag1,2 or multipole data.
			act_len		= min(shape0[0],shape1[0], ml_data_length[sim_index])

			#If BHdiag 1 and 2 are equal in length and equal to multipole length:
			if shape0[0]==shape1[0] and shape0[0]==ml_data_length[sim_index]:
				message('BH_data length is consistent with multipole data length')
			
			#If the shortest data length is different from multipole length.
			if act_len==shape0[0] or act_len==shape1[0]:
				message('BH_diagnostics data is shorter')
			
			#If multipole data is shorter.
			else:
				message('Multipole data is shorter')

			#Update the actual length array.
			act_shape_one.append(act_len)
			
			#message('Data length',act_len)

			#act_shape_one.append()#
				#act_shape_one.append(min(shape1[0],shape2[0]))
				#message(temp0.shape,temp1.shape)

			#Try to load BHdiag3 file if present.
			try:
				
				#Try to load the BH_diagnostics file for BH3.
				temp2=np.genfromtxt(self.ROOTDIR+sim1[sim_index]+self.data_dir+'BH_diagnostics.ah3.gp')[:,np.r_[1,2,3]]
				
				#If merger index is not found from masses set using BH_diag3.
				
				#If the merger_ind from masses is greater than the BHdiag3 data length.

				if mergerind*dt>=int(temp2[0,0]):
					merger_time_one[sim_index]=temp2[0,0]
					mergerind = int(temp2[0,0]/dt)
					ind_merger_one[sim_index] = mergerind
					message('Merger time acquired from BHdiag3',mergerind*dt)
					message('Merger index has been updated with info from BHdiag3')
			except:
				message('Merger time acquired from masses',mergerind*dt)

			message('Merger index',mergerind)

			# FInd shape of data.
			shape0	=	temp0.shape
			shape1	=	temp1.shape

			#if shape0[0] < shape0[1]:
			#	 temp1 = np.transpose(temp1)

			#if shape1[0] < shape1[1]:
			#	 temp2 = np.transpose(temp2)

			# Assign the centroid locations to variables.			 
			
			t0		=	t0#temp0[:, 0]
			x0		=	x0_locs#temp0[:, 1]
			y0		=	y0_locs#temp0[:, 2]

			t1		=	t1#temp1[:, 0]
			x1		=	x1_locs#temp1[:, 1]
			y1		=	y1_locs#temp1[:, 2]

			#t0		 =	 temp0[:, 0]
			#x0		 =	 temp0[:, 1]
			#y0		 =	 temp0[:, 2]
			#
			#t1		 =	 temp1[:, 0]
			#x1		 =	 temp1[:, 1]
			#y1		 =	 temp1[:, 2]

			#if int((x0-x0_locs)[0])!=0:
			#	 message('ERRORRRRRRR!')
			#	 sys.exit(0)
			# Compute lengths

			l0		=	len(t0)
			l1		=	len(t1)
			  
			# Find minimum
			lmin	=	min(l0, l1)
			timeaxis_one.append(t0)

			# Crop data

			if l0<l1:
				t1		=	t1[:lmin]
				x1		=	x1[:lmin]
				y1		=	y1[:lmin]

			elif l1<l0:
				t0		=	t0[:lmin]
				x0		=	x0[:lmin]
				y0		=	y0[:lmin]
	 
			# Compute differences

			dx		=	x0 - x1
			dy		=	y0 - y1
				
			#if sim_index==3:
			#d.append(np.sqrt((temp0[:3014,1]-temp1[:ind_merger[sim_index],1])**2 + (temp0[:3014,2]-temp1[:ind_merger[sim_index],2])**2))

			# Compute the Eucledian distance.
			d_sim	=	np.sqrt( np.power(dx ,2) + np.power(dy, 2))
			d_sim	=	np.array(d_sim)
			d0_sim	=	d_sim[0]
			message('Initial true distance', d0_sim)
			d0_one.append(d0_sim)
			d_sim	=	d_sim/d0_sim
			#d_one.append(np.sqrt((temp0[:act_shape_one[sim_index],1]-temp1[:act_shape_one[sim_index],1])**2 + (temp0[:act_shape_one[sim_index],2]-temp1[:act_shape_one[sim_index],2])**2))
			d_one.append([t0, d_sim])
			message('ml_timeaxis_length: %d, Multipole data length: %d, BHdiag length: %d, Distance length: %d'%(len(timeaxis_one[sim_index]), ml_data_length[sim_index], shape1[0], len(d_one[sim_index])))
			dist_data_length.append(len(d_sim))



		#multipoles = np.array(multipoles)
		#Convert the acquired data lists into numpy arrays.
		masses_one	=	np.array(masses_one)
		M1_one		=	np.array(M1_one)
		M2_one		=	np.array(M2_one)
		M3_one		=	np.array(M3_one)

		#message('Multipoles shape',multipoles.shape)
		
		#message('Masses shape:',masses_one.shape)

		
		d_one		=	np.array(d_one)
		#d_one.shape
		#d_one.append(np.sqrt((temp0[:act_shape_one[sim_index],1]-temp1[:act_shape_one[sim_index],1])**2 + (temp0[:act_shape_one[sim_index],2]-temp1[:act_shape_one[sim_index],2])**2))
		#d_one=np.array(d_one)

		#d_one.shape

		#Plot the distance
		#message('multipoles length',len(multipoles_one))

		#Assign data to sim variables.
		for sim_index in range(0,len(self.aliases)):
			self.multipoles.update(				{ self.aliases[sim_index]	:	multipoles_one[sim_index]})
			self.mass_multipoles.update(		{ self.aliases[sim_index]	:	all_mass_multipoles_one[sim_index]})
			self.spin_multipoles.update(		{ self.aliases[sim_index]	:	all_spin_multipoles_one[sim_index]})
			self.timeaxis.update(				{ self.aliases[sim_index]	:	timeaxis_one[sim_index]})
			self.mass1.update(					{ self.aliases[sim_index]	:	M1_one[sim_index]})
			self.mass2.update(					{ self.aliases[sim_index]	:	M2_one[sim_index]})
			self.mass3.update(					{ self.aliases[sim_index]	:	M3_one[sim_index]})
			self.delta_t.update(				{ self.aliases[sim_index]	:	dt_one[sim_index]})
			self.distance.update(				{ self.aliases[sim_index]	:	d_one[sim_index]})
			self.merger_ind.update(				{ self.aliases[sim_index]	:	ind_merger_one[sim_index]})
			self.actmerger_time.update(			{ self.aliases[sim_index]	:	merger_time_one[sim_index]})
			self.dinit.update(					{ self.aliases[sim_index]	:	d0_one[sim_index]})
			self.data_length.update(			{ self.aliases[sim_index]	:	ml_data_length[sim_index]})
			self.comm_data_length.update(		{ self.aliases[sim_index]	:	act_shape_one[sim_index]})
			self.dist_data_length.update(		{ self.aliases[sim_index]	:	dist_data_length[sim_index]})
		#message(self.multipoles)

		#Resize the multipoles data if merger index was updated from BHdiag3.
		self._resize_multipoles()
		#Reverse the BH1 and BH2 data if BH mass2>mass1.
		self._ifreversal()

		#Plot the distances.
		for alias in self.aliases:
			x		=	self.distance[alias][0]
			y		=	self.distance[alias][1]
			length	=	min(len(x),len(y))
			x		= x[:length]
			y		= y[:length]
			#message(x,y)

			plt.plot(x,y)
			plt.title('Distance vs t/M '+ alias)
			plt.grid(which = 'both', axis='both')
			plt.xlabel('t')
			plt.ylabel(r'$d/d_{init}$')
			plt.show()

	def _resize_multipoles(self):
		''' Private method to resize the (l=2) multipole data. Useful when merger index was updated from BHdiag3.'''
		for alias in self.aliases:
			#Loop over simulations.
			#message(alias)
			#self.timeaxis[alias] = [item[:self.dist_data_length[alias]] for item in self.timeaxis[alias]]
			#Resize the lengths of the data.
			self.multipoles[alias] = [item[:self.dist_data_length[alias]] for item in self.multipoles[alias]]
			#self.data_length.update({alias : len(self.multipoles[alias][0])})
			#self.mass_multipoles[alias] = [item[:self.dist_data_length[alias] for item in self.mass_multipoles[alias]]]
			#self.spin_multipoles[alias] = [item[:self.dist_data_length[alias] for item in self.spin_multipoles[alias]]]


	def _ifreversal(self):
		''' Private method to reverse the (l=2) multipole data if mass2>mass1.

			--------
			Updates:

			sim.multipoles : Resized (l=2) multipole moment data. 
		'''

		#Flag to identify if reversal is required.
		flag = 0
		
		for alias in self.aliases:
			#Loop over simulations.
			message('Check for puncture reversal, %s'%alias)
			if alias!='q1a0_a' and alias!='q1a0_b':
				#Condition for reversal decision.
				if self.mass1[alias]<self.mass2[alias]:
					#Toggle the flag.
					flag=1
					message('**************************************************')
					message('BH 1 and 2 reversal found!!! \n Reversing data...')
					message('**************************************************')
					message('original mass1:%f, mass2: %f'%(self.mass1[alias], self.mass2[alias]))
					#Reverse the data.
					self.mass1[alias], self.mass2[alias]= self.mass2[alias], self.mass1[alias]
					#Unpack the multipoles data.
					time, multipole1, multipole2, multipole3 = self.multipoles[alias]
					#Reverse the multipole data.
					multipole1, multipole2 = multipole2,multipole1
					#Repack the data.
					self.multipoles[alias] = np.array([time, multipole1, multipole2, multipole3])
			if not flag:
				message('Data O.K.')
			return 1.
	


	def load_strain(self, start_index =0):
		''' Method to load the shear data of simulations. '''

		for sim_index in range(0,len(self.aliases)):
			#Loop over simulations.
			alias = self.aliases[sim_index]
			#Set the starting index for data.
			m = 0#start_index = 0#int(190/deltat[j])
			#Load the strain data.
			sim_strain_data = np.genfromtxt(self.WAVDIR+self.aliases[sim_index]+'/' + self.strain_dir +'/strain_'+ str(alias)+ "_wavextcpm.dat")

			#Load the timeaxis, plus and cross polarized data.
			htdat = sim_strain_data[m:,0]
			hpdat = sim_strain_data[m:,1]
			hxdat = sim_strain_data[m:,2]


			message("The strain file is")
			message(self.WAVDIR+self.aliases[sim_index]+'/' + self.strain_dir +'/strain_'+ str(alias)+ "_wavextcpm.dat")

			#Align the peak of the strain with the formation of the common horizon (merger index).

			Lpeak_loc = np.argmax(np.diff(hpdat)**2 + np.diff(hxdat)**2)

			#Load the common horizon location.
			commhor_loc = self.merger_ind[alias]
			#Compute the shift betweeen the Luminosity peak index and common horizon formation location.
			shift = Lpeak_loc-commhor_loc

			#Update the strain_indexshifts.
			self.strain_indexshifts.update({alias: shift})

			#Load the time stepping.
			dt = self.delta_t[alias]

			#Shift the timeaxis and clip the beginning of data.
			htdat = htdat[m:-shift]
			hpdat = hpdat[m+shift:]
			hxdat = hxdat[m+shift:]

			#Update the sim.strain
			self.strain.update({alias: [htdat, hpdat, hxdat]})
			
			if config.print_verbosity>1:
				#Plot the strains.
				plt.plot(htdat, hpdat, label='data '+ alias)
				plt.ylabel('Strain')
				plt.xlabel('Time (s)')
				plt.grid(which='both',axis='both')
				#plt.xlim(-600,400)
				plt.legend()
				plt.show()

		return 1

	def calc_amp_phase(self):
		''' Extract the amplitude and the phase from strain data. '''
		for alias in self.aliases:
			#Loop over simulations.

			#Load the plus and cross polarized strain data.
			hpdat = self.strain[alias][1]
			hxdat = self.strain[alias][2]
			#Load the time stepping.
			dt = self.delta_t[alias]
			#Extract and update the amplitude and phases.
			self.strain_phase.update({ alias: (xtract_cphase(hpdat, hxdat, dt = dt, plot='yes'))})
			self.strain_amplitude.update({ alias : xtract_camp(hpdat, hxdat, dt= dt)})
			self.strain_frequency.update({alias: np.diff(self.strain_phase[alias])/dt})
		return 1

	def ret_horizon_radii(self):
		''' Retrieve the radius of the common horizon at the time of formation. '''

		#Dictionary to hold the areal radii of the horizons.
		self.areal_radii = {}
		for alias in self.aliases:
			#Loop over simulations.

			#Load the BHdiagnostics file to load the radius. 
			ar_rad0 = np.genfromtxt(self.ROOTDIR + alias + '/'+ self.data_dir + 'BH_diagnostics.ah1.gp')[:, 27]
			ar_rad1 = np.genfromtxt(self.ROOTDIR + alias + '/'+ self.data_dir + 'BH_diagnostics.ah2.gp')[:, 27]
			ar_rad2 = 1.75
			try:
				ar_rad2 = np.genfromtxt(self.ROOTDIR + alias + '/'+ self.data_dir + 'BH_diagnostics.ah3.gp')[:, 27]
			except:
				message('No BHdiagnostics 3 file found for %s'%alias)


			#Load the data for this simulation in to a dictionary.
			self.areal_radii.update({alias : [ar_rad0, ar_rad1, ar_rad2]})
			
		return 1


#############################################################################################
# A Class for handling the waveform data
#############################################################################################

class Psi:
	''' A class for handling waveforms.'''

	def __init__(self,
				 timeaxis = None,
				 wavedata = None,
				 base_dir = None,
				 data_dir = None,
				 filename = None):

		self.base_dir = base_dir or ''
		self.data_dir = data_dir or ''
		self.filename = filename or ''
		self.timeaxis = timeaxis or []
		self.wavedata = wavedata or []

	def load_data(self):
		full_path = self.base_dir + self.data_dir + self.filename

		with h5py.File(full_path, "r") as f:
			# List all groups
			keys = list(f.keys())

			index=0
			token=-1


			while (token<0 and index<len(keys)):
				key  = keys[index]
				token = key.find('Psi')
				#message(key)
				index+=1

			if token<0:
				message('Waveform dataset not found')
			else:
				message(key)

			# Get the data
			data   = np.array(f[key])

			self.timeaxis  = data[0]
			self.wavedata  = data[1]

	@property
	def dt(self):
		return self.timeaxis[1]-self.timeaxis[0]

	#@base_dir.setter
	#def base_dir(self, base_dir):
	#	 self.__base_dir = base_dir

