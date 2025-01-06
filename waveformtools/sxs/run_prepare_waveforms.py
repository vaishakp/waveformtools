""" 
MIT License

Copyright (c) 2023 Vaishak Prasad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

A script to use PrepareSXSWaveforms class.

"""

# import scri
# import numpy as np

from waveformtools.sxs.prepare_waveforms import PrepareSXSWaveform

# sims1 = ['ICTSEccParallel01', 'ICTSEccParallel02', 'ICTSEccParallel03']
sims1 = {
    "ICTSEccParallel01": [2, 3],
    "ICTSEccParallel02": [2, 3],
    "ICTSEccParallel03": [2, 3],
    "ICTSEccParallel04": [2, 3],
    "ICTSEccParallel05": [2, 3],
}


# sims1 = []
# sims2 = []
# sims3 = []

# sims2 = ['EccPrecDiff001', 'EccPrecDiff002', 'EccPrecDiff004']
sims2 = {
    "EccPrecDiff001": [2],
    "EccPrecDiff002": [2, 3],
}

sims3 = {
    "EccContPrecDiff001": [3, 4],
    "EccContPrecDiff003": [2],
    "EccContPrecDiff004": [2],
    "EccContPrecDiff005": [2, 3],
    "EccContPrecDiff006": [2, 3],
    "EccContPrecDiff007": [2, 3],
    "EccContPrecDiff008": [2, 3],
}

# sims3 = ['EccContPrecDiff001', 'EccContPrecDiff003',
#        'EccContPrecDiff004', 'EccContPrecDiff005', 'EccContPrecDiff006',
#        'EccContPrecDiff007' 'EccContPrecDiff008']


sims4 = {"eccprecrun4b": [3]}

prefix_dir = "/mnt/pfs/vaishak.p/sims/SpEC/gcc/"

sims1_dir = prefix_dir + "/bfi/ICTSEccParallel"

levs = [3, 4]

eccs = [0]

success_sims = []

for sim in sims1.keys():
    for lev in sims1[sim]:
        print("Sim", sim, "Lev", lev)

        try:
            wfp = PrepareSXSWaveform(sim_name=sim, sim_dir=sims1_dir, lev=lev)

            flag = wfp.prepare_waveform()

            if flag:
                success_sims.append([sim, lev])

        except Exception as excep:
            print(excep)
            print("Failed")


sims2_dir = prefix_dir + "/bfi/EccPrecDiff"

for sim in sims2.keys():
    for lev in sims2[sim]:
        print("Sim", sim, "Lev", lev)

        try:
            wfp = PrepareSXSWaveform(sim_name=sim, sim_dir=sims2_dir, lev=lev)

            flag = wfp.prepare_waveform()
            if flag:
                success_sims.append([sim, lev])

        except Exception as excep:
            print(excep)
            print("Failed")


sims3_dir = prefix_dir + "/bfi/EccContPrecDiff"

for sim in sims3.keys():
    for lev in sims3[sim]:
        print("Sim", sim, "Lev", lev)

        try:
            wfp = PrepareSXSWaveform(sim_name=sim, sim_dir=sims3_dir, lev=lev)
            flag = wfp.prepare_waveform()
            if flag:
                success_sims.append([sim, lev])

        except Exception as excep:
            print(excep)
            print("Failed")


sims4_dir = prefix_dir

for sim in sims4.keys():
    for lev in sims4[sim]:
        print("Sim", sim, "Lev", lev)

        try:
            wfp = PrepareSXSWaveform(sim_name=sim, sim_dir=sims4_dir, lev=lev)
            flag = wfp.prepare_waveform()

            if flag:
                success_sims.append([sim, lev])

        except Exception as excep:
            print(excep)
            print("Failed")

print("Successfully extrapolated: ")
print(success_sims)
