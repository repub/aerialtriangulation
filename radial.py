#Created by Jae Sung Kim (Penn State University Libraries)
#Last modified: 09/01/21
#Example of running: python3 radial.py
#This is for estimating coefficients of radial lens distortion from old USGS camera calibration report without coefficients of radial lens distortion
#The input parameters should be found from USGS camera calibration report and hard coded in this script.

"""
MIT License

Copyright (c) 2021 Penn State University Libraries

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
"""

import numpy as np
from numpy import pi, zeros, eye, matmul
from numpy.linalg import inv
from math import tan, cos, sin
f = 208.043
x0=-0.003
y0=-0.004

K0=0
K1=0
K2=0
K3=0

dist = [[-5, -7, -7, -8],
	[-9, -6, -9, -6],
	[-4, -4, -5, -2],
	[7, 9, 8, 8]]
	
angle=[7.5,15,22.5,30]

L=zeros((4*len(dist),1))
radius=zeros((4*len(dist),1))


for i in range(len(dist)):
	for j in range(len(dist[i])):
		L[4*i+j][0]=dist[i][j] 	
		radius[4*i+j][0]=1000*f*tan(angle[i]/180*pi)

no_pt = len(L)
no_l = len(L)
WW=eye(no_l)
f=zeros((no_l,1))
B=zeros((no_l,4))

for i in range(no_pt):
	dr=L[i][0]
	r=radius[i][0]
	FK0 = -r
	FK1 = -r**3
	FK2 = -r**5
	FK3 = -r**7
	f[i,:] = np.array([-dr])
	B[i,:] = np.array([FK0,FK1,FK2,FK3])
	

N=matmul(matmul(B.T,WW),B)
t=matmul(matmul(B.T,WW),f)
delta=matmul(inv(N),t)
v=(f-matmul(B,delta))

print(delta)
	

	
	
	
	
	

