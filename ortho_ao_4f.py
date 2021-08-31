# Developed by Jae Sung Kim (The Pennsylvania State University Libraries).
# Last modified: 08/31/21
# Example of running: python3 ortho_ao_4f.py io.txt fiducial_mark.txt bba_min_result_1.json ao_result_1.json 31 /mnt/data/input.tif /mnt/data/output.tif

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

import math, csv, json, os, json, subprocess, gdal
import numpy as np
from math import cos, sin
from numpy import zeros, eye, matmul
from numpy.linalg import inv
import sys
import cv2 as cv

def main():

	print('ortho rectification')
	argv=sys.argv
	io_reader = csv.DictReader(open(argv[1]))
	fd_reader = csv.DictReader(open(argv[2]))

	io_data = list(io_reader)[0]
	focal_mm = float(io_data['focal_mm'])
	xp_mm = float(io_data['xp_mm'])
	yp_mm = float(io_data['yp_mm'])
	f5x = float(io_data['f5x'])
	f5y = float(io_data['f5y'])
	f6x = float(io_data['f6x'])
	f6y = float(io_data['f6y'])
	f7x = float(io_data['f7x'])
	f7y = float(io_data['f7y'])
	f8x = float(io_data['f8x'])
	f8y = float(io_data['f8y'])


	with open (argv[3]) as f:
		data = json.load(f)
	
	with open (argv[4]) as f:
		par7_data = json.load(f)
	
	image_id = argv[5]
	
	fd_data = list(fd_reader)
	fd_dict={}
	for fd in fd_data:
		fd_key = fd['image_id']
		fd.pop('image_id')
		fd_dict[fd_key] = fd
		
	
		
	fm=np.array([
			[float(fd_dict[image_id]['x5'])],
			[float(fd_dict[image_id]['y5'])],
			[float(fd_dict[image_id]['x6'])],
			[float(fd_dict[image_id]['y6'])],
			[float(fd_dict[image_id]['x7'])],
			[float(fd_dict[image_id]['y7'])],
			[float(fd_dict[image_id]['x8'])],
			[float(fd_dict[image_id]['y8'])]])
	
	x5=fm[0,0]
	y5=fm[1,0]
	x6=fm[2,0]
	y6=fm[3,0]
	x7=fm[4,0]
	y7=fm[5,0]
	x8=fm[6,0]
	y8=fm[7,0]
	
	xyc=np.array([[f5x],[f5y],[f6x],[f6y],[f7x],[f7y]])
	fpm=np.array([[x5,y5,1,0,0,0],[0,0,0,x5,y5,1],[x6,y6,1,0,0,0],[0,0,0,x6,y6,1],[x7,y7,1,0,0,0],[0,0,0,x7,y7,1]])
	delta=matmul(inv(fpm),xyc)
		
	fc=np.array([[f5x],[f5y],[f6x],[f6y],[f7x],[f7y],[f8x],[f8y]])
	L0=fm
	L=L0
	WW=eye(8)
	last_phi = 10
	keep_going = 1
	iter = 0
	while keep_going == 1:
		AA=zeros([8,8])
		BB=zeros([8,6])
		FF=zeros([8,1])
		a1=delta[0,0]
		a2=delta[1,0]
		a3=delta[2,0]
		a4=delta[3,0]
		a5=delta[4,0]
		a6=delta[5,0]
		for j in range(4):
			rx=L[2*j,0]
			ry=L[2*j+1,0]
			x=fc[2*j,0]
			y=fc[2*j+1,0]
			F_s = np.array([[x-a1*rx-a2*ry-a3],[y-a4*rx-a5*ry-a6]])
			A_s = np.array([[-a1, -a2],[-a4, -a5]])
			B_s = np.array([[-rx, -ry, -1, 0, 0, 0],[0, 0, 0, -rx, -ry, -1]])
			
			FF[2*j:2*j+2,:] = F_s
			AA[2*j:2*j+2,2*j:2*j+2] = A_s
			BB[2*j:2*j+2,:] = B_s
			
		Q=inv(WW)
		ff=-FF-matmul(AA,(L0-L))
		Qe=matmul(matmul(AA,Q),AA.T)
		We=inv(Qe)

		N=matmul(matmul(BB.T,We),BB)
		t=matmul(matmul(BB.T,We),ff)
		ddel=matmul(inv(N),t)
		v=matmul(matmul(matmul(Q,AA.T),We),(ff-matmul(BB,ddel)))
		phi=matmul(matmul(v.T,WW),v)

		obj=abs((last_phi-phi[0,0])/last_phi)
		print("iter nubmer: "+str(iter))
		print("objective function is : "+str(obj))
		#Convergence check
		if obj<0.0001:
			keep_going=0
			print("Converged")
			six_par=delta
				
		L=L0+v
		delta=delta+ddel

		if iter>100:
			keep_going=0
			print("too many iteration")

		last_phi = phi[0,0]
		iter=iter+1
	
	focal = focal_mm
	x_0 = xp_mm
	y_0 = yp_mm

		
	s_list=[]
	for i in range(len(data['scale'][image_id])):
		s_list.append(data['scale'][image_id][i])
	
	k_min = np.min(s_list) #
	o = data['eo'][image_id]['omega']
	p = data['eo'][image_id]['phi']
	k = data['eo'][image_id]['kappa']
	XL = data['eo'][image_id]['XL']
	YL = data['eo'][image_id]['YL']
	ZL = data['eo'][image_id]['ZL']

	M = np.array([[cos(p)*cos(k), cos(o)*sin(k)+sin(o)*sin(p)*cos(k), sin(o)*sin(k)-cos(o)*sin(p)*cos(k)],
			[-cos(p)*sin(k), cos(o)*cos(k)-sin(o)*sin(p)*sin(k), sin(o)*cos(k)+cos(o)*sin(p)*sin(k)],
			[sin(p), -sin(o)*cos(p), cos(o)*cos(p)]])
	

	im_1 = cv.imread(argv[6])
	no_row, no_col, no_band = im_1.shape
	del im_1
	
	x_max=a1*0+a2*0+a3
	y_max=a4*0+a5*0+a6
	x_min=a1*no_col+a2*no_row+a3
	y_min=a4*no_col+a5*no_row+a6
	
	ul=np.array([[x_min],[y_max],[-focal]])
	ur=np.array([[x_max],[y_max],[-focal]])
	ll=np.array([[x_min],[y_min],[-focal]])	
	lr=np.array([[x_max],[y_min],[-focal]])
	
	UL=1/k_min*(matmul(inv(M),ul))+np.array([[XL],[YL],[ZL]])
	UR=1/k_min*(matmul(inv(M),ur))+np.array([[XL],[YL],[ZL]])
	LL=1/k_min*(matmul(inv(M),ll))+np.array([[XL],[YL],[ZL]])
	LR=1/k_min*(matmul(inv(M),lr))+np.array([[XL],[YL],[ZL]])
	
	omega = float(par7_data['par7']['o'])
	phi = float(par7_data['par7']['p'])
	kappa = float(par7_data['par7']['k'])
	S = float(par7_data['par7']['S'])	
	Tx = float(par7_data['par7']['Tx'])	
	Ty = float(par7_data['par7']['Ty'])	
	Tz = float(par7_data['par7']['Tz'])	
	
	M2 = np.array([[cos(phi)*cos(kappa), cos(omega)*sin(kappa)+sin(omega)*sin(phi)*cos(kappa), sin(omega)*sin(kappa)-cos(omega)*sin(phi)*cos(kappa)],
				[-cos(phi)*sin(kappa), cos(omega)*cos(kappa)-sin(omega)*sin(phi)*sin(kappa), sin(omega)*cos(kappa)+cos(omega)*sin(phi)*sin(kappa)],
				[sin(phi), -sin(omega)*cos(phi), cos(omega)*cos(phi)]])
	
	Mi = inv(M2)
	
	ULM = matmul(S*M2,np.array([[UL[0,0]],[UL[1,0]],[UL[2,0]]]))+np.array([[Tx],[Ty],[Tz]])
	URM = matmul(S*M2,np.array([[UR[0,0]],[UR[1,0]],[UR[2,0]]]))+np.array([[Tx],[Ty],[Tz]])
	LLM = matmul(S*M2,np.array([[LL[0,0]],[LL[1,0]],[LL[2,0]]]))+np.array([[Tx],[Ty],[Tz]])
	LRM = matmul(S*M2,np.array([[LR[0,0]],[LR[1,0]],[LR[2,0]]]))+np.array([[Tx],[Ty],[Tz]])			
	
	Xul=ULM[0][0]
	Yul=ULM[1][0]
	
	Xur=URM[0][0]
	Yur=URM[1][0]
	
	Xll=LLM[0][0]
	Yll=LLM[1][0]
	
	Xlr=LRM[0][0]
	Ylr=LRM[1][0]
	
	X_list=[Xul,Xur,Xll,Xlr]
	Y_list=[Yul,Yur,Yll,Ylr]
	X_max = max(X_list)+float(argv[10])
	X_min = min(X_list)-float(argv[11])
	Y_max = max(Y_list)+float(argv[12])
	Y_min = min(Y_list)-float(argv[13])
	
	args = ['gdalwarp', '-t_srs', argv[14], '-te', str(X_min), str(Y_min), str(X_max), str(Y_max), argv[8], argv[9], '-multi', '-overwrite']
	
	p=subprocess.Popen(args)
	p.communicate()
	
	
	dem = gdal.Open(argv[9])
	a0,a1,a2,b0,b1,b2 = dem.GetGeoTransform()
	no_row_2 = dem.RasterYSize
	no_col_2 = dem.RasterXSize
	no_row_2_1=int(no_row_2/3)
	no_row_2_2=int(2*no_row_2/3)
	border=[0,no_row_2_1,no_row_2_2,no_row_2]

	# The loop was created to avoid memory problem for big image.
	for i in range(len(border)-1):
		
		if i>0:
			dem = gdal.Open(argv[9])
		dem_band = dem.GetRasterBand(1)
		dem_array = dem_band.ReadAsArray()	
		Zc = dem_array[border[i]:border[i+1],0:no_col_2]
		del dem_array
		del dem_band
		del dem

		[l1,s1]=np.mgrid[border[i]:border[i+1],0:no_col_2]
		Xc = a0+a1*s1+a2*l1
		Yc = b0+b1*s1+b2*l1
		del s1
		del l1

		Xg = 1/S*(Mi[0,0]*(Xc-Tx)+Mi[0,1]*(Yc-Ty)+Mi[0,2]*(Zc-Tz))
		Yg = 1/S*(Mi[1,0]*(Xc-Tx)+Mi[1,1]*(Yc-Ty)+Mi[1,2]*(Zc-Tz))
		Zg = 1/S*(Mi[2,0]*(Xc-Tx)+Mi[2,1]*(Yc-Ty)+Mi[2,2]*(Zc-Tz))
		del Xc
		del Yc
		del Zc

		U = M[0,0]*(Xg-XL)+M[0,1]*(Yg-YL)+M[0,2]*(Zg-ZL)
		V = M[1,0]*(Xg-XL)+M[1,1]*(Yg-YL)+M[1,2]*(Zg-ZL)
		W = M[2,0]*(Xg-XL)+M[2,1]*(Yg-YL)+M[2,2]*(Zg-ZL)
		del Xg
		del Yg
		del Zg
		xi_1=-focal*U/W+x_0
		yi_1=-focal*V/W+y_0
		
		aa1=six_par[0][0]
		aa2=six_par[1][0]
		aa3=six_par[2][0]
		aa4=six_par[3][0]
		aa5=six_par[4][0]
		aa6=six_par[5][0]
		
		x_px=np.int_(np.round((-aa5*xi_1+aa2*yi_1+aa3*aa5-aa2*aa6)/(-aa1*aa5+aa2*aa4),0))
		y_px=np.int_(np.round((-aa4*xi_1+aa1*yi_1+aa3*aa4-aa1*aa6)/(-aa2*aa4+aa1*aa5),0))
		del xi_1
		del yi_1
		del U
		del V
		del W
		
		if i==0:	
			xi=x_px
			del x_px
			yi=y_px
			del y_px
		elif i>0:	
			xi=np.vstack((xi,x_px))
			del x_px
			yi=np.vstack((yi,y_px))
			del y_px
	yi[yi>=no_row] = no_row-1
	yi[yi<0] = 0
	xi[xi>=no_col] = no_col-1
	xi[xi<0] = 0
	
	im_1 = cv.imread(argv[6])
	im_out=im_1[yi,xi]
	del im_1
	dem = gdal.Open(argv[9])
	driver = gdal.GetDriverByName("GTiff")
	ortho = driver.Create(argv[7],no_col_2,no_row_2, 3, gdal.GDT_Byte)
	ortho.SetGeoTransform(dem.GetGeoTransform())
	ortho.SetProjection(dem.GetProjection())
	ortho.GetRasterBand(1).WriteArray(im_out[:,:,2])
	ortho.GetRasterBand(2).WriteArray(im_out[:,:,1])
	ortho.GetRasterBand(3).WriteArray(im_out[:,:,0])	
	ortho.FlushCache()
	del dem
	del ortho

if __name__=="__main__":
	main()
