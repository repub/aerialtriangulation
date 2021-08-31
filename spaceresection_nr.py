#Created by Jae Sung Kim (The Pennsylvania State University Libraries).
#Last modified: 08/31/21
#This code carries focal length, principal point offset as interior orientation paramteters (no radial lens or other distortion)
#python3 spaceresection_nr.py io_nr.txt control.txt EO.txt fiducial_mark.txt EO_init_1.txt 1 2

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

import math, csv
import numpy as np
from math import cos, sin
from numpy import zeros, eye, matmul
from numpy.linalg import inv
import sys

def main():

	argv=sys.argv
	io_reader = csv.DictReader(open(argv[1]))
	io_data = list(io_reader)[0]
	f = float(io_data['focal_mm'])
	x_0 = float(io_data['xp_mm'])
	y_0 = float(io_data['yp_mm'])
	f1x = float(io_data['f1x'])
	f1y = float(io_data['f1y'])
	f2x = float(io_data['f2x'])
	f2y = float(io_data['f2y'])
	f3x = float(io_data['f3x'])
	f3y = float(io_data['f3y'])
	f4x = float(io_data['f4x'])
	f4y = float(io_data['f4y'])
	f5x = float(io_data['f5x'])
	f5y = float(io_data['f5y'])
	f6x = float(io_data['f6x'])
	f6y = float(io_data['f6y'])
	f7x = float(io_data['f7x'])
	f7y = float(io_data['f7y'])
	f8x = float(io_data['f8x'])
	f8y = float(io_data['f8y'])
	
	D1 = float(io_data['D1'])
	D2 = float(io_data['D2'])
	D3 = float(io_data['D3'])
	D4 = float(io_data['D4'])
	D5 = float(io_data['D5'])
	D6 = float(io_data['D6'])
	D7 = float(io_data['D7'])
	D8 = float(io_data['D8'])
	
	cp_reader = csv.DictReader(open(argv[2]))
	cp_data = list(cp_reader)
	cp_dict = {}
	cp_list=[]
	for cp in cp_data:
		cp_key = cp['point_id']
		if cp_key not in cp_list:
			cp_list.append(cp_key)
			cp_dict[cp_key] = {'X':cp['X'],'Y':cp['Y'],'Z':cp['Z']}

	control = list(filter(lambda x: x['groupid']==argv[6], cp_data))
	check = list(filter(lambda x: x['groupid']==argv[7], cp_data))
	
	pt_data = control
	key_index=[]
	ctrl_json={}

	for pt in pt_data:
		
		if pt['img_id'] not in key_index:
			key_index.append(pt['img_id'])
			ctrl_json[pt['img_id']]=[]
			ctrl_json[pt['img_id']].append(pt)
		else:
			ctrl_json[pt['img_id']].append(pt)

	fd_reader = csv.DictReader(open(argv[4]))
	fd_data = list(fd_reader)
	fd_dict={}
	for fd in fd_data:
		fd_key = fd['image_id']
		fd.pop('image_id')
		fd_dict[fd_key] = fd
	
	reader = csv.DictReader(open(argv[3]))
	img_data = list(reader) 
	no_img=len(img_data)
	six_par={}
	pm_ratio={}
	for i in range(no_img):
		
		fm=np.array([[float(fd_dict[img_data[i]['image_id']]['x1'])],
				[float(fd_dict[img_data[i]['image_id']]['y1'])],
				[float(fd_dict[img_data[i]['image_id']]['x2'])],
				[float(fd_dict[img_data[i]['image_id']]['y2'])],
				[float(fd_dict[img_data[i]['image_id']]['x3'])],
				[float(fd_dict[img_data[i]['image_id']]['y3'])],
				[float(fd_dict[img_data[i]['image_id']]['x4'])],
				[float(fd_dict[img_data[i]['image_id']]['y4'])],
				[float(fd_dict[img_data[i]['image_id']]['x5'])],
				[float(fd_dict[img_data[i]['image_id']]['y5'])],
				[float(fd_dict[img_data[i]['image_id']]['x6'])],
				[float(fd_dict[img_data[i]['image_id']]['y6'])],
				[float(fd_dict[img_data[i]['image_id']]['x7'])],
				[float(fd_dict[img_data[i]['image_id']]['y7'])],
				[float(fd_dict[img_data[i]['image_id']]['x8'])],
				[float(fd_dict[img_data[i]['image_id']]['y8'])]])
		
		x1=fm[0,0]
		y1=fm[1,0]
		x2=fm[2,0]
		y2=fm[3,0]
		x3=fm[4,0]
		y3=fm[5,0]
		x4=fm[6,0]
		y4=fm[7,0]
		x5=fm[8,0]
		y5=fm[9,0]
		x6=fm[10,0]
		y6=fm[11,0]
		x7=fm[12,0]
		y7=fm[13,0]
		x8=fm[14,0]
		y8=fm[15,0]
		
		d1=np.sqrt((x2-x1)**2+(y2-y1)**2)
		s1=D1/d1
		d2=np.sqrt((x4-x3)**2+(y4-y3)**2)
		s2=D2/d2
		d3=np.sqrt((x6-x5)**2+(y6-y5)**2)
		s3=D3/d3
		d4=np.sqrt((x8-x7)**2+(y8-y7)**2)
		s4=D4/d4
		d5=np.sqrt((x3-x1)**2+(y3-y1)**2)
		s5=D5/d5
		d6=np.sqrt((x3-x2)**2+(y3-y2)**2)
		s6=D6/d6
		d7=np.sqrt((x4-x1)**2+(y4-y1)**2)
		s7=D7/d7
		d8=np.sqrt((x4-x2)**2+(y4-y2)**2)
		s8=D8/d8		
		
		r_mean=np.mean([s1,s2,s3,s4,s5,s6,s7,s8])
		print(r_mean)
			
		xyc=np.array([[f1x],[f1y],[f2x],[f2y],[f3x],[f3y]])
		fpm=np.array([[x1,y1,1,0,0,0],[0,0,0,x1,y1,1],[x2,y2,1,0,0,0],[0,0,0,x2,y2,1],[x3,y3,1,0,0,0],[0,0,0,x3,y3,1]])
		delta=matmul(inv(fpm),xyc)	
		fc=np.array([[f1x],[f1y],[f2x],[f2y],[f3x],[f3y],[f4x],[f4y],[f5x],[f5y],[f6x],[f6y],[f7x],[f7y],[f8x],[f8y]])
		L0=fm
		test=delta
		L=L0
		WW=eye(16)
		last_phi = 10
		keep_going = 1
		iter = 0
		while keep_going == 1:
			AA=zeros((16,16))
			BB=zeros((16,6))
			FF=zeros((16,1))
			a1=delta[0,0]
			a2=delta[1,0]
			a3=delta[2,0]
			a4=delta[3,0]
			a5=delta[4,0]
			a6=delta[5,0]
			for j in range(8):
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
				six_par[img_data[i]['image_id']]=delta
				pm_ratio[img_data[i]['image_id']]=r_mean
	
			L=L0+v
			delta=delta+ddel

			
			if iter>100:
				keep_going=0
				print("too many iteration")

			last_phi = phi[0,0]
			iter=iter+1

	EO=zeros((no_img, 6))
	EO_out = []
	for i in range(no_img):
		
		EO[i,0]=float(img_data[i]['XL'])
		EO[i,1]=float(img_data[i]['YL'])
		EO[i,2]=float(img_data[i]['ZL'])
		EO[i,3]=float(img_data[i]['omega'])/180*np.pi
		EO[i,4]=float(img_data[i]['phi'])/180*np.pi
		EO[i,5]=float(img_data[i]['kappa'])/180*np.pi
		img_id=img_data[i]['image_id']
		no_ctrl = len(ctrl_json[img_id])
		L0=zeros((no_ctrl*2,1)) #4points for x and y coordinates for each image
		
		for j in range(no_ctrl):
		
			a1 = six_par[img_data[i]['image_id']][0]
			a2 = six_par[img_data[i]['image_id']][1]
			a3 = six_par[img_data[i]['image_id']][2]
			a4 = six_par[img_data[i]['image_id']][3]
			a5 = six_par[img_data[i]['image_id']][4]
			a6 = six_par[img_data[i]['image_id']][5]
			xx1 = float(ctrl_json[img_id][j]['sample'])
			yy1 = float(ctrl_json[img_id][j]['line'])
			xx = a1*xx1+a2*yy1+a3-x_0
			yy = a4*xx1+a5*yy1+a6-y_0
			L0[2*j,0]= xx
			L0[2*j+1,0]= yy
		
		L=L0
		delta=EO[i,:].T
		delta=delta[:,np.newaxis]
		
		WW=eye(2*no_ctrl)
		
		last_phi = 10
		keep_going = 1
		iter = 0

		while keep_going == 1:
			FF=zeros((no_ctrl*2,1))
			AA=zeros((no_ctrl*2,no_ctrl*2))
			BB=zeros((no_ctrl*2,6))
			for j in range(no_ctrl):
				
				x = L[2*j,0] 
				y = L[2*j+1,0]
				XL = delta[0,0]
				YL = delta[1,0]
				ZL = delta[2,0]
				o = delta[3,0]
				p = delta[4,0]
				k = delta[5,0]
				X=float(ctrl_json[img_id][j]['X'])
				Y=float(ctrl_json[img_id][j]['Y'])
				Z=float(ctrl_json[img_id][j]['Z'])
				M = np.array([[cos(p)*cos(k), cos(o)*sin(k)+sin(o)*sin(p)*cos(k), sin(o)*sin(k)-cos(o)*sin(p)*cos(k)],
							[-cos(p)*sin(k), cos(o)*cos(k)-sin(o)*sin(p)*sin(k), sin(o)*cos(k)+cos(o)*sin(p)*sin(k)],
							[sin(p), -sin(o)*cos(p), cos(o)*cos(p)]])
				
				U = M[0,0]*(X-XL)+M[0,1]*(Y-YL)+M[0,2]*(Z-ZL)
				V = M[1,0]*(X-XL)+M[1,1]*(Y-YL)+M[1,2]*(Z-ZL)
				W = M[2,0]*(X-XL)+M[2,1]*(Y-YL)+M[2,2]*(Z-ZL)
				
				F_s = np.array([[x+f*U/W],[y+f*V/W]])
				
				Uo = M[0,1]*(Z-ZL)-M[0,2]*(Y-YL)	
				Up = -W*cos(k)
				Uk = V
				
				Vo = M[1,1]*(Z-ZL)-M[1,2]*(Y-YL)
				Vp = W*sin(k)
				Vk = -U
				
				Wo = M[2,1]*(Z-ZL)-M[2,2]*(Y-YL)
				Wp = U*cos(k)-V*sin(k)
				Wk = 0
				
				B11 = f*(-M[0,0]+U/W*M[2,0])/W
				B12 = f*(-M[0,1]+U/W*M[2,1])/W
				B13 = f*(-M[0,2]+U/W*M[2,2])/W
				B14 = f*(Uo-U/W*Wo)/W
				B15 = f*(Up-U/W*Wp)/W
				B16 = f*(Uk-U/W*Wk)/W
				
				B21 = f*(-M[1,0]+V/W*M[2,0])/W
				B22 = f*(-M[1,1]+V/W*M[2,1])/W
				B23 = f*(-M[1,2]+V/W*M[2,2])/W
				B24 = f*(Vo-V/W*Wo)/W
				B25 = f*(Vp-V/W*Wp)/W
				B26 = f*(Vk-V/W*Wk)/W				
				
				A_s = np.array([[1,0],[0,1]])
				
				FF[2*j:2*j+2,:] = F_s
				AA[2*j:2*j+2,2*j:2*j+2] = A_s
				BB[2*j:2*j+2,:] = np.array([[B11, B12, B13, B14, B15, B16],[B21, B22, B23, B24, B25, B26]])

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
				EO_out.append([img_id, delta[0,0],delta[1,0],delta[2,0],delta[3,0],delta[4,0],delta[5,0]])
				rmse_vx=np.sqrt(np.sum(v[0::2]**2)/len(v[0::2]))
				rmse_vy=np.sqrt(np.sum(v[1::2]**2)/len(v[1::2]))
				print(rmse_vx)
				print(rmse_vy)	
				
			L=L0+v
			delta=delta+ddel

			if iter>100:
				keep_going=0
				print("too many iteration")

			last_phi = phi[0,0]
			iter=iter+1
	
	with open(argv[5],'w') as output_file:
		csv_writer = csv.writer(output_file)
		csv_writer.writerow(['image_id', 'XL', 'YL', 'ZL', 'omega', 'phi', 'kappa'])
		
		for EO_data in EO_out:
			csv_writer.writerow(EO_data)

if __name__=="__main__":
	main()
