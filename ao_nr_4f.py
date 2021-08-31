#Created by Jae Sung Kim (Penn State University Libraries)
#Last modified: 08/31/21
#Example of running: python3 ao_nr_4f.py io_nr.txt fiducial_mark.txt bba_min_result.json control_1.txt ao_result_1.txt 1 2

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


import math, csv, json, os
import numpy as np
from math import cos, sin, tan, atan2
from numpy import zeros, eye, matmul
from numpy.linalg import inv
import sys

def main():

	print('absoulte orientation')
	
	argv=sys.argv
	io_reader = csv.DictReader(open(argv[1]))
	fd_reader = csv.DictReader(open(argv[2]))
	cp_reader = csv.DictReader(open(argv[4]))
	io_data = list(io_reader)[0]
	f = float(io_data['focal_mm'])
	x_0 = float(io_data['xp_mm'])
	y_0 = float(io_data['yp_mm'])
	f5x = float(io_data['f5x'])
	f5y = float(io_data['f5y'])
	f6x = float(io_data['f6x'])
	f6y = float(io_data['f6y'])
	f7x = float(io_data['f7x'])
	f7y = float(io_data['f7y'])
	f8x = float(io_data['f8x'])
	f8y = float(io_data['f8y'])
	
	fd_data = list(fd_reader)
	fd_dict={}
	for fd in fd_data:
		fd_key = fd['image_id']
		fd.pop('image_id')
		fd_dict[fd_key] = fd


	with open(argv[3],'r') as f_json:
		eo_data = json.load(f_json)
	

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
	
	six_par={}
	im_scale={}
	img_data=[*fd_dict]
	no_img=len(img_data)

	for i in range(no_img):
		
		fm=np.array([
				[float(fd_dict[img_data[i]]['x5'])],
				[float(fd_dict[img_data[i]]['y5'])],
				[float(fd_dict[img_data[i]]['x6'])],
				[float(fd_dict[img_data[i]]['y6'])],
				[float(fd_dict[img_data[i]]['x7'])],
				[float(fd_dict[img_data[i]]['y7'])],
				[float(fd_dict[img_data[i]]['x8'])],
				[float(fd_dict[img_data[i]]['y8'])]])
		
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
				six_par[img_data[i]]=delta
					
			L=L0+v
			delta=delta+ddel

			
			if iter>100:
				keep_going=0
				print("too many iteration")

			last_phi = phi[0,0]
			iter=iter+1
	
	no_ctrl = len(control)
	no_obs = len(check)

	image_list = []
	check_list = []
	control_list = []
	
	for i in range(no_obs):
		if check[i]['img_id'] not in image_list:
			image_list.append(check[i]['img_id'])
		if check[i]['point_id'] not in check_list:
			check_list.append(check[i]['point_id'])
			
	for i in range(no_ctrl):
		if control[i]['point_id'] not in control_list:
			control_list.append(control[i]['point_id'])

	result = {}

	for i in range(len(control_list)):
		result[control_list[i]]=[]

		ck = list(filter(lambda x: x['point_id']==control_list[i], cp_data))
		for j in range(len(ck)-1):
		
			x_1 = float(ck[j]['sample'])
			y_1 = float(ck[j]['line'])
			x_2 = float(ck[j+1]['sample'])
			y_2 = float(ck[j+1]['line'])
			
			f_1 = f
			a1_1 = six_par[ck[j]['img_id']][0][0]
			a2_1 = six_par[ck[j]['img_id']][1][0]
			a3_1 = six_par[ck[j]['img_id']][2][0]
			a4_1 = six_par[ck[j]['img_id']][3][0]
			a5_1 = six_par[ck[j]['img_id']][4][0]
			a6_1 = six_par[ck[j]['img_id']][5][0]
			x1 = a1_1*x_1+a2_1*y_1+a3_1-x_0
			y1 = a4_1*x_1+a5_1*y_1+a6_1-y_0
			
			f_2 = f
			a1_2 = six_par[ck[j+1]['img_id']][0][0]
			a2_2 = six_par[ck[j+1]['img_id']][1][0]
			a3_2 = six_par[ck[j+1]['img_id']][2][0]
			a4_2 = six_par[ck[j+1]['img_id']][3][0]
			a5_2 = six_par[ck[j+1]['img_id']][4][0]
			a6_2 = six_par[ck[j+1]['img_id']][5][0]
			x2 = a1_2*x_2+a2_2*y_2+a3_2-x_0
			y2 = a4_2*x_2+a5_2*y_2+a6_2-y_0
			
			l1 = np.array([[x1],[y1],[-f_1]])
			l2 = np.array([[x2],[y2],[-f_2]])
			
			XL1=eo_data['eo'][ck[j]['img_id']]['XL']
			YL1=eo_data['eo'][ck[j]['img_id']]['YL']
			ZL1=eo_data['eo'][ck[j]['img_id']]['ZL']
			o1=eo_data['eo'][ck[j]['img_id']]['omega']
			p1=eo_data['eo'][ck[j]['img_id']]['phi']
			k1=eo_data['eo'][ck[j]['img_id']]['kappa']
			
			XL2=eo_data['eo'][ck[j+1]['img_id']]['XL']
			YL2=eo_data['eo'][ck[j+1]['img_id']]['YL']
			ZL2=eo_data['eo'][ck[j+1]['img_id']]['ZL']
			o2=eo_data['eo'][ck[j+1]['img_id']]['omega']
			p2=eo_data['eo'][ck[j+1]['img_id']]['phi']
			k2=eo_data['eo'][ck[j+1]['img_id']]['kappa']

			M1 = np.array([[cos(p1)*cos(k1), cos(o1)*sin(k1)+sin(o1)*sin(p1)*cos(k1), sin(o1)*sin(k1)-cos(o1)*sin(p1)*cos(k1)],
							[-cos(p1)*sin(k1), cos(o1)*cos(k1)-sin(o1)*sin(p1)*sin(k1), sin(o1)*cos(k1)+cos(o1)*sin(p1)*sin(k1)],
							[sin(p1), -sin(o1)*cos(p1), cos(o1)*cos(p1)]])
							
			M2 = np.array([[cos(p2)*cos(k2), cos(o2)*sin(k2)+sin(o2)*sin(p2)*cos(k2), sin(o2)*sin(k2)-cos(o2)*sin(p2)*cos(k2)],
							[-cos(p2)*sin(k2), cos(o2)*cos(k2)-sin(o2)*sin(p2)*sin(k2), sin(o2)*cos(k2)+cos(o2)*sin(p2)*sin(k2)],
							[sin(p2), -sin(o2)*cos(p2), cos(o2)*cos(p2)]])
							
			L1 = matmul(M1.T,l1)
			L2 = matmul(M2.T,l2)
			C1 = L1[0,0]/L1[2,0]
			C2 = L1[1,0]/L1[2,0]
			C3 = L2[0,0]/L2[2,0]
			C4 = L2[1,0]/L2[2,0]

			B1=np.array([[1, 0, -C1],[0, 1, -C2],[1, 0, -C3],[0, 1, -C4]])
			F1=np.array([[XL1-C1*ZL1],[YL1-C2*ZL1],[XL2-C3*ZL2],[YL2-C4*ZL2]])
			N1=matmul(B1.T,B1)
			t1=matmul(B1.T,F1)
			XYZ=matmul(inv(N1),t1)
			delta = XYZ

			last_phi = 10
			keep_going = 1
			iter = 0
			L0 = np.array([[x1],[y1],[x2],[y2]])
			L=L0
			delta_0=delta
			WW=eye(4)
			while keep_going == 1:
			
				x1=L[0,0]
				y1=L[1,0]
				x2=L[2,0]
				y2=L[3,0]
				X=delta[0,0]
				Y=delta[1,0]
				Z=delta[2,0]
				
				U1 = M1[0,0]*(X-XL1)+M1[0,1]*(Y-YL1)+M1[0,2]*(Z-ZL1)
				V1 = M1[1,0]*(X-XL1)+M1[1,1]*(Y-YL1)+M1[1,2]*(Z-ZL1)
				W1 = M1[2,0]*(X-XL1)+M1[2,1]*(Y-YL1)+M1[2,2]*(Z-ZL1)
				
				U2 = M2[0,0]*(X-XL2)+M2[0,1]*(Y-YL2)+M2[0,2]*(Z-ZL2)
				V2 = M2[1,0]*(X-XL2)+M2[1,1]*(Y-YL2)+M2[1,2]*(Z-ZL2)
				W2 = M2[2,0]*(X-XL2)+M2[2,1]*(Y-YL2)+M2[2,2]*(Z-ZL2)
					
				B11 = -f_1*(-M1[0,0]+U1/W1*M1[2,0])/W1
				B12 = -f_1*(-M1[0,1]+U1/W1*M1[2,1])/W1
				B13 = -f_1*(-M1[0,2]+U1/W1*M1[2,2])/W1				
				B14 = -f_1*(-M1[1,0]+V1/W1*M1[2,0])/W1
				B15 = -f_1*(-M1[1,1]+V1/W1*M1[2,1])/W1
				B16 = -f_1*(-M1[1,2]+V1/W1*M1[2,2])/W1
				
				B21 = -f_1*(-M2[0,0]+U2/W2*M2[2,0])/W2
				B22 = -f_1*(-M2[0,1]+U2/W2*M2[2,1])/W2
				B23 = -f_1*(-M2[0,2]+U2/W2*M2[2,2])/W2
				B24 = -f_1*(-M2[1,0]+V2/W2*M2[2,0])/W2
				B25 = -f_1*(-M2[1,1]+V2/W2*M2[2,1])/W2
				B26 = -f_1*(-M2[1,2]+V2/W2*M2[2,2])/W2
				
				AA = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
				BB = np.array([[B11, B12, B13], [B14, B15, B16],[B21, B22, B23], [B24, B25, B26]])
				
				FF = np.array([[x1+f_1*U1/W1],[y1+f_1*V1/W1],[x2+f_2*U2/W2],[y2+f_2*V2/W2]])
				
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
					d_json={}
					d_json['X']=str(delta[0][0])
					d_json['Y']=str(delta[1][0])
					d_json['Z']=str(delta[2][0])
				
					result[control_list[i]]=d_json
					
				L=L0+v
				delta=delta+ddel

				if iter>100:
					keep_going=0
					print("too many iteration")

				last_phi = phi[0,0]
				iter=iter+1

	last_phi = 10
	keep_going = 1
	iter = 0
	ctrl=[]
	L0=[]
	converged = False
	
	for i in range(len(control_list)):
	
		ctrl.append([float(cp_dict[control_list[i]]['X'])])
		ctrl.append([float(cp_dict[control_list[i]]['Y'])])
		ctrl.append([float(cp_dict[control_list[i]]['Z'])])
	
	ctrl=np.array(ctrl)
	for i in range(len(control_list)):

		L0.append([float(result[control_list[i]]['X'])])
		L0.append([float(result[control_list[i]]['Y'])])
		L0.append([float(result[control_list[i]]['Z'])])
		
	L0 = np.array(L0)
	L=L0
	
	X1c=float(cp_dict[control_list[0]]['X'])
	Y1c=float(cp_dict[control_list[0]]['Y'])
	Z1c=float(cp_dict[control_list[0]]['Z'])
	X2c=float(cp_dict[control_list[-1]]['X'])
	Y2c=float(cp_dict[control_list[-1]]['Y'])
	Z2c=float(cp_dict[control_list[-1]]['Z'])
	d1=np.sqrt((X2c-X1c)**2+(Y2c-Y1c)**2+(Z2c-Z1c)**2)
	
	
	X1=float(result[control_list[0]]['X'])
	Y1=float(result[control_list[0]]['Y'])
	Z1=float(result[control_list[0]]['Z'])
	X2=float(result[control_list[-1]]['X'])
	Y2=float(result[control_list[-1]]['Y'])
	Z2=float(result[control_list[-1]]['Z'])
	d2=np.sqrt((X2-X1)**2+(Y2-Y1)**2+(Z2-Z1)**2)
			
	o=0
	p=0
	k=0
	S=d1/d2
	Tx=X1c-S*X1
	Ty=Y1c-S*Y1
	Tz=Z1c-S*Z1
	
	delta = np.array([[o],[p],[k],[S],[Tx],[Ty],[Tz]])
	WW=eye(3*len(control_list))
	while keep_going == 1:
		FF=zeros((len(control_list)*3,1))
		AA=zeros((len(control_list)*3,len(control_list)*3))
		BB=zeros((len(control_list)*3,7))
		o=delta[0,0]
		p=delta[1,0]
		k=delta[2,0]
		S=delta[3,0]
		Tx=delta[4,0]
		Ty=delta[5,0]
		Tz=delta[6,0]

		for i in range(len(control_list)):
			x=L[3*i,0]
			y=L[3*i+1,0]
			z=L[3*i+2,0]
			xc=ctrl[3*i,0]
			yc=ctrl[3*i+1,0]
			zc=ctrl[3*i+2,0]
			
			M = np.array([[cos(p)*cos(k), cos(o)*sin(k)+sin(o)*sin(p)*cos(k), sin(o)*sin(k)-cos(o)*sin(p)*cos(k)],
					[-cos(p)*sin(k), cos(o)*cos(k)-sin(o)*sin(p)*sin(k), sin(o)*cos(k)+cos(o)*sin(p)*sin(k)],
					[sin(p), -sin(o)*cos(p), cos(o)*cos(p)]])
					
			A=S*M
			B11=matmul(matmul(S*M,np.array([[0,0,0],[0,0,1],[0,-1,0]])),np.array([[x],[y],[z]]))
			B12=matmul(matmul(S*M,np.array([[0,sin(o),-cos(o)],[-sin(o),0,0],[cos(o),0,0]])),np.array([[x],[y],[z]]))
			B13=matmul(matmul(S*np.array([[0,1,0],[-1,0,0],[0,0,0]]),M),np.array([[x],[y],[z]]))
			B14=matmul(M,np.array([[x],[y],[z]]))
			B15=np.array([[1],[0],[0]])
			B16=np.array([[0],[1],[0]])
			B17=np.array([[0],[0],[1]])
			B=np.concatenate((B11,B12,B13,B14,B15,B16,B17), axis=1)
			F=matmul(S*M,[L[3*i],L[3*i+1],L[3*i+2]])+np.array([[Tx],[Ty],[Tz]])-np.array([[xc],[yc],[zc]])
			FF[3*i:3*i+3,:] = F
			AA[3*i:3*i+3,3*i:3*i+3] = A
			BB[3*i:3*i+3,:] = B 
		
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
			converged = True
			print("o, p, k, S, Tx, Ty, Tz")
			print(delta)
			par7={}
			par7['o']=o
			par7['p']=p
			par7['k']=k
			par7['S']=S
			par7['Tx']=Tx
			par7['Ty']=Ty
			par7['Tz']=Tz
			
			par7_deg={}
			par7_deg['omega']=o*(180/np.pi)
			par7_deg['phi']=p*(180/np.pi)
			par7_deg['kappa']=k*(180/np.pi)
		L=L0+v
		delta=delta+ddel

		
		if iter>1000:
			keep_going=0
			print("too many iteration")
			print(delta)

		last_phi = phi[0,0]
		iter=iter+1

	if converged == True:

		result_ck = {}

		for i in range(len(check_list)):
			result_ck[check_list[i]]=[]
			
			ck = list(filter(lambda x: x['point_id']==check_list[i], cp_data))
			for j in range(len(ck)-1):
				
				x_1 = float(ck[j]['sample'])
				y_1 = float(ck[j]['line'])
				x_2 = float(ck[j+1]['sample'])
				y_2 = float(ck[j+1]['line'])
				
				f_1 = f
				a1_1 = six_par[ck[j]['img_id']][0][0]
				a2_1 = six_par[ck[j]['img_id']][1][0]
				a3_1 = six_par[ck[j]['img_id']][2][0]
				a4_1 = six_par[ck[j]['img_id']][3][0]
				a5_1 = six_par[ck[j]['img_id']][4][0]
				a6_1 = six_par[ck[j]['img_id']][5][0]
				x1 = a1_1*x_1+a2_1*y_1+a3_1-x_0
				y1 = a4_1*x_1+a5_1*y_1+a6_1-y_0
				
				f_2 = f
				a1_2 = six_par[ck[j+1]['img_id']][0][0]
				a2_2 = six_par[ck[j+1]['img_id']][1][0]
				a3_2 = six_par[ck[j+1]['img_id']][2][0]
				a4_2 = six_par[ck[j+1]['img_id']][3][0]
				a5_2 = six_par[ck[j+1]['img_id']][4][0]
				a6_2 = six_par[ck[j+1]['img_id']][5][0]
				x2 = a1_2*x_2+a2_2*y_2+a3_2-x_0
				y2 = a4_2*x_2+a5_2*y_2+a6_2-y_0
			
				l1 = np.array([[x1],[y1],[-f_1]])
				l2 = np.array([[x2],[y2],[-f_2]])
				
				XL1=eo_data['eo'][ck[j]['img_id']]['XL']
				YL1=eo_data['eo'][ck[j]['img_id']]['YL']
				ZL1=eo_data['eo'][ck[j]['img_id']]['ZL']
				o1=eo_data['eo'][ck[j]['img_id']]['omega']
				p1=eo_data['eo'][ck[j]['img_id']]['phi']
				k1=eo_data['eo'][ck[j]['img_id']]['kappa']
				
				XL2=eo_data['eo'][ck[j+1]['img_id']]['XL']
				YL2=eo_data['eo'][ck[j+1]['img_id']]['YL']
				ZL2=eo_data['eo'][ck[j+1]['img_id']]['ZL']
				o2=eo_data['eo'][ck[j+1]['img_id']]['omega']
				p2=eo_data['eo'][ck[j+1]['img_id']]['phi']
				k2=eo_data['eo'][ck[j+1]['img_id']]['kappa']
				
				M1 = np.array([[cos(p1)*cos(k1), cos(o1)*sin(k1)+sin(o1)*sin(p1)*cos(k1), sin(o1)*sin(k1)-cos(o1)*sin(p1)*cos(k1)],
								[-cos(p1)*sin(k1), cos(o1)*cos(k1)-sin(o1)*sin(p1)*sin(k1), sin(o1)*cos(k1)+cos(o1)*sin(p1)*sin(k1)],
								[sin(p1), -sin(o1)*cos(p1), cos(o1)*cos(p1)]])
								
				M2 = np.array([[cos(p2)*cos(k2), cos(o2)*sin(k2)+sin(o2)*sin(p2)*cos(k2), sin(o2)*sin(k2)-cos(o2)*sin(p2)*cos(k2)],
								[-cos(p2)*sin(k2), cos(o2)*cos(k2)-sin(o2)*sin(p2)*sin(k2), sin(o2)*cos(k2)+cos(o2)*sin(p2)*sin(k2)],
								[sin(p2), -sin(o2)*cos(p2), cos(o2)*cos(p2)]])
								
				L1 = matmul(M1.T,l1)
				L2 = matmul(M2.T,l2)
				C1 = L1[0,0]/L1[2,0]
				C2 = L1[1,0]/L1[2,0]
				C3 = L2[0,0]/L2[2,0]
				C4 = L2[1,0]/L2[2,0]

				B1=np.array([[1, 0, -C1],[0, 1, -C2],[1, 0, -C3],[0, 1, -C4]])
				F1=np.array([[XL1-C1*ZL1],[YL1-C2*ZL1],[XL2-C3*ZL2],[YL2-C4*ZL2]])
				N1=matmul(B1.T,B1)
				t1=matmul(B1.T,F1)
				XYZ=matmul(inv(N1),t1)
				delta = XYZ

				last_phi = 10
				keep_going = 1
				iter = 0
				L0 = np.array([[x1],[y1],[x2],[y2]])
				L=L0
				delta_0=delta
				WW=eye(4)
				while keep_going == 1:
					x1=L[0,0]
					y1=L[1,0]
					x2=L[2,0]
					y2=L[3,0]
				
					X=delta[0,0]
					Y=delta[1,0]
					Z=delta[2,0]
					
					U1 = M1[0,0]*(X-XL1)+M1[0,1]*(Y-YL1)+M1[0,2]*(Z-ZL1)
					V1 = M1[1,0]*(X-XL1)+M1[1,1]*(Y-YL1)+M1[1,2]*(Z-ZL1)
					W1 = M1[2,0]*(X-XL1)+M1[2,1]*(Y-YL1)+M1[2,2]*(Z-ZL1)
					
					U2 = M2[0,0]*(X-XL2)+M2[0,1]*(Y-YL2)+M2[0,2]*(Z-ZL2)
					V2 = M2[1,0]*(X-XL2)+M2[1,1]*(Y-YL2)+M2[1,2]*(Z-ZL2)
					W2 = M2[2,0]*(X-XL2)+M2[2,1]*(Y-YL2)+M2[2,2]*(Z-ZL2)
						
					B11 = -f_1*(-M1[0,0]+U1/W1*M1[2,0])/W1
					B12 = -f_1*(-M1[0,1]+U1/W1*M1[2,1])/W1
					B13 = -f_1*(-M1[0,2]+U1/W1*M1[2,2])/W1				
					B14 = -f_1*(-M1[1,0]+V1/W1*M1[2,0])/W1
					B15 = -f_1*(-M1[1,1]+V1/W1*M1[2,1])/W1
					B16 = -f_1*(-M1[1,2]+V1/W1*M1[2,2])/W1
					
					B21 = -f_1*(-M2[0,0]+U2/W2*M2[2,0])/W2
					B22 = -f_1*(-M2[0,1]+U2/W2*M2[2,1])/W2
					B23 = -f_1*(-M2[0,2]+U2/W2*M2[2,2])/W2
					B24 = -f_1*(-M2[1,0]+V2/W2*M2[2,0])/W2
					B25 = -f_1*(-M2[1,1]+V2/W2*M2[2,1])/W2
					B26 = -f_1*(-M2[1,2]+V2/W2*M2[2,2])/W2
					
					AA = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
					BB = np.array([[B11, B12, B13], [B14, B15, B16],[B21, B22, B23], [B24, B25, B26]])
					
					FF = np.array([[x1+f_1*U1/W1],[y1+f_1*V1/W1],[x2+f_2*U2/W2],[y2+f_2*V2/W2]])
					
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
						d_json={}
						d_json['X']=str(delta[0][0])
						d_json['Y']=str(delta[1][0])
						d_json['Z']=str(delta[2][0])
						result_ck[check_list[i]].append(d_json)
						
					L=L0+v
					delta=delta+ddel

					if iter>100:
						keep_going=0
						print("too many iteration")

					last_phi = phi[0,0]
					iter=iter+1
		sum_X2 = 0
		sum_Y2 = 0
		sum_Z2 = 0
		no = 0
		
		for ck_id in result_ck:
			
			M = np.array([[cos(p)*cos(k), cos(o)*sin(k)+sin(o)*sin(p)*cos(k), sin(o)*sin(k)-cos(o)*sin(p)*cos(k)],
					[-cos(p)*sin(k), cos(o)*cos(k)-sin(o)*sin(p)*sin(k), sin(o)*cos(k)+cos(o)*sin(p)*sin(k)],
					[sin(p), -sin(o)*cos(p), cos(o)*cos(p)]])
					
			for i in range(len(result_ck[ck_id])):
				XYZ = matmul(S*M,np.array([[float(result_ck[ck_id][i]['X'])],[float(result_ck[ck_id][i]['Y'])],[float(result_ck[ck_id][i]['Z'])]]))+np.array([[Tx],[Ty],[Tz]])

				ck_item = list(filter(lambda x: x['point_id']==ck_id, check))
				ck_item[0]['X']
				ck_item[0]['Y']
				ck_item[0]['Z']
				dX=XYZ[0][0]-float(ck_item[0]['X'])
				dY=XYZ[1][0]-float(ck_item[0]['Y'])
				dZ=XYZ[2][0]-float(ck_item[0]['Z'])	
				sum_X2 = sum_X2 + dX**2
				sum_Y2 = sum_Y2 + dY**2
				sum_Z2 = sum_Z2 + dZ**2
				no = no + 1
			

		rmse_x = np.sqrt(sum_X2/no)
		rmse_y = np.sqrt(sum_Y2/no)
		rmse_z = np.sqrt(sum_Z2/no)
		
		
		result={}
		result['par7']=par7
		result['par7_deg']=par7_deg
		result['rmse_x']=rmse_x
		result['rmse_y']=rmse_y
		result['rmse_z']=rmse_z
		result['eo']=eo_data['eo']
		output_json = json.dumps(result, indent = 4)
		json_fname = ".".join([os.path.splitext(argv[5])[0],"json"])
		with open(json_fname,'w') as output_file:
			output_file.write(output_json)


if __name__=="__main__":
	main()
