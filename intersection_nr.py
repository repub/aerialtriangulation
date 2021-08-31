#Created by Jae Sung Kim (The Pennsylvania State University Libraries).
#Last modified: 08/31/21
# Example of running: python3 intersection_nr.py io_nr.txt fiducial_mark.txt bba_result_nr_1.json control.txt intersection_result_1.txt 1 2 1

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

	print('intersection')
	argv=sys.argv
	ar=int(argv[8])
	io_reader = csv.DictReader(open(argv[1]))
	fd_reader = csv.DictReader(open(argv[2]))
	cp_reader = csv.DictReader(open(argv[4]))
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
		fm=np.array([[float(fd_dict[img_data[i]]['x1'])],
				[float(fd_dict[img_data[i]]['y1'])],
				[float(fd_dict[img_data[i]]['x2'])],
				[float(fd_dict[img_data[i]]['y2'])],
				[float(fd_dict[img_data[i]]['x3'])],
				[float(fd_dict[img_data[i]]['y3'])],
				[float(fd_dict[img_data[i]]['x4'])],
				[float(fd_dict[img_data[i]]['y4'])],
				[float(fd_dict[img_data[i]]['x5'])],
				[float(fd_dict[img_data[i]]['y5'])],
				[float(fd_dict[img_data[i]]['x6'])],
				[float(fd_dict[img_data[i]]['y6'])],
				[float(fd_dict[img_data[i]]['x7'])],
				[float(fd_dict[img_data[i]]['y7'])],
				[float(fd_dict[img_data[i]]['x8'])],
				[float(fd_dict[img_data[i]]['y8'])]])
		
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
		
		xyc=np.array([[f1x],[f1y],[f2x],[f2y],[f3x],[f3y]])
		fpm=np.array([[x1,y1,1,0,0,0],[0,0,0,x1,y1,1],[x2,y2,1,0,0,0],[0,0,0,x2,y2,1],[x3,y3,1,0,0,0],[0,0,0,x3,y3,1]])
		delta=matmul(inv(fpm),xyc)		
		fc=np.array([[f1x],[f1y],[f2x],[f2y],[f3x],[f3y],[f4x],[f4y],[f5x],[f5y],[f6x],[f6y],[f7x],[f7y],[f8x],[f8y]])
		L0=fm
		L=L0
		WW=eye(16)
		last_phi = 10
		keep_going = 1
		iter = 0
		while keep_going == 1:
			AA=zeros([16,16])
			BB=zeros([16,6])
			FF=zeros([16,1])
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
	
	no_obs = len(check)

	image_list = []		
	check_list = []
	
	for i in range(no_obs):
		if check[i]['img_id'] not in image_list:
			image_list.append(check[i]['img_id'])
		if check[i]['point_id'] not in check_list:
			check_list.append(check[i]['point_id'])

	result = {}
	misc = {}
	for i in range(len(check_list)):
		result[check_list[i]]=[]
		misc[check_list[i]]=[]
		ck = list(filter(lambda x: x['point_id']==check_list[i], cp_data))
		for j in range(len(ck)-1):
			
			f_1 = f
			a1_1 = six_par[ck[j]['img_id']][0][0]
			a2_1 = six_par[ck[j]['img_id']][1][0]
			a3_1 = six_par[ck[j]['img_id']][2][0]
			a4_1 = six_par[ck[j]['img_id']][3][0]
			a5_1 = six_par[ck[j]['img_id']][4][0]
			a6_1 = six_par[ck[j]['img_id']][5][0]
			x_1 = float(ck[j]['sample'])
			y_1 = float(ck[j]['line'])
			x1 = a1_1*x_1+a2_1*y_1+a3_1-x_0
			y1 = a4_1*x_1+a5_1*y_1+a6_1-y_0
			
			f_2 = f
			a1_2 = six_par[ck[j+1]['img_id']][0][0]
			a2_2 = six_par[ck[j+1]['img_id']][1][0]
			a3_2 = six_par[ck[j+1]['img_id']][2][0]
			a4_2 = six_par[ck[j+1]['img_id']][3][0]
			a5_2 = six_par[ck[j+1]['img_id']][4][0]
			a6_2 = six_par[ck[j+1]['img_id']][5][0]
			x_2 = float(ck[j+1]['sample'])
			y_2 = float(ck[j+1]['line'])
			x2 = a1_2*x_2+a2_2*y_2+a3_2-x_0
			y2 = a4_2*x_2+a5_2*y_2+a6_2-y_0
			
			if ar == 1:
				
				rr1 = np.sqrt(x1**2+y1**2)
				HH1 = 0.0003048*float(eo_data['eo'][ck[j]['img_id']]['ZL'])
				hh1 = 0.0003048*float(ck[j]['Z'])
				K_1= (2410*HH1/(HH1**2-6*HH1+250)-2410*hh1/(hh1**2-6*hh1+250)*(hh1/HH1))/1000000
				dr1=-K_1*(rr1+rr1**3/f_1**2)
				x1=x1+x1/rr1*dr1
				y1=y1+y1/rr1*dr1
					
				rr2 = np.sqrt(x2**2+y2**2)
				HH2 = 0.0003048*float(eo_data['eo'][ck[j+1]['img_id']]['ZL'])
				hh2 = 0.0003048*float(ck[j+1]['Z'])
				K_2 = (2410*HH2/(HH2**2-6*HH2+250)-2410*hh2/(hh2**2-6*hh2+250)*(hh2/HH2))/1000000
				dr2=-K_2*(rr2+rr2**3/f_2**2)
				x2=x2+x2/rr2*dr2
				y2=y2+y2/rr2*dr2
			
			
			l1 = np.array([[x1],[y1],[-f_1]])
			l2 = np.array([[x2],[y2],[-f_2]])
			

			XL1=float(eo_data['eo'][ck[j]['img_id']]['XL'])
			YL1=float(eo_data['eo'][ck[j]['img_id']]['YL'])
			ZL1=float(eo_data['eo'][ck[j]['img_id']]['ZL'])
			o1=float(eo_data['eo'][ck[j]['img_id']]['omega'])
			p1=float(eo_data['eo'][ck[j]['img_id']]['phi'])
			k1=float(eo_data['eo'][ck[j]['img_id']]['kappa'])
			
			XL2=float(eo_data['eo'][ck[j+1]['img_id']]['XL'])
			YL2=float(eo_data['eo'][ck[j+1]['img_id']]['YL'])
			ZL2=float(eo_data['eo'][ck[j+1]['img_id']]['ZL'])
			o2=float(eo_data['eo'][ck[j+1]['img_id']]['omega'])
			p2=float(eo_data['eo'][ck[j+1]['img_id']]['phi'])
			k2=float(eo_data['eo'][ck[j+1]['img_id']]['kappa'])
			
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
				xx1=L[0,0]
				yy1=L[1,0]
				xx2=L[2,0]
				yy2=L[3,0]
				X=delta[0,0]
				Y=delta[1,0]
				Z=delta[2,0]
				
				U1 = M1[0,0]*(X-XL1)+M1[0,1]*(Y-YL1)+M1[0,2]*(Z-ZL1)
				V1 = M1[1,0]*(X-XL1)+M1[1,1]*(Y-YL1)+M1[1,2]*(Z-ZL1)
				W1 = M1[2,0]*(X-XL1)+M1[2,1]*(Y-YL1)+M1[2,2]*(Z-ZL1)
				
				U2 = M2[0,0]*(X-XL2)+M2[0,1]*(Y-YL2)+M2[0,2]*(Z-ZL2)
				V2 = M2[1,0]*(X-XL2)+M2[1,1]*(Y-YL2)+M2[1,2]*(Z-ZL2)
				W2 = M2[2,0]*(X-XL2)+M2[2,1]*(Y-YL2)+M2[2,2]*(Z-ZL2)
				AA = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
					
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
				
				
				BB = np.array([[B11, B12, B13], [B14, B15, B16],[B21, B22, B23], [B24, B25, B26]])
				
				FF = np.array([[xx1+f_1*U1/W1],[yy1+f_1*V1/W1],[xx2+f_2*U2/W2],[yy2+f_2*V2/W2]])
				
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
					result[check_list[i]].append(delta)
					
				
				L=L0+v
				delta=delta+ddel

				
				if iter>100:
					keep_going=0
					print("too many iteration")

				last_phi = phi[0,0]
				iter=iter+1

			dX = delta[0][0]-float(ck[0]['X'])
			dY = delta[1][0]-float(ck[0]['Y'])
			dZ = delta[2][0]-float(ck[0]['Z'])
			misc[check_list[i]].append(np.array([[dX],[dY],[dZ]]))
	sum_X2 = 0
	sum_Y2 = 0
	sum_Z2 = 0
	no = 0
	for elements in misc:
		for element in misc[elements]:

			sum_X2 = sum_X2 + element[0]**2
			sum_Y2 = sum_Y2 + element[1]**2
			sum_Z2 = sum_Z2 + element[2]**2	
			no = no + 1
	rmse_x = np.sqrt(sum_X2/no)
	rmse_y = np.sqrt(sum_Y2/no)
	rmse_z = np.sqrt(sum_Z2/no)
	
	
	print(rmse_x)
	print(rmse_y)
	print(rmse_z)
	
	with open(argv[5],'w') as output_file:
		csv_writer = csv.writer(output_file)
		csv_writer.writerow(["Check points"])
		csv_writer.writerow([" "])
		for key in result.keys():
			for element in result[key]:
				csv_writer.writerow(['Point:'+key])			
				csv_writer.writerow(['X: '+str(element[0][0])])
				csv_writer.writerow(['Y: '+str(element[1][0])])
				csv_writer.writerow(['Z: '+str(element[2][0])])
				csv_writer.writerow([" "])
		csv_writer.writerow(['RMSE_X: '+str(rmse_x)])
		csv_writer.writerow(['RMSE_Y: '+str(rmse_y)])
		csv_writer.writerow(['RMSE_Z: '+str(rmse_z)])
		
	result={}
	result['rmse_x']=rmse_x[0]
	result['rmse_y']=rmse_y[0]
	result['rmse_z']=rmse_z[0]
	
	output_json = json.dumps(result, indent = 4)

	json_fname = ".".join([os.path.splitext(argv[5])[0],"json"])
	with open(json_fname,'w') as output_file:
		output_file.write(output_json)

if __name__=="__main__":
	main()
