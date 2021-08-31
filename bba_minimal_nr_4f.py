#Created by Jae Sung Kim (Penn State University Libraries)
#Last modified: 08/31/21
#Example of running: sudo python3 bba_minimal_nr_4f.py io_nr.txt fiducial_mark.txt EO.txt photo_list.txt bba_min_result_nr_1.txt 0.0125 /mnt/data/test
#The unit of ground space coordinate is assumed to be feet.

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
import cv2 as cv

def main():

	print('bundle block adjusetment with minimal constraints')
	argv=sys.argv

	io_reader = csv.DictReader(open(argv[1]))
	fd_reader = csv.DictReader(open(argv[2]))
	eo_reader = csv.DictReader(open(argv[3]))
	res_thre=float(argv[6])
	out_dir = argv[7]
	out_log="/".join([out_dir,"output.log"])
	with open(out_log,'w') as output_file:
		csv_writer = csv.writer(output_file)
		csv_writer.writerow(["Recursive BBA w/MC log"])
			
	image_list = csv.DictReader(open(argv[4]))
	image_data = list(image_list)
	list_len = len(image_data)
	image_list = []
	pair_list = {}
	trip_list = {}
	conj_list = {}
	sr_list = {}
	for i in range(list_len):
		image_list.append(image_data[i]['id'])
		if i < list_len-1:
			pair_list[image_data[i]['id']] = image_data[i+1]['id']
			conj_list[image_data[i]['id']] = ".".join(["_".join([os.path.splitext(os.path.basename(image_data[i]['path']))[0],os.path.splitext(os.path.basename(image_data[i+1]['path']))[0]]),"json"])
		if i < list_len-2:
			trip_list[image_data[i]['id']] = [image_data[i+1]['id'],image_data[i+2]['id']]
			sr_list[image_data[i]['id']] = ".".join([os.path.splitext(os.path.basename(image_data[i+2]['path']))[0],"json"])

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

	eo_data = list(eo_reader)
	eo_dict = {}
	for eo in eo_data:
		eo_key = eo['image_id']
		eo.pop('image_id')
		eo_dict[eo_key] = eo
	sl = {}
	six_par={}
	im_scale={}
	img_data=[*eo_dict]
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
			#Convergence check
			if obj<0.0001:
				keep_going=0
				print("Converged")
				six_par[img_data[i]]=delta
				
			L=L0+v
			delta=delta+ddel

			print("iter nubmer: "+str(iter))
			if iter>100:
				keep_going=0
				print("too many iteration")

			last_phi = phi[0,0]
			iter=iter+1
			
	for key in conj_list:
		pair=pair_list[key]
		
		if key not in sl:
			sl[key]={}
		
		if pair not in sl:
			sl[pair]={}
				
		with open(conj_list[key],'r') as f_json:
			conj = json.load(f_json)
			conjugate_input=[]
			raw_conj_input=[]
			if conj['image_1']!=image_list[0]:
				key_prev=image_list[image_list.index(conj['image_1'])-1]
				
				with open(conj_list[key_prev],'r') as f_prev_json:
					prev_conj = json.load(f_prev_json)
					
					test=prev_conj['conjugate_pt']
					
					for conj_pt_2 in conj['conjugate_pt']:
						raw_conj_input.append(conj_pt_2)
						exist=False
						for conj_pt_1 in prev_conj['conjugate_pt']:
							for key_1 in conj_pt_1:

								for key_2 in conj_pt_2:		
									if conj_pt_1[key_1][2]==conj_pt_2[key_2][0] and conj_pt_1[key_1][3]==conj_pt_2[key_2][1]:
										exist=True
								
						if exist==False:
							conjugate_input.append(conj_pt_2)
			else: 
				conjugate_input=conj['conjugate_pt']
				raw_conj_input=conj['conjugate_pt']

			print(len(raw_conj_input))
			out_log="/".join([out_dir,"output.log"])
			with open(out_log,'a') as output_file:
				csv_writer = csv.writer(output_file)
				csv_writer.writerow(["1st conjugate"])
				csv_writer.writerow([len(raw_conj_input)])
			

			for pt in conjugate_input:
				for pid in pt: 
					
					s1=float(pt[pid][0])
					l1=float(pt[pid][1])
					s2=float(pt[pid][2])
					l2=float(pt[pid][3])
										
					if '2' not in sl[key]:
						sl[key]['2']={}
						
					if '1' not in sl[pair]:
						sl[pair]['1']={}
						
					if pid not in sl[key]['2']:
						sl[key]['2'][pid]={}
						
					if pid not in sl[pair]['1']:
						sl[pair]['1'][pid]={}
						
					sl[key]['2'][pid]['x']=s1
					sl[key]['2'][pid]['y']=l1
					sl[pair]['1'][pid]['x']=s2
					sl[pair]['1'][pid]['y']=l2	
	XYZ={}
	obs={}
	for key in conj_list:
		pair=pair_list[key]
		if key not in XYZ:
			XYZ[key]={}
		if key not in obs:
			obs[key]={}
		if pair not in XYZ:
			XYZ[pair]={}
		if pair not in obs:
			obs[pair]={}
		XL1=float(eo_dict[key]['XL'])
		YL1=float(eo_dict[key]['YL'])
		ZL1=float(eo_dict[key]['ZL'])
		o1=float(eo_dict[key]['omega'])/180*np.pi
		p1=float(eo_dict[key]['phi'])/180*np.pi
		k1=float(eo_dict[key]['kappa'])/180*np.pi
		
		XL2=float(eo_dict[pair]['XL'])
		YL2=float(eo_dict[pair]['YL'])
		ZL2=float(eo_dict[pair]['ZL'])
		o2=float(eo_dict[pair]['omega'])/180*np.pi
		p2=float(eo_dict[pair]['phi'])/180*np.pi
		k2=float(eo_dict[pair]['kappa'])/180*np.pi
				
		with open(conj_list[key],'r') as f_json:
			conj = json.load(f_json)
			conjugate_input=[]
			if conj['image_1']!=image_list[0]:
				key_prev=image_list[image_list.index(conj['image_1'])-1]
				
				with open(conj_list[key_prev],'r') as f_prev_json:
					prev_conj = json.load(f_prev_json)
					
					test=prev_conj['conjugate_pt']
					
					for conj_pt_2 in conj['conjugate_pt']:
						exist=False
						for conj_pt_1 in prev_conj['conjugate_pt']:
							for key_1 in conj_pt_1:

								for key_2 in conj_pt_2:		
									if conj_pt_1[key_1][2]==conj_pt_2[key_2][0] and conj_pt_1[key_1][3]==conj_pt_2[key_2][1]:
										exist=True
								
						if exist==False:
							conjugate_input.append(conj_pt_2)
			else: 
				conjugate_input=conj['conjugate_pt']
							

			print(len(conjugate_input))
			out_log="/".join([out_dir,"output.log"])
			with open(out_log,'a') as output_file:
				csv_writer = csv.writer(output_file)
				csv_writer.writerow(["2nd conjugate"])
				csv_writer.writerow([len(conjugate_input)])

			for pt in conjugate_input:
				for pid in pt: 
					
					x_1=float(pt[pid][0])
					y_1=float(pt[pid][1])
					x_2=float(pt[pid][2])
					y_2=float(pt[pid][3])
					
					f_1 = f
					a1_1 = six_par[conj['image_1']][0][0]
					a2_1 = six_par[conj['image_1']][1][0]
					a3_1 = six_par[conj['image_1']][2][0]
					a4_1 = six_par[conj['image_1']][3][0]
					a5_1 = six_par[conj['image_1']][4][0]
					a6_1 = six_par[conj['image_1']][5][0]
					x1 = a1_1*x_1+a2_1*y_1+a3_1-x_0
					y1 = a4_1*x_1+a5_1*y_1+a6_1-y_0
					
					f_2 = f
					a1_2 = six_par[conj['image_2']][0][0]
					a2_2 = six_par[conj['image_2']][1][0]
					a3_2 = six_par[conj['image_2']][2][0]
					a4_2 = six_par[conj['image_2']][3][0]
					a5_2 = six_par[conj['image_2']][4][0]
					a6_2 = six_par[conj['image_2']][5][0]
					x2 = a1_2*x_2+a2_2*y_2+a3_2-x_0
					y2 = a4_2*x_2+a5_2*y_2+a6_2-y_0
					
					if '2' not in obs[key]:
						obs[key]['2']={}
						
					if '1' not in obs[pair]:
						obs[pair]['1']={}
						
					if pid not in obs[key]['2']:
						obs[key]['2'][pid]={}
						
					if pid not in obs[pair]['1']:
						obs[pair]['1'][pid]={}
						
					obs[key]['2'][pid]['x']=x1
					obs[key]['2'][pid]['y']=y1
					obs[pair]['1'][pid]['x']=x2
					obs[pair]['1'][pid]['y']=y2

					L_1 = np.array([[x1],[y1],[-f_1]])
					L_2 = np.array([[x2],[y2],[-f_2]])
					
					
					M1 = np.array([[cos(p1)*cos(k1), cos(o1)*sin(k1)+sin(o1)*sin(p1)*cos(k1), sin(o1)*sin(k1)-cos(o1)*sin(p1)*cos(k1)],
									[-cos(p1)*sin(k1), cos(o1)*cos(k1)-sin(o1)*sin(p1)*sin(k1), sin(o1)*cos(k1)+cos(o1)*sin(p1)*sin(k1)],
									[sin(p1), -sin(o1)*cos(p1), cos(o1)*cos(p1)]])
									
					M2 = np.array([[cos(p2)*cos(k2), cos(o2)*sin(k2)+sin(o2)*sin(p2)*cos(k2), sin(o2)*sin(k2)-cos(o2)*sin(p2)*cos(k2)],
									[-cos(p2)*sin(k2), cos(o2)*cos(k2)-sin(o2)*sin(p2)*sin(k2), sin(o2)*cos(k2)+cos(o2)*sin(p2)*sin(k2)],
									[sin(p2), -sin(o2)*cos(p2), cos(o2)*cos(p2)]])
									
					L1 = matmul(M1.T,L_1)
					L2 = matmul(M2.T,L_2)
					C1 = L1[0,0]/L1[2,0]
					C2 = L1[1,0]/L1[2,0]
					C3 = L2[0,0]/L2[2,0]
					C4 = L2[1,0]/L2[2,0]

					B1=np.array([[1, 0, -C1],[0, 1, -C2],[1, 0, -C3],[0, 1, -C4]])
					F1=np.array([[XL1-C1*ZL1],[YL1-C2*ZL1],[XL2-C3*ZL2],[YL2-C4*ZL2]])
					N1=matmul(B1.T,B1)
					t1=matmul(B1.T,F1)
					cp=matmul(inv(N1),t1)
					
					if '2' not in XYZ[key]:
						XYZ[key]['2']={}
						
					if '1' not in XYZ[pair]:
						XYZ[pair]['1']={}
						
					if pid not in XYZ[key]['2']:
						XYZ[key]['2'][pid]={}
						
					if pid not in XYZ[pair]['1']:
						XYZ[pair]['1'][pid]={}
						
					XYZ[key]['2'][pid]['X']=cp[0]
					XYZ[key]['2'][pid]['Y']=cp[1]
					XYZ[key]['2'][pid]['Z']=cp[2]
					
					XYZ[pair]['1'][pid]['X']=cp[0]
					XYZ[pair]['1'][pid]['Y']=cp[1]
					XYZ[pair]['1'][pid]['Z']=cp[2]
					
	XYZ_sr={}
	obs_sr={}
	sl_sr={}
	for key in sr_list:
		triple=trip_list[key]
		tr_1=trip_list[key][0]
		tr_2=trip_list[key][1]
		
		if key not in XYZ_sr:
			XYZ_sr[key]={}
		if key not in obs_sr:
			obs_sr[key]={}
		if key not in sl_sr:
			sl_sr[key]={}
		if triple[0] not in XYZ_sr:
			XYZ_sr[triple[0]]={}
		if triple[0] not in obs_sr:
			obs_sr[triple[0]]={}
		if triple[0] not in sl_sr:
			sl_sr[triple[0]]={}
		if triple[1] not in XYZ_sr:
			XYZ_sr[triple[1]]={}
		if triple[1] not in obs_sr:
			obs_sr[triple[1]]={}
		if triple[1] not in sl_sr:
			sl_sr[triple[1]]={}
			
		XL1=float(eo_dict[key]['XL'])
		YL1=float(eo_dict[key]['YL'])
		ZL1=float(eo_dict[key]['ZL'])
		o1=float(eo_dict[key]['omega'])/180*np.pi
		p1=float(eo_dict[key]['phi'])/180*np.pi
		k1=float(eo_dict[key]['kappa'])/180*np.pi
		
		XL2=float(eo_dict[tr_1]['XL'])
		YL2=float(eo_dict[tr_1]['YL'])
		ZL2=float(eo_dict[tr_1]['ZL'])
		o2=float(eo_dict[tr_1]['omega'])/180*np.pi
		p2=float(eo_dict[tr_1]['phi'])/180*np.pi
		k2=float(eo_dict[tr_1]['kappa'])/180*np.pi

		with open(sr_list[key],'r') as f_json:
			sr = json.load(f_json)
			pt = sr['scale_restraint_pt']
			for pid in pt: 
			
				s1=float(pt[pid][0])
				l1=float(pt[pid][1])
	
				s2=float(pt[pid][2])
				l2=float(pt[pid][3])
				
				s3=float(pt[pid][4])
				l3=float(pt[pid][5])
				
				f_1 = f
				a1_1 = six_par[sr['image_1']][0][0]
				a2_1 = six_par[sr['image_1']][1][0]
				a3_1 = six_par[sr['image_1']][2][0]
				a4_1 = six_par[sr['image_1']][3][0]
				a5_1 = six_par[sr['image_1']][4][0]
				a6_1 = six_par[sr['image_1']][5][0]
				x1 = a1_1*s1+a2_1*l1+a3_1-x_0
				y1 = a4_1*s1+a5_1*l1+a6_1-y_0
				
				f_2 = f
				a1_2 = six_par[sr['image_2']][0][0]
				a2_2 = six_par[sr['image_2']][1][0]
				a3_2 = six_par[sr['image_2']][2][0]
				a4_2 = six_par[sr['image_2']][3][0]
				a5_2 = six_par[sr['image_2']][4][0]
				a6_2 = six_par[sr['image_2']][5][0]
				x2 = a1_2*s2+a2_2*l2+a3_2-x_0
				y2 = a4_2*s2+a5_2*l2+a6_2-y_0
				
				f_3 = f
				a1_3 = six_par[sr['image_3']][0][0]
				a2_3 = six_par[sr['image_3']][1][0]
				a3_3 = six_par[sr['image_3']][2][0]
				a4_3 = six_par[sr['image_3']][3][0]
				a5_3 = six_par[sr['image_3']][4][0]
				a6_3 = six_par[sr['image_3']][5][0]
				x3 = a1_3*s3+a2_3*l3+a3_3-x_0
				y3 = a4_3*s3+a5_3*l3+a6_3-y_0
				
				if '3' not in sl_sr[key]:
					sl_sr[key]['3']={}
				
				if '2' not in sl_sr[triple[0]]:
					sl_sr[triple[0]]['2']={}
					
				if '1' not in sl_sr[triple[1]]:
					sl_sr[triple[1]]['1']={}
					
				if pid not in sl_sr[key]['3']:
					sl_sr[key]['3'][pid]={}
					
				if pid not in sl_sr[triple[0]]['2']:
					sl_sr[triple[0]]['2'][pid]={}
					
				if pid not in sl_sr[triple[1]]['1']:
					sl_sr[triple[1]]['1'][pid]={}
				
				sl_sr[key]['3'][pid]['x']=s1
				sl_sr[key]['3'][pid]['y']=l1
				sl_sr[triple[0]]['2'][pid]['x']=s2
				sl_sr[triple[0]]['2'][pid]['y']=l2
				sl_sr[triple[1]]['1'][pid]['x']=s3
				sl_sr[triple[1]]['1'][pid]['y']=l3
				
				if '3' not in obs_sr[key]:
					obs_sr[key]['3']={}
				
				if '2' not in obs_sr[triple[0]]:
					obs_sr[triple[0]]['2']={}
					
				if '1' not in obs_sr[triple[1]]:
					obs_sr[triple[1]]['1']={}
					
				if pid not in obs_sr[key]['3']:
					obs_sr[key]['3'][pid]={}
					
				if pid not in obs_sr[triple[0]]['2']:
					obs_sr[triple[0]]['2'][pid]={}
					
				if pid not in obs_sr[triple[1]]['1']:
					obs_sr[triple[1]]['1'][pid]={}
				
				obs_sr[key]['3'][pid]['x']=x1
				obs_sr[key]['3'][pid]['y']=y1
				obs_sr[triple[0]]['2'][pid]['x']=x2
				obs_sr[triple[0]]['2'][pid]['y']=y2
				obs_sr[triple[1]]['1'][pid]['x']=x3
				obs_sr[triple[1]]['1'][pid]['y']=y3
				
				L_1 = np.array([[x1],[y1],[-f_1]])
				L_2 = np.array([[x2],[y2],[-f_2]])
				
				M1 = np.array([[cos(p1)*cos(k1), cos(o1)*sin(k1)+sin(o1)*sin(p1)*cos(k1), sin(o1)*sin(k1)-cos(o1)*sin(p1)*cos(k1)],
								[-cos(p1)*sin(k1), cos(o1)*cos(k1)-sin(o1)*sin(p1)*sin(k1), sin(o1)*cos(k1)+cos(o1)*sin(p1)*sin(k1)],
								[sin(p1), -sin(o1)*cos(p1), cos(o1)*cos(p1)]])
								
				M2 = np.array([[cos(p2)*cos(k2), cos(o2)*sin(k2)+sin(o2)*sin(p2)*cos(k2), sin(o2)*sin(k2)-cos(o2)*sin(p2)*cos(k2)],
								[-cos(p2)*sin(k2), cos(o2)*cos(k2)-sin(o2)*sin(p2)*sin(k2), sin(o2)*cos(k2)+cos(o2)*sin(p2)*sin(k2)],
								[sin(p2), -sin(o2)*cos(p2), cos(o2)*cos(p2)]])
								
				L1 = matmul(M1.T,L_1)
				L2 = matmul(M2.T,L_2)
				C1 = L1[0,0]/L1[2,0]
				C2 = L1[1,0]/L1[2,0]
				C3 = L2[0,0]/L2[2,0]
				C4 = L2[1,0]/L2[2,0]

				B1=np.array([[1, 0, -C1],[0, 1, -C2],[1, 0, -C3],[0, 1, -C4]])
				F1=np.array([[XL1-C1*ZL1],[YL1-C2*ZL1],[XL2-C3*ZL2],[YL2-C4*ZL2]])
				N1=matmul(B1.T,B1)
				t1=matmul(B1.T,F1)
				sp=matmul(inv(N1),t1)
				
				if '3' not in XYZ_sr[key]:
					XYZ_sr[key]['3']={}
				
				if '2' not in XYZ_sr[triple[0]]:
					XYZ_sr[triple[0]]['2']={}
					
				if '1' not in XYZ_sr[triple[1]]:
					XYZ_sr[triple[1]]['1']={}
	
				if pid not in XYZ_sr[key]['3']:
					XYZ_sr[key]['3'][pid]={}
					
				if pid not in XYZ_sr[triple[0]]['2']:
					XYZ_sr[triple[0]]['2'][pid]={}
					
				if pid not in XYZ_sr[triple[1]]['1']:
					XYZ_sr[triple[1]]['1'][pid]={}
				
				XYZ_sr[key]['3'][pid]['X']=sp[0]
				XYZ_sr[key]['3'][pid]['Y']=sp[1]
				XYZ_sr[key]['3'][pid]['Z']=sp[2]
				
				XYZ_sr[triple[0]]['2'][pid]['X']=sp[0]
				XYZ_sr[triple[0]]['2'][pid]['Y']=sp[1]
				XYZ_sr[triple[0]]['2'][pid]['Z']=sp[2]
				
				XYZ_sr[triple[1]]['1'][pid]['X']=sp[0]
				XYZ_sr[triple[1]]['1'][pid]['Y']=sp[1]
				XYZ_sr[triple[1]]['1'][pid]['Z']=sp[2]

	index_del = []
	index_del_s = [] 
	keep_going_v = 1
	while keep_going_v == 1:
		pt_index_c=[]
		pt_index_s=[]
		
		for obs_key in obs:
			for key in obs[obs_key]:
				for pt_key in obs[obs_key][key]:
					pt_index_c.append([obs_key,key,pt_key])
				
		for obs_key in obs_sr:
			for key in obs_sr[obs_key]:
				obs_sr_keys=list(obs_sr[obs_key][key].keys())
				pt_index_s.append([obs_key,key,obs_sr_keys[0]])
					
		L0_c=[]
		L0_s=[]
		no_c=len(pt_index_c)
		no_s=len(pt_index_s)
		
		L0 = zeros((2*no_c+2*no_s,1))
		Lc_ind={}
		for i in range(len(pt_index_c)):
		
			ind = pt_index_c[i]
			x=obs[ind[0]][ind[1]][ind[2]]['x']
			y=obs[ind[0]][ind[1]][ind[2]]['y']
			L0_c.append([x])
			L0_c.append([y])
			L0[2*i,0]=x
			L0[2*i+1,0]=y
			if ind[0] not in Lc_ind:
				Lc_ind[ind[0]]={}
				
			if ind[1] not in Lc_ind[ind[0]]:
				Lc_ind[ind[0]][ind[1]]={}
				
			if ind[2] not in Lc_ind[ind[0]][ind[1]]:
				Lc_ind[ind[0]][ind[1]][ind[2]]={}
			
			if 'x' not in Lc_ind[ind[0]][ind[1]]:
				Lc_ind[ind[0]][ind[1]][ind[2]]['x']=2*i
				
			if 'y' not in Lc_ind[ind[0]][ind[1]]:
				Lc_ind[ind[0]][ind[1]][ind[2]]['y']=2*i+1
				
				
		Ls_ind={}
		for i in range(len(pt_index_s)):
		
			ind = pt_index_s[i]
			x=obs_sr[ind[0]][ind[1]][ind[2]]['x']
			y=obs_sr[ind[0]][ind[1]][ind[2]]['y']
			L0_s.append([x])
			L0_s.append([y])
			L0[2*no_c+2*i,0]=x
			L0[2*no_c+2*i+1,0]=y
			if ind[0] not in Ls_ind:
				Ls_ind[ind[0]]={}
				
			if ind[1] not in Ls_ind[ind[0]]:
				Ls_ind[ind[0]][ind[1]]={}
				
			if ind[2] not in Ls_ind[ind[0]][ind[1]]:
				Ls_ind[ind[0]][ind[1]][ind[2]]={}
			
			if 'x' not in Ls_ind[ind[0]][ind[1]]:
				Ls_ind[ind[0]][ind[1]][ind[2]]['x']=2*no_c+2*i
				
			if 'y' not in Ls_ind[ind[0]][ind[1]]:
				Ls_ind[ind[0]][ind[1]][ind[2]]['y']=2*no_c+2*i+1
			
		
		no_pt_c=0	
		del_ind_c=[]
		del_ind_s=[]
		for img_id in image_list:	
			if '2' in XYZ[img_id]:
				no_pt_c=no_pt_c+len(XYZ[img_id]['2'])
				for cp in XYZ[img_id]['2']:
					del_ind_c.append([img_id,'2',cp])
		no_pt_s=0
		for img_id in image_list:	
			if '3' in XYZ_sr[img_id]:
				no_pt_s=no_pt_s+len(XYZ_sr[img_id]['3'])
				for cp in XYZ_sr[img_id]['3']:
					del_ind_s.append([img_id,'3',cp])
					
		no_pt = no_pt_c + no_pt_s		
		EO = []
		delta = zeros((6*len(image_list)+3*no_pt,1))
		
		for i in range(len(image_list)):
			EO.append(float(eo_dict[image_list[i]]['XL']))
			EO.append(float(eo_dict[image_list[i]]['YL']))
			EO.append(float(eo_dict[image_list[i]]['ZL']))
			EO.append(float(eo_dict[image_list[i]]['omega'])/180*np.pi)
			EO.append(float(eo_dict[image_list[i]]['phi'])/180*np.pi)
			EO.append(float(eo_dict[image_list[i]]['kappa'])/180*np.pi)
			delta[6*i,0]=float(eo_dict[image_list[i]]['XL'])
			delta[6*i+1,0]=float(eo_dict[image_list[i]]['YL'])
			delta[6*i+2,0]=float(eo_dict[image_list[i]]['ZL'])
			delta[6*i+3,0]=float(eo_dict[image_list[i]]['omega'])/180*np.pi
			delta[6*i+4,0]=float(eo_dict[image_list[i]]['phi'])/180*np.pi
			delta[6*i+5,0]=float(eo_dict[image_list[i]]['kappa'])/180*np.pi
		
		del_c=[]
		del_s=[]
		
		for i in range(len(del_ind_c)):
			ind = del_ind_c[i]
			X=XYZ[ind[0]][ind[1]][ind[2]]['X'][0]
			Y=XYZ[ind[0]][ind[1]][ind[2]]['Y'][0]
			Z=XYZ[ind[0]][ind[1]][ind[2]]['Z'][0]
			del_c.append(X)
			del_c.append(Y)
			del_c.append(Z)
			delta[6*len(image_list)+3*i,0]=X
			delta[6*len(image_list)+3*i+1,0]=Y
			delta[6*len(image_list)+3*i+2,0]=Z
			
		for i in range(len(del_ind_s)):
			ind = del_ind_s[i]
			X=XYZ_sr[ind[0]][ind[1]][ind[2]]['X'][0]
			Y=XYZ_sr[ind[0]][ind[1]][ind[2]]['Y'][0]
			Z=XYZ_sr[ind[0]][ind[1]][ind[2]]['Z'][0]
			del_s.append(X)
			del_s.append(Y)
			del_s.append(Z)
			delta[6*len(image_list)+3*no_pt_c+3*i,0]=X
			delta[6*len(image_list)+3*no_pt_c+3*i+1,0]=Y
			delta[6*len(image_list)+3*no_pt_c+3*i+2,0]=Z
			
		no_obs=len(L0)
		WW = eye(no_obs) 
		Wxx = zeros((6*len(image_list)+3*no_pt,6*len(image_list)+3*no_pt)) 
		Wxyz = zeros((3*no_pt,3*no_pt)) 
		
		for i in range(len(image_list)):
			Wxx[6*i,6*i]=(1/(1.0e8))**2
			Wxx[6*i+1,6*i+1]=(1/(1.0e8))**2
			Wxx[6*i+2,6*i+2]=(1/(1.0e8))**2
			Wxx[6*i+3,6*i+3]=(1/(1.0e8))**2
			Wxx[6*i+4,6*i+4]=(1/(1.0e8))**2
			Wxx[6*i+5,6*i+5]=(1/(1.0e8))**2

		for i in range(no_pt):
			Wxyz[3*i,3*i]=(1/(1.0e8))**2
			Wxyz[3*i+1,3*i+1]=(1/(1.0e8))**2
			Wxyz[3*i+2,3*i+2]=(1/(1.0e8))**2
			
		for i in range(6):
			Wxx[i,i]=(1/(1.0e-8))**2 
		
		Wxx[7,7]=(1/(1.0e-8))**2 
		Wxx[6*len(image_list):6*len(image_list)+3*no_pt,6*len(image_list):6*len(image_list)+3*no_pt] = Wxyz
			
		last_phi = 10
		keep_going = 1
		iter = 0
		L=L0
		delta_0=delta
		
		while keep_going == 1:
			FF=zeros((no_obs,1))
			AA=zeros((no_obs,no_obs))
			BB=zeros((no_obs,6*len(image_list)+3*no_pt))
			for i in range(len(pt_index_c)):
				img_id=pt_index_c[i][0]
				img_ind = image_list.index(img_id)

				if pt_index_c[i][1]=='2':
					point_index=[pt_index_c[i][0],pt_index_c[i][1],pt_index_c[i][2]]
					
				elif pt_index_c[i][1]=='1':
					point_index=[image_list[image_list.index(pt_index_c[i][0])-1],'2',pt_index_c[i][2]]
				
				pt_ind=del_ind_c.index(point_index)
					
				x = L[2*i,0] 
				y = L[2*i+1,0]
				XL = delta[6*img_ind,0]
				YL = delta[6*img_ind+1,0]
				ZL = delta[6*img_ind+2,0]
				o = delta[6*img_ind+3,0]
				p = delta[6*img_ind+4,0]
				k = delta[6*img_ind+5,0]
				
				X=delta[6*len(image_list)+3*pt_ind,0]
				Y=delta[6*len(image_list)+3*pt_ind+1,0]			
				Z=delta[6*len(image_list)+3*pt_ind+2,0]
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
				
				B17 = -B11
				B18 = -B12
				B19 = -B13
				
				B21 = f*(-M[1,0]+V/W*M[2,0])/W
				B22 = f*(-M[1,1]+V/W*M[2,1])/W
				B23 = f*(-M[1,2]+V/W*M[2,2])/W
				B24 = f*(Vo-V/W*Wo)/W
				B25 = f*(Vp-V/W*Wp)/W
				B26 = f*(Vk-V/W*Wk)/W				
				
				B27 = -B21
				B28 = -B22
				B29 = -B23
				
				A_s = np.array([[1,0],[0,1]])
				
				FF[2*i:2*i+2,:] = F_s
				AA[2*i:2*i+2,2*i:2*i+2] = A_s
				BB[2*i:2*i+2,6*img_ind:6*img_ind+6] = np.array([[B11, B12, B13, B14, B15, B16],[B21, B22, B23, B24, B25, B26]])
				BB[2*i:2*i+2,6*len(image_list)+3*pt_ind:6*len(image_list)+3*pt_ind+3] = np.array([[B17, B18, B19],[B27, B28, B29]])

			for i in range(len(pt_index_s)):
				img_id = pt_index_s[i][0]
				img_ind = image_list.index(img_id)
				
				if pt_index_s[i][1]=='3':
					point_index=[pt_index_s[i][0],pt_index_s[i][1],pt_index_s[i][2]]
					
				elif pt_index_s[i][1]=='2':
					point_index=[image_list[image_list.index(pt_index_s[i][0])-1],'3',pt_index_s[i][2]]
				
				elif pt_index_s[i][1]=='1':
					point_index=[image_list[image_list.index(pt_index_s[i][0])-2],'3',pt_index_s[i][2]]			
				
				pt_ind=del_ind_s.index(point_index)

				x = L[2*no_c+2*i,0] 
				y = L[2*no_c+2*i+1,0]
				XL = delta[6*img_ind,0]
				YL = delta[6*img_ind+1,0]
				ZL = delta[6*img_ind+2,0]
				o = delta[6*img_ind+3,0]
				p = delta[6*img_ind+4,0]
				k = delta[6*img_ind+5,0]
				
				X=delta[6*len(image_list)+3*no_pt_c+3*pt_ind,0]
				Y=delta[6*len(image_list)+3*no_pt_c+3*pt_ind+1,0]			
				Z=delta[6*len(image_list)+3*no_pt_c+3*pt_ind+2,0]

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
				
				B17 = -B11
				B18 = -B12
				B19 = -B13
				
				B21 = f*(-M[1,0]+V/W*M[2,0])/W
				B22 = f*(-M[1,1]+V/W*M[2,1])/W
				B23 = f*(-M[1,2]+V/W*M[2,2])/W
				B24 = f*(Vo-V/W*Wo)/W
				B25 = f*(Vp-V/W*Wp)/W
				B26 = f*(Vk-V/W*Wk)/W				
				
				B27 = -B21
				B28 = -B22
				B29 = -B23
				
				A_s = np.array([[1,0],[0,1]])
				
				FF[2*no_c+2*i:2*no_c+2*i+2,:] = F_s
				AA[2*no_c+2*i:2*no_c+2*i+2,2*no_c+2*i:2*no_c+2*i+2] = A_s
				BB[2*no_c+2*i:2*no_c+2*i+2,6*img_ind:6*img_ind+6] = np.array([[B11, B12, B13, B14, B15, B16],[B21, B22, B23, B24, B25, B26]])
				BB[2*no_c+2*i:2*no_c+2*i+2,6*len(image_list)+3*no_pt_c+3*pt_ind:6*len(image_list)+3*no_pt_c+3*pt_ind+3] = np.array([[B17, B18, B19],[B27, B28, B29]])

			Q=inv(WW)
			
			ff=-FF-matmul(AA,(L0-L))
			Qe=matmul(matmul(AA,Q),AA.T)
			We=inv(Qe)
			
			N=matmul(matmul(BB.T,We),BB)
			t=matmul(matmul(BB.T,We),ff)
			fx=delta-delta_0
			ddel=matmul(inv(N+Wxx),(t-matmul(Wxx,fx)))
			
			v=matmul(matmul(AA.T,We),(ff-matmul(BB,ddel)))

			vvx=fx+ddel
			phi=matmul(matmul(v.T,WW),v)+matmul(matmul(vvx.T,Wxx),vvx)
			
			obj=abs((last_phi-phi[0,0])/last_phi)
			print("objective function is : "+str(obj))

			#Convergence check
			if obj<0.0001:
				keep_going=0
				print("Converged")
				v_c = abs(v[:-2*len(pt_index_s)])
				v_large = [[i,x[0]] for i,x in enumerate(v_c) if x>res_thre]
				
				v_s = abs(v[-2*len(pt_index_s):])
				v_large_s = [[i,x[0]] for i,x in enumerate(v_s) if x>res_thre]
				
				if len(v_large)==0 and len(v_large_s)==0:
					keep_going_v = 0
					out_log="/".join([out_dir,"output.log"])
					with open(out_log,'a') as output_file:
						
						csv_writer = csv.writer(output_file)
						csv_writer.writerow(["res_threshold"])
						csv_writer.writerow([res_thre])
						csv_writer.writerow(["max residual x"])
						csv_writer.writerow([max(abs(v[0::2]))])
						csv_writer.writerow(["max residual y"])
						csv_writer.writerow([max(abs(v[1::2]))])
						csv_writer.writerow(["rmse x"])
						csv_writer.writerow([np.sqrt(sum(v[0::2]**2)/len(v[0::2]))])
						csv_writer.writerow(["rmse y"])
						csv_writer.writerow([np.sqrt(sum(v[1::2]**2)/len(v[1::2]))])
						
				else:
					if len(v_large)>0:
					
						del_index = []
						for v_ind in v_large:
							if int(v_ind[0]/2) not in del_index:
								del_index.append(int(v_ind[0]/2))
						
						del_index.sort(reverse=True)
						print('2ray large residuals')
						print(del_index)
						for d_ind in del_index:
							img_id = pt_index_c[d_ind][0]
							pair_id = pt_index_c[d_ind][1]
							pt_id = pt_index_c[d_ind][2]
							
							if (pt_id in obs[img_id][pair_id]) and (pt_id in XYZ[img_id][pair_id]):
								index_del.append([img_id,pair_id,pt_id])
								del obs[img_id][pair_id][pt_id]
								del XYZ[img_id][pair_id][pt_id]
								
								if pair_id=='2':
									img_id = image_list[image_list.index(img_id)+1]
									pair_id = '1'
									index_del.append([img_id,pair_id,pt_id])
									del obs[img_id][pair_id][pt_id]
									del XYZ[img_id][pair_id][pt_id]
									
								elif pair_id=='1':
									img_id = image_list[image_list.index(img_id)-1]
									pair_id = '2'
									index_del.append([img_id,pair_id,pt_id])
									del obs[img_id][pair_id][pt_id]
									del XYZ[img_id][pair_id][pt_id]
							
					if len(v_large_s)>0:

						del_index = []
						for v_ind in v_large_s:
							if int(v_ind[0]/2) not in del_index:
								del_index.append(int(v_ind[0]/2))
						
						del_index.sort(reverse=True)
						print('3ray large residuals')
						print(del_index)
						for d_ind in del_index:
							img_id = pt_index_s[d_ind][0]
							trip_id = pt_index_s[d_ind][1]
							pt_id = pt_index_s[d_ind][2]
							if (pt_id in obs_sr[img_id][trip_id]) and (pt_id in XYZ_sr[img_id][trip_id]):
								index_del_s.append([img_id,trip_id,pt_id])
								del obs_sr[img_id][trip_id][pt_id]
								del XYZ_sr[img_id][trip_id][pt_id]
								
								if trip_id=='3':
									img_id_1 = image_list[image_list.index(img_id)+1]
									index_del_s.append([img_id_1,'2',pt_id])
									del obs_sr[img_id_1]['2'][pt_id]
									del XYZ_sr[img_id_1]['2'][pt_id]
									img_id_2 = image_list[image_list.index(img_id)+2]
									index_del_s.append([img_id_2,'1',pt_id])
									del obs_sr[img_id_2]['1'][pt_id]
									del XYZ_sr[img_id_2]['1'][pt_id]
									
								elif trip_id=='2':
									img_id_1 = image_list[image_list.index(img_id)-1]
									index_del_s.append([img_id_1,'3',pt_id])
									del obs_sr[img_id_1]['3'][pt_id]
									del XYZ_sr[img_id_1]['3'][pt_id]
									img_id_2 = image_list[image_list.index(img_id)+1]
									index_del_s.append([img_id_2,'1',pt_id])
									del obs_sr[img_id_2]['1'][pt_id]
									del XYZ_sr[img_id_2]['1'][pt_id]
									
								elif trip_id=='1':
									img_id_1 = image_list[image_list.index(img_id)-2]
									index_del_s.append([img_id_1,'3',pt_id])
									del obs_sr[img_id_1]['3'][pt_id]
									del XYZ_sr[img_id_1]['3'][pt_id]
									img_id_2 = image_list[image_list.index(img_id)-1]
									index_del_s.append([img_id_2,'2',pt_id])
									del obs_sr[img_id_2]['2'][pt_id]
									del XYZ_sr[img_id_2]['2'][pt_id]				
				print(np.max(np.abs(v)))
				
			if keep_going == 1:
				L=L0+v
				delta=delta+ddel

			print("iter nubmer: "+str(iter))
			if iter>100:
				keep_going=0
				print("too many iteration")

			last_phi = phi[0,0]
			iter=iter+1
	result={}
			
			
	with open(argv[5],'w') as output_file:
		csv_writer = csv.writer(output_file)
		csv_writer.writerow(["Exterior Orientation Parameters"])
		result_eo={}
		for i in range(len(image_list)):
			eo_i={}
			eo_i['XL']=delta[6*i,0]
			eo_i['YL']=delta[6*i+1,0]
			eo_i['ZL']=delta[6*i+2,0]
			eo_i['omega']=delta[6*i+3,0]
			eo_i['phi']=delta[6*i+4,0]
			eo_i['kappa']=delta[6*i+5,0]
			result_eo[image_list[i]]=eo_i
			
			csv_writer.writerow([image_list[i]])
			csv_writer.writerow(['XL(ft): '+str(delta[6*i,0])])
			csv_writer.writerow(['YL(ft): '+str(delta[6*i+1,0])])
			csv_writer.writerow(['ZL(ft): '+str(delta[6*i+2,0])])
			csv_writer.writerow(['omeag(rad): '+str(delta[6*i+3,0])])
			csv_writer.writerow(['phi(rad): '+str(delta[6*i+4,0])])
			csv_writer.writerow(['kappa(rad): '+str(delta[6*i+5,0])])
			csv_writer.writerow(" ")
		
		csv_writer.writerow(["scale factor"])
		scale={}
		scale_index = []
				
		for i in range(len(del_ind_c)):
			
			ind = del_ind_c[i]
			img_ind=ind[0]
		
			if img_ind not in scale_index:
				scale_index.append(img_ind)
				scale[img_ind] = []
			pair_ind=ind[1] 
			pt_ind=ind[2]
			
			x = L[Lc_ind[img_ind][pair_ind][pt_ind]['x']][0]
			y = L[Lc_ind[img_ind][pair_ind][pt_ind]['y']][0]
			
			X = delta[6*len(image_list)+3*i,0]
			Y = delta[6*len(image_list)+3*i+1,0]
			Z = delta[6*len(image_list)+3*i+2,0]
			
			im_index = image_list.index(img_ind)
			XL = delta[6*im_index,0]
			YL = delta[6*im_index+1,0]
			ZL = delta[6*im_index+2,0]
			o = delta[6*im_index+3,0]
			p = delta[6*im_index+4,0]
			k = delta[6*im_index+5,0]
			
			M = np.array([[cos(p)*cos(k), cos(o)*sin(k)+sin(o)*sin(p)*cos(k), sin(o)*sin(k)-cos(o)*sin(p)*cos(k)],
					[-cos(p)*sin(k), cos(o)*cos(k)-sin(o)*sin(p)*sin(k), sin(o)*cos(k)+cos(o)*sin(p)*sin(k)],
					[sin(p), -sin(o)*cos(p), cos(o)*cos(p)]])
			k = matmul(inv(M),np.array([[x],[y],[-f]]))/np.array([[X-XL],[Y-YL],[Z-ZL]])
			scale_pt=np.mean(k)
			scale[img_ind].append(scale_pt)

			img_ind = image_list[image_list.index(ind[0])+1]
			if img_ind not in scale_index:
				scale_index.append(img_ind)
				scale[img_ind] = []
			pair_ind = '1'
			pt_ind = ind[2]
			x = L[Lc_ind[img_ind][pair_ind][pt_ind]['x']][0]
			y = L[Lc_ind[img_ind][pair_ind][pt_ind]['y']][0]
			
			X = delta[6*len(image_list)+3*i,0]
			Y = delta[6*len(image_list)+3*i+1,0]
			Z = delta[6*len(image_list)+3*i+2,0]
			
			im_index = image_list.index(img_ind)
			XL = delta[6*im_index,0]
			YL = delta[6*im_index+1,0]
			ZL = delta[6*im_index+2,0]
			o = delta[6*im_index+3,0]
			p = delta[6*im_index+4,0]
			k = delta[6*im_index+5,0]
			
			M = np.array([[cos(p)*cos(k), cos(o)*sin(k)+sin(o)*sin(p)*cos(k), sin(o)*sin(k)-cos(o)*sin(p)*cos(k)],
				[-cos(p)*sin(k), cos(o)*cos(k)-sin(o)*sin(p)*sin(k), sin(o)*cos(k)+cos(o)*sin(p)*sin(k)],
				[sin(p), -sin(o)*cos(p), cos(o)*cos(p)]])
			k = matmul(inv(M),np.array([[x],[y],[-f]]))/np.array([[X-XL],[Y-YL],[Z-ZL]])
			scale_pt=np.mean(k)

			scale[img_ind].append(scale_pt)
		
		for i in range(len(pt_index_s)):
			img_id = pt_index_s[i][0]
			img_ind = image_list.index(img_id)
			if pt_index_s[i][1]=='3':
				point_index=[pt_index_s[i][0],pt_index_s[i][1],pt_index_s[i][2]]
				
			elif pt_index_s[i][1]=='2':
				point_index=[image_list[image_list.index(pt_index_s[i][0])-1],'3',pt_index_s[i][2]]
			
			elif pt_index_s[i][1]=='1':
				point_index=[image_list[image_list.index(pt_index_s[i][0])-2],'3',pt_index_s[i][2]]			
			
			pt_ind=del_ind_s.index(point_index)
			
			x = L[Ls_ind[pt_index_s[i][0]][pt_index_s[i][1]][pt_index_s[i][2]]['x']][0]
			y = L[Ls_ind[pt_index_s[i][0]][pt_index_s[i][1]][pt_index_s[i][2]]['y']][0]
			
			XL = delta[6*img_ind,0]
			YL = delta[6*img_ind+1,0]
			ZL = delta[6*img_ind+2,0]
			o = delta[6*img_ind+3,0]
			p = delta[6*img_ind+4,0]
			k = delta[6*img_ind+5,0]
			
			X=delta[6*len(image_list)+3*no_pt_c+3*pt_ind,0]
			Y=delta[6*len(image_list)+3*no_pt_c+3*pt_ind+1,0]			
			Z=delta[6*len(image_list)+3*no_pt_c+3*pt_ind+2,0]
			
			M = np.array([[cos(p)*cos(k), cos(o)*sin(k)+sin(o)*sin(p)*cos(k), sin(o)*sin(k)-cos(o)*sin(p)*cos(k)],
				[-cos(p)*sin(k), cos(o)*cos(k)-sin(o)*sin(p)*sin(k), sin(o)*cos(k)+cos(o)*sin(p)*sin(k)],
				[sin(p), -sin(o)*cos(p), cos(o)*cos(p)]])
					
			k = matmul(inv(M),np.array([[x],[y],[-f]]))/np.array([[X-XL],[Y-YL],[Z-ZL]])
			scale_pt=np.mean(k)
			print(k)
			print(pt_index_s[i])
			scale[img_id].append(scale_pt)

		result_cp={}
		result['eo'] = result_eo
		result['cp'] = result_cp
		result['scale'] = scale
		output_json = json.dumps(result, indent = 4)

		json_fname = ".".join([os.path.splitext(argv[5])[0],"json"])
		with open(json_fname,'w') as output_file:
			output_file.write(output_json)
		img_json={}
		for i in range(len(image_data)):
			
			img_json[image_data[i]['id']]=image_data[i]['path']
			
		
		out_log="/".join([out_dir,"output.log"])
		with open(out_log,'a') as output_file:
			
			csv_writer = csv.writer(output_file)
			
			for i_key in conj_list:
				csv_writer.writerow(["3rd conjugate"])
				csv_writer.writerow([len(XYZ[i_key]['2'])])
				
		out_log="/".join([out_dir,"output.log"])
		with open(out_log,'a') as output_file:
			csv_writer = csv.writer(output_file)
			csv_writer.writerow(["deleted 2ray point"])
			csv_writer.writerow([len(index_del)])
			csv_writer.writerow([index_del])
			csv_writer.writerow(["deleted 3ray point"])
			csv_writer.writerow([len(index_del_s)])
			csv_writer.writerow([index_del_s])
		
		"""cnt=0
		for key in conj_list:
			
			pair=pair_list[key]
			
			if cnt==0:
				im3_path=img_json[key]
				im4_path=img_json[pair]
				image3 = cv.imread(im3_path) # queryImage
				image4 = cv.imread(im4_path) # trainImage
				for pt in sl[key]['2']:
					
					if [key,'2', pt] not in index_del:
						image3 = cv.circle(image3, (int(sl[key]['2'][pt]['x']),int(sl[key]['2'][pt]['y'])), 150, (255,0,0), -1)
						
				for pt in sl[pair]['1']:	
					
					if [pair,'1', pt] not in index_del:
						image4 = cv.drawMarker(image4, (int(sl[pair]['1'][pt]['x']),int(sl[pair]['1'][pt]['y'])), color=(0,0,255), markerType=cv.MARKER_TRIANGLE_DOWN, markerSize=300, thickness=50)
						
				out_3 = "/".join([out_dir,"_".join([os.path.splitext(os.path.basename(im3_path))[0],"bba_min.tif"])])
				out_4 = "/".join([out_dir,"_".join([os.path.splitext(os.path.basename(im4_path))[0],"bba_min.tif"])])
				cv.imwrite(out_3,image3)
				cv.imwrite(out_4,image4)
				
			else:
				
				img_id = str(image_list[image_list.index(key)-1])
				im1=img_json[img_id]
				im2=img_json[key]
				im1_path = "/".join([out_dir,"_".join([os.path.splitext(os.path.basename(im1))[0],"bba_min.tif"])])
				im2_path = "/".join([out_dir,"_".join([os.path.splitext(os.path.basename(im2))[0],"bba_min.tif"])])
				im4_path = img_json[pair]
				image3 = cv.imread(im2_path) # im2 and im2 are same.
				image4 = cv.imread(im4_path) # trainImage
				for pt in sl[key]['2']:
					
					if [key,'2', pt] not in index_del:
						image3 = cv.circle(image3, (int(sl[key]['2'][pt]['x']),int(sl[key]['2'][pt]['y'])), 150, (255,0,0), -1)
						
				for pt in sl[pair]['1']:	
					
					if [pair,'1', pt] not in index_del:
						image4 = cv.drawMarker(image4, (int(sl[pair]['1'][pt]['x']),int(sl[pair]['1'][pt]['y'])), color=(0,0,255), markerType=cv.MARKER_TRIANGLE_DOWN, markerSize=300, thickness=50)
						
			

				#first img
				image1 = cv.imread(im1_path) # queryImage
				grp_id_1 = '3'
				pt_id_list_1=list(filter(lambda x: x[0]==img_id and x[1]==grp_id_1, pt_index_s))
				pt_id_1=pt_id_list_1[0][2] #pt_id_list_1 should have only one element
				s1=sl_sr[img_id][grp_id_1][pt_id_1]['x']
				l1=sl_sr[img_id][grp_id_1][pt_id_1]['y']
				image1 = cv.drawMarker(image1, (int(s1),int(l1)), color=(0,0,0), markerType=cv.MARKER_CROSS, markerSize=500, thickness=100)			
				out_1 = im1_path 
				cv.imwrite(out_1,image1)
				del image1
				
				#second img same with the third image (key)
				
				grp_id_3 = '2'
				pt_id_list_3=list(filter(lambda x: x[0]==key and x[1]==grp_id_3, pt_index_s))
				pt_id_3=pt_id_list_3[0][2]
				s3=sl_sr[key][grp_id_3][pt_id_3]['x']
				l3=sl_sr[key][grp_id_3][pt_id_3]['y']
				image3 = cv.drawMarker(image3, (int(s3),int(l3)), color=(0,0,0), markerType=cv.MARKER_CROSS, markerSize=500, thickness=100)
				out_3 = im2_path  # 2 and 3 is same
				cv.imwrite(out_3,image3)
				del image3
				
				#fourth image
				grp_id_4 = '1'
				pt_id_list_4=list(filter(lambda x: x[0]==pair and x[1]==grp_id_4, pt_index_s))
				pt_id_4=pt_id_list_4[0][2]
				s4=sl_sr[pair][grp_id_4][pt_id_4]['x']
				l4=sl_sr[pair][grp_id_4][pt_id_4]['y']
				image4 = cv.drawMarker(image4, (int(s4),int(l4)), color=(0,0,0), markerType=cv.MARKER_CROSS, markerSize=500, thickness=100)
				out_4 = "/".join([out_dir,"_".join([os.path.splitext(os.path.basename(im4_path))[0],"bba_min.tif"])])
				cv.imwrite(out_4,image4)
				del image4
				#pt_index_s[]
			cnt = cnt+1
		
		cnt=0
		for key in conj_list:
			
			pair=pair_list[key]
			
			if cnt==0:
				im3_path=img_json[key]
				im4_path=img_json[pair]
				image3 = cv.imread(im3_path) # queryImage
				image4 = cv.imread(im4_path) # trainImage
				for pt in sl[key]['2']:
					
					if [key,'2', pt] not in index_del:
						image3 = cv.circle(image3, (int(sl[key]['2'][pt]['x']),int(sl[key]['2'][pt]['y'])), 3, (255,0,0), -1)
						
				for pt in sl[pair]['1']:	
					
					if [pair,'1', pt] not in index_del:
						image4 = cv.drawMarker(image4, (int(sl[pair]['1'][pt]['x']),int(sl[pair]['1'][pt]['y'])), color=(0,0,255), markerType=cv.MARKER_TRIANGLE_DOWN, markerSize=3, thickness=2)
						
				out_3 = "/".join([out_dir,"_".join([os.path.splitext(os.path.basename(im3_path))[0],"bba_min_sm.tif"])])
				out_4 = "/".join([out_dir,"_".join([os.path.splitext(os.path.basename(im4_path))[0],"bba_min_sm.tif"])])
				cv.imwrite(out_3,image3)
				cv.imwrite(out_4,image4)
				
			else:
				
				img_id = str(image_list[image_list.index(key)-1])
				im1=img_json[img_id]
				im2=img_json[key]
				im1_path = "/".join([out_dir,"_".join([os.path.splitext(os.path.basename(im1))[0],"bba_min_sm.tif"])])
				im2_path = "/".join([out_dir,"_".join([os.path.splitext(os.path.basename(im2))[0],"bba_min_sm.tif"])])
				im4_path = img_json[pair]
				image3 = cv.imread(im2_path) # im2 and im2 are same.
				image4 = cv.imread(im4_path) # trainImage
				for pt in sl[key]['2']:
					
					if [key,'2', pt] not in index_del:
						image3 = cv.circle(image3, (int(sl[key]['2'][pt]['x']),int(sl[key]['2'][pt]['y'])), 3, (255,0,0), -1)
						
				for pt in sl[pair]['1']:	
					
					if [pair,'1', pt] not in index_del:
						image4 = cv.drawMarker(image4, (int(sl[pair]['1'][pt]['x']),int(sl[pair]['1'][pt]['y'])), color=(0,0,255), markerType=cv.MARKER_TRIANGLE_DOWN, markerSize=3, thickness=2)
						
				#first img
				image1 = cv.imread(im1_path) # queryImage
				grp_id_1 = '3'
				pt_id_list_1=list(filter(lambda x: x[0]==img_id and x[1]==grp_id_1, pt_index_s))
				pt_id_1=pt_id_list_1[0][2] #pt_id_list_1 should have only one element
				s1=sl_sr[img_id][grp_id_1][pt_id_1]['x']
				l1=sl_sr[img_id][grp_id_1][pt_id_1]['y']
				image1 = cv.drawMarker(image1, (int(s1),int(l1)), color=(0,0,0), markerType=cv.MARKER_CROSS, markerSize=500, thickness=100)			
				out_1 = im1_path 
				cv.imwrite(out_1,image1)
				del image1
				
				#second img same with the third image (key)
				
				grp_id_3 = '2'
				pt_id_list_3=list(filter(lambda x: x[0]==key and x[1]==grp_id_3, pt_index_s))
				pt_id_3=pt_id_list_3[0][2]
				s3=sl_sr[key][grp_id_3][pt_id_3]['x']
				l3=sl_sr[key][grp_id_3][pt_id_3]['y']
				image3 = cv.drawMarker(image3, (int(s3),int(l3)), color=(0,0,0), markerType=cv.MARKER_CROSS, markerSize=500, thickness=100)
				out_3 = im2_path  # 2 and 3 is same
				cv.imwrite(out_3,image3)
				del image3
				
				#fourth image
				grp_id_4 = '1'
				pt_id_list_4=list(filter(lambda x: x[0]==pair and x[1]==grp_id_4, pt_index_s))
				pt_id_4=pt_id_list_4[0][2]
				s4=sl_sr[pair][grp_id_4][pt_id_4]['x']
				l4=sl_sr[pair][grp_id_4][pt_id_4]['y']
				image4 = cv.drawMarker(image4, (int(s4),int(l4)), color=(0,0,0), markerType=cv.MARKER_CROSS, markerSize=500, thickness=100)
				out_4 = "/".join([out_dir,"_".join([os.path.splitext(os.path.basename(im4_path))[0],"bba_min_sm.tif"])])
				cv.imwrite(out_4,image4)
				del image4
				#pt_index_s[]
			cnt = cnt+1	"""
		

		
			

if __name__=="__main__":
	main()
