# Created by Jae Sung Kim (The Pennsylvania State University Libraries).
# Last modified: 08/31/21
# Example of running: python3 bba_nr.py io_nr.txt fiducial_mark.txt EO_init_1.txt control.txt bba_result_nr_1.txt 1 2 1 control_uncertainty.txt 
# 1 2 1: 1 control, 2 check, 1 correction for atmospheric refraction
# The unit of ground space coordinate is assumed to be feet.

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

	
	argv=sys.argv
	io_reader = csv.DictReader(open(argv[1]))
	fd_reader = csv.DictReader(open(argv[2]))
	eo_reader = csv.DictReader(open(argv[3]))
	cp_reader = csv.DictReader(open(argv[4]))
	ar=int(argv[8])
	uncertainty_reader = csv.DictReader(open(argv[9]))
	
	with open(argv[5],'w') as output_file:
		csv_writer = csv.writer(output_file)
		csv_writer.writerow(["output"])
	
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

	eo_data = list(eo_reader)
	eo_dict = {}
	for eo in eo_data:
		eo_key = eo['image_id']
		eo.pop('image_id')
		eo_dict[eo_key] = eo

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
	
	uncertainty_data = list(uncertainty_reader)
	
	for uncertainty in uncertainty_data:
		sig_x = float(uncertainty['X'])
		sig_y = float(uncertainty['Y'])
		sig_z = float(uncertainty['Z'])
		
	six_par={}
	img_data=[*eo_dict]
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

	no_obs = len(control)
	L0 = zeros((no_obs*2,1))
	image_list = []		
	control_list = []
	WW = zeros((2*no_obs,2*no_obs))
	for i in range(no_obs):
		if control[i]['img_id'] not in image_list:
			image_list.append(control[i]['img_id'])
		if control[i]['point_id'] not in control_list:
			control_list.append(control[i]['point_id'])
		
		xx1 = float(control[i]['sample'])
		yy1 = float(control[i]['line'])
		a1 = six_par[control[i]['img_id']][0]
		a2 = six_par[control[i]['img_id']][1]
		a3 = six_par[control[i]['img_id']][2]
		a4 = six_par[control[i]['img_id']][3]
		a5 = six_par[control[i]['img_id']][4]
		a6 = six_par[control[i]['img_id']][5]
		
		xx = a1*xx1+a2*yy1+a3-x_0
		yy = a4*xx1+a5*yy1+a6-y_0
		rr = np.sqrt(xx**2+yy**2)
		
		if ar==1:
	
			H = float(eo_dict[control[i]['img_id']]['ZL'])
			HH = 0.0003048*H
			hh = 0.0003048*float(control[i]['Z'])
			K= (2410*HH/(HH**2-6*HH+250)-2410*hh/(hh**2-6*hh+250)*(hh/HH))/1000000
			dr=-K*(rr+rr**3/f**2)
			xx=xx+xx/rr*dr
			yy=yy+yy/rr*dr
			
			
		L0[2*i,0] = xx
		L0[2*i+1,0] = yy
		
		WW[2*i,2*i] = 1
		WW[2*i+1,2*i+1] = 1
		
	Wxx = zeros((6*len(image_list)+3*len(control_list),6*len(image_list)+3*len(control_list))) 
	Wxyz = zeros((3*len(control_list),3*len(control_list)))
	delta = zeros((6*len(image_list)+3*len(control_list),1))
	
	for i in range(len(image_list)):
		Wxx[6*i,6*i]=(1/(3.28084*100))**2
		Wxx[6*i+1,6*i+1]=(1/(3.28084*100))**2
		Wxx[6*i+2,6*i+2]=(1/(3.28084*100))**2
		Wxx[6*i+3,6*i+3]=(1/(10/(180*np.pi)))**2
		Wxx[6*i+4,6*i+4]=(1/(10/(180*np.pi)))**2
		Wxx[6*i+5,6*i+5]=(1/(10/(180*np.pi)))**2
	 
	for i in range(len(control_list)):
		Wxyz[3*i,3*i]=(1/sig_x)**2
		Wxyz[3*i+1,3*i+1]=(1/sig_y)**2
		Wxyz[3*i+2,3*i+2]=(1/sig_z)**2

	Wxx[6*len(image_list):6*len(image_list)+3*len(control_list),6*len(image_list):6*len(image_list)+3*len(control_list)] = Wxyz
	
	for i in range(len(image_list)):
		delta[6*i,0]=float(eo_dict[image_list[i]]['XL'])
		delta[6*i+1,0]=float(eo_dict[image_list[i]]['YL'])
		delta[6*i+2,0]=float(eo_dict[image_list[i]]['ZL'])
		delta[6*i+3,0]=float(eo_dict[image_list[i]]['omega'])
		delta[6*i+4,0]=float(eo_dict[image_list[i]]['phi'])
		delta[6*i+5,0]=float(eo_dict[image_list[i]]['kappa'])

	for i in range(len(control_list)):
		delta[6*len(image_list)+3*i,0]=float(cp_dict[control_list[i]]['X'])
		delta[6*len(image_list)+3*i+1,0]=float(cp_dict[control_list[i]]['Y'])
		delta[6*len(image_list)+3*i+2,0]=float(cp_dict[control_list[i]]['Z'])
		
	last_phi = 10
	keep_going = 1
	iter = 0
	L=L0
	delta_0=delta
	while keep_going == 1:
		FF=zeros((no_obs*2,1))
		AA=zeros((no_obs*2,no_obs*2))
		BB=zeros((no_obs*2,6*len(image_list)+3*len(control_list)))
		for i in range(no_obs):
			img_ind=image_list.index(control[i]['img_id'])
			pt_ind=control_list.index(control[i]['point_id'])
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


		Q=inv(WW)
		ff=-FF-matmul(AA,(L0-L))
		Qe=matmul(matmul(AA,Q),AA.T)
		We=inv(Qe)
		
		N=matmul(matmul(BB.T,We),BB)
		t=matmul(matmul(BB.T,We),ff)
		fx=delta-delta_0
		ddel=matmul(inv(N+Wxx),(t-matmul(Wxx,fx)))
		
		v=matmul(matmul(matmul(Q,AA.T),We),(ff-matmul(BB,ddel)))
		
		vvx=fx+ddel
		phi=matmul(matmul(v.T,WW),v)+matmul(matmul(vvx.T,Wxx),vvx)
		obj=abs((last_phi-phi[0,0])/last_phi)
		print("iter nubmer: "+str(iter))
		print("objective function is : "+str(obj))
		#Convergence check
		if obj<0.0001:
			keep_going=0
			print("Converged")
			
			rmse_vx=np.sqrt(np.sum(v[0::2]**2)/len(v[0::2]))
			rmse_vy=np.sqrt(np.sum(v[1::2]**2)/len(v[1::2]))
			
			print(rmse_vx)
			print(rmse_vy)		
			
			with open(argv[5],'a') as output_file:
				csv_writer = csv.writer(output_file)
				csv_writer.writerow(["residuals"])
				csv_writer = csv.writer(output_file)
				csv_writer.writerow(["rmse x"])
				csv_writer.writerow([np.sqrt(sum(v[0::2]**2)/len(v[0::2]))])
				csv_writer.writerow(["rmse y"])
				csv_writer.writerow([np.sqrt(sum(v[1::2]**2)/len(v[1::2]))])
						
			for i in range(no_obs):
				img_ind=image_list.index(control[i]['img_id'])
				pt_ind=control_list.index(control[i]['point_id'])
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

		if keep_going==1:	
			L=L0+v
			delta=delta+ddel

		if iter>100:
			keep_going=0
			print("too many iteration")

		last_phi = phi[0,0]
		iter=iter+1
		result={}
	with open(argv[5],'a') as output_file:
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
		for i in range(no_obs):
			img_ind=image_list.index(control[i]['img_id'])
			pt_ind=control_list.index(control[i]['point_id'])
			if control[i]['img_id'] not in scale.keys():
				scale[control[i]['img_id']]=[]
			
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
			k = matmul(inv(M),np.array([[x],[y],[-f]]))/np.array([[X-XL],[Y-YL],[Z-ZL]])
			csv_writer.writerow(['k: '+str(k.T)+" for image:"+str(image_list[img_ind])+" for point:"+str(pt_ind)])
			scale_pt={}
			scale_pt[control[i]['point_id']]=np.mean(k)
			scale[control[i]['img_id']].append(scale_pt)
			
		csv_writer.writerow(["Control Points, Pass Points Coordinates"])
		result_cp={}
		for i in range(len(control_list)):
			cp_i={}
			cp_i['X']=delta[6*len(image_list)+3*i,0]
			cp_i['Y']=delta[6*len(image_list)+3*i,0]
			cp_i['Z']=delta[6*len(image_list)+3*i,0]
			result_cp[control_list[i]]=eo_i		
			csv_writer.writerow([control_list[i]])
			csv_writer.writerow(['X(ft): '+str(delta[6*len(image_list)+3*i,0])])
			csv_writer.writerow(['Y(ft): '+str(delta[6*len(image_list)+3*i+1,0])])
			csv_writer.writerow(['Z(ft): '+str(delta[6*len(image_list)+3*i+2,0])])
			csv_writer.writerow(" ")
		result['eo'] = result_eo
		result['cp'] = result_cp
		result['scale'] = scale
		output_json = json.dumps(result, indent = 4)

		json_fname = ".".join([os.path.splitext(argv[5])[0],"json"])
		with open(json_fname,'w') as output_file:
			output_file.write(output_json)



if __name__=="__main__":
	main()
