#Developed by Jae Sung Kim (Penn State University Libraries)
#Last modified: 08/31/21

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
import cv2 as cv
import matplotlib.pyplot as plt
import sys, json, os, csv

def main():
	argv = sys.argv
	img1 = cv.imread(argv[1],cv.IMREAD_GRAYSCALE)          
	img2 = cv.imread(argv[2],cv.IMREAD_GRAYSCALE) 	 
	img3 = cv.imread(argv[3],cv.IMREAD_GRAYSCALE) 	 
	
	out_dir = argv[8]
	
	top_r = int(argv[4])
	bot_r = int(argv[5])
	[row_1,col_1]=img1.shape
	row_1_t=int((row_1-bot_r-top_r)*0.8+top_r)
	row_1_b=row_1
	col_1_l=int(col_1*0.4)
	col_1_r=int(col_1*0.6)
	img_1_p = img1[row_1_t:row_1_b,col_1_l:col_1_r]

	[row_2,col_2]=img2.shape
	row_2_t=int((row_2-bot_r-top_r)*0.4+top_r)
	row_2_b=int((row_2-bot_r-top_r)*0.6+top_r)
	col_2_l=int(col_2*0.4)
	col_2_r=int(col_2*0.6)
	img_2_p = img2[row_2_t:row_2_b,col_2_l:col_2_r]

	[row_3,col_3]=img3.shape
	row_3_t=top_r
	row_3_b=int((row_3-bot_r-top_r)*0.2+top_r)
	col_3_l=int(col_3*0.4)
	col_3_r=int(col_3*0.6)
	img_3_p = img3[row_3_t:row_3_b,col_3_l:col_3_r]

	orb = cv.ORB_create(int(argv[6]))
	kp1, des1 = orb.detectAndCompute(img_1_p,None)
	kp2, des2 = orb.detectAndCompute(img_2_p,None)
	kp3, des3 = orb.detectAndCompute(img_3_p,None)
	del img1
	del img2
	del img3

	bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

	matches_1 = bf.match(des1,des2)
	matches_1 = sorted(matches_1, key = lambda x:x.distance)
	
	matches_2 = bf.match(des2,des3)
	matches_2 = sorted(matches_2, key = lambda x:x.distance)

	src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches_1 ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches_1 ]).reshape(-1,1,2)
	M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,float(argv[7]))
	matchesMask = mask.ravel().tolist()
	match_idx = [i for i, j in enumerate(matchesMask) if j == 1]
	
	out_log="/".join([out_dir,"output_sr.log"])
	with open(out_log,'w') as output_file:
		csv_writer = csv.writer(output_file)
		csv_writer.writerow(["Initial no corner"])
		csv_writer.writerow([argv[6]])
		csv_writer.writerow(["RANSAC threshold"])
		csv_writer.writerow([argv[7]])
		csv_writer.writerow(["number of corner detected first model"])
		csv_writer.writerow([str(len(match_idx))])
		
		 	
	conj_1 = []
	for match_pt in match_idx:
		
		col1=src_pts[match_pt][0][0]+col_1_l
		row1=src_pts[match_pt][0][1]+row_1_t
		col2=dst_pts[match_pt][0][0]+col_2_l
		row2=dst_pts[match_pt][0][1]+row_2_t
		conj_1.append([col1, row1, col2, row2])
		
		
	src_pts = np.float32([ kp2[m.queryIdx].pt for m in matches_2 ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp3[m.trainIdx].pt for m in matches_2 ]).reshape(-1,1,2)
	M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,float(argv[7]))
	matchesMask = mask.ravel().tolist()
	match_idx = [i for i, j in enumerate(matchesMask) if j == 1]
	with open(out_log,'w') as output_file:
		csv_writer = csv.writer(output_file)
		csv_writer.writerow(["Initial no corner"])
		csv_writer.writerow([argv[6]])
		csv_writer.writerow(["RANSAC threshold"])
		csv_writer.writerow([argv[7]])
		csv_writer.writerow(["number of corner detected second model"])
		csv_writer.writerow([str(len(match_idx))])

	conj_2 = []
	for match_pt in match_idx:
		
		col1=src_pts[match_pt][0][0]+col_2_l
		row1=src_pts[match_pt][0][1]+row_2_t
		col2=dst_pts[match_pt][0][0]+col_3_l
		row2=dst_pts[match_pt][0][1]+row_3_t
		conj_2.append([col1, row1, col2, row2])
		

	sr_list={}
	cnt=0
	for i in range(len(conj_1)):
		for j in range(len(conj_2)):
			if int(conj_1[i][2])==int(conj_2[j][0]) and int(conj_1[i][3])==int(conj_2[j][1]):
				sc_1=conj_1[i]
				sc_2=conj_2[j]
				print(sc_1)
				print(sc_2)
				sr_list[cnt]=[sc_1[0],sc_1[1],sc_1[2],sc_1[3],sc_2[2],sc_2[3]]
				cnt=cnt+1

	result={}
	image_list = csv.DictReader(open(argv[9]))
	img_data = list(image_list)
	img_id_1=list(filter(lambda x: x['path']==argv[1], img_data))
	img_id_2=list(filter(lambda x: x['path']==argv[2], img_data))
	img_id_3=list(filter(lambda x: x['path']==argv[3], img_data))
	
	result['image_1'] = img_id_1[0]['id']
	result['image_2'] = img_id_2[0]['id']
	result['image_3'] = img_id_3[0]['id']
	
	result['scale_restraint_pt'] = sr_list
	output_json = json.dumps(result, indent = 4)
	json_fname = ".".join([os.path.splitext(os.path.basename(argv[3]))[0],"json"])
	print(json_fname)
	with open(json_fname,'w') as output_file:
		output_file.write(output_json)

	image1 = cv.imread(argv[1]) # queryImage
	image2 = cv.imread(argv[2]) # trainImage
	
	image1 = cv.drawMarker(image1, (int(sc_1[0]),int(sc_1[1])), color=(255,0,0), markerType=cv.MARKER_CROSS, markerSize=300, thickness=50)
	image2 = cv.drawMarker(image2, (int(sc_1[2]),int(sc_1[3])), color=(255,0,0), markerType=cv.MARKER_CROSS, markerSize=300, thickness=50)	
	
	cv.namedWindow('result_1', cv.WINDOW_NORMAL)
	cv.resizeWindow('result_1', 900, 900)
	cv.imshow('result_1',image1)

	cv.namedWindow('result_2', cv.WINDOW_NORMAL)
	cv.resizeWindow('result_2', 900, 900)
	cv.imshow('result_2',image2)
	
	del image1
	del image2

	o_path=os.path.split(argv[8])
	out_dir_2 = os.path.join(o_path[0],str(int(o_path[1])+1))
	
	image3 = cv.imread(argv[2]) # queryImage
	image4 = cv.imread(argv[3]) # trainImage
	
	image3 = cv.drawMarker(image3, (int(sc_2[0]),int(sc_2[1])), color=(255,0,0), markerType=cv.MARKER_CROSS, markerSize=300, thickness=50)
	image4 = cv.drawMarker(image4, (int(sc_2[2]),int(sc_2[3])), color=(255,0,0), markerType=cv.MARKER_CROSS, markerSize=300, thickness=50)	
		
	cv.namedWindow('result_3', cv.WINDOW_NORMAL)
	cv.resizeWindow('result_3', 900, 900)
	cv.imshow('result_3',image3)
	
	cv.namedWindow('result_4', cv.WINDOW_NORMAL)
	cv.resizeWindow('result_4', 900, 900)
	cv.imshow('result_4',image4)

	del image3
	del image4

	if cv.waitKey(0):
		cv.destroyAllWindows()
	
if __name__=="__main__":
	main()
	

# Parts of the codes (ORB, matching, RANSAC) are from https://github.com/opencv/opencv/blob/master/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.markdown 
# and https://github.com/opencv/opencv/blob/master/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.markdown (last accessed 08/31/21)
# Following statement was added according to the condition (BSD-3) of opencv v.4.2.0

"""
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2020, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
Copyright (C) 2019-2020, Xperience AI, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage."""
