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
	top_r = int(argv[3])
	bot_r = int(argv[4])
	out_dir = argv[7]
	out_1 = "/".join([out_dir,"_".join([os.path.splitext(os.path.basename(argv[1]))[0],"conj.tif"])])
	out_2 = "/".join([out_dir,"_".join([os.path.splitext(os.path.basename(argv[2]))[0],"conj.tif"])])
	[row_1,col_1]=img1.shape
	row_1_t=int((row_1-bot_r-top_r)*0.4+top_r)
	row_1_b=row_1
	col_1_l=int(col_1/3)
	col_1_r=int(2*col_1/3)
	img_1_p_1 = img1[row_1_t:row_1_b,0:col_1_l]
	img_1_p_2 = img1[row_1_t:row_1_b,col_1_l:col_1_r]
	img_1_p_3 = img1[row_1_t:row_1_b,col_1_r:]
	[row_2,col_2]=img2.shape
	row_2_t=top_r
	row_2_b=int((row_2-bot_r-top_r)*0.6+top_r)
	col_2_l=int(col_2/2)
	col_2_r=int(2*col_2/3)
	img_2_p_1 = img2[row_2_t:row_2_b,0:col_2_l]
	img_2_p_2 = img2[row_2_t:row_2_b,col_2_l:col_2_r]
	img_2_p_3 = img2[row_2_t:row_2_b,col_2_r:]
	
	orb = cv.ORB_create(int(argv[5]))
	kp1, des1 = orb.detectAndCompute(img_1_p_1,None)
	kp2, des2 = orb.detectAndCompute(img_2_p_1,None)
	print(len(kp1))
	print(len(kp2))
	
	del img_1_p_1
	del img_2_p_1
	del img1
	del img2

	bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)

	src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
	M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,float(argv[6]))
	matchesMask = mask.ravel().tolist()
	match_idx = [i for i, j in enumerate(matchesMask) if j == 1]
	if len(match_idx)>int(argv[9]):
		match_idx=match_idx[:int(argv[9])]
	out_log="/".join([out_dir,"output.log"])
	with open(out_log,'a') as output_file:
		csv_writer = csv.writer(output_file)
		csv_writer.writerow(["Initial no corner"])
		csv_writer.writerow([argv[5]])
		csv_writer.writerow(["RANSAC threshold"])
		csv_writer.writerow([argv[6]])
		csv_writer.writerow(["number of corner detected"])
		csv_writer.writerow([str(len(match_idx))])
		
	print(len(match_idx))

	conj_list=[]
	cnt=0
	for match_pt in match_idx:
		
		col1 = src_pts[match_pt][0][0]
		row1 = src_pts[match_pt][0][1]+row_1_t	
		col2 = dst_pts[match_pt][0][0]
		row2 = dst_pts[match_pt][0][1]+row_2_t
		conj_list.append({cnt:[int(col1), int(row1), int(col2), int(row2)]})
		cnt=cnt+1
	image1 = cv.imread(argv[1])
	image2 = cv.imread(argv[2])
	
	for match_pt in match_idx:
		
		[col1,row1]=src_pts[match_pt][0]
		[col2,row2]=dst_pts[match_pt][0]
		image1 = cv.circle(image1, (int(col1),int(row1)+row_1_t), 150, (255,0,0), -1)
		image2 = cv.circle(image2, (int(col2),int(row2)+row_2_t), 150, (255,0,0), -1)	
	
	orb = cv.ORB_create(int(argv[5]))
	kp1, des1 = orb.detectAndCompute(img_1_p_2,None)
	kp2, des2 = orb.detectAndCompute(img_2_p_2,None)
	print(len(kp1))
	print(len(kp2))
	
	del img_1_p_2
	del img_2_p_2

	bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)


	src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
	M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,float(argv[6]))
	matchesMask = mask.ravel().tolist()
	match_idx = [i for i, j in enumerate(matchesMask) if j == 1]

	if len(match_idx)>int(argv[9]):
		match_idx=match_idx[:int(argv[9])]
	out_log="/".join([out_dir,"output.log"])
	with open(out_log,'a') as output_file:
		csv_writer = csv.writer(output_file)
		csv_writer.writerow(["Initial no corner"])
		csv_writer.writerow([argv[5]])
		csv_writer.writerow(["RANSAC threshold"])
		csv_writer.writerow([argv[6]])
		csv_writer.writerow(["number of corner detected"])
		csv_writer.writerow([str(len(match_idx))])
		
	print(len(match_idx))

	for match_pt in match_idx:
		
		col1 = src_pts[match_pt][0][0]+col_1_l
		row1 = src_pts[match_pt][0][1]+row_1_t	
		col2 = dst_pts[match_pt][0][0]+col_2_l
		row2 = dst_pts[match_pt][0][1]+row_2_t
		conj_list.append({cnt:[int(col1), int(row1), int(col2), int(row2)]})
		cnt=cnt+1
	
	
	for match_pt in match_idx:
		
		[col1,row1]=src_pts[match_pt][0]
		[col2,row2]=dst_pts[match_pt][0]
		image1 = cv.circle(image1, (int(col1+col_1_l),int(row1)+row_1_t), 150, (255,0,0), -1)
		image2 = cv.circle(image2, (int(col2+col_2_l),int(row2)+row_2_t), 150, (255,0,0), -1)
		
	orb = cv.ORB_create(int(argv[5]))
	kp1, des1 = orb.detectAndCompute(img_1_p_3,None)
	kp2, des2 = orb.detectAndCompute(img_2_p_3,None)
	print(len(kp1))
	print(len(kp2))
	
	del img_1_p_3
	del img_2_p_3

	bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)

	src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
	M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,float(argv[6]))
	matchesMask = mask.ravel().tolist()
	match_idx = [i for i, j in enumerate(matchesMask) if j == 1]
	if len(match_idx)>int(argv[9]):
		match_idx=match_idx[:int(argv[9])]
	out_log="/".join([out_dir,"output.log"])
	with open(out_log,'a') as output_file:
		csv_writer = csv.writer(output_file)
		csv_writer.writerow(["Initial no corner"])
		csv_writer.writerow([argv[5]])
		csv_writer.writerow(["RANSAC threshold"])
		csv_writer.writerow([argv[6]])
		csv_writer.writerow(["number of corner detected"])
		csv_writer.writerow([str(len(match_idx))])
		
	print(len(match_idx))

	for match_pt in match_idx:
		
		col1 = src_pts[match_pt][0][0]+col_1_r
		row1 = src_pts[match_pt][0][1]+row_1_t	
		col2 = dst_pts[match_pt][0][0]+col_2_r
		row2 = dst_pts[match_pt][0][1]+row_2_t
		conj_list.append({cnt:[int(col1), int(row1), int(col2), int(row2)]})
		cnt=cnt+1
		
	result={}
	image_list = csv.DictReader(open(argv[8]))
	img_data = list(image_list)
	img_id_1=list(filter(lambda x: x['path']==argv[1], img_data))
	img_id_2=list(filter(lambda x: x['path']==argv[2], img_data))
	result['image_1'] = img_id_1[0]['id']
	result['image_2'] = img_id_2[0]['id']
	result['conjugate_pt'] = conj_list
	output_json = json.dumps(result, indent = 4)

	json_fname = ".".join(["_".join([os.path.splitext(os.path.basename(argv[1]))[0],os.path.splitext(os.path.basename(argv[2]))[0]]),"json"])
	with open(json_fname,'w') as output_file:
		output_file.write(output_json)
	
	
	for match_pt in match_idx:
		
		[col1,row1]=src_pts[match_pt][0]
		[col2,row2]=dst_pts[match_pt][0]
		image1 = cv.circle(image1, (int(col1+col_1_r),int(row1)+row_1_t), 150, (255,0,0), -1)
		image2 = cv.circle(image2, (int(col2+col_2_r),int(row2)+row_2_t), 150, (255,0,0), -1)
		
	cv.namedWindow('result_3', cv.WINDOW_NORMAL)
	cv.resizeWindow('result_3', 900, 900)
	cv.imshow('result_3',image1)
	
	cv.namedWindow('result_4', cv.WINDOW_NORMAL)
	cv.resizeWindow('result_4', 900, 900)
	cv.imshow('result_4',image2)
	
	del image1
	del image2

	if cv.waitKey(0):
		cv.destroyAllWindows()

if __name__=="__main__":
	main()

# Parts of the codes (ORB, matching, RANSAC) are from https://github.com/opencv/opencv/blob/master/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.markdown (last accessed 08/31/21)
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

