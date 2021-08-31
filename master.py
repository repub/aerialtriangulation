#Developed by Jae Sung Kim (Penn State University Libraries)
#Last modified 08/31/21

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

import csv, subprocess, os

def main():
	image_list = csv.DictReader(open("photo_list.txt"))
	img_data = list(image_list)
	list_len = len(img_data)
	
	for i in range(list_len-1):
		
		out_dir = "".join(['/mnt/data/output/conj/',str(i)])
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		args = ['python3','orb_aerial.py', img_data[i]['path'], img_data[i+1]['path'], '338','1707', '3000', '10', out_dir, 'photo_list.txt','40']
		print(args)
		p = subprocess.Popen(args)
		outs, errs = p.communicate()
		print(outs)
		

	for i in range(list_len-2):
		out_dir = "".join(['/mnt/data/output/conj/',str(i)])
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		args = ['python3', 'scale.py', img_data[i]['path'], img_data[i+1]['path'], img_data[i+2]['path'], '338','1707','2000', '10', out_dir,'photo_list.txt']
		print(args)
		p = subprocess.Popen(args)
		
		outs, errs = p.communicate()
		print(outs)
	
	out_dir_2 = "/mnt/data/output/bba/"
	if not os.path.exists(out_dir_2):
		os.makedirs(out_dir_2)
	args = ['python3', 'bba_minimal_nr.py', 'io_nr.txt', 'fiducial_mark.txt', 'EO.txt', 'photo_list.txt', 'bba_min_result_nr.txt','0.01',out_dir_2]
	p = subprocess.Popen(args)
	outs, errs = p.communicate()
	
	args = ['python3', 'ao_nr.py', 'io_nr.txt', 'fiducial_mark.txt', 'bba_min_result_nr.json', 'control_2.txt', 'ao_result_nr.txt','2','1']
	p = subprocess.Popen(args)
	outs, errs = p.communicate()
	
	for i in range(list_len):
		args = ['python3', 'ortho_ao.py', 'io_nr.txt', 'fiducial_mark.txt', 'bba_min_result_nr.json', 'ao_result_nr.json', img_data[i]['id'], img_data[i]['path'],  
		".".join(["_".join(["/mnt/data/output/ortho",img_data[i]['id']]),"tif"]), '/mnt/data/bba/dem_merged_2271.tif', 'dem_centre.tif','1000','1000','1000','0','EPSG:2271']
		print(args)
		p = subprocess.Popen(args)
		
		outs, errs = p.communicate()
		print(outs)
if __name__=="__main__":
	main()

