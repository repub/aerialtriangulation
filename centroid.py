# Developed by Jae Sung Kim (The Pennsylvania State University Libraries).
# Last modified: 09/01/21
# python3 centroid.py input.txt /mnt/data/input_image.tif /mnt/data/input_dem.tif output.txt
import sys, gdal, ogr, json,csv
import numpy as np
from numpy.linalg import inv
from numpy import matmul
from numpy import zeros
def main():
	argv=sys.argv
	control_path = argv[1]
	raster_path = argv[2]
	dem_path = argv[3]
	
	img = gdal.Open(raster_path)
	a0,a1,a2,b0,b1,b2 = img.GetGeoTransform()
	no_row = img.RasterYSize
	no_col = img.RasterXSize
	img_band = img.GetRasterBand(1)
	img_array = img_band.ReadAsArray()
	reader = csv.DictReader(open(control_path))
	img_data = list(reader) 
	
	dem = gdal.Open(dem_path)
	c0,c1,c2,d0,d1,d2 = dem.GetGeoTransform()
	no_row_2 = dem.RasterYSize
	no_col_2 = dem.RasterXSize
	dem_band = dem.GetRasterBand(1)
	dem_array = dem_band.ReadAsArray()
	
	
	for i in range(len(img_data)):
		
		x=float(img_data[i]['X'])
		y=float(img_data[i]['Y'])
		z=float(img_data[i]['Z'])
		sample = int((x-a0)/a1)+0.5
		line = int((b0-y)/(-b2))+0.5
		x2 = a0 + sample*a1 + line*a2
		y2 = b0 + sample*b1 + line*b2
		sl = matmul(inv(np.array([[c1, c2],[d1, d2]])),np.array([[x2-c0],[y2-d0]]))
		s = int(sl[0,0]) 
		l = int(sl[1,0])
		z2 = float(dem_array[l][s])
		img_data[i]['X']=x2
		img_data[i]['Y']=y2
		img_data[i]['Z']=z2
		print([x,x2,y,y2,z,z2])
		
	with open(argv[4], 'w', newline='') as output_file:
		csv_wr = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
		csv_wr.writeheader()
		for row in img_data:
			csv_wr.writerow(row)
			


if __name__=="__main__":
	main()
	
#The references for the GDAL script are in https://gdal.org/tutorials/raster_api_tut.html, https://gdal.org/tutorials/geotransforms_tut.html (last accessed 09/01/21)
#For the license (MIT/X) of reference,

"""In general GDAL/OGR is licensed under an MIT/X style license with the following terms:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE."""
