# Developed by Jae Sung Kim (The Pennsylvania State University Libraries).
# Last modified: 09/01/21
#python3 assign_z.py /mnt/data/test_image/control_geom_1.shp /mnt/data/test_image/dem.tif

import sys, gdal, ogr, json
import numpy as np
from numpy.linalg import inv
from numpy import matmul

def main():
	argv=sys.argv
	shp_path = argv[1]
	raster_path = argv[2]	
	
	input_shapefile = ogr.GetDriverByName("ESRI Shapefile").Open(shp_path, 1)
	dem = gdal.Open(raster_path)
	a0,a1,a2,b0,b1,b2 = dem.GetGeoTransform()
	no_row = dem.RasterYSize
	no_col = dem.RasterXSize
	dem_band = dem.GetRasterBand(1)
	dem_array = dem_band.ReadAsArray()
	vector_layer = input_shapefile.GetLayer(0)
	z_field_ft = ogr.FieldDefn("Elev_ft",ogr.OFTReal)
	vector_layer.CreateField(z_field_ft)
	
	for vector_feature in vector_layer:
		geom = vector_feature.GetGeometryRef()
		Xg, Yg = json.loads(geom.ExportToJson())['coordinates']
		sl = matmul(inv(np.array([[a1, a2],[b1, b2]])),np.array([[Xg-a0],[Yg-b0]]))

		s = int(sl[0,0]) # No need to round since we read inside pixel regardless the location of points in pixels
		l = int(sl[1,0])
		z_ft = float(dem_array[l][s])
		vector_feature.SetField("Elev_ft",z_ft)
		vector_layer.SetFeature(vector_feature)
		
	
 

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
