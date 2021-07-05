import nctoolkit as nc
data = nc.open_data("myfile.nc")
data.crop(lon = [x1,x2], lat = [y1, y2])
data.to_nc("crop.nc")