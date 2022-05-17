# from pyproj import CRS,transform, Proj
import csv
import time
from pyproj import Transformer, transform

# epsg:4548 shunyi
# epsg:4547 hengyang
transformer = Transformer.from_crs(4547, 4326)  #epsg:4548 ==> epsg:4326

x,y = 477529.86, 4451684.57
print("time0-1: {}".format(time.time()))
latitude, longitude = transformer.transform(y, x)
print("time0-2: {}".format(time.time()))
print("longitude = {0}, latitude = {1}".format(longitude, latitude))

csv_reader_data = csv.reader(open("road_cloundpoints.csv"))
csv_writer_data = csv.writer(open("road_longlats.csv", mode='w'))
counter = 0
for data_row in csv_reader_data:
    counter += 1
    if len(data_row) > 0:
        # print("{0}: {1}".format(counter, data_row))
        if counter % 100 ==0: 
            print("line: {}".format(counter))
        if counter == 1:    # first line is camera coordinate
            camera_point = data_row
            latitude, longitude = transformer.transform(camera_point[1], camera_point[0])
            csv_writer_data.writerow([longitude, latitude, camera_point[2]])
        else:               # others are road coordinates
            road_points = data_row
            road_pointsnum_row = int(len(data_row) / 3)
            # print("road_pointsnum_row = {}".format(road_pointsnum_row))
            longlat_row = []
            print_flag = False
            for p_index in range(road_pointsnum_row):
                road_point = [road_points[p_index * 3], road_points[p_index * 3 + 1], road_points[p_index * 3 + 2]]
                if float(road_point[0]) == float('inf'):
                    latitude = 100000
                    longitude = 100000
                else:
                    latitude, longitude = transformer.transform(road_point[1], road_point[0])

                if latitude < 0 or longitude < 0:
                    print("err data: pos({0}, {1}) = {2}, {3}, 3d pos = ({4}, {5}, {6}".format(p_index, counter, longitude, latitude, road_point[0], road_point[1], road_point[2]))
                    print_flag = True
                else:
                    if print_flag:
                        print("err data's next: pos({0}, {1}) = {2}, {3}, 3d pos = ({4}, {5}, {6}\n".format(p_index, counter, longitude, latitude, road_point[0], road_point[1], road_point[2]))
                        print_flag = False

                longlat_row.append(longitude)
                longlat_row.append(latitude)
            csv_writer_data.writerow(longlat_row)

