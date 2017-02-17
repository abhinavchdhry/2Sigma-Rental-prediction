import json
import urllib2
from math import radians, cos, sin, asin, sqrt

header = {
	"key" : "AIzaSyCcGLbzzfIm5USs_3GY3euooefdcs-qFGI",
#	"radius" : "1000"
	"rankby" : "distance"
}

f = open("train.json", "r")
data = json.load(f)

lat = data["latitude"]
lon = data["longitude"]

# Data we need:
# ** Nearest dist. to convenience store/department store
# ** Nearest transit station
# ** Transit stations within 500m
# ** Nearest shopping mall
# ** Restaurants nearby
# ** Average restaurant rating
# ** Average restaurant pricing
# ** Distance to nearest airport
# ** Party scene in neighborhood? Bars/night clubs nearby
# ** Nearest Distance to university

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

def send_request(header, _type):
	baseURL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
	header["type"] = _type
	for param in header:
		baseURL = baseURL + param + "=" + header[param] + "&"
	response = urllib2.urlopen(baseURL).read()
	results = json.loads(response)["results"]
	filtered = []
	if "results" in results:
		results = results["results"]
		for res in results:
			d = {}
			d["location"] = res["geometry"]["location"]
			d["name"] = res["name"]
			d["rating"] = res["rating"] if "rating" in res else None
			d["types"] = res["types"] if "types" in res else None
			filtered.append(d)
	return(filtered)

_place_types = ["subway_station", "transit_station", "train_station", "bus_station", "taxi_stand", "shopping_mall", "restaurant", "convenience_store", "department_store", "park", "night_club", "bar", "airport", "atm", "university"]

content = []
count = 0
outfile = open("data.json", "w")

for key in lat:
	print("Key: " + str(key) + "...")
	loc = str(lat[key]) + "," + str(lon[key])
	header["location"] = loc
	neighborhood = {}		# A directory for neighborhood information

	nearest_transit = 50
	for _type in ["subway_station", "transit_station", "train_station", "bus_station"]:
		out = send_request(header, _type)
		if len(out) > 0:
			nearest = out[0]
			current.lon = nearest["geometry"]["location"]["lng"]
			current.lat = nearest["geometry"]["location"]["lat"]
			nearest.dist = haversine(lon[key], lat[key], current.lon, current.lat)
			if nearest.dist < nearest_transit:
				nearest_transit = nearest.dist
	neighborhood["nearest_transit"] = nearest_transit if nearest_transit < 50 else None

	for _place in ["shopping_mall", "airport", "university"]:
		out = send_request(header, _place)
		nearest.dist = 50
		if len(out) > 0:
			nearest = out[0]
			current.lon = nearest["geometry"]["location"]["lng"]
			current.lat = nearest["geometry"]["location"]["lat"]
			nearest.dist = haversine(lon[key], lat[key], current.lon, current.lat)
		neighborhood["nearest_" + _place] = nearest.dist if nearest.dist < 50 else None

	nearest_transit = 50
        for _type in ["convenience_store", "department_store"]:
                out = send_request(header, _type)
                if len(out) > 0:
                        nearest = out[0]
                        current.lon = nearest["geometry"]["location"]["lng"]
                        current.lat = nearest["geometry"]["location"]["lat"]
                        nearest.dist = haversine(lon[key], lat[key], current.lon, current.lat)
                        if nearest.dist < nearest_transit:
                                nearest_transit = nearest.dist
        neighborhood["nearest_store"] = nearest_transit if nearest_transit < 50 else None

	del header["rankby"]
	header["radius"] = "1000"
	# TODO: Now compute average rating for restaurants, clubs, bars, etc in 1 km radius

	content.append({ key : neighborhood })
#	print("...Done")
	count += 1

	if count % 200 == 0:
		print("Dumping to file...")
		for entity in content:
			json.dump(entity, outfile)
			outfile.write("\n")
		content = []
		print("Done.")
