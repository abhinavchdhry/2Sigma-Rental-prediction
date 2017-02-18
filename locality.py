import json
import urllib2
from math import radians, cos, sin, asin, sqrt

header = {
	"key" : "AIzaSyCcGLbzzfIm5USs_3GY3euooefdcs-qFGI",
#	"radius" : "1000"
	"rankby" : "distance"
}

#f = open("train.json", "r")
#data = json.load(f)
f = open("location4.data", "r")
data = []
for line in f:
	data.append([x.strip() for x in line.split(",")])

lat = []
lon = []

lat = [float(x[1]) for x in data]
lon = [float(x[2]) for x in data]

#lat = data["latitude"]
#lon = data["longitude"]

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
#	print("QUERY: " + baseURL)
	response = urllib2.urlopen(baseURL).read()
	results = json.loads(response)
	filtered = []
	if "results" in results:
		results = results["results"]
		for res in results:
			d = {}
			d["location"] = res["geometry"]["location"]
			d["name"] = res["name"]
			d["id"] = res["id"]
			if "rating" in res:
				d["rating"] = res["rating"]
			if "types" in res:
				d["types"] = res["types"]
			filtered.append(d)
	return(filtered)

content = []
count = 0
outfile = open("data4.json", "w")

for i in range(0,len(data)):
	key = data[i][0]
	print("Key: " + str(key) + "done = " + str(count) + "...")
	loc = str(lat[i]) + "," + str(lon[i])
	header["location"] = loc
	neighborhood = {}		# A directory for neighborhood information

#	print("Location: " + loc)

	nearest_transit = 50
	for _type in ["subway_station", "transit_station", "train_station", "bus_station"]:
		out = send_request(header, _type)
#		print("OutLen = " + str(len(out)))
		if len(out) > 0:
			nearest = out[0]
			current_lon = nearest["location"]["lng"]
			current_lat = nearest["location"]["lat"]
			nearest_dist = haversine(lon[i], lat[i], current_lon, current_lat)
#			print("HAVERSINE: " + str(nearest_dist))
			if nearest_dist < nearest_transit:
				nearest_transit = nearest_dist
	neighborhood["nearest_transit"] = nearest_transit if nearest_transit < 50 else None
#	print("Nearest transit: " + str(nearest_transit))

	for _place in ["shopping_mall", "university"]:
		out = send_request(header, _place)
		nearest_dist = 50
		if len(out) > 0:
			nearest = out[0]
			current_lon = nearest["location"]["lng"]
			current_lat = nearest["location"]["lat"]
			nearest_dist = haversine(lon[i], lat[i], current_lon, current_lat)
		neighborhood["nearest_" + _place] = nearest_dist if nearest_dist < 50 else None

	nearest_transit = 50
        for _type in ["convenience_store", "department_store"]:
                out = send_request(header, _type)
                if len(out) > 0:
                        nearest = out[0]
                        current_lon = nearest["location"]["lng"]
                        current_lat = nearest["location"]["lat"]
                        nearest_dist = haversine(lon[i], lat[i], current_lon, current_lat)
                        if nearest_dist < nearest_transit:
                                nearest_transit = nearest_dist
        neighborhood["nearest_store"] = nearest_transit if nearest_transit < 50 else None

#	del header["rankby"]
#	header["radius"] = "1000"
	# TODO: Now compute average rating for restaurants, clubs, bars, etc in 1 km radius
	out = send_request(header, "night_club")
	night_clubs_nearby = len(out)
        sum_of_ratings = 0.0
        clubs_with_ratings = 0
	if night_clubs_nearby > 0:
		for night_club in out:
			if "rating" in night_club:
				sum_of_ratings += night_club["rating"]
				clubs_with_ratings += 1
	avg_club_rating = None if clubs_with_ratings == 0 else (sum_of_ratings/float(clubs_with_ratings))
	neighborhood["night_clubs_nearby"] = night_clubs_nearby
	neighborhood["avg_club_rating"] = avg_club_rating
	neighborhood["clubs_with_no_rating"] = night_clubs_nearby - clubs_with_ratings
	

	barsAndRestos = send_request(header, "bar") + send_request(header, "restaurant")
	barDirectory = {}
	for bar in barsAndRestos:
		barDirectory[bar["id"]] = bar
	ids = barDirectory.keys()

#	barDirectory = [{bar["id"].encode('ascii', 'ignore'):bar} for bar in barsAndRestos]
#	ids = [bar["id"].encode('ascii', 'ignore') for bar in barsAndRestos]
#	ids = set(ids)
	neighborhood["bars_nearby"] = len(ids)
	sum_of_ratings = 0.0
	bars_with_ratings = 0
	for _id in ids:
		bar = barDirectory[_id]
		if "rating" in bar:
			bars_with_ratings += 1
			sum_of_ratings += bar["rating"]
	neighborhood["avg_bar_rating"] = None if bars_with_ratings == 0 else (sum_of_ratings/float(bars_with_ratings))
	neighborhood["bars_with_no_rating"] = len(ids) - bars_with_ratings

	# Compute distance to 3 major NY airports: JFK, LAG and NAL
	JFKloc = [40.6413, -73.7781]
	LAGloc = [40.7769, -73.8740]
	NALloc = [40.6895, -74.1745]

	neighborhood["dist_to_JFK"] = haversine(lon[i], lat[i], JFKloc[1], JFKloc[0])
	neighborhood["dist_to_LAG"] = haversine(lon[i], lat[i], LAGloc[1], LAGloc[0])
	neighborhood["dist_to_NAL"] = haversine(lon[i], lat[i], NALloc[1], NALloc[0])

#	print(neighborhood)
	content.append({ key : neighborhood })
	count += 1
	
	if count % 100 == 0:
		print("Dumping to file...")
		for entity in content:
			json.dump(entity, outfile)
			outfile.write("\n")
		content = []
		print("Done.")
