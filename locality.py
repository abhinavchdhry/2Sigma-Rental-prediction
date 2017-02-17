import json
import urllib2

header = {
	"key" : "AIzaSyCcGLbzzfIm5USs_3GY3euooefdcs-qFGI",
	"radius" : "1000"
}

f = open("train.json", "r")
data = json.load(f)

lat = data["latitude"]
lon = data["longitude"]

def send_request(header, _type):
	baseURL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
	header["type"] = _type
	for param in header:
		baseURL = baseURL + param + "=" + header[param] + "&"
	response = urllib2.urlopen(baseURL).read()
	results = json.loads(response)["results"]
	filtered = []
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
	_place_listings = {}
	for _type in _place_types:
		print(_type)
		out = send_request(header, _type)
		_place_listings[_type] = out
	content.append({ key : _place_listings })
	print("...Done")
	
	if count % 500 == 0:
		print("Dumping to file...")
		for entity in content:
			json.dump(entity, outfile)
			outfile.write("\n")
		content = []
		print("Done.")
