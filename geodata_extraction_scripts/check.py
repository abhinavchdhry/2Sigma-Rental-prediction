import json

f = open("location1.data", "r")

l = []
for line in f:
	_id = int(line.split(",")[0])
	l.append(_id)

l2 = []
f1 = open("data1.complete.json", "r")
for line in f1:
	d = json.loads(line)
	key = int(d.keys()[0])
	l2.append(key)

if (l == l2):
	print("Verified")
