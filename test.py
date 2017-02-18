f = open("data2.json", "r")
fout = open("data2_new.json", "w")

N = 3381
arr = []

count = 0
for line in f:
	arr.append(line)
	
arr = arr[0:N]

for line in arr:
	fout.write(line)
#	fout.write("\n")

f.close()
fout.close()
