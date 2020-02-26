from datetime import datetime


file = open("output.txt", "r")
for line in file:
    spl = line.split(",")
    spl = [item.strip() for item in spl]

    date = datetime.utcfromtimestamp(int(spl[0].split(".")[0])).strftime('%Y-%m-%dT%H:%M:%SZ')
    spl[0] = date
    print(spl)
