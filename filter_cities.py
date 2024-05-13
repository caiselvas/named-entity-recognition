import pandas as pd

data = pd.read_csv('./data/fullcities15000.txt', sep='\t', header=None, encoding='utf-8')
# geonameid         : integer id of record in geonames database
# name              : name of geographical point (utf8) varchar(200)
# asciiname         : name of geographical point in plain ascii characters, varchar(200)
# alternatenames    : alternatenames, comma separated, ascii names automatically transliterated, convenience attribute from alternatename table, varchar(10000)
# latitude          : latitude in decimal degrees (wgs84)
# longitude         : longitude in decimal degrees (wgs84)
# feature class     : see http://www.geonames.org/export/codes.html, char(1)
# feature code      : see http://www.geonames.org/export/codes.html, varchar(10)
# country code      : ISO-3166 2-letter country code, 2 characters
# cc2               : alternate country codes, comma separated, ISO-3166 2-letter country code, 200 characters
# admin1 code       : fipscode (subject to change to iso code), see exceptions below, see file admin1Codes.txt for display names of this code; varchar(20)
# admin2 code       : code for the second administrative division, a county in the US, see file admin2Codes.txt; varchar(80) 
# admin3 code       : code for third level administrative division, varchar(20)
# admin4 code       : code for fourth level administrative division, varchar(20)
# population        : bigint (8 byte int) 
# elevation         : in meters, integer
# dem               : digital elevation model, srtm3 or gtopo30, average elevation of 3''x3'' (ca 90mx90m) or 30''x30'' (ca 900mx900m) area in meters, integer. srtm processed by cgiar/ciat.
# timezone          : the iana timezone id (see file timeZone.txt) varchar(40)
# modification date : date of last modification in yyyy-MM-dd format

data.columns = ['geonameid', 'name', 'asciiname', 'alternatenames', 'latitude', 'longitude', 'feature class', 'feature code', 'country code', 'cc2', 'admin1 code', 'admin2 code', 'admin3 code', 'admin4 code', 'population', 'elevation', 'dem', 'timezone', 'modification date']

print(data['country code'].unique())

# Filter out only the cities with country code AD, AR, AT, BE, CA, CH, CL, DE, DK, CU, EC, ES, FR, GB, IE, IT, MC, NL, PT, US
data = data[data['country code'].isin(['AD', 'AR', 'AT', 'BE', 'CA', 'CH', 'CL', 'DE', 'DK', 'CU', 'EC', 'ES', 'FR', 'GB', 'IE', 'IT', 'MC', 'NL', 'PT', 'US', 'CU', 'CZ'])]

# Filter only where the population is greater than 50000
data = data[data['population'] >= 50000]

# Keep only the name column
data = data[['name']]

# To list
data = data['name'].tolist()

# Remove duplicates
data = list(set(data))

# Save to file
print(len(data))

with open('./data/cities50000.txt', 'w', encoding='utf-8') as f:
	for city in data:
		f.write(city + '\n')