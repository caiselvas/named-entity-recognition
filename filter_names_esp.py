import pandas as pd

hombres = pd.read_excel('./data/nombres_por_edad_media.xls', header=None, sheet_name='Hombres')
mujeres = pd.read_excel('./data/nombres_por_edad_media.xls', header=None, sheet_name='Mujeres')

# Remove the first 7 rows
hombres = hombres.iloc[7:]
mujeres = mujeres.iloc[7:]

# Column names are order, name, frequency, age
hombres.columns = ['order', 'name', 'frequency', 'age']
mujeres.columns = ['order', 'name', 'frequency', 'age']

# Filter only the names with frequency greater than 500
hombres = hombres[hombres['frequency'] >= 500]
mujeres = mujeres[mujeres['frequency'] >= 500]

# Keep only the name column
hombres = hombres[['name']]
mujeres = mujeres[['name']]

# To list
hombres = hombres['name'].tolist()
mujeres = mujeres['name'].tolist()

# Remove duplicates
hombres = list(set(hombres))
mujeres = list(set(mujeres))

# Title case
hombres = [name.title() for name in hombres]
mujeres = [name.title() for name in mujeres]

# Keep only strings
hombres = [name for name in hombres if isinstance(name, str)]
mujeres = [name for name in mujeres if isinstance(name, str)]

# Save to file
print(len(hombres) + len(mujeres))

with open('./data/names500_esp.txt', 'w', encoding='utf-8') as f:
	for name in hombres:
		f.write(name + '\n')
	for name in mujeres:
		f.write(name + '\n')



