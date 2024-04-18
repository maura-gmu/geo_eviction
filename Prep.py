# Installation and Setup
import warnings

warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point

# DC ward data
ward_sf = gpd.read_file('data/Wards_by_Population_in_2002_nlogo.shp')
ward_sf = ward_sf.to_crs('EPSG:4326')

ward_sf.unique_id = ward_sf.WARD_1 
(ward_sf.columns)
ward_sf = ward_sf[['WARD_1', 'geometry', 'SHAPE_Area']]
len(ward_sf)

## Filtered and Imputed Household Pulse Survey data
filter_df = pd.read_csv('data/df_filtered_MARCH2024.csv')
## Properties
prop_df = pd.read_csv('data/DC_lots_merged.csv')
prop_df['Units'].fillna(1, inplace=True)
prop_df['Units'] = prop_df['Units'].astype(int)
prop_df['vacancies'] = prop_df['Units'].astype(int)

filter_df.columns

# HPS income brackets
income_table = {
    1: [0, 24999],
    2: [25000, 34999],
    3: [35000, 49999],
    4: [50000, 74999],
    5: [75000, 99999],
    6: [100000, 149999],
    7: [150000, 199999],
    8: [200000, float('inf')]  # 200,000+
}


# Calculate monthly income and budget
filter_df['monthly_income'] = filter_df['income'] / 12 
filter_df['homeless'] = False
filter_df['evicted'] = False

# unemployed == 1 is true
filter_df['original_income'] = filter_df['monthly_income']
filter_df.loc[filter_df['unemployed'] == 1, 'monthly_income'] = 0.001

# Set housing_cost to 0 for those who paid off their homes or don't pay rent
filter_df.loc[~filter_df['housing_status'].isin([2, 3]), 'housing_cost'] = 0

# Set wealth to 0, debt to housing_cost, and strikes to 1 for those who owe backrent or backmortgage
# owe_backrent and owe_backmortgage: 2 is true, 1 is false
filter_df['wealth'] = np.where((filter_df['owe_backrent'] == 2) | (filter_df['owe_backmortgage'] == 2), 0, filter_df['monthly_income'])
filter_df['debt'] = np.where((filter_df['owe_backrent'] == 2) | (filter_df['owe_backmortgage'] == 2), filter_df['housing_cost'], 0)
filter_df['strikes'] = np.where((filter_df['owe_backrent'] == 2) | (filter_df['owe_backmortgage'] == 2), 1, 0)
filter_df['credit'] = 0 
filter_df['search'] = 0

init_num_households = len(filter_df)
init_num_properties = len(prop_df)
init_num_units = sum(prop_df['Units'])
init_num_vacancies = sum(prop_df['Units'])
#print(init_num_households, init_num_properties, init_num_units, init_num_vacancies)
filter_df['rounded_weight' ] = filter_df['weight'].round().astype(int)
households_df = filter_df.loc[np.repeat(filter_df.index.values, filter_df['rounded_weight'])]

# Reset index once household weights are applied
households_df.reset_index(drop=True, inplace=True)
households_df['unique_id'] = households_df.index.astype(str) + '_household'
households_df['rent'] = (households_df['housing_status'] == 3) | (households_df['housing_status'] == 4)
households_df.drop(columns=['rounded_weight'], inplace=True)
#print(len(households_df), init_num_properties, init_num_units, init_num_vacancies)

renter_proptypes = [
    "Residential-Multifamily (Misce",
    "Mixed Use",
    "Residential-Transient (Misce",
    "Residential-Conversion (5 Unit",
    "Residential-Conversion (Less T",
    "Residential-Conversion (More T",
    "Residential-Apartment (Walkup)",
    "Residential-Apartment (Elevato)",
]

owner_proptypes = [ 
    "Residential-Single Family (Det",
    "Residential-Single Family (Mis",
    "Residential-Single Family (NC",
    "Residential-Single Family (Row",
    "Residential-Single Family (Sem",
    "Residential-Cooperative (Horiz", 
    "Residential-Cooperative (Verti", 
    "Residential-Flats (Less Than 5", 
    "Residential-Condominium (Horiz", 
]

sfr = [
    "Residential-Single Family (Det",
    "Residential-Single Family (Mis",
    "Residential-Single Family (NC",
    "Residential-Single Family (Row",
    "Residential-Single Family (Sem",
]

condo_coop = [
    "Residential-Condominium (Horiz", 
    "Residential-Cooperative (Horiz", 
    "Residential-Cooperative (Verti", 
]

rental_res = [
    "Residential-Multifamily (Misce",
    "Mixed Use",
    "Residential-Transient (Misce",
    "Residential-Conversion (5 Unit",
    "Residential-Conversion (Less T",
    "Residential-Conversion (More T",
    "Residential-Apartment (Walkup)",
    "Residential-Apartment (Elevato)",
]

crs = "EPSG:4326"


#Create geopandas dataframe from properties df
patches = gpd.GeoDataFrame(
    prop_df,
    geometry=gpd.points_from_xy(x=prop_df.long, y=prop_df.lat),
    crs = crs,
)
# There are duplicate OBJECTIDs so we will use the index as the unique id
patches['unique_id'] = patches.index.astype(str) + '_property'

patches.loc[patches['Units'] == 0, 'Units'] = 1
patches.loc[patches['PROPTYPE'].isin(renter_proptypes), 'rent'] = True  #used for merging later
patches.loc[patches['PROPTYPE'].isin(owner_proptypes), 'rent'] = False  # used for merging later

# Create geopandas dataframe from households df
# Set geometry to be center of the DC wards space
center = Point(-8573211.872777091, 4706482.125258839)
households_df['geometry'] = center
households_gdf = gpd.GeoDataFrame(
            households_df,
            geometry = 'geometry',
            crs = crs
            )

# ward-based unemployment table 
unemployment_table = {
    #    B1 change   B0 intercept
    1: [-0.2464, 7.1286],   # R² = 0.6864, B1 is change in percentage each month
    2: [-0.1357, 4.5714],   # R² = 0.3156
    3: [-0.0964, 4.2429],   # R² = 0.2212
    4: [-0.4179, 11.057],   # R² = 0.8388
    5: [-0.3036, 11.671],   # R² = 0.6621
    6: [-0.1321, 7.6],      # R² = 0.227
    7: [-0.2857, 14.886],   # R² = 0.5319
    8: [-0.4071, 19.514]    # R² = 0.6182
}

# zipcode-based housing price changes
zip_housing_prices = {
    20032: [1, 1, 4383.5, 374552, -2332, 195178],
    20019: [35.257, 1750.5, 4564.1, 387280, 2375.3, 180570],
    20020: [1, 1, 4501.9, 414048, 2107.8, 153003],
    20004: [-105.3, 2577.8, 721.54, 502893, -58.179, 488166],
    20036: [1, 1, -889.04, 556837, 44.893, 392832],
    20017: [-75.71, 2256.8, 3191.2, 624241, -984.86, 375456],
    20018: [1, 1, 4773.4, 624419, 2251.1, 358541],
    20005: [-70.82, 2205.5, -280.29, 652685, 1038.8, 518039],
    20024: [-58.53, 2170.3, 1137.7, 721606, -583.21, 439986],
    20011: [-9.498, 2045.6, 5693.5, 721868, 280.57, 416499],
    20002: [-43.02, 2011.1, 3581.9, 779966, 1234.6, 497272],
    20012: [1, 1, 11952, 785645, 1991.9, 446918],
    20010: [-28.58, 2114.7, 5356.4, 816199, 733.18, 517056],
    20001: [-48.54, 2338.5, 2453.6, 844599, 1209, 568671],
    20009: [-44.35, 2291.2, 1451.9, 908303, 845.82, 527844],
    20003: [-75.41, 2232.6, 3285.4, 909852, 1832.6, 542341],
    20037: [2.6782, 2322.9, 974.36, 921650, -710.61, 573140],
    20015: [-44.34, 1834, 16675, 1E+06, -403.79, 365482],
    20016: [4.2909, 1901.8, 13249, 1E+06, 1208.9, 428127],
    20007: [1, 1, 7539.8, 1E+06, 1445, 443493],
    20052: [1, 1, 1847.3, 1E+06, 1, 1],
    20008: [-46.65, 1979.1, 15262, 2E+06, 1412.4, 455688],
    20006: [28.939, 2604.5, 1, 1, -467.21, 256691]
}
