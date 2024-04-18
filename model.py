# Mesa and Mesa Geo
import mesa
import mesa_geo as mg
import numpy as np
from shapely.geometry import Point
from model_prep import ward_sf, patches, households_gdf, renter_proptypes, owner_proptypes, unemployment_table, zip_housing_prices, rental_res, condo_coop, sfr



# Math and Data Cleaning/Wrangling
import math
import random
from typing import Dict



"""# Eviction Model"""

class WardAgent(mg.GeoAgent):
    """
    Agent representing a ward in the eviction model. It has fixed location and geometry.

    Attributes:
        num_properties (int): Indicates number of properties in ward.
    """

    def __init__(self, unique_id, model, geometry, crs):
        super().__init__(unique_id, model, geometry, crs)
        self.unemployment_rate = 0.0
        self.num_parcels = 0
        self.mean_housing = 0
        #print(f'Ward {self.unique_id} starts with mean housing costs of {self.mean_housing}') # good


class PropAgent(mg.GeoAgent):
    """An agent representing a property parcel with fixed location and geometry."""

    def __init__(self, unique_id, model, geometry, crs):
        super().__init__(unique_id, model, geometry, crs)
        self.value = 0


class HouseholdAgent(mg.GeoAgent):
    """An agent representing a household respondent from the HPS."""

    def __init__(self, unique_id, model, geometry, crs):

        # Pass the parameters to the parent class
        super().__init__(unique_id, model, geometry, crs)


class EvictionSpace(mg.GeoSpace):
    """
      Custom GeoSpace class representing DC wards.

      Attributes:
        _id_ward_map (Dict[str, WardAgent]): A dictionary mapping ward IDs to WardAgent instances.
        num_households (int): Total number of households in the DC wards.
    """
    _id_ward_map = Dict[str, WardAgent]
    num_households = int
    num_parcels = int


    def __init__(self):
        """
        Initialize Space.

        Initializes _id_property_map as an empty dictionary, with num_households and num_properties == 0.
        """
        self.crs = "EPSG:4326"
        super().__init__(warn_crs_conversion=False)
        self.num_households = 0
        self.num_parcels = 0
        self.all_housing_values = 0

    def add_wards(self, wards):
      """
      Add ward agents to the Space object.

      Args:
        wards (list): List of WardAgent instances to be added.

      Notes:
        Calculates the total area of all wards and updates the SHAPE_AREA of each ward relative to the total area.
      """
      super().add_agents(wards)

      
    def add_parcels_to_ward(self, parcels, ward):
        """
        Add parcels agents to the Space object.

        Args:
            parcels (list): List of PropAgent instances to be added.
            ward (int): The ward that parcels are being added to.

        Notes:

        """

        self.num_parcels += len(parcels) # space-level parcel count
        ward.num_parcels = len(parcels) # ward-level parcel count

        super().add_agents(parcels)

    def add_households_to_parcel(self, households, parcel):
        """
        Add household agents to the Space object.

        Args:
            households (list): List of HouseholdAgent instances to be added.
            parcel (int): Parcel the households are joining.

        """
        super().add_agents(households)



class EvictionModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, moratorium = True, perc_savings = 3, permanent_stimulus = False, release_stimulus = False, stimulus_value = 500, initial_occupancy_rate = 80, prop_count = 100, moratorium_expiration_week = 3, leniency = 3, housing_perc_income = 30, other_weekly_costs = 10):
        
        super().__init__()
        self.month = 0
        self.landlord_income = 0
        self.schedule = mesa.time.RandomActivation(self)
        self.space = EvictionSpace()
        #print(f"self.space.crs is {self.space.crs} and total bounds are {self.space.total_bounds}")
        self.leniency = leniency # scale
        self.moratorium = moratorium # switch
        self.housing_perc_income = housing_perc_income # scale
        self.other_weekly_costs = other_weekly_costs # scale
        self.moratorium_expiration_week = moratorium_expiration_week # scale
        self.perc_homeless = 0
        self.missed_payments = 0
        self.running = True
        self.prop_count = prop_count
        self.initial_occupancy_rate = initial_occupancy_rate
        self.wh_blk_dissim = 0
        self.wh_asn_dissim = 0
        self.wh_oth_dissim = 0
        self.blk_asn_dissim = 0
        self.blk_oth_dissim = 0
        self.asn_oth_dissim = 0
        self.lat_dissim = 0
        self.stimulus_value = stimulus_value
        self.release_stimulus = release_stimulus
        self.permanent_stimulus = permanent_stimulus
        self.perc_savings = perc_savings

        
        wards = mg.AgentCreator(WardAgent, model=self).from_GeoDataFrame(
            gdf=ward_sf,
            unique_id="WARD_1",
            set_attributes=True
            )
        self.space.add_agents(wards)

        # Shuffle the index of the patches and households GDFs
        #random_state = 4
        #shuffled_patches = patches.sample(frac=1, random_state=random_state)   
        #shuffled_households = households_gdf.sample(frac=1, random_state=random_state)
        
        shuffled_patches = patches.sample(frac=1)
        parcels = mg.AgentCreator(PropAgent, model=self).from_GeoDataFrame(
            gdf=shuffled_patches.head(prop_count), # take first x rows
            set_attributes=True,
            )
        
        # Filter parcels belonging to each ward
        ward_parcels = {ward.unique_id: [] for ward in wards}
   
        compatible_parcels_by_rent = {}

        for parcel in parcels:
            ward_parcels[parcel.WARD_1].append(parcel)
            
            # Assign households by whether they rent or own
            compatible_parcels_by_rent.setdefault(parcel.rent, []).append(parcel)

    
        
        # How many units are available
        unit_count = sum([parcel.Units for parcel in parcels])
        
        
        
        assigned_households = []
        total_assigned_units = 0
        
        shuffled_households = households_gdf.sample(frac=1)
        for household in mg.AgentCreator(HouseholdAgent, model=self).from_GeoDataFrame(
                #gdf=shuffled_households.head(prop_count),  # to improve efficiency, but makes property values worse
                gdf=shuffled_households,  
                set_attributes=True,
                #agent_kwargs={"x": x},
                ):
            
            # Check if there are compatible parcels for this household's rent
            compatible_parcels = compatible_parcels_by_rent.get(household.rent)
            if compatible_parcels:
                # Find a random compatible parcel for the household with vacancies
                selected_parcel = next((parcel for parcel in compatible_parcels if parcel.vacancies > 0), None)
                if selected_parcel:
                # Assign the household to the selected parcel
                
                    # Add initial savings to wealth (base wealth assumed to be one month's worth of income)
                    household.wealth += (self.perc_savings * household.original_income)
                    household.budget = household.original_income * (self.housing_perc_income / 100)
                    household.housing_cost = household.budget if household.housing_status in [2, 3] else 0 
                    household.credit_limit = household.housing_cost if household.use_credit_cards == 1 else 0
                    household.parcel_id = selected_parcel.unique_id
                    household.geometry = selected_parcel.geometry
                    household.ward_id = selected_parcel.WARD_1
                    household.zipcode = selected_parcel.ZIPCODE
                    self.space.add_agents(household)
                    self.schedule.add(household)
                    assigned_households.append(household)
                    selected_parcel.vacancies -= 1
                    
                    total_assigned_units += 1
                
                    # Once initial occupancy rate is met, break
                    if total_assigned_units == unit_count:
                        break

        
        households = [ agent for agent in self.schedule.agents if isinstance(agent, HouseholdAgent) ]            
        valued_parcels = []
        for parcel in parcels:
            # Initially you need all households to help set the values of the properties
            self.schedule.add(parcel)
            parcel_households = [household for household in assigned_households if household.parcel_id == parcel.unique_id]
            
            if parcel_households:
                parcel.value = np.mean([household.budget for household in parcel_households if household.budget > 1])
                valued_parcels.append(parcel)
        
        # After values are set, remove households to meet occupancy rate
    
        for parcel in valued_parcels:
            parcel_households = [household for household in assigned_households if household.parcel_id == parcel.unique_id]
            for unit in range(parcel.Units):
                if random.random() > initial_occupancy_rate / 100:
                    if parcel_households:
                        selected_household = random.choice(parcel_households)
                        parcel_households.remove(selected_household)
                        parcel.vacancies += 1
                        self.schedule.remove(selected_household)
                        self.space.remove_agent(selected_household)
                        if not parcel_households:
                            break
        empty_parcels = [parcel for parcel in parcels if parcel.value <= 1]                    
                
        # Set value of parcels occupied by non-payers
        for parcel in empty_parcels:
            other_parcels = [ other_parcel for other_parcel in valued_parcels if other_parcel.unique_id != parcel.unique_id and other_parcel.WARD_1 == parcel.WARD_1]
            if other_parcels:
                parcel.value = np.mean([other_parcel.value for other_parcel in other_parcels])
                valued_parcels.append(parcel)
            else:
                parcel.value = np.mean([parcel.value for parcel in valued_parcels])
                    
                    

        demographic_counts = {}
        total_wh = sum(1 for household in households if household.race == 1)
        total_blk = sum(1 for household in households if household.race == 2)
        total_asn = sum(1 for household in households if household.race == 3)
        total_oth = sum(1 for household in households if household.race == 4)
        total_Latino = sum(1 for household in households if household.latino == 2)
        total_nonLatino = sum(1 for household in households if household.latino == 1)

        for ward in wards:
            ## Add parcels to the ward in the Space Object
            self.space.add_parcels_to_ward(ward_parcels[ward.unique_id], ward)
            # Update num_properties and add parcels to the respective ward

            ward_households = [household for household in assigned_households if household.ward_id == ward.unique_id]
            ward.num_households = len(ward_households)
            ward.mean_housing = np.mean([household.budget for household in ward_households])
            #print(f'There are {ward.num_households} households in ward {ward.unique_id}')
            #print(f'The mean housing cost for ward {ward.unique_id} is {ward.mean_housing}')
            wh_count = sum(1 for household in ward_households if household.race == 1)
            blk_count = sum(1 for household in ward_households if household.race == 2)
            asn_count = sum(1 for household in ward_households if household.race == 3)
            oth_count = sum(1 for household in ward_households if household.race == 4)
            
            Lat_count = sum(1 for household in ward_households if household.latino == 2)
            nonLat_count = sum(1 for household in ward_households if household.latino == 1)
            
            ward_counts = {
                "White" : wh_count,
                "Black" : blk_count,
                "Asian" : asn_count,
                "Other" : oth_count,
                "Latino" : Lat_count,
                "Non Latino" : nonLat_count
                }
        
            demographic_counts[ward.unique_id] = ward_counts
            self.schedule.add(ward)

        
        if total_wh > 0 and total_blk > 0:
            self.wh_blk_dissim = .5 * (
            abs( (demographic_counts[1]['White'] / total_wh) - (demographic_counts[1]['Black'] / total_blk) ) +
            abs( (demographic_counts[2]['White'] / total_wh) - (demographic_counts[2]['Black'] / total_blk) ) +
            abs( (demographic_counts[3]['White'] / total_wh) - (demographic_counts[3]['Black'] / total_blk) ) +
            abs( (demographic_counts[4]['White'] / total_wh) - (demographic_counts[4]['Black'] / total_blk) ) +
            abs( (demographic_counts[5]['White'] / total_wh) - (demographic_counts[5]['Black'] / total_blk) ) +
            abs( (demographic_counts[6]['White'] / total_wh) - (demographic_counts[6]['Black'] / total_blk) ) +
            abs( (demographic_counts[7]['White'] / total_wh) - (demographic_counts[7]['Black'] / total_blk) ) +
            abs( (demographic_counts[8]['White'] / total_wh) - (demographic_counts[8]['Black'] / total_blk) ) 
            )
            
        if total_wh> 0 and total_asn > 0:
            self.wh_asn_dissim = .5 * (
            abs( (demographic_counts[1]['White'] / total_wh) - (demographic_counts[1]['Asian'] / total_asn) ) +
            abs( (demographic_counts[2]['White'] / total_wh) - (demographic_counts[2]['Asian'] / total_asn) ) +
            abs( (demographic_counts[3]['White'] / total_wh) - (demographic_counts[3]['Asian'] / total_asn) ) +
            abs( (demographic_counts[4]['White'] / total_wh) - (demographic_counts[4]['Asian'] / total_asn) ) +
            abs( (demographic_counts[5]['White'] / total_wh) - (demographic_counts[5]['Asian'] / total_asn) ) +
            abs( (demographic_counts[6]['White'] / total_wh) - (demographic_counts[6]['Asian'] / total_asn) ) +
            abs( (demographic_counts[7]['White'] / total_wh) - (demographic_counts[7]['Asian'] / total_asn) ) +
            abs( (demographic_counts[8]['White'] / total_wh) - (demographic_counts[8]['Asian'] / total_asn) ) 
            )
            
        if total_wh> 0 and total_oth > 0:
            self.wh_oth_dissim = .5 * (
            abs( (demographic_counts[1]['White'] / total_wh) - (demographic_counts[1]['Other'] / total_oth) ) +
            abs( (demographic_counts[2]['White'] / total_wh) - (demographic_counts[2]['Other'] / total_oth) ) +
            abs( (demographic_counts[3]['White'] / total_wh) - (demographic_counts[3]['Other'] / total_oth) ) +
            abs( (demographic_counts[4]['White'] / total_wh) - (demographic_counts[4]['Other'] / total_oth) ) +
            abs( (demographic_counts[5]['White'] / total_wh) - (demographic_counts[5]['Other'] / total_oth) ) +
            abs( (demographic_counts[6]['White'] / total_wh) - (demographic_counts[6]['Other'] / total_oth) ) +
            abs( (demographic_counts[7]['White'] / total_wh) - (demographic_counts[7]['Other'] / total_oth) ) +
            abs( (demographic_counts[8]['White'] / total_wh) - (demographic_counts[8]['Other'] / total_oth) ) 
            )
                
        if total_blk > 0 and total_asn > 0:
            self.blk_asn_dissim = .5 * (
            abs( (demographic_counts[1]['Black'] / total_blk) - (demographic_counts[1]['Asian'] / total_asn) ) +
            abs( (demographic_counts[2]['Black'] / total_blk) - (demographic_counts[2]['Asian'] / total_asn) ) +
            abs( (demographic_counts[3]['Black'] / total_blk) - (demographic_counts[3]['Asian'] / total_asn) ) +
            abs( (demographic_counts[4]['Black'] / total_blk) - (demographic_counts[4]['Asian'] / total_asn) ) +
            abs( (demographic_counts[5]['Black'] / total_blk) - (demographic_counts[5]['Asian'] / total_asn) ) +
            abs( (demographic_counts[6]['Black'] / total_blk) - (demographic_counts[6]['Asian'] / total_asn) ) +
            abs( (demographic_counts[7]['Black'] / total_blk) - (demographic_counts[7]['Asian'] / total_asn) ) +
            abs( (demographic_counts[8]['Black'] / total_blk) - (demographic_counts[8]['Asian'] / total_asn) ) 
            )
                
        if total_blk > 0 and total_oth > 0:
            self.blk_oth_dissim = .5 * (
            abs( (demographic_counts[1]['Black'] / total_blk) - (demographic_counts[1]['Other'] / total_blk) ) +
            abs( (demographic_counts[2]['Black'] / total_blk) - (demographic_counts[2]['Other'] / total_blk) ) +
            abs( (demographic_counts[3]['Black'] / total_blk) - (demographic_counts[3]['Other'] / total_blk) ) +
            abs( (demographic_counts[4]['Black'] / total_blk) - (demographic_counts[4]['Other'] / total_blk) ) +
            abs( (demographic_counts[5]['Black'] / total_blk) - (demographic_counts[5]['Other'] / total_blk) ) +
            abs( (demographic_counts[6]['Black'] / total_blk) - (demographic_counts[6]['Other'] / total_blk) ) +
            abs( (demographic_counts[7]['Black'] / total_blk) - (demographic_counts[7]['Other'] / total_blk) ) +
            abs( (demographic_counts[8]['Black'] / total_blk) - (demographic_counts[8]['Other'] / total_blk) ) 
            )
                
        if total_asn > 0 and total_oth > 0:
            self.asn_oth_dissim = .5 * (
            abs( (demographic_counts[1]['Asian'] / total_asn) - (demographic_counts[1]['Other'] / total_oth) ) +
            abs( (demographic_counts[2]['Asian'] / total_asn) - (demographic_counts[2]['Other'] / total_oth) ) +
            abs( (demographic_counts[3]['Asian'] / total_asn) - (demographic_counts[3]['Other'] / total_oth) ) +
            abs( (demographic_counts[4]['Asian'] / total_asn) - (demographic_counts[4]['Other'] / total_oth) ) +
            abs( (demographic_counts[5]['Asian'] / total_asn) - (demographic_counts[5]['Other'] / total_oth) ) +
            abs( (demographic_counts[6]['Asian'] / total_asn) - (demographic_counts[6]['Other'] / total_oth) ) +
            abs( (demographic_counts[7]['Asian'] / total_asn) - (demographic_counts[7]['Other'] / total_oth) ) +
            abs( (demographic_counts[8]['Asian'] / total_asn) - (demographic_counts[8]['Other'] / total_oth) ) 
            )
                
        if total_Latino > 0 and total_nonLatino > 0:
            self.lat_dissim = .5 * (
            abs( (demographic_counts[1]['Latino'] / total_Latino) - (demographic_counts[1]['Non Latino'] / total_nonLatino) ) +
            abs( (demographic_counts[2]['Latino'] / total_Latino) - (demographic_counts[2]['Non Latino'] / total_nonLatino) ) +
            abs( (demographic_counts[3]['Latino'] / total_Latino) - (demographic_counts[3]['Non Latino'] / total_nonLatino) ) +
            abs( (demographic_counts[4]['Latino'] / total_Latino) - (demographic_counts[4]['Non Latino'] / total_nonLatino) ) +
            abs( (demographic_counts[5]['Latino'] / total_Latino) - (demographic_counts[5]['Non Latino'] / total_nonLatino) ) +
            abs( (demographic_counts[6]['Latino'] / total_Latino) - (demographic_counts[6]['Non Latino'] / total_nonLatino) ) +
            abs( (demographic_counts[7]['Latino'] / total_Latino) - (demographic_counts[7]['Non Latino'] / total_nonLatino) ) +
            abs( (demographic_counts[8]['Latino'] / total_Latino) - (demographic_counts[8]['Non Latino'] / total_nonLatino) ) 
            )
                
            
        
        self.datacollector = mesa.DataCollector(
            # Model-level reporters
            {"Current month" : "month", 
             "Landlord's income" : "landlord_income",
             "Missing Payments" : "missed_payments",
             "% DC Households now Homeless" : "perc_homeless",
             "white-black dissimilarity index" : "white_black_dissimilarity",
             "white_asian_dissimilarity index" : "white_asian_dissimilarity",
             "white_other_dissimilarity index" : "white_other_dissimilarity",
             "black_asian_dissimilarity index" : "black_asian_dissimilarity",
             "black_other_dissimilarity index" : "black_other_dissimilarity",
             "asian_other_dissimilarity index" : "asian_other_dissimilarity",
             "latino_dissimilarity index " : "latino_dissimilarity",
             "Household Location": lambda agent: agent.geometry if isinstance(agent, HouseholdAgent) else None,
             "Household Ward": lambda agent: agent.ward_id if isinstance(agent, HouseholdAgent) else None,
             "Household Race": lambda agent: agent.race if isinstance(agent, HouseholdAgent) else None,
             "Household Latino" : lambda agent: agent.Latino if isinstance(agent, HouseholdAgent) else None,
             "Ward Unemployment_Rate": lambda agent: agent.unemployment_rate if isinstance(agent, WardAgent) else None,
             "Ward Num Properties": lambda agent: agent.num_parcels if isinstance(agent, WardAgent) else None,
             "Ward Mean Housing" : lambda agent: agent.mean_housing if isinstance(agent, WardAgent) else None,
             "Property Value": lambda agent: agent.value if isinstance(agent, PropAgent) else None,
            }
            )
    
    @property
    def white_black_dissimilarity(self):
        return self.wh_blk_dissim
    
    @property
    def white_asian_dissimilarity(self):
        return self.wh_asn_dissim
    
    @property
    def white_other_dissimilarity(self):
        return self.wh_oth_dissim
    
    @property
    def black_asian_dissimilarity(self):
        return self.blk_asn_dissim
    
    @property
    def black_other_dissimilarity(self):
        return self.blk_oth_dissim
    
    @property
    def asian_other_dissimilarity(self):
        return self.asn_oth_dissim
    
    @property
    def latino_dissimilarity(self):
        return self.lat_dissim
    
    
    @property
    def homeless(self):
      return self.perc_homeless
    
    @property
    def show_month(self):
      return self.month
        
    @property 
    def income(self):
        #self.landlord_expenses = 0
        return self.landlord_income  
    
    @property 
    def missing_payments(self):
        #self.landlord_expenses = 0
        return self.missed_payments  

    def step(self):
        '''
        Execute one step of the model.

        '''
        # Execute one step
        self.schedule.step()
        
        if self.moratorium:
            if self.moratorium_expiration_week >= (self.schedule.steps / 2):
                self.moratorium = False
        
        # Increment the month counter every two steps
        if self.schedule.steps % 2 == 0:
            self.month += 1
            # For each agent, set their unemployment rate for the current month

        [ self.update_rent(agent) for agent in self.schedule.agents if isinstance(agent, PropAgent) ]
        [ self.unemploy(agent) for agent in self.schedule.agents if isinstance(agent, WardAgent) ]
        
        [ self.move(agent) for agent in self.schedule.agents if isinstance(agent, HouseholdAgent) ]
        [ self.earn_income(agent) for agent in self.schedule.agents if isinstance(agent, HouseholdAgent) and self.schedule.steps % 2 == 0 ]
        [ self.pay_rent(agent) for agent in self.schedule.agents if isinstance(agent, HouseholdAgent) ]
        num_homeless = sum(1 for agent in self.schedule.agents if isinstance(agent, HouseholdAgent) and agent.homeless)
        
        total_households = sum(1 for agent in self.schedule.agents if isinstance(agent, HouseholdAgent))
        self.perc_homeless = (num_homeless / total_households) * 100
        
        
        # Collect data
        self.datacollector.collect(self)
    
        
        self.release_stimulus = False
        # Stop running if all households become homeless
        if self.perc_homeless == 100:
            self.running = False
        if self.month == 13:
            self.running = False


    def move(self, household):
        #print(f'Search # {household.search}')
        
        if not household.homeless: # homeless == False
            if household.search >= 2:
                household.homeless = True
                print(f'Household {household.unique_id} is now homeless.')
                household.geometry = Point(-8573211.872777091, 4706482.125258839)
                household.parcel_id = None
                household.zipcode = None
                household.ward_id = None
                
            
            else: # household.search < 2
                parcels = [parcel for parcel in self.schedule.agents if isinstance(parcel, PropAgent) and parcel.vacancies > 0 and parcel.value <= household.budget]
                if household.housing_status in [3, 4]: # if household is renter
                    #print(f'The household is a renter')
                    target_patches = [parcel for parcel in parcels if parcel.PROPTYPE in renter_proptypes ]
                    

                else: # if household is owner
                    target_patches = [parcel for parcel in parcels if isinstance(parcel, PropAgent) and parcel.vacancies > 0 and parcel.PROPTYPE in owner_proptypes]
                
                
                # if a household cannot find a home with their same type (renter or owner), they open the search wider
                if not target_patches:
                    target_patches = parcels
                    if not target_patches:
                        
                        # If household still cannot find a home, they widen their search to anything within their savings
                        if household.wealth > household.budget:
                            target_patches = [parcel for parcel in self.schedule.agents if isinstance(parcel, PropAgent) and parcel.vacancies > 0 and parcel.value <= household.wealth]

                    
                if household.evicted:
                    household.search += 1
                    if target_patches:
                        target_patch = random.choice(target_patches)
                        household.geometry = target_patch.geometry
                        household.parcel_id = target_patch.unique_id
                        household.zipcode = target_patch.ZIPCODE
                        household.ward_id = target_patch.WARD_1
                        household.rent = target_patch.rent

                        target_patch.vacancies -= 1
                        household.search = 0
                        household.evicted = False
                        print(f'Household {household.unique_id} has moved.')
                        # Update housing cost based on new property's value
                        household.housing_cost = target_patch.value
                    else: 
                        print(f'Household {household.unique_id} cannot find a place to move during this search.')
    


    def earn_income(self, household):
        household.wealth += household.monthly_income
        
        # If stimulus check released:
        if (self.release_stimulus or self.permanent_stimulus):
            household.wealth += self.stimulus_value
            print(f"Household received stimulus of {self.stimulus_value}.")
    

    def pay_other_expenses(self, household):
      # Assume food comes first
        food_expenses = (household.home_food_costs + household.outside_food_costs) * 2 # counting food costs every two weeks
        household.food_expenses = food_expenses
        additional_costs = food_expenses + ((household.monthly_income /2) * (self.other_weekly_costs * 2)) # every two weeks


        if additional_costs > household.wealth:
            if household.credit < household.credit_limit:
                household.credit += additional_costs
            else: 
                household.food_insecure = True
            
        else: household.wealth -= additional_costs

    def pay_rent(self, household):
        self.pay_other_expenses(household) # every two weeks
    
        if self.schedule.steps % 2 == 0:
            if household.housing_status in [2,3]:
                if household.wealth < household.housing_cost:
                    self.use_credit(household)
                else: 
                    household.wealth -= household.housing_cost
                    self.landlord_income += household.housing_cost


    def use_credit(self, household):
        if household.strikes >= self.leniency:
            if not self.moratorium:
                household.evicted = True
        else: # household strikes < leniency
            if household.credit > household.credit_limit:
                household.owe_backrent = True
                household.strikes += 1
                household.debt += household.housing_cost
                self.missed_payments += household.housing_cost
            else:
                household.credit += household.housing_cost
                self.landlord_income += household.housing_cost


    def set_rent_change(self, parcel):
        month = self.month
        if month == 1:
            pass
        elif month > 1:
            zip_prices = zip_housing_prices.get(parcel.ZIPCODE)
            if zip_prices:
                change_rental = zip_prices[0] * math.log(month)
                change_sfr = zip_prices[2] * month
                change_condocoop = zip_prices[4] * month

            if parcel.PROPTYPE in rental_res:
                parcel.value += change_rental
            elif parcel.PROPTYPE in condo_coop:
                parcel.value += change_condocoop
            elif parcel.PROPTYPE in sfr:
                parcel.value += change_sfr

    def update_rent(self, parcel):
        if parcel.PROPTYPE != 0: 
            self.set_rent_change(parcel)

    def unemploy(self, ward):
        month = self.month
        # Store the current unemployment rate as the old unemployment rate
        ward.old_unemployment_rate = ward.unemployment_rate
        # Get unemployment rate for ward at current month 
        b1, b0 = unemployment_table.get(ward.unique_id)
        ward.unemployment_rate = (b0 + (b1 * month)) / 100
        ward.change_unemployment = b1 * month

        employed_households = [agent for agent in self.schedule.agents if isinstance(agent, HouseholdAgent) and agent.ward_id == ward.unique_id and agent.unemployed == 0]
        unemployed_households = [agent for agent in self.schedule.agents if isinstance(agent, HouseholdAgent) and agent.ward_id == ward.unique_id and agent.unemployed == 1]

        y = int(round(ward.change_unemployment * len(employed_households) / 100))
        x = int(round(abs(ward.change_unemployment * len(unemployed_households) / 100)))

        ward_earning_households = [agent for agent in self.schedule.agents if isinstance(agent, HouseholdAgent) and agent.monthly_income > 0.001 and agent.ward_id == ward.unique_id]
        ward.mean_income = np.mean([agent.monthly_income for agent in ward_earning_households])

        

        if ward.change_unemployment > 0:
            for i in range(y):
                household = random.choice(employed_households)
                print('Household {household.unique_id} lost its job and income.')
                household.unemployed = 1 # unemployed
                household.monthly_income = 0
                household.budget = 0
        elif ward.change_unemployment < 0:
            for i in range(x):
                household = random.choice(unemployed_households)
                household.unemployed = 0 # employed
                household.monthly_income = ward.mean_income
                household.budget = household.monthly_income * (self.housing_perc_income / 100)

        else:
            pass
    


