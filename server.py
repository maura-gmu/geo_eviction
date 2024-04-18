import mesa
import mesa_geo as mg
import xyzservices.providers as xyz
from model import EvictionModel, PropAgent, HouseholdAgent, WardAgent
#from mesa.visualization.ModularVisualization import ModularServer




class HomelessElement(mesa.visualization.TextElement):
    """
    Text element indicating number of homeless people.
    """

    def __init__(self):
        super().__init__()

    def render(self, model):
        return f"Landlord income: {model.income}, % homeless: {model.perc_homeless}"

# value, min, max, step
model_params = {
    "other_weekly_costs" : mesa.visualization.Slider("How much of weekly income is spent on other costs?", 20, 20, 50, 10),
    "housing_perc_income" : mesa.visualization.Slider("How much of income is spent on housing?", 30, 10, 80, 5),
    "moratorium" : mesa.visualization.Checkbox("Moratorium policy is in place", True),
    "moratorium_expiration_week" : mesa.visualization.Slider("How many weeks into simulation does Moratorium expire?", 3, 1, 5, 1), # if moratorium == True
    "leniency": mesa.visualization.Slider("How many missed housing payments before eviction?", 3, 1, 5, 1),
    "prop_count" : mesa.visualization.Slider("# of Properties", 1000, 1000, 111423, 10000),
    "initial_occupancy_rate" : mesa.visualization.Slider("% of Properties initially occupied", 80, 10, 100, 10),
    "stimulus_value" : mesa.visualization.Slider("Set Stimulus Check Value", 1000, 500, 2000, 500),
    "release_stimulus" : mesa.visualization.Checkbox("Release Stimulus Check Once", False),
    "permanent_stimulus" : mesa.visualization.Checkbox("Release Stimulus Every Month", False),
    "perc_savings" : mesa.visualization.Slider("How many month's worth of savings do households have?", 2, 0, 6, 1)
}



color_map = {
    1: "#ff0000",  # Red
    2: "#008000",  # Green
    3: "#0000ff",  # Blue
    4: "#ffc0cb",  # Pink
    5: "#ffff00",  # Yellow
    6: "#ffa500",  # Orange
    7: "#00ff00",  # Lime
    8: "#40e0d0"   # Turquoise
}



def eviction_draw(agent):
    """
    Portrayal Method for canvas
    """
    portrayal = {}
    
    if isinstance(agent, HouseholdAgent):
        portrayal["radius"] = .001
        portrayal["shape"] = "circle"
        if not agent.evicted:
            if agent.race == 1:
                portrayal["color"] = "Lime"
            elif agent.race == 2:
                portrayal["color"] = "Violet"
            elif agent.race == 3:
                portrayal["color"] = "Turquoise"
            elif agent.latino == 2:
                portrayal["color"] = "Orange"
            else:
                portrayal["color"] = "Yellow"
        else: portrayal["color"] = "Red"
     
        portrayal["layer"] = 2
        
    if isinstance(agent, PropAgent):
        portrayal["color"] = "Gray"
        portrayal["layer"] = 1
        portrayal["filled"] = False
        
        
    if isinstance(agent, WardAgent):

        if agent.unique_id in color_map:
          portrayal["color"] = color_map[agent.unique_id] if agent.unique_id in color_map else "#808080"            

    return portrayal
  

homeless_element = HomelessElement()
map_element = mg.visualization.MapModule(eviction_draw, zoom = 4, tiles=xyz.CartoDB.Positron)


homeless_chart = mesa.visualization.ChartModule([{"Label": "% DC Households now Homeless", "Color": "Black"}])
index_dissimilarity_chart = mesa.visualization.ChartModule([
    {"Label": "white-black dissimilarity index" , "Color": "Black"},
    {"Label": "white_asian_dissimilarity index" , "Color": "Blue"},
    {"Label": "white_other_dissimilarity index" , "Color": "Red"},
    {"Label": "black_asian_dissimilarity index" , "Color": "Green"},
    {"Label": "black_other_dissimilarity index" , "Color": "Orange"},
    {"Label": "asian_other_dissimilarity index" , "Color": "Brown"},
    {"Label": "latino_dissimilarity index" , "Color": "Yellow"},
    ])

server = mesa.visualization.ModularServer(
  EvictionModel, 
  [
   map_element, 
   homeless_element, 
   homeless_chart,
   index_dissimilarity_chart
   ], 
  "Eviction", 
  model_params)


