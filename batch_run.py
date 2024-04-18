
# parameter search
#from mesa.batchrunner import BatchRunner
import ray
ray.shutdown()
from ray import tune, train
from model import EvictionModel
from ray.tune.schedulers import HyperBandScheduler
def run_eviction_experiment(config):
    model = EvictionModel(
         moratorium = config["moratorium"],
         moratorium_expiration_week = config["moratorium_expiration_week"],
         perc_savings = config["perc_savings"],
         initial_occupancy_rate = config["initial_occupancy_rate"],
         permanent_stimulus = config["permanent_stimulus"],
         release_stimulus = config["release_stimulus"],
         leniency = config["leniency"],
         housing_perc_income = config["housing_perc_income"],
         stimulus_value = config["stimulus_value"],
         prop_count = config["prop_count"],
         other_weekly_costs = config["other_weekly_costs"],
         )
    
    for i in range(28):
        model.step()
    perc_homeless = model.perc_homeless
    month = model.month
    landlord_income = model.landlord_income
    missed_payments = model.missed_payments
    white_black_dissimilarity = model.white_black_dissimilarity
    white_asian_dissimilarity = model.white_asian_dissimilarity
    white_other_dissimilarity = model.white_other_dissimilarity
    black_asian_dissimilarity = model.black_asian_dissimilarity
    black_other_dissimilarity = model.black_other_dissimilarity
    asian_other_dissimilarity = model.asian_other_dissimilarity
    latino_dissimilarity = model.latino_dissimilarity
    train.report({"perc_homeless" : perc_homeless,
                  "month" : month, 
                   "landlord_income" : landlord_income,
                   "missed_payments" : missed_payments,
                   "white_black_dissimilarity" : white_black_dissimilarity,
                   "white_asian_dissimilarity" : white_asian_dissimilarity,
                   "white_other_dissimilarity" : white_other_dissimilarity,
                   "black_asian_dissimilarity" : black_asian_dissimilarity,
                   "black_other_dissimilarity" : black_other_dissimilarity,
                   "asian_other_dissimilarity" : asian_other_dissimilarity,
                   "latino_dissimilarity" : latino_dissimilarity
                  })
 
# Run experiment
model_params = {
    "moratorium" : tune.choice([True, False]),
    "moratorium_expiration_week": tune.grid_search(list(range(1, 21, 1))),
    "housing_perc_income": tune.grid_search(list(range(20, 101, 5))),
    "perc_savings": tune.grid_search(list(range(1, 6, 1))),
    "permanent_stimulus" : tune.choice([True, False]),
    "release_stimulus" : tune.choice([True, False]),
    "leniency": tune.grid_search(list(range(1, 6, 1))),
    "initial_occupancy_rate": tune.grid_search(list(range(20, 101, 10))),
    "stimulus_value": tune.grid_search(list(range(500, 2001, 500))),
    #"prop_count": tune.grid_search(list(range(1000, 111423, 1000))),
    "prop_count": tune.grid_search([1000]),
    "other_weekly_costs": tune.grid_search(list(range(10, 50, 10))),
}


ray.init(local_mode = True)
analysis = tune.run(
    run_eviction_experiment,
    config=model_params,
    num_samples=5,
    metric="perc_homeless",  
    mode="min",  # minimize homelessness
    scheduler = HyperBandScheduler(max_t=100, reduction_factor=3)

)
best_result = analysis.get_best_trial(metric="perc_homeless", mode="min")
print("Best parameters to minimize homelessness:", best_result.config)
print("Lowest rate of homelessness", best_result.last_result)

ray.shutdown()