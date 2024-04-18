#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:14:55 2024

@author: MauraLapoff
"""
# parameter search
#from mesa.batchrunner import BatchRunner
import ray
ray.shutdown()
from ray import tune, train
from model import EvictionModel

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
    train.report({"perc_homeless" : perc_homeless})
 
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
    "prop_count": tune.grid_search(list(range(1000, 111423, 1000))),
    "other_weekly_costs": tune.grid_search(list(range(10, 50, 10))),
}


ray.init(local_mode = True)
analysis = tune.run(
    run_eviction_experiment,
    config=model_params,
    num_samples=5,
    metric="perc_homeless",  
    mode="min",  # minimize homelessness
)

best_result = analysis.get_best_trial(metric="perc_homeless", mode="min")
print("Best parameters to minimize homelessness:", best_result.config)
print("Lowest rate of homelessness", best_result.last_result)

ray.shutdown()