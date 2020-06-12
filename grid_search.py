import os
import json
import time
import logging
import pandas as pd

def load_json(file_path):
    with open(file_path, 'r') as stream:    
        return json.load(stream)

def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def change_config(config, key_value, value):
    if key_value == "network":
        config["model"]["network"] = value
    
    if key_value == "checkpoint_path":
        config["model"]["checkpoint_path"] = value
        
    if key_value == "loss_function":
        config["loss"]["loss_function"] = value    

    if key_value == "weights":
        config["loss"]["weights"] = value   

    if key_value == "optimization_method":
        config["optimizer"]["optimization_method"] = value    
    if key_value == "optimization_parameters":
        config["optimizer"]["optimization_method"] = value    
    if key_value == "lr":
        config["optimizer"]["optimization_method"]["lr"] = value  

    if key_value == "weight_decay":
        config["optimizer"]["optimization_method"]["weight_decay"] = value    
    
    if key_value == "num_epochs":
        config["training"]["num_epochs"] = value      
    
    if key_value == "patience":
        config["training"]["call_back"]["patience"] = value   
    
    if key_value == "lr_scheduler":
        config["training"]["lr_scheduler"] = value   

    if key_value == "batch_size":
        config["data_loader"]["batch_size"] = value   
    
    if key_value == "dynamic_range":
        config["data_loader"]["dynamic_range"] = value  

    return config
            
def run_differnet_settings(config_path):
    configs = load_json(config_path)
    for k in configs:
        logging.info("%s : %s \n" % (k,configs[k]))

    grid = dict()
    grid["data_dir"] = ["/pstore/home/shetabs1/data/CellCycle"] 
    grid["model_name"] = ["squeezenet1_0"] 
    grid["lr"] = [   0.001  ]  
    grid["weight_decay"] = [0 ] 
    grid["optimization_method"] = ["adam"]
    grid["weights"] = ["frequency", None] 

    number_of_parameters = 1
    for k in grid:
        number_of_parameters = len(grid[k]) * number_of_parameters

    full_grid = pd.DataFrame(index = range(number_of_parameters), 
                                        columns = grid.keys())

    for i in range(number_of_parameters):
        for j in grid:

    for i in range(number_of_parameters):
        for j in grid:
            config = change_config(config, key_value, value)

                        
                        save_json(config_path, configs)
                        time.sleep(5)
                        os.system('bash launch_code.sh')

if __name__ == "__main__":
    #run_differnet_settings("configs/sample_config.json")
    # for i in range(1113000,1116000):
    #     os.system('scancel ' + str(i))
