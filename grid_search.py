import os
import json
import time
import logging

def load_json(file_path):
    with open(file_path, 'r') as stream:    
        return json.load(stream)

def save_json(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def change_config(config, key_value, value):
    if key_value == "data_dir":
        config["data"]["data_dir"] = value

    elif key_value == "test_data_dir":
        config["data"]["test_data_dir"] = value

    elif key_value == "model_name":
        config["machine_learning"]["model_name"] = value

    elif key_value == "lr":
        config["machine_learning"]["optimization_parameters"]["lr"] = value

    elif key_value == "weight_decay":
        config["machine_learning"]["optimization_parameters"]["weight_decay"] = value

    elif key_value == "optimization_method":
        config["machine_learning"]["optimization_method"] = value
        
    elif key_value == "loss_function":
        config["machine_learning"]["loss_function"] = value
    
    elif key_value == "weights":
        config["machine_learning"]["weights"] = value

    return config
            
def run_differnet_settings(config_path):
    configs = load_json(config_path)
    for k in configs:
        logging.info("%s : %s \n" % (k,configs[k]))

    grid = dict()
    grid["data_dir"] = ["/pstore/home/shetabs1/data/CellCycle"] # , 
                       # "/pstore/home/shetabs1/data/Exp12_labeled_images/Donor13/",
                       # "/pstore/home/shetabs1/data/Exp12_labeled_images/Donor23/"]

    grid["model_name"] = ["squeezenet1_0"] # "squeezenet1_0",  "resnet18", "densenet121"
    grid["lr"] = [   0.001  ]  
    grid["weight_decay"] = [0 ]  #[0.9, 0.1, 0.01 ,0.001 ,0.0001 , 0] 
    grid["optimization_method"] = ["adam"] #["adam", "rmsprop" ] 
    grid["weights"] = ["frequency", None] #["adam", "rmsprop" ] 

    for a in grid["data_dir"]:
        configs = change_config(configs, "data_dir", a) 
        #b = a.replace("Donor12", "Donor3").replace("Donor13", "Donor2").replace("Donor23", "Donor1")
        #logging.info(b)
        #configs = change_config(configs, "test_data_dir", b) 
        for c in grid["model_name"]: 
            configs = change_config(configs, "model_name", c)
            for d in grid["lr"]: 
                configs = change_config(configs, "lr", d)
                for e in grid["weight_decay"]: 
                    configs = change_config(configs, "weight_decay", e)
                    for f in grid["optimization_method"]: 
                        configs = change_config(configs, "optimization_method", f) 
                        save_json(config_path, configs)
                        time.sleep(5)
                        os.system('bash launch_code.sh')

if __name__ == "__main__":
    run_differnet_settings("configs/sample_config.json")
    # for i in range(1113000,1116000):
    #     os.system('scancel ' + str(i))
