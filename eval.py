
from utils import *
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn 
import logging
import time
import os
import pandas as pd


def get_checkpoint(file_path):
    if file_path is not None:
        checkpoint = torch.load(file_path)
        checkpoint["classes"] = ['Doublet_with_interaction', 'Doublet_with_no_interaction', 'Doublet_with_no_interaction_functional', 'Multiplet', 'SingleBcell', 'SingleTcell', 'SingleTcell_Functional'] 
        checkpoint["channels"] = ['Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch6', 'Ch7']
    else:
        checkpoint = None
    return checkpoint

def elapsed_time_print(start_time, message, epoch):
    """
    function to print the elapsed fime
    """
    elapsed_time = time.time() - start_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    to_be_printed = "epoch %d: " + message 
    logging.info(10*"---")
    logging.info(to_be_printed % (epoch, elapsed_time))
    logging.info(10*"---")
    return None

def predict(configs):
    
    device = configs["model"]["device"]

    # seperating the configs part
    model_configs = configs["model"]
    data_loader_configs = configs["data_loader"]
    output_folder = configs["data_loader"]["ouput_folder"]

    checkpoint = get_checkpoint(model_configs["checkpoint_path"])
    
    data_loader = DataLoaderGenerator(data_loader_configs)
    data_loader.existing_channels = checkpoint["channels"] 

    data_loader.data_frame_creator()
    
    logging.info(data_loader.df)

    
    # number of exsting channels and output classes
    number_of_channels = len(checkpoint["channels"])
    number_of_classes = len(checkpoint["classes"])

    # initialize the model
    model = get_model(  model_configs,
                        device,
                        checkpoint,
                        number_of_channels ,
                        number_of_classes)

    data_loader.data_loader(model.image_size, checkpoint)
    
    uncertainty_samples = 100
    for j in range(uncertainty_samples):
        data_loader.df["prediction_" + str(j)] = -1
    # the evaluation phase
 
    logging.info("starting the evaluation")
    with torch.no_grad():  
        
        percentage = 0.
        for i, data in enumerate(data_loader.validationloader, 0): 
            model.eval()
            # finding the file in the dataframe
            idx = data["idx"].cpu().numpy()   
            percentage = percentage + len(idx) / data_loader.df.shape[0]
            logging.info("Completion Percentage %.2f" % percentage)
            inputs, labels = data["image"], data["label"]
            inputs, labels = inputs.to(device) , labels.to(device)
                
            inputs = inputs.float() 
            labels = labels.reshape(labels.shape[0])
                
            outputs = model(inputs)
                
            outputs_probability = F.softmax(outputs).cpu().numpy()  
            _, predicted = torch.max(outputs.data, 1) 

            data_loader.df.loc[idx,"prediction"] = predicted.cpu().numpy() 

            logging.info("probabilities")
            for k, cl in enumerate(checkpoint["classes"],0):
                data_loader.df.loc[idx,cl + "_probability"] = outputs_probability[:,k]

            logging.info("starting the uncertainty calculation")
            model.train()
            logging.info("model.train is on")
            for j in range(uncertainty_samples):
                outputs = model(inputs)  
                _, predicted = torch.max(outputs.data, 1)    
                data_loader.df.loc[idx, "prediction_" + str(j)] = predicted.cpu().numpy()
                
        
        mode = data_loader.df.loc[:, [("prediction_" + str(j)) for j in range(100) ] ].mode(axis = 1)[0]
        for i in range(0,data_loader.df.shape[0]):
            data_loader.df.loc[i,"uncertainty"] = 1. - ((data_loader.df.loc[i, [("prediction_" + str(j)) for j in range(100) ] ] == mode[i]).sum())/100.
        # 
        # save the label of all images and their predictions
        data_loader.df.to_csv(os.path.join(output_folder,
                                        "granular_results.csv"), index = False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser( \
                            description='Starting the deep learning code')
    parser.add_argument('-c',\
                        '--config', \
                        help='config yaml file address', \
                        default="configs/sample_config_evaluation.json", \
                        type=str)

    args = vars(parser.parse_args())
    logger(2)
    
    configs = load_json(args['config'])
    for k in configs:
        logging.info("%s : %s \n" % (k,configs[k]))
    predict(configs)

