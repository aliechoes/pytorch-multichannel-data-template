
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn 
from train.metrics import metric_history
from train.lr_schedulers import GetLRScheduler
import time
import os
import pandas as pd
import logging

def make_model_sequential(model, device):
    embedding_generator = model.embedding_generator
    image_size = model.image_size 
    model = model.module 
    model = model.to(device)
    model.image_size  =  image_size 
    model.embedding_generator = embedding_generator 
    return model, eval(model.embedding_generator)

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

def early_stopping(validation_criteria, patience):
    """
    early stopping in case the model does not improve after some epochs.
    It only looks at the validation set
    """
    n0 = len(validation_criteria) - patience 
    n1 = len(validation_criteria) + 1 
    validation_difference = validation_criteria.iloc[n0:n1] - \
                                    validation_criteria.iloc[n0] 
                                    
    model_is_improved = (validation_difference > 0.).sum()
    if model_is_improved > 0:
        return False
    else:
        return True

def saving_model(model_path, epoch, model, optimizer ,data_loader,  criteria, current_criteria_value):
    """
    Saving the model with all the parameters
    Args:
        epoch(int): current epoch
        model_path(str): output path for the pth file
        model(torch model): trained model
        optimizer(torch optimizer): torch optimizer 
        data_loader(torch data loader): dataloader with all the parameters
        criteria(str): criteria for optimization
        current_criteria_value(float): current criteria value for this epoch
    outputs:
        None
    """
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'classes': data_loader.classes,
                'channels': data_loader.existing_channels,
                'scaling_factor': data_loader.scaling_factor,
                'statistics': data_loader.statistics,
                'criteria': criteria,
                'current_criteria_value': current_criteria_value,
                'data_map': data_loader.data_map
            }, model_path)            

def train(  model,   
            data_loader, 
            optimizer,
            criterion,  
            writer, 
            model_folder,
            training_configs,
            device):
    """
    the function which trains the model and evaluates it over the whole dataset
    Args:
        model: architecture   
        data_loader: dataloader including the train and validation loader
        optimizer: the selected method
        criterion: loss function
        metric_dataframe: dataframe for tracking the progress of the network per epoch    
        metrics_of_interest: list of metrics which we would like to track
        num_epochs: number of epochs
        writer: tensorboard writer 
        model_folder: folder to save the models per epoch
        call_back: 
            saving_period(int): period of saving models
            patience(int): period of checking whether the training should stop
            criteria(str): the metric to track for early stopping
        device(str): either cpu or cuda
    """

    saving_period = training_configs["call_back"]["saving_period"]
    patience = training_configs["call_back"]["patience"]
    criteria = training_configs["call_back"]["criteria"]
    best_criteria_value = 0.
    metrics_of_interest = training_configs["metrics_of_interest"]
    num_epochs = training_configs["num_epochs"]
    lr_scheduler_config = training_configs["lr_scheduler"]
    # creating a dataframe which will contain all the metrics per set per epoch
    metric_dataframe = pd.DataFrame(columns= ["epoch","set", "metric", "value"])
    train_index = data_loader.train_dataset.old_index
    validation_index = data_loader.validation_dataset.old_index

    scheduler = GetLRScheduler(optimizer,lr_scheduler_config) 
    
    for epoch in range(1,num_epochs+1) :  # loop over the dataset multiple times
        logging.info(10*"---")
        logging.info("epoch: %d , learing rate: %.8f" % (epoch, optimizer.state_dict()["param_groups"][0]["lr"] )) 
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(data_loader.trainloader, 0):
            
            idx = data["idx"].cpu().numpy()  

            inputs, labels = data["image"], data["label"]
            inputs, labels = inputs.to(device), labels.to(device)
            
            inputs = inputs.float()
            labels = labels.reshape(labels.shape[0])
            

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize 
            outputs = model(inputs) 
            outputs_probability = F.softmax(outputs).detach().cpu().numpy()  
            _, predicted = torch.max(outputs.data, 1) 

            data_loader.train_dataset.df.loc[idx,"prediction"] = predicted.detach().cpu().numpy().astype(int) 

            for k, cl in enumerate(data_loader.classes,0):
                data_loader.train_dataset.df.loc[idx,cl + "_probability"] = outputs_probability[:,k]

            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # print loss every 5 minibatches
            if i % 5 == 4:
                logging.info('[epoch: %d, minibatch %5d] loss: %.8f' % (epoch, i + 1, running_loss / 5))
                running_loss = 0.0
        elapsed_time_print(start_time, "Training took %s", epoch)



        # the evaluation phase
        with torch.no_grad():  
            model.eval()
            start_time_validation = time.time()
            for i, data in enumerate(data_loader.validationloader, 0): 

                # finding the file in the dataframe
                idx = data["idx"].cpu().numpy()   

                inputs, labels = data["image"], data["label"]
                inputs, labels = inputs.to(device) , labels.to(device)
                
                inputs = inputs.float() 
                labels = labels.reshape(labels.shape[0])
                
                outputs = model(inputs)
                
                outputs_probability = F.softmax(outputs).cpu().numpy()  
                _, predicted = torch.max(outputs.data, 1) 

                data_loader.validation_dataset.df.loc[idx,"prediction"] = predicted.cpu().numpy().astype(int)

                for k, cl in enumerate(data_loader.classes,0):
                    data_loader.validation_dataset.df.loc[idx,cl + "_probability"] = outputs_probability[:,k]
                
            data_loader.df.loc[train_index,"prediction"] =  data_loader.train_dataset.df["prediction"].tolist()
            data_loader.df.loc[validation_index ,"prediction"] =  data_loader.validation_dataset.df["prediction"].tolist()
            
            for k, cl in enumerate(data_loader.classes,0):
                data_loader.df.loc[train_index,cl + "_probability"] =  data_loader.train_dataset.df[cl + "_probability"].tolist()
                data_loader.df.loc[validation_index ,cl + "_probability"] =  data_loader.validation_dataset.df[cl + "_probability"].tolist()

            # adding the results to the metric dataframe 
            metric_dataframe = metric_history(data_loader.df, 
                            metric_dataframe, 
                            epoch, 
                            metrics_of_interest,
                            data_loader.classes )
            

            writer.add_metrics(metric_dataframe,metrics_of_interest ,epoch)


        # TODO saving the best model so far
        indx =  (metric_dataframe["set"] == "validation") & \
                        (metric_dataframe["metric"] == criteria )
        current_criteria_value = metric_dataframe.loc[indx, "value"].iloc[-1]
        scheduler.step(current_criteria_value)

        # saving the model
        if  epoch % saving_period == (saving_period ):
            start_time = time.time()
            model_path = os.path.join(model_folder, "epoch_" + str(epoch ) + ".pth" )
            saving_model(model_path, epoch, model, optimizer ,data_loader,  criteria, current_criteria_value)  
            elapsed_time_print(start_time, "Saving Model took %s", epoch)

        if best_criteria_value < current_criteria_value:
            logging.info("The validation %s has improved from %.4f to %.4f" % \
                            (criteria,best_criteria_value, current_criteria_value)  )
            writer.add_images( data_loader, epoch )
            writer.add_pr_curve( data_loader, epoch )
            writer.add_confusion_matrix( data_loader, epoch ) 

            best_epoch = epoch
            best_criteria_value = current_criteria_value
            start_time = time.time()
            model_path = os.path.join(model_folder, "best_model.pth" )
            saving_model(model_path, epoch, model, optimizer ,data_loader,  criteria, current_criteria_value)   
            elapsed_time_print(start_time, "Saving the best Model took %s", epoch)


        # check for early stopping
        if epoch > 1.5*patience:
            indx =  (metric_dataframe["set"] == "validation") & \
                        (metric_dataframe["metric"] == criteria )
            validation_criteria = metric_dataframe.loc[indx, "value"]
            if early_stopping(validation_criteria, patience):
                logging.info("The training has stopped as the early stopping is triggered")
                break
        
        elapsed_time_print(start_time_validation, "Evaluating Model took %s", epoch)
    
    
    ## the feature extractor only can be done when the weights are calculated.
    # the formula to get the feature extractor is included in th model.embedding_generator 
    # however, it has to be evaluated separately and cannot be part of the model as 
    # pytorch makes mistakes with new architecures in the model
    
    
    model, feature_extractor = make_model_sequential(model, device)
    writer.add_graph(model, data_loader)
    model = None
    writer.add_embedding_with_images( feature_extractor, data_loader, epoch, device)
    writer.add_embedding_without_images( feature_extractor, data_loader, epoch, device)

    logging.info('Finished Training')
    return metric_dataframe, best_criteria_value, best_epoch 