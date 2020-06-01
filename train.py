
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn 
from machine_learning.metrics import metric_history
import time
import os
import pandas as pd
 

def elapsed_time_print(start_time, message, epoch):
    """
    function to print the elapsed fime
    """
    elapsed_time = time.time() - start_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    to_be_printed = "epoch %d: " + message 
    print(4*"---")
    print(to_be_printed % (epoch, elapsed_time))
    print(4*"---")
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

def train(  model,   
            data_loader, 
            optimizer,
            criterion,  
            writer, 
            model_folder,
            configs):
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

    saving_period = configs["validation"]["call_back"]["saving_period"]
    patience = configs["validation"]["call_back"]["patience"]
    criteria = configs["validation"]["call_back"]["criteria"]
    best_criteria_value = 0
    metrics_of_interest = configs["validation"]["metrics_of_interest"]
    num_epochs = configs["machine_learning"]["num_epochs"]
    device =  configs["machine_learning"]["device"]

    # creating a dataframe which will contain all the metrics per set per epoch
    metric_dataframe = pd.DataFrame(columns= ["epoch","set", "metric", "value"])

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(12*"-*-")
        print("EPOCH: %d" % epoch)
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(data_loader.trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            
            inputs, labels = data["image"], data["label"]
            inputs,labels = inputs.to(device), labels.to(device)
            
            inputs = inputs.float()
            labels = labels.reshape(labels.shape[0])
            

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize 
            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # print loss every 5 minibatches
            if i % 5 == 4:
                print('[epoch: %d, minibatch %5d] loss: %.3f' % (epoch, i + 1, running_loss / 5))
                running_loss = 0.0
        elapsed_time_print(start_time, "Training took %s", epoch)

        # saving the model
        if  epoch % saving_period == (saving_period ):
            start_time = time.time()
            model_path = os.path.join(model_folder, "epoch_" + str(epoch ) + ".pth" )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss ,
                'channels': data_loader.existing_channels,
                'statistics': data_loader.statistics,
                'data_map': data_loader.data_map
            }, model_path)  
            elapsed_time_print(start_time, "Saving Model took %s", epoch)

        # the evaluation phase
        with torch.no_grad():  
            model.eval()
            start_time = time.time()
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

                data_loader.df.loc[idx,"prediction"] = predicted.cpu().numpy() 

                for k, cl in enumerate(data_loader.classes,0):
                    data_loader.df.loc[idx,cl + "_probability"] = outputs_probability[:,k]

            # adding the results to the metric dataframe 
            metric_dataframe = metric_history(data_loader.df, 
                            metric_dataframe, 
                            epoch, 
                            metrics_of_interest )

            writer.add_metrics(metric_dataframe,metrics_of_interest ,epoch)

        elapsed_time_print(start_time, "Evaluating Model took %s", epoch)

        # TODO saving the best model so far
        indx =  (metric_dataframe["set"] == "validation") & \
                        (metric_dataframe["metric"] == criteria )
        current_criteria_value = metric_dataframe.loc[indx, "value"].iloc[-1]
        
        if best_criteria_value < current_criteria_value:
            #writer.add_images( data_loader, epoch )
            writer.add_pr_curve( data_loader, epoch )
            writer.add_confusion_matrix( data_loader, epoch ) 

            best_epoch = epoch
            best_criteria_value = current_criteria_value
            start_time = time.time()
            model_path = os.path.join(model_folder, "best_model.pth" )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss ,
                'channels': data_loader.existing_channels,
                'statistics': data_loader.statistics,
                'criteria': criteria,
                'current_criteria_value': current_criteria_value,
                'data_map': data_loader.data_map
            }, model_path)  
            elapsed_time_print(start_time, "Saving the best Model took %s", epoch)


        # check for early stopping
        if epoch > 1.5*patience:
            indx =  (metric_dataframe["set"] == "validation") & \
                        (metric_dataframe["metric"] == criteria )
            validation_criteria = metric_dataframe.loc[indx, "value"]
            if early_stopping(validation_criteria, patience):
                print("The training has stopped as the early stopping is triggered")
                break
    
    ## the feature extractor only can be done when the weights are calculated.
    # the formula to get the feature extractor is included in th model.embedding_generator 
    # however, it has to be evaluated separately and cannot be part of the model as 
    # pytorch makes mistakes with new architecures in the model
    for i in range(10):
        try:
            writer.add_hparams(configs, best_criteria_value, best_epoch, optimizer.state_dict() )
            break
        except:
            print("there is a problem with hparam")
    #feature_extractor = eval(model.embedding_generator)
    #writer.add_embedding( feature_extractor, data_loader, epoch, device)
    #writer.add_graph(model, data_loader)
    print('Finished Training')
    return  model, metric_dataframe 