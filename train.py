
import torch
import torchvision
import torch.nn.functional as F
from machine_learning.metrics import metric_history
import time
import os

def early_stopping(validation_criteria, patience):
    n0 = len(validation_criteria) - patience 
    n1 = len(validation_criteria) + 1 
    validation_difference = validation_criteria.iloc[n0:n1] - \
                                    validation_criteria.iloc[n0] 
    print(validation_difference > 0.)
    model_is_improved = (validation_difference > 0.).sum()
    if model_is_improved > 0:
        return False
    else:
        return True

def train(  model,   
            data_loader, 
            optimizer,
            criterion,
            metric_dataframe ,    
            metrics_of_interest,
            num_epochs,
            writer, 
            model_folder,
            call_back,
            device = 'cpu'):
    
    saving_period = call_back["saving_period"]
    patience = call_back["patience"]
    criteria = call_back["criteria"]

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
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
                print('[epoch: %d, minibatch %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

        if  epoch % saving_period == (saving_period - 1):
            model_path = os.path.join(model_folder, "epoch_" + str(epoch + 1) + ".pth" )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss ,
                'mean': data_loader.mean,
                'std': data_loader.std
            }, model_path)  

        with torch.no_grad():  
            
            for i, data in enumerate(data_loader.validationloader, 0): 
 
                idx = data["idx"].cpu().numpy()   

                inputs, labels = data["image"], data["label"]
                inputs, labels = inputs.to(device) , labels.to(device)
                
                inputs = inputs.float() 
                labels = labels.reshape(labels.shape[0])
                
                outputs = model(inputs)
                #embedding = model.embedding_generator(inputs)
                
                outputs_probability = F.softmax(outputs).cpu().numpy()  
                _, predicted = torch.max(outputs.data, 1) 

                data_loader.df.loc[idx,"prediction"] = predicted.cpu().numpy() 

                for k, cl in enumerate(data_loader.classes,0):
                    data_loader.df.loc[idx,cl + "_probability"] = outputs_probability[:,k]


            metric_dataframe = metric_history(data_loader.df, 
                            metric_dataframe, 
                            epoch, 
                            metrics_of_interest )

            writer.add_metrics(metric_dataframe,metrics_of_interest ,epoch)
            writer.add_images( data_loader, epoch )
            writer.add_pr_curve( data_loader, epoch )
        if epoch > 1.5*patience:
            indx =  (metric_dataframe["set"] == "validation") & \
                        (metric_dataframe["metric"] == criteria )
            validation_criteria = metric_dataframe.loc[indx, "value"]
            if early_stopping(validation_criteria, patience):
                print("The training has stopped as the early stopping is triggered")
                break
    
    
    writer.add_embedding( model, data_loader, epoch, device)
    writer.add_graph(model, data_loader)
    print('Finished Training')
    return  model, metric_dataframe 