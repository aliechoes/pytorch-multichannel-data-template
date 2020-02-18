import torch
import torchvision
from machine_learning.metrics import metric_history
import time
import os

def train(  model,   
            data_loader, 
            optimizer,
            criterion,
            metric_dataframe ,    
            metrics_of_interest,
            num_epochs,
            writer, 
            model_folder,
            device = 'cpu'):
    

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
            
            model_path = os.path.join(model_folder, "epoch_" + str(epoch + 1) + ".pth" )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss 
            }, model_path) 

        with torch.no_grad():  
            
            for i, data in enumerate(data_loader.validationloader, 0): 
                idx = data["idx"].cpu().numpy()[0]  
                inputs, labels = data["image"], data["label"]
                inputs, labels = inputs.to(device) , labels.to(device)
                
                inputs = inputs.float() 
                labels = labels.reshape(labels.shape[0])
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                data_loader.df.loc[idx,"prediction"] = predicted.cpu().numpy()[0]
                
 
            
            metric_dataframe = metric_history(data_loader.df, 
                            metric_dataframe, 
                            epoch, 
                            metrics_of_interest )

            writer.add_metrics(metric_dataframe,metrics_of_interest ,epoch)
            writer.add_embedding( model, data_loader, epoch, device)
            writer.add_images( data_loader, epoch )


    print('Finished Training')
    writer.add_graph(model, data_loader)

    
    
    return  model, metric_dataframe 