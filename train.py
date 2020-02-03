import torch
import torchvision
from machine_learning.metrics import metric_history
import time


def train(  model,   
            data_loader_generator, 
            optimizer,
            criterion,
            metric_dataframe ,    
            metrics_of_interest,
            num_epochs,
            writer, 
            device = 'cpu'):
    

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader_generator.trainloader, 0):
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
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 5))
                
                writer.add_loss('training loss',
                            running_loss / 5,
                            epoch * len(data_loader_generator.trainloader) + i)
                running_loss = 0.0
        

        with torch.no_grad():  
            
            for i, data in enumerate(data_loader_generator.validationloader, 0): 
                idx = data["idx"].to(device).numpy()[0]  
                inputs, labels = data["image"], data["label"]
                inputs, labels = inputs.to(device) , labels.to(device)
                
                inputs = inputs.float() 
                labels = labels.reshape(labels.shape[0])
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                data_loader_generator.df.loc[idx,"prediction"] = predicted.numpy()[0]
                
 

            metric_dataframe = metric_history(data_loader_generator.df, 
                            metric_dataframe, 
                            epoch, 
                            metrics_of_interest )

            writer.add_metrics(metric_dataframe,metrics_of_interest ,epoch)

    print('Finished Training')
    writer.add_graph(model, data_loader_generator)
    writer.add_embedding( model, data_loader_generator)
    return  model, metric_dataframe 