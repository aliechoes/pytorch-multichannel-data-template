import pandas as pd 
import torch
from sklearn.metrics import  accuracy_score, cohen_kappa_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import log_loss
import numpy as np
from sklearn.metrics import confusion_matrix
import logging

def custom_log_loss(y_true, y_pred, existing_labels): 
    custom_y_pred = np.zeros((y_true.shape[0], len(existing_labels)))
    
    for i in range(0, len(y_true)):
        j = y_pred[i]
        custom_y_pred[i,j ] = 1  
    return log_loss(y_true, custom_y_pred, labels=existing_labels)


def top_k_accuracy(df_temp, classes, k = 3): 
    prob_classes = []
    for cl in classes:
        prob_classes.append(cl  + "_probability")

    y_true =  torch.tensor(df_temp["label"].tolist())
    y_true = y_true.reshape(len(y_true), 1 )
    possible_k = min(len(classes) - 1, k )
    topk = torch.topk(torch.from_numpy(df_temp.loc[:, prob_classes].to_numpy().astype(np.float64)), possible_k)[1]
    results = (topk == y_true).sum() / float(len(y_true))
    return results.numpy()

def metric_history(df, metric_dataframe, epoch, metrics_of_interest, classes ):
 
    existing_labels = df["label"].unique().tolist()
    for s in ["train", "validation", "test"]:
        logging.info(10*"---") 
        
        df_temp = df[ df["set"]==s].reset_index(drop = True).copy() 
        
        if df_temp.shape[0] > 0:
            y_true = df_temp["label"].astype(int)
            y_pred = df_temp["prediction"].astype(int)
            
            assert (y_pred == -1).sum()== 0  
            

            if "accuracy" in metrics_of_interest:
                results_temp = {
                    'epoch': epoch,
                    'set':   s,
                    'metric': 'accuracy',
                    'value': accuracy_score(y_true, y_pred ) 
                }
                logging.info("epoch %d: accuracy for the %s set is %f" % (epoch, s, accuracy_score(y_true, y_pred ) ) )
                metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)

            if "top_2_accuracy" in metrics_of_interest:
                results_temp = {
                    'epoch': epoch,
                    'set':   s,
                    'metric': 'top_2_accuracy',
                    'value': top_k_accuracy(df_temp, classes, k = 2 ) 
                }
                logging.info("epoch %d: top-2 accuracy for the %s set is %f" % (epoch, s, top_k_accuracy(df_temp, classes, k = 2 ) ) )
                metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)

            if "top_3_accuracy" in metrics_of_interest:
                results_temp = {
                    'epoch': epoch,
                    'set':   s,
                    'metric': 'top_3_accuracy',
                    'value': top_k_accuracy(df_temp, classes, k = 3 ) 
                }
                logging.info("epoch %d: top-3 accuracy for the %s set is %f" % (epoch, s, top_k_accuracy(df_temp, classes, k = 3 ) ) )
                metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)

            if "cross_entropy" in metrics_of_interest:
                    results_temp = {
                        'epoch': epoch,
                        'set':   s,
                        'metric': 'cross_entropy',
                        'value': custom_log_loss(y_true, y_pred, existing_labels) 
                    }
                    logging.info("epoch %d: cross_entropy for the %s set is: %f" % (epoch, s,  \
                        custom_log_loss(y_true, y_pred, existing_labels )  ) )
                    metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)
    
            
            if "f1_macro" in metrics_of_interest:
                results_temp = {
                    'epoch': epoch,
                    'set':   s,
                    'metric': 'f1_macro',
                    'value': f1_score(y_true, y_pred, average='macro' ) 
                }
                logging.info("epoch %d: f1_macro for the %s set is: %f" % (epoch, s, f1_score(y_true, y_pred, average='macro' ) ) )
                metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)
            
            if "f1_micro" in metrics_of_interest:
                results_temp = {
                    'epoch': epoch,
                    'set':   s,
                    'metric': 'f1_micro',
                    'value': f1_score(y_true, y_pred, average='micro' ) 
                }
                logging.info("epoch %d: f1_micro for the %s set is: %f" % (epoch, s, f1_score(y_true, y_pred, average='micro' ) ) )
                metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)
                
            if "f1_weighted" in metrics_of_interest:
                results_temp = {
                    'epoch': epoch,
                    'set':   s,
                    'metric': 'f1_weighted',
                    'value': f1_score(y_true, y_pred, average='weighted' ) 
                }
                logging.info("epoch %d: f1_weighted for the %s set is: %f" % (epoch, s, f1_score(y_true, y_pred, average='weighted' ) ) )
                metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)

            if "recall_macro" in metrics_of_interest:
                results_temp = {
                    'epoch': epoch,
                    'set':   s,
                    'metric': 'recall_macro',
                    'value': recall_score(y_true, y_pred, average='macro' ) 
                }
                logging.info("epoch %d: recall_macro for the %s set is: %f" % (epoch, s, recall_score(y_true, y_pred, average='macro' ) ) )
                metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)
            
            if "recall_micro" in metrics_of_interest:
                results_temp = {
                    'epoch': epoch,
                    'set':   s,
                    'metric': 'recall_micro',
                    'value': recall_score(y_true, y_pred, average='micro' ) 
                }
                logging.info("epoch %d: recall_micro for the %s set is: %f" % (epoch, s, recall_score(y_true, y_pred, average='micro' ) ) )
                metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)
                
            if "recall_weighted" in metrics_of_interest:
                results_temp = {
                    'epoch': epoch,
                    'set':   s,
                    'metric': 'recall_weighted',
                    'value': recall_score(y_true, y_pred, average='weighted' ) 
                }
                logging.info("epoch %d: recall_weighted for the %s set is: %f" % (epoch, s, recall_score(y_true, y_pred, average='weighted' ) ) )
                metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)

            if "precision_macro" in metrics_of_interest:
                results_temp = {
                    'epoch': epoch,
                    'set':   s,
                    'metric': 'precision_macro',
                    'value': precision_score(y_true, y_pred, average='macro' ) 
                }
                logging.info("epoch %d: precision_macro for the %s set is: %f" % (epoch, s, precision_score(y_true, y_pred, average='macro' ) ) )
                metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)
            
            if "precision_micro" in metrics_of_interest:
                results_temp = {
                    'epoch': epoch,
                    'set':   s,
                    'metric': 'precision_micro',
                    'value': precision_score(y_true, y_pred, average='micro' ) 
                }
                logging.info("epoch %d: precision_micro for the %s set is: %f" % (epoch, s, precision_score(y_true, y_pred, average='micro' ) ) )
                metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)
                
            if "precision_weighted" in metrics_of_interest:
                results_temp = {
                    'epoch': epoch,
                    'set':   s,
                    'metric': 'precision_weighted',
                    'value': precision_score(y_true, y_pred, average='weighted' ) 
                }
                logging.info("epoch %d: precision_weighted for the %s set is: %f" % (epoch, s, precision_score(y_true, y_pred, average='weighted' ) ) )
                metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)
            
            if "cohen_kappa_score" in metrics_of_interest:
                results_temp = {
                    'epoch': epoch,
                    'set':   s,
                    'metric': 'cohen_kappa_score',
                    'value': cohen_kappa_score(y_true, y_pred ) 
                }
                logging.info("epoch %d: cohen_kappa_score for the %s set is: %f" % (epoch, s, cohen_kappa_score(y_true, y_pred )  ) )
                metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)

        logging.info(10*"---")
    return metric_dataframe