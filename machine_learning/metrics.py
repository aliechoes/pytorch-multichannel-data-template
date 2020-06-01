import pandas as pd 
from sklearn.metrics import  accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
import numpy as np
from sklearn.metrics import confusion_matrix


def custom_log_loss(y_true, y_pred, existing_labels): 
    custom_y_pred = np.zeros((y_true.shape[0], len(existing_labels)))
    
    for i in range(0, len(y_true)):
        j = y_pred[i]
        custom_y_pred[i,j ] = 1  
    return log_loss(y_true, custom_y_pred, labels=existing_labels)


def metric_history(df, metric_dataframe, epoch, metrics_of_interest ):
 
    existing_labels = df["label"].unique().tolist()
    for s in ["train", "validation", "test"]:
        print(4*"---")
        df_temp = df[ df["set"]==s].reset_index(drop = True).copy()
        y_true = df_temp["label"].astype(int)
        y_pred = df_temp["prediction"].astype(int)
        
        assert (y_pred == -1).sum()== 0  
        

        if "accuracy" in metrics_of_interest:
            results_temp = {
                'epoch': epoch+1,
                'set':   s,
                'metric': 'accuracy',
                'value': accuracy_score(y_true, y_pred ) 
            }
            print("epoch %d: accuracy for the %s set is %f" % (epoch, s, accuracy_score(y_true, y_pred ) ) )
            metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)

        if "cross_entropy" in metrics_of_interest:
                results_temp = {
                    'epoch': epoch+1,
                    'set':   s,
                    'metric': 'cross_entropy',
                    'value': custom_log_loss(y_true, y_pred, existing_labels) 
                }
                print("cross_entropy for the %s set is: %f" % (s,  \
                    custom_log_loss(y_true, y_pred, existing_labels )  ) )
                metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)
  
        
        if "f1_macro" in metrics_of_interest:
            results_temp = {
                'epoch': epoch+1,
                'set':   s,
                'metric': 'f1_macro',
                'value': f1_score(y_true, y_pred, average='macro' ) 
            }
            print("f1_macro for the %s set is: %f" % (s, f1_score(y_true, y_pred, average='macro' ) ) )
            metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)
          
        
        if "f1_weighted" in metrics_of_interest:
            results_temp = {
                'epoch': epoch+1,
                'set':   s,
                'metric': 'f1_weighted',
                'value': f1_score(y_true, y_pred, average='weighted' ) 
            }
            print("f1_weighted for the %s set is: %f" % (s, f1_score(y_true, y_pred, average='weighted' ) ) )
            metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)

    print(4*"---")
    return metric_dataframe