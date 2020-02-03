import pandas as pd 
from sklearn.metrics import  accuracy_score
from sklearn.metrics import f1_score


def metric_history(df, metric_dataframe, epoch, metrics_of_interest ):
    print(("epoch", epoch))
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
            print("accuracy for the %s set is: %f" % (s, accuracy_score(y_true, y_pred ) ) )
            metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)

        
        if "dice" in metrics_of_interest:
            results_temp = {
                'epoch': epoch+1,
                'set':   s,
                'metric': 'dice',
                'value': f1_score(y_true, y_pred, average='macro' ) 
            }
            print("dice for the %s set is: %f" % (s, f1_score(y_true, y_pred, average='macro' ) ) )
            metric_dataframe = metric_dataframe.append(results_temp, ignore_index=True)
    print(4*"---")
    return metric_dataframe