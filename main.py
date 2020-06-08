from utils import *

def main(configs):
    """
    This is the main function which includes the main logic of the script.
    
    Args    
        configs: dictionary file with the format of the config file. a sample
                 file can be find at ./configs/sample-config.json
    """
    # seperating the configs part
    data_configs = configs["data"]
    ml_configs = configs["machine_learning"]
    validation_configs = configs["validation"]
    
    tensorboard_path = validation_configs["tensorboard_path"]


    # check whether there is a model to continue for transfer learning
    checkpoint = get_checkpoint(ml_configs["checkpoint_path"])

    # creating a unique name for the model
    run_name = create_name( ml_configs["model_name"], 
                            ml_configs["optimization_method"] , 
                            ml_configs["optimization_parameters"]["lr"] )
                         
    # creating the tensorboard
    writer = TensorBoardSummaryWriter( os.path.join(tensorboard_path, run_name ) )
    
    
    # creating the folder for the models to be saved per epoch
    model_folder = os.path.join(tensorboard_path, run_name, "models/")
    make_folders(model_folder)

    
    # creating the dataloader
    data_loader = DataLoaderGenerator(data_configs) 
    data_loader.data_frame_creator()
    # number of exsting channels and output classes
    number_of_channels = len(data_loader.existing_channels)
    number_of_classes = len(data_loader.nb_per_class.keys())

    # initialize the model
    model = get_model(  ml_configs,
                        checkpoint,
                        number_of_channels ,
                        number_of_classes)
    
    data_loader.data_loader(model.image_size, checkpoint)

    ## load the optimzer
    optimizer = get_optimizer(  ml_configs, 
                                model, 
                                checkpoint) 
    
    ## load the loss
    criterion = get_loss(ml_configs, data_loader) 

    # train the model and record the results in the metric_dataframe
    _ , metric_dataframe = train(model,   
                                    data_loader, 
                                    optimizer,
                                    criterion,  
                                    writer, 
                                    model_folder,
                                    configs)
    
    # save the dataset with train/validation/test per epoch
    output_folder = os.path.join(tensorboard_path, run_name, "output_files/")
    make_folders(output_folder)
    metric_dataframe.to_csv(os.path.join(output_folder,
                                    "aggregated_results.csv"), index = False)
    
    # save the label of all images and their predictions
    data_loader.df.to_csv(os.path.join(output_folder,
                                    "granular_results.csv"), index = False)






if __name__ == "__main__":
    parser = argparse.ArgumentParser( \
                            description='Starting the deep learning code')
    parser.add_argument('-c',\
                        '--config', \
                        help='config yaml file address', \
                        default="configs/sample_config.json", \
                        type=str)

    args = vars(parser.parse_args())
    
    configs = load_json(args['config'])
    for k in configs:
        print("%s : %s \n" % (k,configs[k]))
    main(configs)

