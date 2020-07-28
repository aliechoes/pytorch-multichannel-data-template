from utils import *
from train.train import train
import logging


def main(configs):
    """
    This is the main function which includes the main logic of the script.
    
    Args    
        configs: dictionary file with the format of the config file. a sample
                 file can be find at ./configs/sample-config.json
    """
    
    logging.info("Starting the main pipeline")
    system_info()
    
    device = configs["training"]["device"]

    # seperating the configs part
    model_configs = configs["model"]
    loss_configs = configs["loss"]
    optimizer_configs = configs["optimizer"]
    training_configs = configs["training"]
    data_loader_configs = configs["data_loader"]
    tensorboard_configs = configs["tensorboard"]
     


    # check whether there is a model to continue for transfer learning
    checkpoint = get_checkpoint(model_configs["checkpoint_path"])

    # creating a unique name for the model
    run_name = create_name( model_configs["network"], 
                            optimizer_configs["optimization_method"] , 
                            optimizer_configs["optimization_parameters"]["lr"] )
                         
    # creating the tensorboard
    writer = TensorBoardSummaryWriter(tensorboard_configs, run_name  )
    
    
    # creating the folder for the models to be saved per epoch
    model_folder = os.path.join(writer.tensorboard_path, run_name, "models/")
    make_folders(model_folder)

    
    # creating the dataloader
    data_loader = DataLoaderGenerator(data_loader_configs) 
    data_loader.data_frame_creator()

    # number of exsting channels and output classes
    number_of_channels = len(data_loader.existing_channels)
    number_of_classes = len(data_loader.classes)

    # initialize the model
    model = get_model(  model_configs,
                        device,
                        checkpoint,
                        number_of_channels ,
                        number_of_classes)
    
    data_loader.data_loader(model.image_size, checkpoint)

    ## load the optimzer
    optimizer = get_optimizer(  optimizer_configs, 
                                model, 
                                checkpoint) 
    
    ## load the loss
    criterion = get_loss(loss_configs, data_loader, device) 

    # train the model and record the results in the metric_dataframe
    metric_dataframe, best_criteria_value, best_epoch = train(   model, 
                                                                data_loader, 
                                                                optimizer,
                                                                criterion,  
                                                                writer, 
                                                                model_folder,
                                                                training_configs,
                                                                device)
    
    
    writer.add_hparams(configs, best_criteria_value, best_epoch )
    # save the dataset with train/validation/test per epoch
    output_folder = os.path.join(writer.tensorboard_path, 
                                        run_name, "output_files/")
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
    logger(configs["training"]["verbosity"])

    for k in configs:
        logging.info("%s : %s \n" % (k,configs[k]))
    
    
    main(configs)

