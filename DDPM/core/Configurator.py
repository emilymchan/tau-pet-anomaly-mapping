"""
Configurator.py

Default class for configuring DL experiments

"""
import copy
import logging
import torch

from dl_utils.config_utils import check_config_file, import_module, set_seed


class DLConfigurator(object):
    """
    DL Configurator
        - parametrization of deep learning via 'config.yaml'
        - initializes Trainer:
            routine for training neural networks
        - initializes FedDownstreamTask:
            routing for downstream evaluations, e.g., classification, anomaly detection
        - starts the experiments
    """
    def __init__(self, config_file, log_wandb=False):
        """
        :param config_file: file
            config.yaml file that contains the DL configuration
        """
        self.dl_config = check_config_file(config_file)
        if self.dl_config is None:
            print('[Configurator::init] ERROR: Invalid configuration file. Configurator will exit...')
            return

        # set seeds
        set_seed(2109)

        # init model and device
        dev = self.dl_config['device']
        self.device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu') if dev == 'gpu' else 'cpu'
        model_class = import_module(self.dl_config['model']['module_name'], self.dl_config['model']['class_name'])
        self.model = model_class(**(self.dl_config['model']['params']))

    def start_training(self, global_model: dict = dict()):
        # Trainer Init
        data = self.load_data(self.dl_config['trainer']['data_loader'], train=True)
        trainer_class = import_module(self.dl_config['trainer']['module_name'], self.dl_config['trainer']['class_name'])
        self.dl_config['trainer']['params']['rescale'] = self.dl_config['trainer']['data_loader']['params']['args']['rescale'] 
        self.trainer = trainer_class(training_params=self.dl_config['trainer']['params'], model=copy.deepcopy(self.model),
                                     data=data, device=self.device, log_wandb=True)
        # Train
        model_state, opt_state, epoch = None, None, 0
        if self.trainer is None:
            logging.error("[Configurator::train::ERROR]: Trainer not defined! Shutting down...")
            exit()
        if 'model_weights' in global_model.keys():
            model_state = global_model['model_weights']
            logging.info("[Configurator::train::INFO]: Model weights loaded!")
        if 'optimizer_weights' in global_model.keys():
            opt_state = global_model['optimizer_weights']
        if 'epoch' in global_model.keys():
            epoch = global_model['epoch']

        logging.info("[Configurator::train]: ################ Starting training ################")
        trained_model_state, trained_opt_state = self.trainer.train(model_state, opt_state, epoch)
        logging.info("[Configurator::train]: ################ Finished training ################")
        logging.info("[Configurator::train]: ################ Starting testing ################")
        self.trainer.test(trained_model_state, data.test_dataloader(), task='Test')
        logging.info("[Configurator::train]: ################ Finished testing ################")
        # self.start_evaluations(trained_model_state)


    def start_evaluations(self, global_model, weight_dir=None):
        data = self.load_data(self.dl_config['trainer']['data_loader'], train=False)

        predictor_class = import_module(self.dl_config['predict']['module_name'], self.dl_config['predict']['class_name'])
        self.dl_config['trainer']['params']['rescale'] = self.dl_config['trainer']['data_loader']['params']['args']['rescale'] 
        self.predictor = predictor_class(training_params=self.dl_config['trainer']['params'], model=copy.deepcopy(self.model),
                                     data=data, device=self.device, log_wandb=True)

        logging.info("[Configurator::train]: ################ Predicting test set ################")
        self.predictor.predict(global_model, data.test_dataloader(), task='test', weight_dir=weight_dir)
        logging.info("[Configurator::train]: ################ Predicting validation set ################")        
        self.predictor.predict(global_model, data.val_dataloader(), task='val', weight_dir=weight_dir)
        logging.info("[Configurator::train]: ################ Predicting training set ################")
        self.predictor.predict(global_model, data.train_dataloader(), task='train', weight_dir=weight_dir)
        

    @staticmethod
    def load_data(data_loader_config, train=True):
        """
        :param data_loader_config: dict
            parameters for data loaders -  must include module/class name and params
        :param train: bool
            True if the datasets are used for training, False otherwise
        :return: list of
           train, val, test datasets if train is True, dict with downstream datasets otherwise
        """
        data_loader_module = import_module(data_loader_config['module_name'], data_loader_config['class_name'])

        # if train:
        return data_loader_module(**(data_loader_config['params']))

