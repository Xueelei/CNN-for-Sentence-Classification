class Evaluator:
    def __init__(self, params, utils):
        self.params = params
        self.utils = utils
        self.log_time = {}

    def evaluate_CNN(self):
        config = 'CNN use pre-trained embeddings'
        print('------{}------'.format(config))
        training_time = self.utils.train(save_plots_as=config)
        self.log_time[config] = training_time
        
        config = 'Predict for test file'
        print('------{}------'.format(config))
        self.utils.test()
