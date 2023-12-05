class ParameterCarrier:

    def __init__(self, args):
        self.algorithm = "privtrace"
        # self.dataset_file_name = args['dataset_file_name']
        self.total_epsilon = args['total_epsilon']
        self.epsilon_partition = args['epsilon_partition']
        self.trajectory_number_to_generate = args['trajectory_number_to_generate']
        self.lat_range = args['lat_range']
        self.lon_range = args['lon_range']
        self.fixed_divide_parameter = args['fixed_divide_parameter']
        self.save_name = args['save_name']
        self.dataset = args['dataset']
        self.data_name = args['data_name']
        self.training_data_name = args['training_data_name']
        self.level2_max_divide = args['level2_max_divide']
        self.n_bins = args['n_bins']
        self.seed = args['seed']


