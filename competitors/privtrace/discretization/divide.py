import numpy as np
from config.parameter_carrier import ParameterCarrier


class Divide:

    def __init__(self, cc: ParameterCarrier):
        self.cc = cc

    # divide parameter, output is array[x_divide_number, y_divide_number, x_increase, y_increase]
    def level1_divide_parameter(self, total_density, trajectory_number, border2):
        divide_threshold = 60
        initial_parameter = 600
#         initial_parameter = 1
        top = border2[0]
        bot = border2[1]
        lef = border2[2]
        rig = border2[3]
        print("total_density:", total_density)
        para = np.floor(np.sqrt(total_density / initial_parameter))
        if self.cc.fixed_divide_parameter != 0:
            para = self.cc.fixed_divide_parameter
            print(f"para is fixed: {para}")
        else:
            self.cc.level2_max_divide = (self.cc.n_bins+2) / para
        assert para > 1, 'need no dividing'
        if para > divide_threshold:
            para = divide_threshold

        x_divide_number = para
        y_divide_number = para
        x_divide_number = int(x_divide_number)
        y_divide_number = int(y_divide_number)
        print("(x_divide, y_divide)", (x_divide_number, y_divide_number))
        # we round the number of subcells to power of 2 for fair comparison
        x_divide_number_lower = int(2**np.floor(np.log2(x_divide_number)))
        x_divide_number_upper = int(2**np.ceil(np.log2(x_divide_number)))
        y_divide_number_lower = int(2**np.floor(np.log2(y_divide_number)))
        y_divide_number_upper = int(2**np.ceil(np.log2(y_divide_number)))
        x_divide_number = x_divide_number_lower if x_divide_number - x_divide_number_lower < x_divide_number_upper - x_divide_number else x_divide_number_upper
        y_divide_number = y_divide_number_lower if y_divide_number - y_divide_number_lower < y_divide_number_upper - y_divide_number else y_divide_number_upper
        assert x_divide_number == y_divide_number, "x_divide_number != y_divide_number"
        print("FOR FAIR COMPARISON, THE DIVIDE VALUE IS ROUNDED TO THE CLOSEST POWER OF 2", (x_divide_number, y_divide_number))
        x_increase = 1 / x_divide_number * (rig - lef)
        y_increase = 1 / y_divide_number * (top - bot)
        divide_parameter1 = np.array([x_divide_number, y_divide_number, x_increase, y_increase])
        return divide_parameter1

    def subdividing_parameter(self, noisy_density):
        initial_parameter = 20000 / 13
        subdivide_parameter1_ = int(np.ceil(np.sqrt(noisy_density / initial_parameter)))
        # for fair comparison
        subdivide_parameter1_ = min(subdivide_parameter1_, int(self.cc.level2_max_divide))
        subdivide_parameter1 = 2**np.floor(np.log2(subdivide_parameter1_))
        print("FOR FAIR COMPARISON, THE VALUE IS ROUNDED TO POWER OF 2,_gen", subdivide_parameter1_, "->", subdivide_parameter1)
        return subdivide_parameter1