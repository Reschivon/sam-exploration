import matplotlib.pyplot as plt
import numpy as np
import sys

_eps = 200

#####################################################################
# Defination of class

class result():
    def __init__(self, path):
        self.file = np.load(path, allow_pickle=True).squeeze()
        self.eps = _eps
        self.pass_eps = _eps
        this_rer, this_ce, this_pe = 0, 0, 0
        self.rer, self.ce, self.pe, self.fail = 0, 0, 0, 0
        self.rer_list, self.ce_list, self.pe_list, self.cmd_list, self.bandwidth_list, self.coverage_list, self.overlap_list =\
              [], [], [], [], [], [], []

        for i in range(_eps):
            cmd_count = 0
            bf = self.file[i]
            data = bf[-1]
            cmd_count = len(bf)
            # calculate rer and ce
            this_rer = data['repetitive_exploration_rate'] - 1
            this_ce = float(data['explored_area']) / cmd_count
            # this_bandwidth = data['bandwidth']
            this_coverage = data['ratio_explored']
            this_overlap = data['overlapped_ratio']

            self.rer += this_rer
            self.ce += this_ce

            # RER
            self.rer_list.append(this_rer)
            # CE
            self.ce_list.append(this_ce)
            # PE
            if data['cumulative_distance'] >= 1:
                this_pe = float(data['explored_area']) / data['cumulative_distance']
                self.pe += this_pe
                self.pe_list.append(this_pe)
            else:
                self.pe_list.append(0.0)
            # CMD
            self.cmd_list.append(cmd_count)

            # BANDWIDTH
            # self.bandwidth_list.append(this_bandwidth)

            #COVERAGE
            self.coverage_list.append(this_coverage)

            # overlap
            self.overlap_list.append(this_overlap)


            # calculate fail rate
            if data['cube_found'] == False:
                self.fail += 1
            
            self.np_rer_list = np.asarray(self.rer_list)
            self.np_ce_list = np.asarray(self.ce_list)
            self.np_pe_list = np.asarray(self.pe_list)
            self.np_cmd_list = np.asarray(self.cmd_list)
            self.np_bandwidth_list = np.asarray(self.bandwidth_list)
            self.np_coverage_list = np.asarray(self.coverage_list)
            self.np_overlap_list = np.asarray(self.overlap_list)
    
    def print_stats(self):
        def print_array(arr):
            print('\t', np.mean(arr), 'std:', np.std(arr))

        print('RER:')
        print_array(self.np_rer_list)

        print('CE:')
        print_array(self.np_ce_list)

        print('PE:')
        print_array(self.np_pe_list)

        print('Commands:')
        print_array(self.np_cmd_list)

        print('Overlap Ratio:')
        print_array(self.np_overlap_list)

        # print('    Bandwidth:')
        # print_array(self.np_bandwidth_list)

        print('Coverage:') 
        # only the last
        print_array(self.np_coverage_list[-1])

        print('not_found:\n\t', self.fail)

#####################################################################
# Create results
def visualize(eval_path):
    res = result(eval_path)
    res.print_stats()

if __name__ == '__main__':
    visualize(sys.argv[1])

#####################################################################
# Make the plot

# # color list
# color_list = ['b-', 'g-', 'r-', 'y-', 'co', 'mo']

# # create x axis
# x_axis = range(_eps)
# fig, axs = plt.subplots(3, 1)

# # Upper image
# for i in range(len(result_list)):
#     res = result_list[i]
#     axs[0].plot(x_axis, res.np_rer_lst, color_list[i], label=res.name)
# axs[0].set(ylabel = 'GRER')
# axs[0].set_title('The GRERs of SAM-VFM, SAM, and ST-COM over 200 Testing Episodes')
# axs[0].legend()

# # Lower image
# for i in range(len(result_list)):
#     res = result_list[i]
#     axs[1].plot(x_axis, res.np_ce_lst, color_list[i], label=res.name)
# axs[1].set(ylabel = 'CE')
# axs[1].set_title('The GEs of SAM-VFM, SAM, and ST-COM over 200 Testing Episodes')
# #axs[1].legend()

# # Lower image
# for i in range(len(result_list)):
#     res = result_list[i]
#     axs[2].plot(x_axis, res.np_pe_lst, color_list[i], label=res.name)
# axs[2].set(xlabel='episodes', ylabel = 'PE')
# axs[2].set_title('The PEs of SAM-VFM, SAM, and ST-COM over 200 Testing Episodes')
# #axs[2].legend()

# plt.show()
