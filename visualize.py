import matplotlib.pyplot as plt
import numpy as np

_seed = 9
_eps = 200

#####################################################################
# Defination of class

class result():
    def __init__(self, _fn):
        self.path = "./eval/20220926T024451181854-large_columns_sigmoid.npy"
        self.file = np.load(self.path, allow_pickle=True).squeeze()
        self.name = _fn
        self.eps = _eps
        self.pass_eps = _eps
        this_rer, this_ce, this_pe = 0, 0, 0
        self.rer, self.ce, self.pe, self.fail = 0, 0, 0, 0
        self.rer_lst, self.ce_lst, self.pe_lst = [], [], []

        for i in range(_eps):
            cmd_count = 0
            bf = self.file[i]
            dat = bf[-1]
            cmd_count = len(bf)
            # calculate rer and ce
            this_rer = dat['repetitive_exploration_rate'] - 1
            this_ce = float(dat['explored_area']) / cmd_count
            self.rer += this_rer
            self.ce += this_ce
            self.rer_lst.append(this_rer)
            self.ce_lst.append(this_ce)
            # calculate pe
            if dat['cumulative_distance'] >= 1:
                this_pe = float(dat['explored_area']) / dat['cumulative_distance']
                self.pe += this_pe
                self.pe_lst.append(this_pe)
            else:
                self.pe_lst.append(0.0)
            # calculate fail rate
            if dat['cube_found'] == False:
                self.fail += 1
            
            self.np_rer_lst = np.asarray(self.rer_lst)
            self.np_ce_lst = np.asarray(self.ce_lst)
            self.np_pe_lst = np.asarray(self.pe_lst)
    
    def print_stats(self):
        print(self.name)
        print('    rer:', self.rer / self.pass_eps, ' std:', np.std(self.np_rer_lst))
        print('    ce:', self.ce / self.pass_eps, ' std:', np.std(self.np_ce_lst))
        print('    pe:', self.pe / len(self.pe_lst), ' std:', np.std(self.np_pe_lst))
        print('    not_found:', self.fail)

#####################################################################
# Create results

result_list = []
# result_list.append(result("SAM"))
# result_list.append(result("ST-COM"))
# result_list.append(result("SAM-VFM (A)"))
# result_list.append(result("SAM-VFM (B)"))
# result_list.append(result("RAND"))
result_list.append(result("Eval to 200 episodes"))

for res in result_list:
    res.print_stats()

#####################################################################
# Make the plot

# color list
color_list = ['b-', 'g-', 'r-', 'y-', 'co', 'mo']

# create x axis
x_axis = range(_eps)
fig, axs = plt.subplots(3, 1)

# Upper image
for i in range(len(result_list)):
    res = result_list[i]
    axs[0].plot(x_axis, np.sort(res.np_rer_lst), color_list[i], label=res.name)
axs[0].set(ylabel = 'GRER')
axs[0].set_title('The GRER for SAM-IVFM over 200 Testing Episodes')
axs[0].legend()

# Lower image
for i in range(len(result_list)):
    res = result_list[i]
    axs[1].plot(x_axis, np.sort(res.np_ce_lst), color_list[i], label=res.name)
axs[1].set(ylabel = 'CE')
axs[1].set_title('The GEs for SAM-IVFM over 200 Testing Episodes')
#axs[1].legend()

# Lower image
for i in range(len(result_list)):
    res = result_list[i]
    axs[2].plot(x_axis, np.sort(res.np_pe_lst), color_list[i], label=res.name)
axs[2].set(xlabel='episodes', ylabel = 'PE')
axs[2].set_title('The PEs for SAM-IVFM over 200 Testing Episodes')
#axs[2].legend()

print(f"GRER avg: {np.mean(res.np_rer_lst)}, " +
        f"GE avg {np.mean(res.np_ce_lst)}, " +
        f"PE avg: {np.mean(res.np_pe_lst)}")

plt.show()
