from importlib.resources import path
import matplotlib.pyplot as plt
import numpy as np
import sys

_seed = 9

#####################################################################
# Defination of class

class result():
    def __init__(self, pathname, name):
        self.path = pathname
        self.file = np.load(self.path, allow_pickle=True).squeeze()
        self.name = name
        
        self.rer, self.ce, self.pe, self.fail, self.cmd = 0, 0, 0, 0, 0
        self.rer_lst, self.ce_lst, self.pe_lst, self.cmd_1st = [], [], [], []

        self.eps = len(self.file)
        self.pass_eps = self.eps

        for i in range(self.eps):
            bf = self.file[i]
            dat = bf[-1]

            # calculate rer and ce
            cmd_count = len(bf)
            this_rer = dat['repetitive_exploration_rate'] - 1
            this_ce = float(dat['explored_area']) / cmd_count

            self.rer_lst.append(this_rer)
            self.ce_lst.append(this_ce)
            self.cmd_1st.append(cmd_count)

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
            self.np_cmd_1st = np.asarray(self.cmd_1st)

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
res = result(sys.argv[1], '2 Agent')
result_list.append(res)

for res in result_list:
    res.print_stats()

#####################################################################
# Make the plot

# color list
color_list = ['b-', 'g-', 'r-', 'y-', 'co', 'mo']

# create x axis
x_axis = range(res.eps)
fig, axs = plt.subplots(4, 1)

# Upper image
for i in range(len(result_list)):
    res = result_list[i]
    axs[0].plot(x_axis, np.sort(res.np_rer_lst), color_list[i], label=res.name)
axs[0].set(ylabel = 'GRER')
axs[0].set_title(f'The GRER over {res.eps} Testing Episodes for one agent')
axs[0].legend()

# Lower image
for i in range(len(result_list)):
    res = result_list[i]
    axs[1].plot(x_axis, np.sort(res.np_ce_lst), color_list[i], label=res.name)
axs[1].set(ylabel = 'CE')
axs[1].set_title(f'The GEs over {res.eps} Testing Episodes for one agent')
#axs[1].legend()

# Lower image
for i in range(len(result_list)):
    res = result_list[i]
    axs[2].plot(x_axis, np.sort(res.np_pe_lst), color_list[i], label=res.name)
axs[2].set(xlabel='episodes', ylabel = 'PE')
axs[2].set_title(f'The PEs over {res.eps} Testing Episodes for one agent')
#axs[2].legend()

for i in range(len(result_list)):
    res = result_list[i]
    axs[3].plot(x_axis, np.sort(res.np_cmd_1st), color_list[i], label=res.name)
axs[3].set(xlabel='episodes', ylabel = 'Cmds')
axs[3].set_title(f'The Cmds over {res.eps} Testing Episodes for one agent')

print(f"GRER avg: {np.mean(res.np_rer_lst)}, " +
      f"CE avg {np.mean(res.np_ce_lst)}, " +
      f"PE avg: {np.mean(res.np_pe_lst)}"
    )

plt.show()
