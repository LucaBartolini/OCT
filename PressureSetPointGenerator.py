import numpy as np
import os
import matplotlib.pyplot as plt


class waveform:
    """A class to generate an arbitrary waveform
    """

    def __init__(self, **kwargs):
        # frequency with which setpoints will be given out
        self.freq = kwargs.get('Bscan_RepRate', 33.333)
        self.delta_t = 1/self.freq  # Delta_t between setpoints
        self.waveform = np.array([])  # waveform
        self.max_suction = 600  # mbar
        print(f"B-scan Repetition rate set at {self.freq:.5} Hz")
        print(f"The setpoints will be spaced  {self.delta_t:.5} seconds")
        print("========= END INITIALIZATION =========\n")

    def add_flat(self, time, level=None):
        if level == None:
            if self.waveform.size != 0:
                level = self.waveform[-1]  # keeps the same level
            else:
                print('You have to provide a level at which to keep the')
        assert (level >= 0), "`level` must be positive"
        N_pts = int(np.around(time/self.delta_t))
        flat = np.full((N_pts, ), level)
        self.waveform = np.append(self.waveform, flat)
        return self.waveform

    def jump_to(self, suction):
        assert (suction >= 0), "`level` must be positive"
        self.waveform = np.append(self.waveform, [suction])
        return self.waveform

    def add_ramp(self, to_suction, time):
        if self.waveform.size == 0:
            self.waveform = np.asarray([0])
        ramp_start = self.waveform[-1]
        N_pts = int(np.around(time/self.delta_t))
        ramp = np.linspace(ramp_start, to_suction, N_pts)
        self.waveform = np.append(self.waveform, ramp)
        return self.waveform

    def add_oscillations(self, freq, min_lvl, max_lvl, N_osc, initial_phase_deg=90):
        assert min_lvl >= 0, "`p_min` must be positive"
        assert max_lvl <= self.max_suction,  "`p_max` must be below 1000 mbar"
        assert min_lvl < max_lvl, "`p_min` must me smaller than `p_max`"
        assert type(N_osc) == int, "N_osc must be integer"
        period = 1/freq
        N_pts = int(np.around(period/self.delta_t))  # in one period
        phases = np.linspace(0, 2*np.pi, num=N_pts)
        phases += 2*np.pi*initial_phase_deg/360  # so the oscillation starts smooth
        amplitude = (max_lvl - min_lvl)/2
        offset = (max_lvl + min_lvl)/2
        oscillation = offset + amplitude*np.cos(phases)

        oscillation = np.tile(oscillation, N_osc)

        self.waveform = np.append(self.waveform, oscillation)

        return self.waveform

    def to_csv(self, filename):
        if not filename.endswith('.csv'):
            filename += '.csv'
        self.waveform = np.append(self.freq, self.waveform)
        np.savetxt(filename, self.waveform, delimiter=",")
        return f"File `{filename}` saved at: \n{os.getcwd()}\n===================================="

    def from_csv(self, filename):
        if not filename.endswith('.csv'):
            filename += '.csv'
        array = np.genfromtxt(filename, delimiter=',')
        self.freq, self.waveform = array[0], array[1:]
        print(f"File '{filename}' successfully read")
        print(f"{len(self.waveform)/self.freq:.5} second long waveform, with sampling {self.freq:.5} Hz.")

    def __len__(self):
        return (self.waveform.size)

    def plot(self):
        ## Let's see how the waveform looks live
        ## creation of x-axis (time axis)
        time = np.linspace(0, self.delta_t*len(self.waveform), num = len(self.waveform))

        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(time, self.waveform)

        sup_title = f"Time series of the setpoint for Suction (mbar below atmospheric pressure)"
        fig.suptitle(sup_title, fontsize=13)

        ax.set_ylabel('Pressure Setpoint (mbar)', fontsize=12)
        ax.set_xlabel('Time (s)', fontsize=12)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')        
        
        return fig
if __name__ == '__main__':
    print('`PressureSetPointGenerator` compiled successfully')
