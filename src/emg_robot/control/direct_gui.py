from threading import Thread
import tkinter as tk
from tkinter import ttk

from emg_robot.defaults import I2C_ADDRESSES, ROBOT_IP, EMG_CHANNEL_NAMES


def create_weight_row(tab, row, name, w, th):
    w_var = tk.DoubleVar(value=w)
    th_var = tk.DoubleVar(value=th)
    act_var = tk.DoubleVar(value=0.)

    label = ttk.Label(tab, text=name)
    label.grid(column=0, row=row, sticky="nw")

    vert_layout = tk.Frame(tab)
    vert_layout.grid(column=1, row=row, sticky="nsew")

    w_slider = tk.Scale(vert_layout, variable=w_var, from_=-1., to=1., resolution=0.05, showvalue=False, orient=tk.HORIZONTAL)
    w_slider.pack(fill=tk.BOTH, expand=True)

    channel_activity = ttk.Progressbar(vert_layout, variable=act_var, mode='determinate', orient=tk.HORIZONTAL)
    channel_activity.pack(fill=tk.BOTH, expand=True)    

    w_spin = ttk.Spinbox(tab, textvariable=w_var, from_=-1., to=1., increment=0.05)
    w_spin.grid(column=2, row=row, sticky="ne")

    th_spin = ttk.Spinbox(tab, textvariable=th_var, from_=0., to=10000., increment=10.)
    th_spin.grid(column=3, row=row, sticky="ne")

    return w_var, th_var, act_var


class DirectControllerGUI(tk.Tk):
    def __init__(self, 
                 controller,
                 channel_names):
        super().__init__()
        #ttk.Style(self).theme_use('default')

        self.controller = controller
        self.control_thread = None
        self.running = False

        # Special variables that will be linked to GUI elements
        self.pitch_weights = []
        self.pitch_thresholds = []
        self.pitch_f = tk.DoubleVar(value = controller.pitch_f)
        self.pitch_activity = []
        self.roll_weights = []
        self.roll_thresholds = []
        self.roll_f = tk.DoubleVar(value = controller.roll_f)
        self.roll_activity = []

        self.title("Control Weights")
        self.geometry("640x250")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Gains for transforming the emg signals
        tabs = ttk.Notebook(self)
        tab_pitch = ttk.Frame(tabs)
        tab_roll = ttk.Frame(tabs)
        tabs.add(tab_pitch, text="Pitch")
        tabs.add(tab_roll, text="Roll")
        tabs.grid(column=0, row=0, columnspan=2, sticky="nsew", pady=(5, 10))


        # Some general options
        ttk.Label(self, text="Pitch factor").grid(column=0, row=2, sticky="nw")
        pf_spin = ttk.Spinbox(self, textvariable=self.pitch_f, from_=0., to=10., increment=0.01)
        pf_spin.grid(column=1, row=2, sticky="ne")
        ttk.Label(self, text="Roll factor").grid(column=0, row=3, sticky="nw")
        rf_spin = ttk.Spinbox(self, textvariable=self.roll_f, from_=0., to=10., increment=0.01)
        rf_spin.grid(column=1, row=3, sticky="ne")

        self.btn_start_stop = ttk.Button(self, text="Start", command=self.on_start_stop)
        self.btn_start_stop.grid(column=1, row=4, sticky="ew", pady=(7, 7))
        self.progress = ttk.Progressbar(self, mode="indeterminate", orient=tk.HORIZONTAL)
        self.progress.grid(column=0, row=5, columnspan=2, sticky="sew")
        self.grid_columnconfigure(0, weight=3)
        self.grid_rowconfigure(5, weight=1)

        
        # Create column labels for each tab
        for t in (tab_pitch, tab_roll):
            t.grid_columnconfigure(1, weight=3)
            ttk.Label(t, text="Muscle").grid(column=0, row=0, sticky="ew")
            ttk.Label(t, text="Weight").grid(column=1, row=0, columnspan=2, sticky="ew")
            ttk.Label(t, text="Threshold").grid(column=3, row=0, sticky="ew")

        # Just for convenience
        cpw = controller.pitch_weights
        cpth = controller.pitch_thresholds
        crw = controller.roll_weights
        crth = controller.roll_thresholds

        for i, channel in enumerate(channel_names):
            pw, pth, pa = create_weight_row(tab_pitch, i+1, channel, cpw[i], cpth[i])
            rw, rth, ra = create_weight_row(tab_roll, i+1, channel, crw[i], crth[i])
            self.pitch_weights.append(pw)
            self.pitch_thresholds.append(pth)
            self.pitch_activity.append(pa)
            self.roll_weights.append(rw)
            self.roll_thresholds.append(rth)
            self.roll_activity.append(ra)

    def on_start_stop(self):
        if not self.running:
            self.btn_start_stop.config(text="Stop")
            self.progress.start()
            self.start_controller()
        else:
            self.stop_controller()
            self.progress.stop()
            self.btn_start_stop.config(text="Start")

    def on_closing(self):
        self.stop_controller()
        self.destroy()

    def start_controller(self):
        self.running = True
        self.control_thread = Thread(target=self.control_loop)
        self.control_thread.start()

    def stop_controller(self):
        self.running = False
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join()
        self.control_thread = None

    def control_loop(self):
        c = self.controller
        pw = self.pitch_weights
        pth = self.pitch_thresholds
        pf = self.pitch_f
        pa = self.pitch_activity
        rw = self.roll_weights
        rth = self.roll_thresholds
        rf = self.roll_f
        ra = self.roll_activity

        try:
            while self.running:
                for i in range(len(pw)):
                    c.pitch_weights[i] = pw[i].get()
                    c.pitch_thresholds[i] = pth[i].get()
                    c.roll_weights[i] = rw[i].get()
                    c.roll_thresholds[i] = rth[i].get()
                c.pitch_f = pf.get()
                c.roll_f = rf.get()
                c.run_once()

                act = c.emg_activity() * 100
                for i in range(len(pa)):
                    pa[i].set(act[i])
                    ra[i].set(act[i])
        except:
            pass


def start_gui(controller, channel_names):
    gui = DirectControllerGUI(controller, channel_names)
    gui.mainloop()


if __name__ == '__main__':
    from .controller_direct import DirectController
    controller = DirectController(I2C_ADDRESSES, ROBOT_IP)
    start_gui(controller, EMG_CHANNEL_NAMES)
