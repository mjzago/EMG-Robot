from threading import Thread
import tkinter as tk
from tkinter import ttk

from emg_robot.defaults import I2C_ADDRESSES, ROBOT_IP, EMG_CHANNEL_NAMES


def create_weight_row(tab, row, name, w, th):
    w_var = tk.DoubleVar(value=w)
    th_var = tk.DoubleVar(value=th)

    label = tk.Label(tab, text=name)
    label.grid(column=0, row=row, sticky="nw")

    w_slider = tk.Scale(tab, variable=w_var, from_=-1., to=1., resolution=0.05, orient=tk.HORIZONTAL)
    w_slider.grid(column=1, row=row, sticky="ew")

    w_spin = ttk.Spinbox(tab, textvariable=w_var, from_=-1., to=1., increment=0.05)
    w_spin.grid(column=2, row=row, sticky="ne")
    
    th_spin = ttk.Spinbox(tab, textvariable=th_var, from_=0., to=1., increment=0.05)
    th_spin.grid(column=3, row=row, sticky="ne")

    return w_var, th_var


class DirectControllerGUI(tk.Tk):
    def __init__(self, 
                 controller,
                 channel_names):
        super().__init__()

        self.controller = controller
        self.control_thread = None
        self.running = False

        # Special variables that will be linked to GUI elements
        self.pitch_weights = []
        self.pitch_thresholds = []
        self.pitch_f = tk.DoubleVar(value = controller.pitch_f)
        self.roll_weights = []
        self.roll_thresholds = []
        self.roll_f = tk.DoubleVar(value = controller.roll_f)

        self.title("Control Weights")
        self.geometry("280x500")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Some general options
        tk.Label(self, text="Pitch factor").grid(column=0, row=0, sticky="nw")
        pf_spin = ttk.Spinbox(self, textvariable=self.pitch_f, from_=0., to=10., increment=0.05)
        pf_spin.grid(column=1, row=0, sticky="ne")
        tk.Label(self, text="Roll factor").grid(column=0, row=1, sticky="nw")
        rf_spin = ttk.Spinbox(self, textvariable=self.roll_f, from_=0., to=10., increment=0.05)
        rf_spin.grid(column=1, row=1, sticky="ne")

        self.progress = ttk.Progressbar(self, mode="indeterminate", orient=tk.HORIZONTAL)
        self.progress.grid(column=0, row=2, sticky="ew")
        self.grid_columnconfigure(0, weight=3)
        self.btn_start_stop = tk.Button(self, text="Start", command=self.on_start_stop)
        self.btn_start_stop.grid(column=1, row=2, sticky="ew")
        ttk.Separator(self).grid(column=0, row=3, columnspan=2, sticky="ew")

        # Gains for transforming the emg signals
        tabs = ttk.Notebook(self)
        tab_pitch = ttk.Frame(tabs)
        tab_roll = ttk.Frame(tabs)
        tabs.add(tab_pitch, text="Pitch")
        tabs.add(tab_roll, text="Roll")
        tabs.grid(column=0, row=4, columnspan=2, sticky="nsew")
        
        # Create column labels for each tab
        for t in (tab_pitch, tab_roll):
            t.grid_columnconfigure(1, weight=3)
            tk.Label(t, text="Muscle").grid(column=0, row=0, sticky="ew")
            tk.Label(t, text="Weight").grid(column=1, row=0, columnspan=2, sticky="ew")
            tk.Label(t, text="Threshold").grid(column=3, row=0, sticky="ew")

        # Just for convenience
        cpw = controller.pitch_weights
        cpth = controller.pitch_thresholds
        crw = controller.roll_weights
        crth = controller.roll_thresholds

        for i, channel in enumerate(channel_names):
            pw, pth = create_weight_row(tab_pitch, i+1, channel, cpw[i], cpth[i])
            rw, rth = create_weight_row(tab_roll, i+1, channel, crw[i], crth[i])
            self.pitch_weights.append(pw)
            self.pitch_thresholds.append(pth)
            self.roll_weights.append(rw)
            self.roll_thresholds.append(rth)

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
        self.quit()

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
        rw = self.roll_weights
        rth = self.roll_thresholds
        rf = self.roll_f

        try:
            while self.running:
                for i in range(len(pw)):
                    c.pitch_weights[i] = pw[i].get()
                    c.pitch_thresholds[i] = pth[i].get()
                    c.pitch_f = pf.get()
                    c.roll_weights[i] = rw[i].get()
                    c.roll_thresholds[i] = rth[i].get()
                    c.roll_f = rf.get()
                c.run_once()
        except:
            pass


def start_gui(controller, channel_names):
    gui = DirectControllerGUI(controller, channel_names)
    gui.mainloop()


if __name__ == '__main__':
    from .controller_direct import DirectController
    controller = DirectController(I2C_ADDRESSES, ROBOT_IP)
    start_gui(controller, EMG_CHANNEL_NAMES)
