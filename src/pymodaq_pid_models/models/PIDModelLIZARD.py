from pymodaq.pid.utils import PIDModelGeneric, OutputToActuator, InputFromDetector, main
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, pyqtSlot, QThread
from pyqtgraph.dockarea import Dock
#from pymodaq.daq_utils.daq_utils import linspace_step, check_modules, getLineInfo
from pymodaq.daq_move.daq_move_main import DAQ_Move
from pymodaq.daq_utils.plotting.viewer1D.viewer1D_main import Viewer1D
from pymodaq_plugins.daq_move_plugins.daq_move_Mock import DAQ_Move_Mock
from pymodaq.daq_utils import math_utils
from pymodaq.daq_utils.h5modules import H5Saver
import time
from datetime import datetime
import numpy as np
from collections import OrderedDict


class PIDModelLIZARD(PIDModelGeneric):

    params = [
        {'title': 'Stabilization', 'name': 'stabilization', 'type': 'group', 'children': [
            {'title': 'Ellipse phase (rad):', 'name': 'ellipse_phase', 'type': 'float', 'value': 0},
            {'title': 'Error (rad):', 'name': 'error', 'type': 'float', 'value': 0},
            {'title': 'Delay line absolute position (microns):', 'name': 'delay_line_absolute_position',
             'type': 'float', 'value': 0},
            {'title': 'Relative order to actuator (microns):', 'name': 'relative_order_to_actuator', 'type': 'float',
             'value': 0},
            {'title': 'Laser wavelength (microns):', 'name': 'laser_wl', 'type': 'float', 'value': 0.8},
            {'title': 'Correction sign:', 'name': 'correction_sign', 'type': 'int', 'value': 1, 'min': -1, 'max': 1,
             'step': 2},
            {'title': 'Offsets', 'name': 'offsets', 'type': 'group', 'children': [
                {'title': 'Phase time zero (rad):', 'name': 'time_zero_phase', 'type': 'float', 'value': 0},
                {'title': 'Piezo (nm):', 'name': 'piezo_offset', 'type': 'float', 'value': 0}]},
        ]},
        {'title': 'Calibration', 'name': 'calibration', 'type': 'group', 'expanded': True, 'visible': True,
         'children': [
            {'title': 'Start calibration:', 'name': 'start_calibration', 'type': 'bool', 'value': False},
            {'title': 'Timeout:', 'name': 'timeout', 'type': 'int', 'value': 10000},
            {'title': 'Calibration', 'name': 'calibration_move', 'type': 'group', 'expanded': False, 'visible': True,
             'children': [
                {'title': 'Start pos:', 'name': 'start', 'type': 'float', 'value': -10040.},
                {'title': 'Stop pos:', 'name': 'stop', 'type': 'float', 'value': -10041.},
                {'title': 'Number of steps:', 'name': 'Nstep', 'type': 'int', 'value': 11},
                {'title': 'Averaging:', 'name': 'average', 'type': 'int', 'value': 1}]},
            {'title': 'Ellipse params', 'name': 'calibration_ellipse', 'type': 'group', 'expanded': False,
             'visible': True, 'children': [
                 {'title': 'Dx:', 'name': 'dx', 'type': 'float', 'value': 0.0},
                 {'title': 'Dy:', 'name': 'dy', 'type': 'float', 'value': 0.0},
                 {'title': 'x0:', 'name': 'x0', 'type': 'float', 'value': 0.00},
                 {'title': 'y0:', 'name': 'y0', 'type': 'float', 'value': 0.00},
                 {'title': 'theta (°):', 'name': 'theta', 'type': 'float','value': 0.00}
            ]}
         ]},
        {'title': 'Stabilized scan', 'name': 'stabilized_scan', 'type': 'group', 'expanded': False, 'visible': True,
         'children': [
            {'title': 'Do stabilized scan:', 'name': 'do_stabilized_scan', 'type': 'bool', 'value': False},
            {'title': 'Timeout (ms):', 'name': 'timeout', 'type': 'int', 'value': 5000},
            {'title': 'Stabilized scan parameters', 'name': 'stabilized_scan_parameters', 'type': 'group',
             'expanded': False, 'visible': True, 'children': [
                {'title': 'Length (microns):', 'name': 'length', 'type': 'float', 'value': 1},
                {'title': 'Step size (microns):', 'name': 'step_size', 'type': 'float', 'value': 0.02},
                {'title': 'Iterations per setpoint:', 'name': 'iterations_per_setpoint', 'type': 'int', 'value': 2}
            ]}
         ]}
    ]

    def __init__(self, pid_controller):
        super().__init__(pid_controller)

        self.dock_calib = Dock('Calibration')
        widget_calib = QtWidgets.QWidget()
        self.viewer_calib = Viewer1D(widget_calib)
        widget_ellipse = QtWidgets.QWidget()
        self.viewer_ellipse = Viewer1D(widget_ellipse)
        self.viewer_ellipse.show_data([np.zeros((10,)), np.zeros((10,))])
        self.dock_calib.addWidget(widget_calib)
        self.dock_calib.addWidget(widget_ellipse, row=0, col=1)
        self.pid_controller.dock_area.addDock(self.dock_calib)
        self.dock_calib.float()

        self.curr_time = time.perf_counter()
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        #  self.timer.timeout.connect(self.timeout)
        self.lsqe = math_utils.LSqEllipse()

        #self.h5_saver = H5Saver()
        self.channel_pytables_array = None
        self.channel_pytables_array_time = None
        self.current_scan_path = None
        self.scope_spectrum_shape = None
        self.calibration_scan_shape = None
        self.stabilized_scan_shape = None

        self.calibration_scan_running = False
        self.stabilized_scan_running = False
        self.det_done_flag = False
        self.move_done_flag = False
        self.timeout_scan_flag = False
        self.calibration_done = False
        self.time_zero_phase_defined = False

        # The time zero corresponds to the temporal overlap of the XUV and IR pulses. More precisely here it is defined
        # as the starting position of the delay stage during a calibration scan. Thus this position is related to a
        # particular calibration scan and defined by the user.
        # The time_zero_phase is set at the first iteration of the feedback loop and will be offset to define the
        # delay_phase which is then zero at time zero.
        # The delay phase is unwrapped to keep track of the past from the begining of the feedback loop. Thus there is
        # a direct proportional relation between delay_phase and the pump/probe delay.
        self.time_zero_phase = 0  # within [-pi, +pi]
        self.ellipse_phase = 0  # within [-pi, +pi]
        self.delay_phase = 0  # unwrapped phase
        # This attribute is used to record every phase measurement (within [-pi, +pi]). One can track the history of
        # the evolution of the phase using the unwrap function (see convert_input).
        self.phase_buffer = np.array([])

        self.delay_line_absolute_position = 0
        self.current_setpoint = 0
        self.error = 0

        self.log_spectrum_node_initialized = False

    def update_settings(self, param):
        """ Get a parameter instance whose value has been modified by the user on the UI

        Parameters
        ----------
        param: (Parameter) instance of Parameter object
        """
        if param.name() == 'start_calibration':
            if param.value():
                self.calibration_scan_running = True
                self.start_calibration()
                QtWidgets.QApplication.processEvents()
            else:
                self.calibration_scan_running = False

    def ini_model(self):
        """ Initialize the PID model

        Defines all the action to be performed on the initialized modules (PID parameters, actuators, detectors).
        Either here for specific things (ROI, ...) or within the preset of the current model.

        We suppose that the modules have been initialized within the preset.
        """
        super().ini_model()

        self.pid_controller.settings.child('main_settings', 'pid_controls', 'output_limits',
            'output_limit_min').setValue(0)  # in microns
        self.pid_controller.settings.child('main_settings', 'pid_controls', 'output_limits',
            'output_limit_min_enabled').setValue(False)
        self.pid_controller.settings.child('main_settings', 'pid_controls', 'output_limits',
            'output_limit_max').setValue(15)  # in microns
        self.pid_controller.settings.child('main_settings', 'pid_controls', 'output_limits',
            'output_limit_max_enabled').setValue(False)

        self.pid_controller.settings.child('main_settings', 'pid_controls', 'pid_constants', 'kp').setValue(0.3)
        self.pid_controller.settings.child('main_settings', 'pid_controls', 'pid_constants', 'ki').setValue(0.05)
        self.pid_controller.settings.child('main_settings', 'pid_controls', 'pid_constants', 'kd').setValue(0)

        # Launch the acquisition of the scope.
        # The name should correspond to the one given in the preset.
        self.pid_controller.modules_manager.get_mod_from_name("Scope").ui.grab_pb.click()

        # Load ROI configuration file
        self.pid_controller.modules_manager.get_mod_from_name("Scope").ui.viewers[0].roi_manager.load_ROI()

    def start_calibration(self):
        """ Launch a calibration scan

        This method is called by ticking the "start_calibration" parameter of the UI.
        It should be called after the ROIs have been properly defined.
        It performs a scan of the delay line position and acquire the values of the modulated signals provided by the
        ROIs.
        The start position of this scan defines the time zero (the temporal superposition of the IR and XUV pulses).
        At the end of the scan the parameters of the ellipse that fit the modulated signals in XY representation are
        set in the parameters and a window displays the result.
        The phase at time zero is also set.
        At the end of the scan the delay line returns to the time zero (the start position).
        """
        #self.pid_controller.log_signal.emit('A calibration scan has been launched !')

        steps = np.linspace(self.settings.child('calibration', 'calibration_move', 'start').value(),
                            self.settings.child('calibration', 'calibration_move', 'stop').value(),
                            self.settings.child('calibration', 'calibration_move', 'Nstep').value())

        detector_data = np.zeros((steps.size, 2))

        # Connect the signal emitted by the delay line when reaching its target to the move_done method
        delay_line_module = self.pid_controller.modules_manager.get_mod_from_name("Delay line", mod="act")
        delay_line_module.move_done_signal.connect(self.move_done)

        # Connect the signal emitted by the scope at the end of its acquisition to the det_done method
        scope_module = self.pid_controller.modules_manager.get_mod_from_name("Scope")
        scope_module.grab_done_signal.connect(self.det_done)

        # Use the ActiveDSO ClearDevice method. This method will send a device clear signal to the oscilloscope.
        # Any unread response currently in the device output buffer will be cleared. See Lecroy ActiveDSO Developer’s
        # Guide.
        # SPECIFIC TO THE SCOPE WE USE
        # scope_module.controller.DeviceClear(False)

        QtWidgets.QApplication.processEvents()
        QThread.msleep(1000)
        QtWidgets.QApplication.processEvents()

        for ind_step, step in enumerate(steps):
            self.move_done_flag = False
            delay_line_module.move_Abs(step)
            self.wait_for_move_done()

            self.det_done_flag = False
            scope_module.grab_data()
            self.wait_for_det_done()

            raw_spectrum_from_scope = []
            #raw_spectrum_from_scope = scope_module.data_to_save_export['data1D']['Scope_Lecroy Waverunner_CH000']['data']
            raw_spectrum_from_scope = scope_module.data_to_save_export['data1D']['Scope_Mock1_CH000']['data']

            QtWidgets.QApplication.processEvents()
            QThread.msleep(300)
            QtWidgets.QApplication.processEvents()

            # The signal offset is taken from ROI_02 (mean value) which should be out of any electron signal
            signal_offset = scope_module.ui.viewers[0].measure_data_dict['Lineout_ROI_02:']
            spectrum_without_offset = []
            spectrum_without_offset = [elt - signal_offset for elt in raw_spectrum_from_scope]

            # The XUV flux normalization corresponds to the integral of the all spectrum (i.e. the total number of
            # photoelectrons).
            # Dividing by this value, we suppress the effect of the fluctuations of the intensity of the XUV source.
            xuv_flux_normalization = np.trapz(spectrum_without_offset)

            # Measurement from ROI_00 (mean)
            # Correspond to the value of the first modulated signal m1
            # raw_m1 is the sum of the values in the ROI. Thus the ROI should be defined properly by the user.
            raw_m1 = scope_module.ui.viewers[0].measure_data_dict["Lineout_ROI_00:"]
            m1 = (raw_m1 - signal_offset) / xuv_flux_normalization

            # Measurement from ROI_01 (mean)
            raw_m2 = scope_module.ui.viewers[0].measure_data_dict["Lineout_ROI_01:"]
            m2 = (raw_m2 - signal_offset) / xuv_flux_normalization

            detector_data[ind_step, 0] = m1
            detector_data[ind_step, 1] = m2

            if not self.calibration_scan_running:
                break

        self.viewer_calib.show_data([detector_data[:, 0], detector_data[:, 1]])

        self.lsqe.fit([detector_data[:, 0], detector_data[:, 1]])
        center, width, height, theta = self.lsqe.parameters()
        ellipse_x, ellipse_y = self.get_ellipse_fit(center, width, height, theta)

        self.viewer_ellipse.plot_channels[0].setData(x=detector_data[:, 0], y=detector_data[:, 1])
        self.viewer_ellipse.plot_channels[1].setData(x=ellipse_x, y=ellipse_y)

        self.settings.child('calibration', 'calibration_ellipse', 'x0').setValue(center[0])
        self.settings.child('calibration', 'calibration_ellipse', 'y0').setValue(center[1])
        self.settings.child('calibration', 'calibration_ellipse', 'dx').setValue(width)
        self.settings.child('calibration', 'calibration_ellipse', 'dy').setValue(height)
        self.settings.child('calibration', 'calibration_ellipse', 'theta').setValue(np.rad2deg(theta))

        QtWidgets.QApplication.processEvents()

        # The actuator should go back to the starting position (time zero) at the end of the calibration scan.
        self.move_done_flag = False
        delay_line_module.move_Abs(steps[0])
        self.wait_for_move_done()

        self.calibration_done = True
        # self.settings.child('calibration', 'start_calibration').setValue(False)

        # Disconnect the signals
        delay_line_module.move_done_signal.disconnect(self.move_done)
        scope_module.grab_done_signal.disconnect(self.det_done)

        # self.pid_controller.logger.emit('The calibration scan is finished! You can INIT, PLAY and uncheck P

    pyqtSlot(str, float)
    def move_done(self, actuator_title, position):
        """ Triggered by a signal from the delay line (DAQ_Move object) when it reaches its targeted position

        The move_done_flag is set to False at the begining of each iteration in the calibration loop.

        Parameters
        ----------
        actuator_title: (str) title of the actuator module
        position: (float) position of the actuator in microns
        """
        self.delay_line_absolute_position = position
        self.move_done_flag = True

    pyqtSlot()
    def det_done(self):
        """ Called each time the oscilloscope finished his acquisition

        The det_done_flag is set to False at the begining of each iteration in the calibration loop.
        """
        self.det_done_flag = True

    def wait_for_move_done(self):
        """ Wait for the delay line to have reached its position or timeout
        """
        self.timeout_scan_flag = False
        self.timer.start(self.settings.child('calibration', 'timeout').value())
        while not(self.move_done_flag or self.timeout_scan_flag):
            QtWidgets.QApplication.processEvents()
        self.timer.stop()

    def wait_for_det_done(self):
        """ Wait for the scope to be ready or timeout
        """
        self.timeout_scan_flag = False
        self.timer.start(self.settings.child('calibration', 'timeout').value())
        while not(self.det_done_flag or self.timeout_scan_flag):
            QtWidgets.QApplication.processEvents()
        self.timer.stop()

    def get_ellipse_fit(self, center, width, height, theta):
        """ Construct the arrays to plot the fitted ellipse after the calibration using the ellipse parameters
        """
        t = np.linspace(0, 2*np.pi, 1000)
        ellipse_x = (center[0] + width*np.cos(t)*np.cos(theta) - height*np.sin(t)*np.sin(theta))
        ellipse_y = (center[1] + width*np.cos(t)*np.sin(theta) + height*np.sin(t)*np.cos(theta))

        return ellipse_x, ellipse_y

    def convert_input(self, data):
        """ Return a measured phase (delay) from the oscilloscope spectrum

        Convert the measurements from the ROIs in the electronic spectrum to a measured phase in rad (same
        dimensionality as the setpoint). The output feeds the PID module (external library).

        Parameters
        ----------
        data: (dict) Dictionary from which the current spectrum is extracted

        Returns
        -------
        (InputFromDetector) Stores the unwrapped phase in radians. The origin of the phase axis is the phase at time
            zero.
        """
        # Spectrum from the oscilloscope
        raw_spectrum_from_scope = data['Scope']['data1D']['Scope_Mock1_CH000']['data']

        scope_viewer = self.pid_controller.modules_manager.get_mod_from_name("Scope").ui.viewers[0]
        # The signal offset is taken from ROI_02 (mean value), which should be out of any electron signal.
        signal_offset = scope_viewer.measure_data_dict['Lineout_ROI_02:']
        spectrum_without_offset = [elt - signal_offset for elt in raw_spectrum_from_scope]

        # The XUV flux normalization corresponds to the integral of the all spectrum (i.e. the total number of
        # photoelectrons).
        # Dividing by this value, we suppress the effect of the fluctuations of the intensity of the XUV source.
        xuv_flux_normalization = np.trapz(spectrum_without_offset)

        # Measurement from ROI_00 (mean value). Correspond to the value of the first modulated signal m1.
        raw_m1 = scope_viewer.measure_data_dict['Lineout_ROI_00:']
        m1 = (raw_m1 - signal_offset)/xuv_flux_normalization
        # Measurement from ROI_01 (mean value). Correspond to the value of the second modulated signal m2.
        raw_m2 = scope_viewer.measure_data_dict['Lineout_ROI_01:']
        m2 = (raw_m2 - signal_offset)/xuv_flux_normalization

        # The phase returned by get_phi_from_xy is within [-pi, +pi].
        phi = self.get_phi_from_xy(m1, m2)

        # The first call of convert_input method is done after the user push the PLAY button (launch of the PID loop).
        # It should be done just after the calibration scan. Which means that we are at time zero.
        # That is how we define the phase at time zero.
        if not self.time_zero_phase_defined:
            self.time_zero_phase = phi
            self.settings.child('stabilization', 'offsets', 'time_zero_phase').setValue(self.time_zero_phase)
            self.time_zero_phase_defined = True

        # Record the measured phases. The phases in self.phase_buffer should be within [-pi,+pi].
        self.phase_buffer = np.append(self.phase_buffer, [phi])

        # Perform the unwrap operation and offset by time_zero_phase. Thus delay_phase is zero at time zero.
        phase_buffer_unwrapped = np.unwrap(self.phase_buffer - self.time_zero_phase)
        unwrapped_phase = phase_buffer_unwrapped[-1]
        self.delay_phase = unwrapped_phase

        return InputFromDetector([unwrapped_phase])

    def convert_output(self, phase_correction):
        """ Convert the phase correction from the PID module to an order in absolute value for the actuator.

        Parameters
        ----------
        phase_correction: (float) output value from the PID module. This phase correction in radians should be within
        [-pi,+pi].

        Returns
        -------
        (OutputToActuator) Stores the absolute value in microns for the piezo actuator.
        """
        # Laser wavelength in microns
        laser_wl = self.settings.child('stabilization', 'laser_wl').value()
        # This parameter is used to easily change the sign of the correction on the fly, while the feedback loop is
        # running
        correction_sign = self.settings.child('stabilization', 'correction_sign').value()

        # Get the current position of the actuator
        self.delay_line_absolute_position = self.pid_controller.actuator_modules[0].current_position

        absolute_order_to_actuator = (self.delay_line_absolute_position
                                      + correction_sign*phase_correction*laser_wl/(8*np.pi))

        self.settings.child('stabilization', 'delay_line_absolute_position').setValue(self.delay_line_absolute_position)

        # Get the value of the error (in rad)
        self.current_setpoint = self.pid_controller.settings.child('main_settings', 'pid_controls', 'set_point').value()
        self.error = self.delay_phase - self.current_setpoint
        self.settings.child('stabilization', 'error').setValue(self.error)

        self.settings.child('stabilization', 'relative_order_to_actuator').setValue(
            absolute_order_to_actuator - self.delay_line_absolute_position)

        return OutputToActuator(mode="abs", values=[absolute_order_to_actuator])


if __name__ == '__main__':
    main("BeamSteering.xml")
