from pymodaq.pid.utils import PIDModelGeneric, OutputToActuator, InputFromDetector, main
from scipy.ndimage import center_of_mass


class PIDModelLIZARD(PIDModelGeneric):
    limits = dict(max=dict(state=True, value=10),
                  min=dict(state=True, value=-10), )
    konstants = dict(kp=1, ki=0.01, kd=0.001)

    setpoint_ini = [0.]

    actuators_name = ["Delay line"]
    detectors_name = ["TOF"]

    Nsetpoints = 1

    params = [
        {'title': 'Stabilization', 'name': 'stabilization', 'type': 'group', 'children': [
            {'title': 'Ellipsis phase (rad):', 'name': 'ellipsis_phase', 'type': 'float', 'value': 0},
            {'title': 'Error (rad):', 'name': 'error', 'type': 'float', 'value': 0},
            {'title': 'Actuator absolute position (microns):', 'name': 'actuator_absolute_position', 'type': 'float',
             'value': 0},
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
            {'title': 'Do calibration:', 'name': 'do_calibration', 'type': 'bool', 'value': False},
            {'title': 'Timeout:', 'name': 'timeout', 'type': 'int', 'value': 10000},
            {'title': 'Calibration', 'name': 'calibration_move', 'type': 'group', 'expanded': True, 'visible': True,
             'children': [
                {'title': 'Start pos:', 'name': 'start', 'type': 'float', 'value': -9544.},
                {'title': 'Stop pos:', 'name': 'stop', 'type': 'float', 'value': -9542.},
                {'title': 'Step size:', 'name': 'step', 'type': 'float', 'value': 0.04},
                {'title': 'Averaging:', 'name': 'average', 'type': 'int', 'value': 1}]},
            {'title': 'Ellipse params', 'name': 'calibration_ellipse',
             'type': 'group', 'expanded': True, 'visible': True, 'children': [
                 {'title': 'Dx:', 'name': 'dx', 'type': 'float', 'value': 0.0},
                 {'title': 'Dy:', 'name': 'dy', 'type': 'float', 'value': 0.0},
                 {'title': 'x0:', 'name': 'x0', 'type': 'float',
                  'value': 0.00},
                 {'title': 'y0:', 'name': 'y0', 'type': 'float',
                  'value': 0.00},
                 {'title': 'theta (°):', 'name': 'theta', 'type': 'float',
                  'value': 0.00}]}
         ]},
        {'title': 'Stabilized scan', 'name': 'stabilized_scan',
         'type': 'group', 'expanded': True, 'visible': True, 'children': [
            {'title': 'Do stabilized scan:', 'name': 'do_stabilized_scan',
             'type': 'bool', 'value': False},
            {'title': 'Timeout (ms):', 'name': 'timeout', 'type': 'int',
             'value': 5000},
            {'title': 'Stabilized scan parameters',
             'name': 'stabilized_scan_parameters', 'type': 'group',
             'expanded': True, 'visible': True, 'children': [
                {'title': 'Length (microns):', 'name': 'length',
                 'type': 'float', 'value': 1},
                {'title': 'Step size (microns):', 'name': 'step_size',
                 'type': 'float', 'value': 0.02},
                {'title': 'Iterations per setpoint:',
                 'name': 'iterations_per_setpoint', 'type': 'int',
                 'value': 2}]}
         ]}
    ]

    def __init__(self, pid_controller):
        super().__init__(pid_controller)

    def update_settings(self, param):
        """
        Get a parameter instance whose value has been modified by a user on the UI
        Parameters
        ----------
        param: (Parameter) instance of Parameter object
        """
        if param.name() == '':
            pass

    def ini_model(self):
        super().ini_model()

    def convert_input(self, measurements):
        """
        Convert the measurements in the units to be fed to the PID (same dimensionality as the setpoint)
        Parameters
        ----------
        measurements: (Ordereddict) Ordereded dict of object from which the model extract a value of the same units as the setpoint

        Returns
        -------
        float: the converted input

        """
        # print('input conversion done')
        image = measurements['Camera']['data2D']['Camera_Mock2DPID_CH000']['data']
        image = image - self.settings.child('threshold').value()
        image[image < 0] = 0
        x, y = center_of_mass(image)
        self.curr_input = [y, x]
        return InputFromDetector([y, x])

    def convert_output(self, outputs, dt, stab=True):
        """
        Convert the output of the PID in units to be fed into the actuator
        Parameters
        ----------
        output: (float) output value from the PID from which the model extract a value of the same units as the actuator

        Returns
        -------
        list: the converted output as a list (if there are a few actuators)

        """
        # print('output converted')

        self.curr_output = outputs
        return OutputToActuator(mode='rel', values=outputs)


if __name__ == '__main__':
    main("BeamSteering.xml")
