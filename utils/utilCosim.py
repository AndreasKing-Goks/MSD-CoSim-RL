import numpy as np
import matplotlib.pyplot as plt
from libcosimpy.CosimExecution import CosimExecution
from libcosimpy.CosimSlave import CosimLocalSlave
from libcosimpy.CosimManipulator import CosimManipulator
from libcosimpy.CosimObserver import CosimObserver
from libcosimpy.CosimEnums import CosimVariableType

#########################################################
############### Co-Simulation Interfacing ###############
#########################################################

def GetVariableIndex(variables, name):
    try:
        index = 0
        for var in variables:
            if var.name == str.encode(name):
                return index
            index +=1
    except:
        raise Exception("Could not locate the variable %s in the model" %(name))

def GetVariableInfo(variables, name):
    try:
        index = GetVariableIndex(variables,name)
        vr = variables[index].reference
        # print(variables[index])
        var_type = CosimVariableType(variables[index].type)
        return vr, var_type
    except:
        raise Exception("Could not locate the variable %s in the model" %(name))

class ObserverStruct:
    var = None
    var_vr = None
    var_type = None
    slave = None
    variable = None

class CoSimInstance:
    '''
        instanceName is the name of the co-simulation instance (str)
        stopTime is the stop time of the co-simulation (seconds)
        stepSize is the macro step size of the co-simulation (seconds)
    '''
    def __init__(self, instanceName: str="simulation", stopTime: float=1.0, stepSize: float = 0.01):
        self.instanceName = instanceName
        self.stopTime = int(stopTime*1e9)
        self.stepSize = int(stepSize*1e9)
        self.time=0

        self.observer_time_series_struct = {}

        self.slaves = {}
        self.slaves_index = {}
        self.slaves_variables = {}

        self.slave_input_var = []
        self.slave_input_name = []

        self.slave_output_var = []
        self.slave_output_name = []

        self.execution = CosimExecution.from_step_size(self.stepSize)
        self.manipulator = CosimManipulator.create_override()
        self.execution.add_manipulator(self.manipulator)

        self.observer_time_series = CosimObserver.create_time_series()
        self.execution.add_observer(self.observer_time_series)

        self.observer_last_value = CosimObserver.create_last_value()
        self.execution.add_observer(self.observer_last_value)

        self.first_plot = True

        self.fromExternalSlaveName = []
        self.fromExternalSlaveVar = []
        self.fromExternalSlaveFunc = []
    

    def AddObserverTimeSeries(self, name: str, slaveName: str, variable: str):
        self.observer_time_series_struct[name]             = ObserverStruct()
        self.observer_time_series_struct[name].slave       = slaveName
        self.observer_time_series_struct[name].var         = variable
        self.observer_time_series_struct[name].var_vr, self.observer_time_series_struct[name].var_type = GetVariableInfo(self.slaves_variables[slaveName],
                                                                                                           variable)
        self.observer_time_series.start_time_series(self.slaves_index[slaveName],
                                                                   value_reference=self.observer_time_series_struct[name].var_vr,
                                                                   variable_type=self.observer_time_series_struct[name].var_type)

    def GetObserverTimeSeries(self, name: str, sample_count: int = 10000, from_step: int = 0):
        vr = self.observer_time_series_struct[name].var_vr
        var_type = self.observer_time_series_struct[name].var_type
        slave = self.slaves_index[self.observer_time_series_struct[name].slave]
        if var_type == CosimVariableType.REAL:
            time_points, step_numbers, samples = self.observer_time_series.time_series_real_samples(slave_index = slave,
                                                                                     value_reference = vr,
                                                                                     sample_count = sample_count,
                                                                                     from_step = from_step)
            time_seconds = [x*1e-9 for x in time_points]
            return time_seconds, step_numbers, samples
        elif var_type == CosimVariableType.BOOLEAN:
            time_points, step_numbers, samples = self.observer_time_series.time_series_boolean_samples(slave_index = slave,
                                                                                        value_reference = vr,
                                                                                        sample_count = sample_count,
                                                                                        from_step = from_step)
            time_seconds = [x*1e-9 for x in time_points]
            return time_seconds, step_numbers, samples
        elif var_type == CosimVariableType.INTEGER:
            time_points, step_numbers, samples = self.observer_time_series.time_series_integer_samples(slave_index = slave,
                                                                                        value_reference = vr,
                                                                                        sample_count = sample_count,
                                                                                        from_step = from_step)
            time_seconds = [x*1e-9 for x in time_points]
            return time_seconds, step_numbers, samples
        else:
            time_points, step_numbers, samples = self.observer_time_series.time_series_string_samples(slave_index = slave,
                                                                                       value_reference = vr,
                                                                                       sample_count = sample_count,
                                                                                       from_step = from_step)
            time_seconds = [x*1e-9 for x in time_points]
            return time_seconds, step_numbers, samples

    def PlotTimeSeries(self, separate_plots: bool = False, create_window: bool = True, show: bool = True, legend: bool = True, plot_legend: str=""):
        for key in self.observer_time_series_struct:
            time_points, step_number, samples = self.GetObserverTimeSeries(key)
            if create_window:
                if self.first_plot:
                    plt.figure()
                    self.first_plot = False
                else:
                    if separate_plots:
                        plt.legend()
                        plt.grid()
                        plt.xlabel("Time [s]")
                        plt.title("Time series form co-simulation instance \"%s\"" %(self.instanceName))
                        plt.figure()

            label = plot_legend + ": " + self.instanceName + ": " + str(key)
            plt.plot(time_points, samples, label=label)
        if legend:
            plt.legend()
        plt.xlabel("Time [s]")
        plt.title("Time series form co-simulation instance \"%s\"" %(self.instanceName))
        plt.grid(True)
        if show:
            plt.show()

    def AddSlave(self, path: str, name: str):
        self.slaves[name] = CosimLocalSlave(fmu_path=path, instance_name = name)
        self.slaves_index[name] = self.execution.add_local_slave(local_slave = self.slaves[name])
        self.slaves_variables[name] = self.execution.slave_variables(slave_index = self.slaves_index[name])

    def AddSlaveConnection(self, slaveInputName: str, slaveInputVar: str, slaveOutputName: str, slaveOutputVar: str):
        self.slave_input_name.append(slaveInputName)
        self.slave_input_var.append(slaveInputVar)
        self.slave_output_name.append(slaveOutputName)
        self.slave_output_var.append(slaveOutputVar)

    def GetLastValue(self, slaveName: str, slaveVar: str):
        out_vr, out_type = GetVariableInfo(self.slaves_variables[slaveName], slaveVar)
        if out_type == CosimVariableType.REAL:
            return self.observer_last_value.last_real_values(slave_index = self.slaves_index[slaveName], 
                                                             variable_references = [out_vr])[0]
        if out_type == CosimVariableType.BOOLEAN:
            return self.observer_last_value.last_boolean_values(slave_index = self.slaves_index[slaveName], 
                                                             variable_references = [out_vr])[0]
        if out_type == CosimVariableType.INTEGER:
            return self.observer_last_value.last_integer_values(slave_index = self.slaves_index[slaveName], 
                                                             variable_references = [out_vr])[0]
        if out_type == CosimVariableType.STRING:
            return self.observer_last_value.last_string_values(slave_index = self.slaves_index[slaveName], 
                                                             variable_references = [out_vr])[0]


    def CoSimManipulate(self):
        for i in range(0,len(self.slave_input_name)):
            out_vr, out_type = GetVariableInfo(self.slaves_variables[self.slave_output_name[i]], self.slave_output_var[i])
            out_val = [self.GetLastValue(slaveName = self.slave_output_name[i],
                                        slaveVar = self.slave_output_var[i])]

            if out_type == CosimVariableType.REAL:
                in_vr, in_type = GetVariableInfo(self.slaves_variables[self.slave_input_name[i]], self.slave_input_var[i])
                if out_type == in_type:
                    self.manipulator.slave_real_values(self.slaves_index[self.slave_input_name[i]], [in_vr], out_val)
            elif out_type == CosimVariableType.BOOLEAN:
                in_vr, in_type = GetVariableInfo(self.slaves_variables[self.slave_input_name[i]], self.slave_input_var[i])
                if out_type == in_type:
                    self.manipulator.slave_boolean_values(self.slaves_index[self.slave_input_name[i]], [in_vr], out_val)
            elif out_type == CosimVariableType.INTEGER:
                in_vr, in_type = GetVariableInfo(self.slaves_variables[self.slave_input_name[i]], self.slave_input_var[i])
                if out_type == in_type:
                    self.manipulator.slave_integer_values(self.slaves_index[self.slave_input_name[i]], [in_vr], out_val)
            else:
                in_vr, in_type = GetVariableInfo(self.slaves_variables[self.slave_input_name[i]], self.slave_input_var[i])
                if out_type == in_type:
                    self.manipulator.slave_string_values(self.slaves_index[self.slave_input_name[i]], [in_vr], out_val)

    def AddInputFromExternal(self, slaveName: str, slaveVar: str, func):    
        self.fromExternalSlaveName.append(slaveName)
        self.fromExternalSlaveVar.append(slaveVar)
        self.fromExternalSlaveFunc.append(func)


    def SetInputFromExternal(self):    
        for i in range(0,len(self.fromExternalSlaveName)):
            var_vr, var_type = GetVariableInfo(self.slaves_variables[self.fromExternalSlaveName[i]], self.fromExternalSlaveVar[i])
            val =[self.fromExternalSlaveFunc[i]()]
            if var_type == CosimVariableType.REAL:
                self.manipulator.slave_real_values(self.slaves_index[self.fromExternalSlaveName[i]], [var_vr], val)

            if var_type == CosimVariableType.BOOLEAN:
                self.manipulator.slave_boolean_values(self.slaves_index[self.fromExternalSlaveName[i]], [var_vr], val)

            if var_type == CosimVariableType.INTEGER:
                self.manipulator.slave_integer_values(self.slaves_index[self.fromExternalSlaveName[i]], [var_vr], val)

            if var_type == CosimVariableType.STRING:
                self.manipulator.slave_string_values(self.slaves_index[self.fromExternalSlaveName[i]], [var_vr], val)

    def SetInitialValue(self, slaveName: str, slaveVar: str, initValue):
        var_vr, var_type = GetVariableInfo(self.slaves_variables[slaveName], slaveVar)
        if var_type == CosimVariableType.REAL:
            self.execution.real_initial_value(slave_index = self.slaves_index[slaveName],
                                              variable_reference = var_vr, value = initValue)

        if var_type == CosimVariableType.BOOLEAN:
            self.execution.boolean_initial_value(slave_index = self.slaves_index[slaveName],
                                              variable_reference = var_vr, value = initValue)

        if var_type == CosimVariableType.INTEGER:
            self.execution.integer_initial_value(slave_index = self.slaves_index[slaveName],
                                              variable_reference = var_vr, value = initValue)

        if var_type == CosimVariableType.STRING:
            self.execution.string_initial_value(slave_index = self.slaves_index[slaveName],
                                              variable_reference = var_vr, value = initValue)

    def PreSolverFunctionCall(self):
        pass

    def PostSolverFunctionCall(self):
        pass

    def Simulate(self):
        while self.time < self.stopTime:
            self.CoSimManipulate()
            self.SetInputFromExternal()
            self.PreSolverFunctionCall()
            self.execution.step()
            self.PostSolverFunctionCall()
            self.time +=self.stepSize

    # Custom Function
    def ApplyActionForce(self, F_action):
        var_vr, var_type = GetVariableInfo(self.slaves_variables["MASS1D"], "F_3")
        self.manipulator.slave_real_values(self.slaves_index["MASS1D"], [var_vr], [F_action])

#########################################################
############ Mass-Spring-Damper Environment #############
#########################################################

## Classes
class ObservationSpaceMSD:
    def __init__(self, 
                 obsPos,        # float 
                 obsVel):       # float
        # Observation Space
        self.MSDPosition = [-obsPos, obsPos]
        self.MSDVelocity = [-obsVel, obsVel]
        self.n = 2 # [y, y_dot]

class TerminalStateBoundMSD:
    def __init__(self, 
                 y_desired,                 # float
                 y_tsBound,
                 MSDPosition):              # float, y metre tolerance up and down for termination
        
        # Check if terminal state is within the the observation space
        # y = 0 is a point where the spring do not exert any forces
        tsPosDown = (y_desired - y_tsBound) if (y_desired - y_tsBound)>MSDPosition[0] else MSDPosition[0] # Down is negative value
        tsPosUp   = (y_desired + y_tsBound) if (y_desired + y_tsBound)<MSDPosition[1] else MSDPosition[1] # Up is positive value
        self.MSDPositionTerminal = [tsPosDown, tsPosUp]

class ActionSpaceMSD:
    def __init__(self, 
                 force):
        self.force = force                                  # Newton
        self.all_force = [0, self.force]
        self.n = len(self.all_force)                        # Number of action space
    
    def sample(self):
        magnitude = int(np.random.choice(self.all_force))
        action = self.all_force.index(magnitude)
        return action
    
class EnvironmentMSD:
    def __init__(self,
                 y_desired,                             # y desired between -10 to 10 ideally
                 obsPos:                float=10,
                 obsVel:                float=10,
                 mass:                  float=1,
                 stiffCoef:            float=1,
                 dampCoef:             float=1,
                 binNumbers:             int=10,
                 stopTime:              float=100,
                 fps:                   float=60,
                 initMSDPos:            float=0,
                 initMSDVel:            float=0,
                 force:                 float=10,
                 y_tsBound:             float=5.0,      # Terminal state bound
                 y_desiredTolerance:    float=0.25):    # Target height tolerance
        
        ############ Description ############
        self.name = "MSD-RL"
        self.binNumbers = binNumbers
        self.y_desired = y_desired                      # The desired location we want the mass to be
        self.y_desiredTolerance = y_desiredTolerance    # The tolerance of the desired location
        self.mass = mass
        self.stiffCoef = stiffCoef
        self.dampCoef = dampCoef
        
        # Time
        self.stopTime = stopTime
        self.stepSize = 1/fps
        self.tspan = [0, self.stopTime]

        # Observation space
        self.observation_space = ObservationSpaceMSD(obsPos, obsVel)

        # Terminal state bound
        self.terminal_state = TerminalStateBoundMSD(y_desired, y_tsBound, self.observation_space.MSDPosition)

        # Action space
        self.action_space = ActionSpaceMSD(force)

        # Action taken
        self.action_taken =[]

        # Target tolerance
        downToleranceLow  = (self.y_desired - self.y_desiredTolerance) if (self.y_desired - self.y_desiredTolerance)>self.observation_space.MSDPosition[0] else self.observation_space.MSDPosition[0]
        upperToleranceLow = (self.y_desired + self.y_desiredTolerance) if (self.y_desired + self.y_desiredTolerance)<self.observation_space.MSDPosition[1] else self.observation_space.MSDPosition[1]
        self.y_desiredBoundLow = [downToleranceLow, upperToleranceLow]

        downToleranceMed  = (self.y_desired - self.y_desiredTolerance/2) if (self.y_desired - self.y_desiredTolerance/2)>self.observation_space.MSDPosition[0] else self.observation_space.MSDPosition[0]
        upperToleranceMed = (self.y_desired + self.y_desiredTolerance/2) if (self.y_desired + self.y_desiredTolerance/2)<self.observation_space.MSDPosition[1] else self.observation_space.MSDPosition[1]
        self.y_desiredBoundMed = [downToleranceMed, upperToleranceMed]

        downToleranceHi  = (self.y_desired - self.y_desiredTolerance/4) if (self.y_desired - self.y_desiredTolerance/4)>self.observation_space.MSDPosition[0] else self.observation_space.MSDPosition[0]
        upperToleranceHi = (self.y_desired + self.y_desiredTolerance/4) if (self.y_desired + self.y_desiredTolerance/4)<self.observation_space.MSDPosition[1] else self.observation_space.MSDPosition[1]
        self.y_desiredBoundHi = [downToleranceHi, upperToleranceHi]

        ############ Initial States ############
        ## Continuous Space
        # Initial states
        self.initial_states = [initMSDPos, initMSDVel]

        # Current states
        self.states = self.initial_states

        ############ Initial CoSimInstance ############
        self.CoSimInstance = None
    
    def InitializeCoSim(self):
        # Also for resetting the environment
        ############ Co-simulation Setup ############
        # Instantiation
        self.CoSimInstance = CoSimInstance(instanceName=self.name, 
                                      stopTime=self.stopTime,
                                      stepSize=self.stepSize)
        
        # Adding slaves
        self.CoSimInstance.AddSlave(name="MASS1D"  , path="01_FMUs/Mass1D.fmu")
        self.CoSimInstance.AddSlave(name="SPRING1D", path="01_FMUs/Spring1D.fmu")
        self.CoSimInstance.AddSlave(name="DAMPER1D", path="01_FMUs/Damper1D.fmu")

        # Setup Observer (observer record the position and velocity)
        self.CoSimInstance.AddObserverTimeSeries(name="position", slaveName="MASS1D", variable="position")
        self.CoSimInstance.AddObserverTimeSeries(name="velocity", slaveName="MASS1D", variable="velocity")

        # Add model connections
        # Mass to spring and damper
        self.CoSimInstance.AddSlaveConnection(slaveInputName="MASS1D", slaveInputVar="F_1", slaveOutputName="SPRING1D", slaveOutputVar="F_a")
        self.CoSimInstance.AddSlaveConnection(slaveInputName="MASS1D", slaveInputVar="F_2", slaveOutputName="DAMPER1D", slaveOutputVar="F_a")

        # Spring and damper to mass
        self.CoSimInstance.AddSlaveConnection(slaveInputName="SPRING1D", slaveInputVar="v_a", slaveOutputName="MASS1D", slaveOutputVar="v")
        self.CoSimInstance.AddSlaveConnection(slaveInputName="DAMPER1D", slaveInputVar="v_a", slaveOutputName="MASS1D", slaveOutputVar="v")
        
        ## Co-Simulation parameters
        # Mass
        self.CoSimInstance.SetInitialValue(slaveName="MASS1D", slaveVar="m", initValue=self.mass)

        # Stiffness coefficient
        self.CoSimInstance.SetInitialValue(slaveName="SPRING1D", slaveVar="k", initValue=self.stiffCoef)

        # Damping Coefficient
        self.CoSimInstance.SetInitialValue(slaveName="DAMPER1D", slaveVar="c", initValue=self.dampCoef)

        # Initial Position
        self.CoSimInstance.SetInitialValue(slaveName="MASS1D", slaveVar="x", initValue=self.initial_states[0])
        # self.CoSimInstance.SetInitialValue(slaveName="MASS1D", slaveVar="position", initValue=self.initial_states[0])

        # Initial Velocity
        self.CoSimInstance.SetInitialValue(slaveName="MASS1D", slaveVar="v", initValue=self.initial_states[1])
        # self.CoSimInstance.SetInitialValue(slaveName="MASS1D", slaveVar="velocity", initValue=self.initial_states[1])

    def DiscretizeState(self, states, checkBins = False, shiftRight=False):
        # Continuous state shape (number of bins per state variable)
        state_shape = (self.binNumbers, self.binNumbers)

        # Unpacks states container
        y, y_dot = states

        # Define bin edges and max_bin_idx
        binNumbers = self.binNumbers
        binEdges = binNumbers + 1
        max_bin_idx = binNumbers - 1

        # Define bin edges for discretization
        MSDPosBins = np.linspace(self.observation_space.MSDPosition[0], self.observation_space.MSDPosition[1], binEdges)
        MSDVelBins = np.linspace(self.observation_space.MSDVelocity[0], self.observation_space.MSDVelocity[1], binEdges)

        # Print bins description
        if checkBins:
            # Set print options for compact display
            np.set_printoptions(precision=4, suppress=True)

            # Use array2string to format without excessive spaces
            print(f"Mass-Spring-Damper Position Bins: {np.array2string(MSDPosBins, separator=',')}")
            print(f"Mass-Spring-Damper Velocity Bins: {np.array2string(MSDVelBins, separator=',')}")

        # Discretize each part of the state
        y_discrete = np.clip(np.digitize(y, MSDPosBins, right= not shiftRight),0 , max_bin_idx)
        y_dot_discrete = np.clip(np.digitize(y_dot, MSDVelBins, right= not shiftRight),0 , max_bin_idx)

        states_discretized = [y_discrete, y_dot_discrete] # Made a vector of discretized vector

        # Flat indexing of the discretized vector
        discrete_states = int(np.ravel_multi_index((states_discretized[0], states_discretized[1]), state_shape))

        return discrete_states
    
    def RewardFunction(self, states):
        ## Use continuous state to compute the obtained reward

        # Check the termination status of the current states
        done = (
            states[0] < self.terminal_state.MSDPositionTerminal[0] or       # y < tsPosDown
            states[0] > self.terminal_state.MSDPositionTerminal[1]          # y > tsPosUp
        )

        # Check if targe tolerance condition met
        lowDownCond = states[0] > self.y_desiredBoundLow[0]
        lowUpCond = states[0] < self.y_desiredBoundLow[1]

        medDownCond = states[0] > self.y_desiredBoundMed[0]
        medUpCond = states[0] < self.y_desiredBoundMed[1]

        hiDownCond = states[0] > self.y_desiredBoundHi[0]
        hiUpCond = states[0] < self.y_desiredBoundHi[1]

        reward = -2

        if lowUpCond and lowDownCond:
            reward = 1

        if medDownCond and medUpCond:
            reward = 2

        if hiDownCond and hiUpCond:
            reward = 4

        return reward, done
    
    def step(self, action):
        # Report the action taken
        self.action_taken.append("Idle" if action==0 else "Push")

        # Retrieve the action force based on the action
        F_action = self.action_space.all_force[action]

        # Setup CoSimManipulate before applying forces
        self.CoSimInstance.CoSimManipulate()
        
        # Apply the action force as the external force
        self.CoSimInstance.ApplyActionForce(F_action)

        # Advance the simulation by one step
        self.CoSimInstance.execution.step()

        # Get the new state (position, velocity)
        next_position = self.CoSimInstance.GetLastValue(slaveName="MASS1D", slaveVar="x")
        next_velocity = self.CoSimInstance.GetLastValue(slaveName="MASS1D", slaveVar="v")
        next_states = [next_position, next_velocity]

        # Compute the reward per step
        reward, done = self.RewardFunction(next_states)

        # Set the next states as current states
        self.states = next_states

        return next_states, reward, done

    def reset(self):
        # Initialize the Co-simulation
        self.InitializeCoSim()

        # Set the initial states
        self.states = self.initial_states