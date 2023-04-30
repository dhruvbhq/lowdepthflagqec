import numpy as np
import enum
from mpi4py import MPI
from datetime import datetime

comm = MPI.COMM_WORLD
num_cores = comm.Get_size()
my_rank = comm.Get_rank()

#############################################################

class syn_ex_status(enum.Enum):
    """
    Enum for specifying status of syndrome extraction, i.e. what the protocol
    is currently doing.
    """
    IDLE = enum.auto()
    MEAS_GEN_WITH_FLAG = enum.auto()
    MEAS_GEN_WITHOUT_FLAG = enum.auto()
    DET_RAISED_FLAG = enum.auto()
    DET_NONZERO_SYNDROME = enum.auto()
    DET_UNRAISED_FLAG_AND_ZERO_SYNDROME = enum.auto()
    
#############################################################

class error_model_enum(enum.Enum):
    """
    Enum for specifying the type of error model.
    """
    ONE_STOCHASTIC_PAULI_ERROR = enum.auto()
    CODE_CAPACITY_NOISE = enum.auto()
    CHAO_CKT_LVL_NOISE_WITHOUT_IDLING = enum.auto()

#############################################################

class error_spec:
    """
    Helper class to allow injecting different kinds of errors manually during
    syndrome extraction, by overriding depolarizing errors. Purpose is only to
    test the implementation.

    error_loc is the location where error is to be added - it is not
    necessarily a bad location, but it is a location after a two qubit gate (we
    treat XNOT as a single 2 qubit gate).
    """
    def __init__(self,
            inject_error:bool = False,
            error_loc:int = None,
            qubit_idx1:int = None,
            qubit_idx2:int = None,
            pauli_idx1:int = 0,
            pauli_idx2:int = 0):

        self.inject_error = inject_error
        self.error_loc = error_loc
        self.qubit_idx1 = qubit_idx1
        self.qubit_idx2 = qubit_idx2
        self.pauli_idx1 = pauli_idx1
        self.pauli_idx2 = pauli_idx2

#############################################################
# This class provides functionality of a quantum circuit, in the sense that it
# maintains a binary vector to store the accumulated Pauli operator acting on the
# state, and implements error propagation rules for quantum gates, as well as 
# injecting errors themselves on different locations.

class qec_flag_base_ckt_class:
    def __init__(self,
                 num_data_qubits,
                 num_anc_qubits,
                 num_flag_qubits,
                 p_phys,
                 samples_per_point,
                 error_scale_factor_idling,
                 error_scale_factor_two_qubit_gate,
                 error_scale_factor_single_qubit_gate,
                 error_scale_factor_prep,
                 error_scale_factor_meas,
                 verbose=False,
                 debug=False):

        # functionality, such as collecting measurement
        # outcomes and reset ancillas/flags only works for the case of 1
        # ancilla and 1 flag qubit.
        self.num_data_qubits = num_data_qubits
        self.num_anc_qubits = num_anc_qubits
        self.num_flag_qubits = num_flag_qubits
        self.num_all_qubits = self.num_data_qubits + self.num_anc_qubits + self.num_flag_qubits
        self.p_phys = p_phys
        self.samples_per_point = samples_per_point
        self.error_scale_factor_idling = error_scale_factor_idling
        self.error_scale_factor_two_qubit_gate = error_scale_factor_two_qubit_gate
        self.error_scale_factor_single_qubit_gate = error_scale_factor_single_qubit_gate
        self.error_scale_factor_prep = error_scale_factor_prep
        self.error_scale_factor_meas = error_scale_factor_meas
        self.verbose = verbose
        self.debug = debug
        
        # Binary vector to store overall pauli operator acting on qubits, due to errors, faults and propagation
        # in binary symplectic representation of Paulis
        self.pauli_accumulator = np.array([0]*(2*(self.num_all_qubits)))
        
        self.data_qubits = [i for i in range(self.num_data_qubits)]
        self.anc_qubits = [i for i in range(self.num_data_qubits, self.num_data_qubits+self.num_anc_qubits)]
        self.flag_qubits = [i for i in range(self.num_data_qubits+self.num_anc_qubits, self.num_all_qubits)]
    
    ########################################################################### 
    def clear_pauli_accumulator(self):
        self.pauli_accumulator = np.array([0]*(2*(self.num_all_qubits)))
    
    ########################################################################### 
    def single_qubit_X_error(self, qubit_idx, p_err):
        # Intended to be used for preparation errors for |0> state
        if(np.random.uniform() < p_err):
            # At this point, it has been decided that an error has to be
            # injected. 
            error_string = np.array([0]*(2*(self.num_all_qubits)))
            error_string[qubit_idx] = 1
            self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string).astype(int)
            if self.debug:
                print("DEBUG: ###INJECTING### X error on qubit ", qubit_idx)

    ###########################################################################
    def single_qubit_Z_error(self, qubit_idx, p_err):
        # Intended to be used for preparation errors for |+> state
        if(np.random.uniform() < p_err):
            # At this point, it has been decided that an error has to be
            # injected. 
            error_string = np.array([0]*(2*(self.num_all_qubits)))
            error_string[qubit_idx+self.num_all_qubits] = 1
            self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string).astype(int)
            if self.debug:
                print("DEBUG: ###INJECTING### Z error on qubit ", qubit_idx)
    
    ###########################################################################
    def single_qubit_Y_error(self, qubit_idx, p_err):
        if(np.random.uniform() < p_err):
            # At this point, it has been decided that an error has to be
            # injected. 
            error_string = np.array([0]*(2*(self.num_all_qubits)))
            error_string[qubit_idx] = 1
            error_string[qubit_idx+self.num_all_qubits] = 1
            self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string).astype(int)
            if self.debug:
                print("DEBUG: ###INJECTING### Y error on qubit ", qubit_idx)

    ###########################################################################
    def single_qubit_pauli_error(self, pauli_idx, qubit_idx):
        """
        Helper function to inject directed Pauli errors on qubits.
        pauli_idx* == 1: X error
        pauli_idx* == 2: Y error
        pauli_idx* == 3: Z error
        """
        error_string = np.array([0]*(2*(self.num_all_qubits)))
        
        if(pauli_idx == 1):
            error_string[qubit_idx] = 1
        elif(pauli_idx == 2):
            error_string[qubit_idx] = 1
            error_string[qubit_idx+self.num_all_qubits] = 1
        elif(pauli_idx == 3):
            error_string[qubit_idx+self.num_all_qubits] = 1
            
        self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string).astype(int)
            
    ########################################################################### 
    def single_qubit_depol_error(self, qubit_idx, p_err):
        
        if(np.random.uniform() < p_err):
            # At this point, it has been decided that an error has to be
            # injected. Now, decide which Pauli error is to be injected.
            dec = np.random.uniform()
            if dec < (1/3):
                self.single_qubit_pauli_error(1, qubit_idx)
                if(self.debug):
                    print("DEBUG: injecting X error on qubit ", qubit_idx)
            elif (dec >= (1/3)) and (dec < (2/3)):
                self.single_qubit_pauli_error(2, qubit_idx)
                if(self.debug):
                    print("DEBUG: injecting Y error on qubit ", qubit_idx)
            elif dec >= (2/3):
                self.single_qubit_pauli_error(3, qubit_idx)
                if(self.debug):
                    print("DEBUG: injecting Z error on data qubit ", qubit_idx)
            else:
                assert False, "Error in function single_qubit_depol_error"            

    ########################################################################### 
    def one_stochastic_pauli_error_data_qubits(self, p_err):
        """An additional function to help inject code capacity-style errors at various points."""
        
        # This list just keeps track of errors injected on data qubits. Each 
        # entry corresponds to a data qubit. A value of 0 means that no error
        # was injected, 1 is an X error, 2 is a Y error, 3 is a Z error.
        # This can be printed by setting debug = True in constructor.
        err_track = np.zeros(self.num_data_qubits)

        for n in range(self.num_data_qubits):
            if(np.random.uniform() < p_err):
                # At this point, it has been decided that an error has to be
                # injected on a particular data qubit. Now, decide which
                # Pauli error is to be injected.
                dec = np.random.uniform()
                if(dec < (1/3)):
                    self.single_qubit_pauli_error(1, self.data_qubits[n])
                    if(self.debug):
                        print("DEBUG: injecting X error on qubit ", n)
                    err_track[n] = 1
                elif((dec >= (1/3)) and (dec < (2/3))):
                    self.single_qubit_pauli_error(2, self.data_qubits[n])
                    if(self.debug):
                        print("DEBUG: injecting Y error on data qubit ", n)
                    err_track[n] = 2
                elif(dec >= (2/3)):
                    self.single_qubit_pauli_error(3, self.data_qubits[n])
                    if(self.debug):
                        print("DEBUG: injecting Z error on data qubit ", n)
                    err_track[n] = 3
                else:
                    assert False, "Error in function one_stochastic_pauli_error_data_qubits"
    
        if(self.debug):
            print("DEBUG: ERR_TRACK = ", err_track)
        
    ###########################################################################
    def two_qubit_pauli_error(self, pauli_idx1, pauli_idx2, qubit_idx1, qubit_idx2):
        """
        Helper function to inject directed Pauli errors on qubits.
        pauli_idx* == 1: X error
        pauli_idx* == 2: Y error
        pauli_idx* == 3: Z error
        """
        error_string1 = np.array([0]*(2*(self.num_all_qubits)))
        error_string2 = np.array([0]*(2*(self.num_all_qubits)))
        
        if(pauli_idx1 == 1):
            error_string1[qubit_idx1] = 1
        elif(pauli_idx1 == 2):
            error_string1[qubit_idx1] = 1
            error_string1[qubit_idx1+self.num_all_qubits] = 1
        elif(pauli_idx1 == 3):
            error_string1[qubit_idx1+self.num_all_qubits] = 1

        if(pauli_idx2 == 1):
            error_string2[qubit_idx2] = 1
        elif(pauli_idx2 == 2):
            error_string2[qubit_idx2] = 1
            error_string2[qubit_idx2+self.num_all_qubits] = 1
        elif(pauli_idx2 == 3):
            error_string2[qubit_idx2+self.num_all_qubits] = 1
            
        self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string1).astype(int)
        self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string2).astype(int)
        
    ########################################################################### 
    def two_qubit_gate_depol_error(self, qubit_idx1, qubit_idx2, p_err, location=None):
        
        if(np.random.uniform() < p_err):
            # At this point, it has been decided that an error has to be
            # injected. Now, decide which Pauli error is to be injected.
            dec = np.random.uniform()
            if self.debug:
                print("DEBUG: ###INJECTING### two_qubit_gate_depol_error at location = ", location)
            if dec < (1/15) :
                self.two_qubit_pauli_error(0, 1, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting I \otimes X error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            elif (dec >= (1/15)) and (dec < (2/15)) :
                self.two_qubit_pauli_error(0, 2, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting I \otimes Y error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            elif (dec >= (2/15)) and (dec < (3/15)) :
                self.two_qubit_pauli_error(0, 3, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting I \otimes Z error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            elif (dec >= (3/15)) and (dec < (4/15)) :
                self.two_qubit_pauli_error(1, 0, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting X \otimes I error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            elif (dec >= (4/15)) and (dec < (5/15)) :
                self.two_qubit_pauli_error(1, 1, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting X \otimes X error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            elif (dec >= (5/15)) and (dec < (6/15)) :
                self.two_qubit_pauli_error(1, 2, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting X \otimes Y error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            elif (dec >= (6/15)) and (dec < (7/15)) :
                self.two_qubit_pauli_error(1, 3, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting X \otimes Z error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            elif (dec >= (7/15)) and (dec < (8/15)) :
                self.two_qubit_pauli_error(2, 0, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting Y \otimes I error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            elif (dec >= (8/15)) and (dec < (9/15)) :
                self.two_qubit_pauli_error(2, 1, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting Y \otimes X error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            elif (dec >= (9/15)) and (dec < (10/15)) :
                self.two_qubit_pauli_error(2, 2, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting Y \otimes Y error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            elif (dec >= (10/15)) and (dec < (11/15)) :
                self.two_qubit_pauli_error(2, 3, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting Y \otimes Z error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            elif (dec >= (11/15)) and (dec < (12/15)) :
                self.two_qubit_pauli_error(3, 0, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting Z \otimes I error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            elif (dec >= (12/15)) and (dec < (13/15)) :
                self.two_qubit_pauli_error(3, 1, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting Z \otimes X error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            elif (dec >= (13/15)) and (dec < (14/15)) :
                self.two_qubit_pauli_error(3, 2, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting Z \otimes Y error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            elif(dec >= (14/15)):
                self.two_qubit_pauli_error(3, 3, qubit_idx1, qubit_idx2)
                if(self.debug):
                    print("DEBUG: injecting Z \otimes Z error on q1 = ", qubit_idx1, " q2 = ", qubit_idx2)
            else:
                assert False, "Error in function two_qubit_gate_depol_error"

    ########################################################################### 
    def two_qubit_gate_error(self,
            test_config:"error_spec"=None,
            error_loc:int=None,
            depol_err_qubit_idx1:int=None,
            depol_err_qubit_idx2:int=None,
            p_err=0):
        """Helper function to inject errors. By default, depolarizing
        errors will be added after two qubit gates, at the qubit indices
        specified by depol_* parameters, else the specified error at the
        specified location, based on test_config."""

        if(test_config is not None):
            # injecting manually specified error to make it possible to write a unit test
            if((test_config.inject_error) and (test_config.error_loc == error_loc)):
                self.two_qubit_pauli_error(test_config.pauli_idx1,
                                           test_config.pauli_idx2,
                                           test_config.qubit_idx1,
                                           test_config.qubit_idx2)
        else:
            # two qubit depol gate error, as per error model
            if self.debug:
                print("DEBUG: before injecting two qubit error at location ", error_loc)
            self.two_qubit_gate_depol_error(depol_err_qubit_idx1, depol_err_qubit_idx2, p_err, error_loc)
    
    ########################################################################### 
    def prepare_Z_basis(self, qubit_idx, p_err):
        # Clear errors
        self.pauli_accumulator[qubit_idx] = 0
        self.pauli_accumulator[qubit_idx+self.num_all_qubits] = 0
        # Add X error after preparation
        self.single_qubit_X_error(qubit_idx=qubit_idx, p_err=self.error_scale_factor_prep*p_err)

    ########################################################################### 
    def prepare_X_basis(self, qubit_idx, p_err):
        # Clear errors
        self.pauli_accumulator[qubit_idx] = 0
        self.pauli_accumulator[qubit_idx+self.num_all_qubits] = 0
        # Add Z error after preparation
        self.single_qubit_Z_error(qubit_idx=qubit_idx, p_err=self.error_scale_factor_prep*p_err)
               
    ########################################################################### 
    def reset_ancilla(self, p_err):

        for i in range(self.num_anc_qubits):
            # Clear errors
            self.pauli_accumulator[self.num_data_qubits+i] = 0
            self.pauli_accumulator[self.num_data_qubits+i+self.num_all_qubits] = 0
            # Add a preparation error
            self.single_qubit_X_error(self.num_data_qubits+i, self.error_scale_factor_prep*p_err)
        
    ########################################################################### 
    def reset_flag(self, p_err):
        # to reinitialize the flag every time to |+>.
        
        # Clear all errors
        for i in range(self.num_flag_qubits):
            self.pauli_accumulator[self.num_data_qubits+self.num_anc_qubits+i] = 0
            self.pauli_accumulator[self.num_data_qubits+self.num_anc_qubits+i+self.num_all_qubits] = 0
            # Add a preparation error
            self.single_qubit_Z_error(self.num_data_qubits+self.num_anc_qubits+i, self.error_scale_factor_prep*p_err)
        
        if(self.debug):
            print("DEBUG: flag has been reset A")
            
    ###########################################################################
    # Quantum gates are now just rules to propagate Paulis to other Paulis 
    # (since they are Cliffords)
    def h(self, qubit_idx, p_err, test_config, error_loc):
        # Modify an X error to a Z error and vice versa
        self.pauli_accumulator[qubit_idx], self.pauli_accumulator[qubit_idx+self.num_all_qubits] = \
            self.pauli_accumulator[qubit_idx+self.num_all_qubits], self.pauli_accumulator[qubit_idx]
        
        # Add single qubit gate error
        self.single_qubit_depol_error(qubit_idx=qubit_idx, p_err=self.error_scale_factor_single_qubit_gate*p_err)
        
    ###########################################################################
    def cnot(self, control_idx, target_idx, p_err, test_config, error_loc):
        error_string1 = np.array([0]*(2*(self.num_all_qubits)))
        error_string2 = np.array([0]*(2*(self.num_all_qubits)))

        # An X on control propagates to X on target
        if self.pauli_accumulator[control_idx] == 1:
            error_string1[target_idx] = 1
        
        # A Z on the target propagates to a Z on the control
        if self.pauli_accumulator[target_idx+self.num_all_qubits] == 1:
            error_string2[control_idx+self.num_all_qubits] = 1
        
        self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string1).astype(int)
        self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string2).astype(int)
        
        # Add a two qubit gate error
        # Not using test_config functionality for now
        self.two_qubit_gate_error(None, error_loc, control_idx, target_idx, self.error_scale_factor_two_qubit_gate*p_err)
        
    ###########################################################################
    def xnot(self, oplus_idx1, oplus_idx2, p_err, test_config, error_loc):
        error_string1 = np.array([0]*(2*(self.num_all_qubits)))
        error_string2 = np.array([0]*(2*(self.num_all_qubits)))
        
        # A Z on 1 qubit propagates to an X to the other
        if self.pauli_accumulator[oplus_idx1+self.num_all_qubits] == 1:
            error_string1[oplus_idx2] = 1
        
        if self.pauli_accumulator[oplus_idx2+self.num_all_qubits] == 1:
            error_string2[oplus_idx1] = 1
        
        self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string1).astype(int)
        self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string2).astype(int)
        
        # Add a two qubit gate error
        self.two_qubit_gate_error(None, error_loc, oplus_idx1, oplus_idx2, self.error_scale_factor_two_qubit_gate*p_err)
        
    ###########################################################################
    def ynot(self, oplus_idx, y_idx, p_err, test_config, error_loc):
        error_string1 = np.array([0]*(2*(self.num_all_qubits)))
        error_string2 = np.array([0]*(2*(self.num_all_qubits)))
        error_string3 = np.array([0]*(2*(self.num_all_qubits)))
        
        # A Z on oplus end propagates to Y on y end
        if self.pauli_accumulator[oplus_idx+self.num_all_qubits] == 1:
            error_string1[y_idx] = 1
            error_string1[y_idx+self.num_all_qubits] = 1
        
        # An X on y end propagates to X on oplus end
        if self.pauli_accumulator[y_idx] == 1:
            error_string2[oplus_idx] = 1

        # A Z on y end propagates to X on oplus end
        if self.pauli_accumulator[y_idx+self.num_all_qubits] == 1:
            error_string3[oplus_idx] = 1
            
        self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string1).astype(int)
        self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string2).astype(int)
        self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string3).astype(int)
        
        # Add a two qubit gate error
        self.two_qubit_gate_error(None, error_loc, oplus_idx, y_idx, self.error_scale_factor_two_qubit_gate*p_err)
        
    ###########################################################################
    def cz(self, qubit_idx1, qubit_idx2, p_err, test_config, error_loc):
        error_string1 = np.array([0]*(2*(self.num_all_qubits)))
        error_string2 = np.array([0]*(2*(self.num_all_qubits)))
        
        # X on one qubit propagates to a Z on the other
        if self.pauli_accumulator[qubit_idx1] == 1:
            error_string1[qubit_idx2+self.num_all_qubits] = 1

        if self.pauli_accumulator[qubit_idx2] == 1:
            error_string2[qubit_idx1+self.num_all_qubits] = 1
            
        self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string1).astype(int)
        self.pauli_accumulator = np.logical_xor(self.pauli_accumulator, error_string2).astype(int)
        
        # Add a two qubit gate error
        self.two_qubit_gate_error(None, error_loc, qubit_idx1, qubit_idx2, self.error_scale_factor_two_qubit_gate*p_err)
        
    ###########################################################################
    def measure_Z_basis(self, qubit_idx, p_err):
        # If there is an X error present in the pauli_accumulator, it will show
        # up in the Z measurement as a nontrivial outcome (-1 eigenvalue).
        meas_outcome = int(self.pauli_accumulator[qubit_idx] == 1)
        
        # Simulate measurement error by flipping this bit probabilistically
        if np.random.uniform() < self.error_scale_factor_meas*p_err:
            meas_outcome = 1 - meas_outcome
            
        return meas_outcome
    
    ###########################################################################
    def measure_X_basis(self, qubit_idx, p_err):
        # If there is a Z error present in the pauli_accumulator, it will show
        # up in the X measurement as a nontrivial outcome (-1 eigenvalue).
        meas_outcome = int(self.pauli_accumulator[qubit_idx+self.num_all_qubits] == 1)
        
        # Simulate measurement error by flipping this bit probabilistically
        if np.random.uniform() < self.error_scale_factor_meas*p_err:
            meas_outcome = 1 - meas_outcome
            
        return meas_outcome

#############################################################

class qec_flag_stabilizer_base:
    def __init__(self,
                 num_data_qubits,
                 num_anc_qubits,
                 num_flag_qubits,
                 syndrome_lookup_table,
                 syndrome_lookup_table_no_flag,
                 logical_ops,
                 p_phys,
                 samples_per_point=10**3,
                 error_model=error_model_enum.CHAO_CKT_LVL_NOISE_WITHOUT_IDLING,
                 error_scale_factor_idling=1, #idling is not implemented yet
                 seed_error_injection=None,
                 verbose=False,
                 debug=False):

        # some functionality, such as collecting measurement
        # outcomes and reset ancillas/flags only works for the case of 1
        # ancilla and 1 flag qubit.
        self.num_data_qubits = num_data_qubits
        self.num_anc_qubits = num_anc_qubits
        self.num_flag_qubits = num_flag_qubits
        self.num_all_qubits = self.num_data_qubits + self.num_anc_qubits + self.num_flag_qubits
        self.syndrome_lookup_table = syndrome_lookup_table
        self.syndrome_lookup_table_no_flag = syndrome_lookup_table_no_flag
        self.p_phys = p_phys
        self.samples_per_point = samples_per_point
        self.logical_ops = logical_ops
        self.error_model = error_model
        self.logical_error_counts = [None]*len(p_phys)
        self.verbose = verbose
        self.debug = debug
        self.syndrome_ex_status = syn_ex_status.IDLE # Syndrome extraction status
        self.current_syndrome_n_flag = None # Might or might not have flag info, based on subround
        self.syndrome_n_flag_1st_subround = None
        self.syndrome_2nd_subround = None

        if self.error_model == error_model_enum.ONE_STOCHASTIC_PAULI_ERROR or \
            self.error_model == error_model_enum.CODE_CAPACITY_NOISE:
            # will automatically inject independent Pauli error on data qubits
            # ONE_STOCHASTIC_PAULI_ERROR: just after "encoding"
            # CODE_CAPACITY_NOISE: before every stabilizer measurement circuit (flagged/unflagged)
            self.error_scale_factor_idling = 0
            self.error_scale_factor_two_qubit_gate = 0
            self.error_scale_factor_single_qubit_gate = 0
            self.error_scale_factor_prep = 0
            self.error_scale_factor_meas = 0
        elif self.error_model == error_model_enum.CHAO_CKT_LVL_NOISE_WITHOUT_IDLING:
            self.error_scale_factor_idling = 0
            self.error_scale_factor_two_qubit_gate = 1.0
            # Chao uses 0 error on hadamard
            self.error_scale_factor_single_qubit_gate = 0
            self.error_scale_factor_prep = (4.0/15)
            self.error_scale_factor_meas = (4.0/15)
        else:
            assert False, "Error in error model specification."
            
        if(seed_error_injection is not None):
            np.random.seed(seed_error_injection)
        
        self.qec_flag_base_ckt = qec_flag_base_ckt_class(self.num_data_qubits, 
                                                         self.num_anc_qubits, 
                                                         self.num_flag_qubits,
                                                         self.p_phys,
                                                         self.samples_per_point,
                                                         self.error_scale_factor_idling,
                                                         self.error_scale_factor_two_qubit_gate,
                                                         self.error_scale_factor_single_qubit_gate,
                                                         self.error_scale_factor_prep,
                                                         self.error_scale_factor_meas)
    
    ########################################################################### 
    def create_circuit(self):
        
        if hasattr(self, 'qec_flag_base_ckt'):
            del self.qec_flag_base_ckt
        
        self.qec_flag_base_ckt = qec_flag_base_ckt_class(self.num_data_qubits, 
                                                         self.num_anc_qubits, 
                                                         self.num_flag_qubits,
                                                         self.p_phys,
                                                         self.samples_per_point,
                                                         self.error_scale_factor_idling,
                                                         self.error_scale_factor_two_qubit_gate,
                                                         self.error_scale_factor_single_qubit_gate,
                                                         self.error_scale_factor_prep,
                                                         self.error_scale_factor_meas)
        
    def print_pauli_accumulator(self):
        print(self.qec_flag_base_ckt.pauli_accumulator)
        
    ########################################################################### 
    def init_state(self, p_err=0):
        pass
    
    
    ########################################################################### 
    def measure_ancilla_and_flag(self, with_flag, p_err=0):
        """
        Measures ancilla qubit and flag qubit (if with_flag is true). Saves the
        measurement outcome to self.current_syndrome_n_flag, as an np array. The
        outcome is either a single bit of syndrome value, or a list of two
        values with first entry being the syndrome and second entry the flag. 

        Note: This implementation only works for the case of 1 ancilla qubit
        and 1 flag qubit.
        """
        
        if(with_flag):
            syndrome = self.qec_flag_base_ckt.measure_Z_basis(self.qec_flag_base_ckt.anc_qubits[0], p_err)
            flag = self.qec_flag_base_ckt.measure_X_basis(self.qec_flag_base_ckt.flag_qubits[0], p_err)
            self.current_syndrome_n_flag = np.atleast_2d(np.array([syndrome, flag]))
        else:
            syndrome = self.qec_flag_base_ckt.measure_Z_basis(self.qec_flag_base_ckt.anc_qubits[0], p_err)
            self.current_syndrome_n_flag = np.array([syndrome])    

    ########################################################################### 
    def measure_full_syndrome_without_flags(self, test_config:"error_spec"=None, p_err=0):
        pass

    ########################################################################### 
    def update_syn_ex_status(self):
        """
        Helper function for syndrome_extraction. Updates the status variable
        depending on the observed values of syndrome bit and flag.
        """
        # If flag is measured as 1 (i.e. |->), change status to DET_RAISED_FLAG
        if(self.current_syndrome_n_flag[0][1] == 1):
            self.syndrome_ex_status = syn_ex_status.DET_RAISED_FLAG
        # Else, if syndrome bit is nonzero, change status to DET_NONZERO_SYNDROME 
        elif(self.current_syndrome_n_flag[0][0] == 1):
            self.syndrome_ex_status = syn_ex_status.DET_NONZERO_SYNDROME
        # Else, if both flag and syndrome are 0, change status to
        # DET_UNRAISED_FLAG_AND_ZERO_SYNDROME
        elif((self.current_syndrome_n_flag[0][1] == 0) and
            (self.current_syndrome_n_flag[0][0] == 0)):
            self.syndrome_ex_status = syn_ex_status.DET_UNRAISED_FLAG_AND_ZERO_SYNDROME
        if self.debug:
            print("DEBUG: current_syndrome_n_flag = ", self.current_syndrome_n_flag, " syndrome_ex_status changed to ", self.syndrome_ex_status)
        return

    ########################################################################### 
    def syndrome_extraction(self, test_config:"error_spec"=None, p_err=0):
        """
        The flag protocol for extracting syndrome as well as flag qubits.
        Expected implementation is in child class.
        """

        # This is expected to be the place where the final syndrome will be decided.
        pass

    ########################################################################### 
    def syndrome_decoding(self):
        
        # Note: this only works for the case of 1 ancilla qubit and 1 flag
        # qubit.

        # This function actually applies the correction, which is assumed to
        # be error-free.

        # If syndrome is not present in look up table, don't correct.
        if self.debug:
            print("DEBUG: in SYNDROME_DECODING, syndrome_n_flag_1st_subround = ", self.syndrome_n_flag_1st_subround, " syndrome_2nd_subround = ", self.syndrome_2nd_subround)
        if (self.syndrome_n_flag_1st_subround in self.syndrome_lookup_table) and\
            (self.syndrome_2nd_subround in\
                self.syndrome_lookup_table[self.syndrome_n_flag_1st_subround]):
                # The correction is a binary symplectic vector representing
                # the correction Pauli string, only on the data qubits
                correction = self.syndrome_lookup_table[self.syndrome_n_flag_1st_subround][self.syndrome_2nd_subround]
                # Adding zeros for ancilla and flag qubits
                correction = np.concatenate((correction[0:self.num_data_qubits],\
                                             [0]*(self.num_anc_qubits+self.num_flag_qubits),\
                                                 correction[self.num_data_qubits:2*self.num_data_qubits],\
                                                     [0]*(self.num_anc_qubits+self.num_flag_qubits)))
                    
                self.qec_flag_base_ckt.pauli_accumulator = \
                    np.logical_xor(self.qec_flag_base_ckt.pauli_accumulator, correction).astype(int)

    ########################################################################### 
    def logical_error_tracking(self, j):
        
	# This step has an additional error-free decoding step in the end to
	# remove the O(p) errors Ref: Chao and Reichardt, Chamberland.
        
         
        # Project the state back to codespace, possibly with a logical error
        if self.debug:
            print("DEBUG: Applying error-free QEC cycle")
        pauli_accumulator_before_noiseless_qec = np.copy(self.qec_flag_base_ckt.pauli_accumulator)
        self.syndrome_extraction(test_config=None, p_err=-1)
        self.syndrome_decoding()

        # To decide if there has been a logical error on the data qubits:
        # If the pauli accumulator commutes with all logical operators, 
        # there has been no logical error
        pauli_accumulator_data_qubits = \
            np.concatenate((self.qec_flag_base_ckt.pauli_accumulator[0:self.num_data_qubits],
                           self.qec_flag_base_ckt.pauli_accumulator[self.num_all_qubits:self.num_all_qubits+self.num_data_qubits]))
        n = self.num_data_qubits
        iden = np.eye(n)
        zero = np.zeros([n,n])
        L = np.block([[zero,iden],[iden,zero]])
        
        commutation = (np.matmul(pauli_accumulator_data_qubits, np.matmul(L, np.transpose(self.logical_ops)))%2).astype(int)

        # If the error anticommutes with even 1 logical operator, count it as a logical error.        
        if np.any(commutation):
            if(self.debug):
                print("DEBUG: counting as a logical error, commutation = ", commutation)
            self.logical_error_counts[j] += 1
            self.create_circuit()
        else:
            if(self.debug):
                print("DEBUG: NOT counting as a logical error")
            # Restore the state to the one before the artificial noiseless qec cycle
            
            self.qec_flag_base_ckt.pauli_accumulator = np.copy(pauli_accumulator_before_noiseless_qec)
            
        if(self.debug):
            print("#######################################################")
        
    ########################################################################### 
    def logical_error_rate_reporting(self):
        print("logical_error_counts = ", self.logical_error_counts)
        self.logical_error_probs = [logical_error_count/self.samples_per_point for logical_error_count in self.logical_error_counts]
        print("logical_error_probs = ", self.logical_error_probs)

    ########################################################################### 
    def cleanup(self):
        self.syndrome_ex_status = syn_ex_status.IDLE
        self.current_syndrome_n_flag = None
        self.syndrome_n_flag_1st_subround = None
        self.syndrome_2nd_subround = None
        
    ########################################################################### 
    def p_phys_sweep_simulation(self):
        
        for j in range(len(self.p_phys)):
    
            # This print is just to check if the simulation is progressing
            print("Simulating for p_phys = ", self.p_phys[j])
            self.logical_error_counts[j] = 0
            self.create_circuit()
            
            # Error correction cycles
            for i in range(self.samples_per_point):
                if(i % (self.samples_per_point/4) == 0):
                    print("NOTE: sample = ", i, " #####################")

                self.init_state(self.p_phys[j])
                self.syndrome_extraction(p_err=self.p_phys[j])
                # This function also applies the recovery/correction operation.
                self.syndrome_decoding()
                self.logical_error_tracking(j)
                self.cleanup()

    ########################################################################## 
    def p_phys_sweep_simulation_mpi(self):
        
        batch_size = self.samples_per_point // num_cores
        remainder = self.samples_per_point % num_cores
        if my_rank < remainder:
            batch_size += 1
        if my_rank == 0:
            self.results_per_batch_per_p_phys = {}

        for j in range(len(self.p_phys)):
    
            # This print is just to check if the simulation is progressing
            print("NOTE: Simulating for p_phys = ", self.p_phys[j], " rank = ", my_rank, " batch_size = ", batch_size, " current time = ", datetime.now().time())

            self.logical_error_counts[j] = 0
            self.create_circuit()
            
            for i in range(batch_size):
                if(i % (self.samples_per_point/4) == 0):
                    print("NOTE: sample = ", i, " rank = ", my_rank, " #####################", " current time = ", datetime.now().time())

                self.init_state(self.p_phys[j])
                self.syndrome_extraction(p_err=self.p_phys[j])
                self.syndrome_decoding()

                # To infer whether a logical error has occured, an additional
                # error-free decoding step is implemented inside this function.
                self.logical_error_tracking(j)
                
                self.cleanup()

            # Send a dictionary with my_rank, p_phys, samples, and logical error counts to my_rank=0 
            local_result_dict = {
                    "rank":my_rank,
                    "p_phys":self.p_phys[j],
                    "batch_size":batch_size,
                    "logical_error_counts":self.logical_error_counts[j]
                    }
            if my_rank > 0:
                if(self.debug):
                    print("DEBUG: sending dict from rank = ", my_rank, "dict = ", local_result_dict, " current time = ", datetime.now().time())
                comm.send(local_result_dict, dest=0, tag=1000+my_rank)
                if(self.debug):
                    print("DEBUG: after send statement from rank = ", my_rank, " current time = ", datetime.now().time())
            else:
                self.results_per_batch_per_p_phys["rank_"+str(my_rank)+"_p_phys_idx_"+str(j)] = local_result_dict

        # Collect all results in rank=0 process (core)
        if my_rank == 0:
            for k in range(1, num_cores):
                for j in range(len(self.p_phys)):
                    if(self.debug):
                        print("DEBUG: before recv statement from rank = ", my_rank, " current time = ", datetime.now().time())
                    self.results_per_batch_per_p_phys["rank_"+str(k)+"_p_phys_idx_"+str(j)] = comm.recv(source=k, tag=1000+k)
                    if(self.debug):
                        print("DEBUG: after recv statement from rank = ", my_rank, " current time = ", datetime.now().time())
            self.complete_results = {}
            # Total samples = samples_per_point * size of p_phys
            self.complete_results["total_samples"] = 0
            for r in self.results_per_batch_per_p_phys:
                if(self.debug):
                    print("DEBUG: in MPI method, r = ", r, "val = ", self.results_per_batch_per_p_phys[r], "\n")
                if str(self.results_per_batch_per_p_phys[r]["p_phys"]) in self.complete_results:
                    self.complete_results[str(self.results_per_batch_per_p_phys[r]["p_phys"])] += \
                        self.results_per_batch_per_p_phys[r]["logical_error_counts"]
                else:
                    self.complete_results[str(self.results_per_batch_per_p_phys[r]["p_phys"])] = \
                        self.results_per_batch_per_p_phys[r]["logical_error_counts"]
                self.complete_results["total_samples"] += self.results_per_batch_per_p_phys[r]["batch_size"]
            print("NOTE: in MPI method, complete_results = ", self.complete_results, " rank = ", my_rank)
