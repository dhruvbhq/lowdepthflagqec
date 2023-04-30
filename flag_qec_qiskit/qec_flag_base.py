import qiskit
import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, execute, IBMQ
from qiskit.quantum_info import partial_trace, state_fidelity
import enum
from mpi4py import MPI
from datetime import datetime

aer_sim = Aer.get_backend('aer_simulator')
statevector_sim = Aer.get_backend('statevector_simulator')
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

class qec_flag_base:
    def __init__(self,
                 num_data_qubits,
                 num_anc_qubits,
                 num_flag_qubits,
                 syndrome_lookup_table,
                 syndrome_lookup_table_no_flag,
                 p_phys,
                 rounds=10**3,
                 seed_simulator=None,
                 seed_error_injection=None,
                 verbose=False,
                 debug=False,
                 barrier=False):

        # functionality, such as collecting measurement
        # outcomes and reset ancillas/flags only works for the case of 1
        # ancilla and 1 flag qubit.
        self.num_data_qubits = num_data_qubits
        self.num_anc_qubits = num_anc_qubits
        self.num_flag_qubits = num_flag_qubits
        self.syndrome_lookup_table = syndrome_lookup_table
        self.syndrome_lookup_table_no_flag = syndrome_lookup_table_no_flag
        self.p_phys = p_phys
        self.rounds = rounds
        self.logical_error_counts = [None]*len(p_phys)
        self.verbose = verbose
        self.debug = debug
        self.barrier = barrier
        self.syndrome_ex_status = syn_ex_status.IDLE # Syndrome extraction status
        self.current_syndrome_n_flag = None # Might or might not have flag info, based on subround
        self.syndrome_n_flag_1st_subround = None
        self.syndrome_2nd_subround = None

        self.error_scale_factor_cnot = 1.0
        self.error_scale_factor_hadamard = 0 
        self.error_scale_factor_prep = (4.0/15)
        self.error_scale_factor_meas = (4.0/15)

        # By default, qiskit chooses a different random seed every time
        # execute(backend) is invoked to simulate. This causes the state to
        # collapse along a different path each time, which messes up tracking
        # the state. self.seed_simulator is used so that every time execute()
        # is called, it takes the same seed, whether it is the
        # statevector_simulator or an aer_simulator backend. The seed is still
        # chosen randomly, or supplied by the user.

        if(seed_error_injection is not None):
            np.random.seed(seed_error_injection)
        if(seed_simulator == None):
            seed_simulator = np.random.randint(1,10**9)
        self.seed_simulator = seed_simulator
    
    ########################################################################### 
    def create_circuit(self):
        self.data_qubits = QuantumRegister(self.num_data_qubits, 'data_qubits')
        self.anc_qubits = QuantumRegister(self.num_anc_qubits, 'anc_qubits')
        self.flag_qubits = QuantumRegister(self.num_flag_qubits, 'flag_qubits')
        self.syndrome_bits = ClassicalRegister(self.num_anc_qubits, 'syndrome_bits')
        self.flag_bits = ClassicalRegister(self.num_flag_qubits, 'flag_bits')
        self.qec_flag_base_ckt = QuantumCircuit(self.data_qubits,
                                                self.anc_qubits,
                                                self.flag_qubits,
                                                self.syndrome_bits,
                                                self.flag_bits)
        if(self.barrier):
            self.qec_flag_base_ckt.barrier()
        
    ########################################################################### 
    def init_state(self, p_err=0):
        pass
    
    ########################################################################### 
    def encoding_circuit(self):
        # Expected implementation is in child class
        pass
    
    ########################################################################### 
    def add_barrier(self):
        if(self.barrier):
            self.qec_flag_base_ckt.barrier()
        return
    ########################################################################### 
    def state_sim(self):
        result = execute(self.qec_flag_base_ckt, statevector_sim, seed_simulator=self.seed_simulator).result()
        state_qec = result.get_statevector(self.qec_flag_base_ckt)
        # Trace out ancilla qubits
        self.current_state = partial_trace(state_qec, [x for x in 
                                                       range(self.data_qubits.size,
                                                             self.data_qubits.size + self.anc_qubits.size + self.flag_qubits.size)])
    
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
        self.qec_flag_base_ckt.measure(self.anc_qubits, self.syndrome_bits)

        if(with_flag):
            # Measure in X basis for flag qubit
            for i in range(self.flag_qubits.size):
                self.qec_flag_base_ckt.h(self.data_qubits.size + self.anc_qubits.size + i)
            self.qec_flag_base_ckt.measure(self.flag_qubits, self.flag_bits)
        
        result = execute(self.qec_flag_base_ckt,
                         aer_sim,
                         shots=1,
                         seed_simulator=self.seed_simulator).result()
        counts = result.get_counts(self.qec_flag_base_ckt)

        # Reversing the order to get it as [syndrome, flag]
        # Storing syndrome in reverse order, because of qiskit's ordering.
        # Values returned by qiskit in syndrome register are in the order
        # ['flag_qubit[0] anc_qubit[0]'] etc.
        # i.e. temp_syndrome[0][-1] (last entry) corresponds to anc_qubit[0], 
        # which is the opposite ordering. Therefore, the next line reverses the 
        # order of the obtained value in syndrome register. This way, the syndrome_lookup_table
        # can be specified in the physically intuitive way, and there is no need to reverse
        # the order when the ancillas are reset (if that is implemented).

        # replace() is needed because qiskit returns the two bits as a string
        # '0 0', separated with a space
        temp_syndrome = list(counts.keys())[0][::-1].replace(' ', '')
        if(with_flag):
            self.current_syndrome_n_flag = np.atleast_2d(np.array([int(temp_syndrome[0]), int(temp_syndrome[1])]))
            
            # Error: this models measurement error
            if np.random.uniform() < self.error_scale_factor_meas*p_err:
                # Flip the flag outcome
                self.current_syndrome_n_flag[0][1] = 1 - self.current_syndrome_n_flag[0][1]
            # Error: this models measurement error
            if np.random.uniform() < self.error_scale_factor_meas*p_err:
                # Flip the ancilla(syndrome) outcome
                self.current_syndrome_n_flag[0][0] = 1 - self.current_syndrome_n_flag[0][0]
        else:
            self.current_syndrome_n_flag = np.array([int(temp_syndrome[0])])
            # Error: this models measurement error
            if np.random.uniform() < self.error_scale_factor_meas*p_err:
                # Flip the ancilla(syndrome) outcome
                self.current_syndrome_n_flag[0] = 1 - self.current_syndrome_n_flag[0]

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

        # If syndrome is not present in look up table, don't correct.
        if self.debug:
            print("DEBUG: in SYNDROME_DECODING, syndrome_n_flag_1st_subround = ", self.syndrome_n_flag_1st_subround, " syndrome_2nd_subround = ", self.syndrome_2nd_subround)
        if (self.syndrome_n_flag_1st_subround in self.syndrome_lookup_table) and\
            (self.syndrome_2nd_subround in\
                self.syndrome_lookup_table[self.syndrome_n_flag_1st_subround]):
                correction = self.syndrome_lookup_table[self.syndrome_n_flag_1st_subround][self.syndrome_2nd_subround]
                if self.debug:
                    print("DEBUG: correction = ", correction)
                for idx, op in enumerate(correction):
                    if(op == 'I'):
                        pass
                    elif(op == 'X'):
                        self.qec_flag_base_ckt.x(self.data_qubits[idx])
                    elif(op == 'Y'):
                        self.qec_flag_base_ckt.y(self.data_qubits[idx])
                    elif(op == 'Z'):
                        self.qec_flag_base_ckt.z(self.data_qubits[idx])
                    else:
                        assert False, """Error in syndrome lookup table specification.""" 

    ########################################################################### 
    def reset_ancilla(self, p_err=0):
        # This function resets the ancilla qubits by applying an X gate
        # wherever the last syndrome had a bit value 1

        syndrome_bit = np.atleast_2d(self.current_syndrome_n_flag)
        if(syndrome_bit[0][0] == 1):
            if(self.barrier):
                self.qec_flag_base_ckt.barrier()
            self.qec_flag_base_ckt.x(self.anc_qubits[0])
            if(self.debug):
                print("DEBUG: ancilla has been reset")
            if(self.barrier):
                self.qec_flag_base_ckt.barrier()
        self.single_qubit_X_error(self.anc_qubits[0], self.error_scale_factor_prep*p_err)
        
    ########################################################################### 
    def reset_flag(self, p_err=0):
        # This function resets the flag qubits by applying a X gate followed by
        # H gate wherever the last flag had a bit value 1, else only H gate is
        # applied.  Note that for measurement in X basis, We are going back to
        # Z basis via Hadamard, so it is required to reinitialize the flag
        # every time to |+>.

        if(self.current_syndrome_n_flag[0][1] == 1):
            self.qec_flag_base_ckt.x(self.flag_qubits[0])
            # Error - this models preparation error. With this probability, the
            # flag gets prepared in |-> instead of |+>.
            self.single_qubit_X_error(self.flag_qubits[0], self.error_scale_factor_prep*p_err)
            self.qec_flag_base_ckt.h(self.flag_qubits[0])
            if(self.debug):
                print("DEBUG: flag has been reset A")
        else:
            # Error - this models preparation error. With this probability, the
            # flag gets prepared in |-> instead of |+>.
            self.single_qubit_X_error(self.flag_qubits[0], self.error_scale_factor_prep*p_err)
            self.qec_flag_base_ckt.h(self.flag_qubits[0])
            if(self.debug):
                print("DEBUG: flag has been reset B")

        if(self.barrier):
            self.qec_flag_base_ckt.barrier()

    ########################################################################### 
    def logical_error_tracking(self, j):
        
	# This has an error-free decoding step in the end to remove the
	# remaining O(p) errors.  This is similar to Chao and Reichardt's
	# implementation.

        # Project the state back to codespace, possibly with a logical error
        if self.debug:
            print("DEBUG: Applying error-free QEC cycle")
        self.syndrome_extraction(test_config=None, p_err=0)
        self.syndrome_decoding()

        # Simulate current statevector to determine if there has been a decoding error
        self.state_sim()
        # If state of data qubits is not close enough to the expected (encoded) state,
        # count it as a logical error
        if(not np.isclose(state_fidelity(self.current_state, self.ideal_initial_state), 1.0)):
            if(self.debug):
                print("DEBUG: counting as a logical error, fidelity = ", state_fidelity(self.current_state, self.ideal_initial_state))
            self.logical_error_counts[j] += 1
        else:
            if(self.debug):
                print("DEBUG: NOT counting as a logical error")
        if(self.debug):
            print("#######################################################")
        
    ########################################################################### 
    def logical_error_rate_reporting(self):
        print("logical_error_counts = ", self.logical_error_counts)
        self.logical_error_probs = [logical_error_count/self.rounds for logical_error_count in self.logical_error_counts]
        print("logical_error_probs = ", self.logical_error_probs)
    
    ########################################################################### 
    def stochastic_pauli_X_error_data_qubits(self, j):
        # This list just keeps track of errors injected on data qubits. Each 
        # entry corresponds to a data qubit. A value of 0 means that no error
        # was injected, 1 is an X error, 2 is a Y error, 3 is a Z error.
        # This can be printed by setting debug = True in constructor.
        err_track = np.zeros(self.num_data_qubits)

        for n in range(self.num_data_qubits):
            if(np.random.uniform() < self.p_phys[j]):
                # Only a Pauli X error
                self.qec_flag_base_ckt.x(self.data_qubits[n])

                if(self.debug):
                    print("DEBUG: injecting X error on data qubit ", n)
                err_track[n] = 1
        
        if(self.debug):
            print("DEBUG: ERR_TRACK = ", err_track)
        if(self.barrier):
            self.qec_flag_base_ckt.barrier()
        
    ########################################################################### 
    def stochastic_full_pauli_error_data_qubits(self, p_err):
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
                    self.qec_flag_base_ckt.x(self.data_qubits[n])
                    if(self.debug):
                        print("DEBUG: injecting X error on data qubit ", n)
                    err_track[n] = 1
                elif((dec >= (1/3)) and (dec < (2/3))):
                    self.qec_flag_base_ckt.y(self.data_qubits[n])
                    if(self.debug):
                        print("DEBUG: injecting Y error on data qubit ", n)
                    err_track[n] = 2
                elif(dec >= (2/3)):
                    self.qec_flag_base_ckt.z(self.data_qubits[n])
                    if(self.debug):
                        print("DEBUG: injecting Z error on data qubit ", n)
                    err_track[n] = 3
                else:
                    assert False, "Error in function stochastic_full_pauli_error_data_qubits"
        
        if(self.debug):
            print("DEBUG: ERR_TRACK = ", err_track)
        if(self.barrier):
            self.qec_flag_base_ckt.barrier()
        
    ########################################################################### 
    def single_qubit_gate_depol_error(self, qubit_idx, p_err):
        if(np.random.uniform() < p_err):
            # At this point, it has been decided that an error has to be
            # injected. Now, decide which Pauli error is to be injected.
            dec = np.random.uniform()
            if dec < (1/3) :
                self.qec_flag_base_ckt.x(qubit_idx)
                if(self.debug):
                    print("DEBUG: injecting X error on qubit ", qubit_idx)
            elif (dec >= (1/3)) and (dec < (2/3)) :
                self.qec_flag_base_ckt.y(qubit_idx)
                if(self.debug):
                    print("DEBUG: injecting Y error on data qubit ", qubit_idx)
            elif dec >= (2/3) :
                self.qec_flag_base_ckt.z(qubit_idx)
                if(self.debug):
                    print("DEBUG: injecting Z error on data qubit ", qubit_idx)
            else:
                assert False, "Error in function single_qubit_gate_depol_error"
        
        if(self.barrier):
            self.qec_flag_base_ckt.barrier()

    ########################################################################### 
    def single_qubit_X_error(self, qubit_idx, p_err):
        # Intended to be used for preparation errors
        if(np.random.uniform() < p_err):
            # At this point, it has been decided that an error has to be
            # injected. 
            self.qec_flag_base_ckt.x(qubit_idx)
            if self.debug:
                print("DEBUG: ###INJECTING### X error on qubit ", qubit_idx)
        
            if(self.barrier):
                self.qec_flag_base_ckt.barrier()
    ########################################################################### 

    def two_qubit_pauli_error(self, pauli_idx1, pauli_idx2, qubit_idx1, qubit_idx2):
        """
        Helper function to inject directed Pauli errors on qubits.
        """
        if(pauli_idx1 == 1):
            self.qec_flag_base_ckt.x(qubit_idx1)
        elif(pauli_idx1 == 2):
            self.qec_flag_base_ckt.y(qubit_idx1)
        elif(pauli_idx1 == 3):
            self.qec_flag_base_ckt.z(qubit_idx1)

        if(pauli_idx2 == 1):
            self.qec_flag_base_ckt.x(qubit_idx2)
        elif(pauli_idx2 == 2):
            self.qec_flag_base_ckt.y(qubit_idx2)
        elif(pauli_idx2 == 3):
            self.qec_flag_base_ckt.z(qubit_idx2)

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

        if(self.barrier):
            self.qec_flag_base_ckt.barrier()

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
                if self.debug:
                    print("DEBUG in two_qubit_gate_error, applying user-defined test_config error")
                if(self.barrier):
                    self.qec_flag_base_ckt.barrier()
                self.two_qubit_pauli_error(test_config.pauli_idx1,
                                           test_config.pauli_idx2,
                                           test_config.qubit_idx1,
                                           test_config.qubit_idx2)
                if(self.barrier):
                    self.qec_flag_base_ckt.barrier()
        else:
            # two qubit depol gate error, as per error model
            if self.debug:
                print("DEBUG: before injecting two qubit error at location ", error_loc)
            self.two_qubit_gate_depol_error(depol_err_qubit_idx1, depol_err_qubit_idx2, p_err, error_loc)

    ########################################################################### 
    def xnot_subckt_err(self,
            qubit_idx1:int=None,
            qubit_idx2:int=None,
            p_err=0,
            test_config:"error_spec"=None,
            error_loc:int=None):
        """Helper function to implement XNOT gate with errors. By default, depolarizing
        errors will be added after two qubit gates, at the specified qubit indices,
        else the specified error at the specified location, based on test_config."""

        self.qec_flag_base_ckt.h(qubit_idx1)
        # Error
        self.single_qubit_gate_depol_error(qubit_idx1, self.error_scale_factor_hadamard*p_err)
        self.qec_flag_base_ckt.cnot(qubit_idx1, qubit_idx2)
        # Error
        self.two_qubit_gate_error(test_config, error_loc, qubit_idx1, qubit_idx2, self.error_scale_factor_cnot*p_err)
        self.qec_flag_base_ckt.h(qubit_idx1)
        # Error
        self.single_qubit_gate_depol_error(qubit_idx1, self.error_scale_factor_hadamard*p_err)

    ########################################################################### 
    def ynot_subckt_err(self,
            qubit_idx1:int=None,
            qubit_idx2:int=None,
            p_err=0,
            test_config:"error_spec"=None,
            error_loc:int=None):
        """Helper function to implement YNOT gate with errors. By default, depolarizing
        errors will be added after two qubit gates, at the specified qubit indices,
        else the specified error at the specified location, based on test_config."""

        self.qec_flag_base_ckt.h(qubit_idx1)
        # Error
        self.single_qubit_gate_depol_error(qubit_idx1, self.error_scale_factor_hadamard*p_err)
        self.qec_flag_base_ckt.cy(qubit_idx1, qubit_idx2)
        # Error
        self.two_qubit_gate_error(test_config, error_loc, qubit_idx1, qubit_idx2, self.error_scale_factor_cnot*p_err)
        self.qec_flag_base_ckt.h(qubit_idx1)
        # Error
        self.single_qubit_gate_depol_error(qubit_idx1, self.error_scale_factor_hadamard*p_err)

    ########################################################################### 
    def cnot_subckt_err(self,
            qubit_idx1:int=None,
            qubit_idx2:int=None,
            p_err=0,
            test_config:"error_spec"=None,
            error_loc:int=None):
        """Helper function to implement CNOT gate with errors. By default, depolarizing
        errors will be added after two qubit gates, at the specified qubit indices,
        else the specified error at the specified location, based on test_config."""

        self.qec_flag_base_ckt.cnot(qubit_idx1, qubit_idx2)
        # Error
        self.two_qubit_gate_error(test_config, error_loc, qubit_idx1, qubit_idx2, self.error_scale_factor_cnot*p_err)

    ########################################################################### 
    def cleanup(self):
        self.syndrome_ex_status = syn_ex_status.IDLE
        self.current_syndrome_n_flag = None
        self.syndrome_n_flag_1st_subround = None
        self.syndrome_2nd_subround = None
        
    ########################################################################### 
    def p_phys_sweep_simulation(self):
        
        # This part is just to get the initial state vector after encoding, to
        # use it as a reference state for tracking logical errors, so there is
        # no need to run it in a loop, and no need to inject an error

        self.create_circuit()
            
        self.init_state(0)
            
        self.encoding_circuit()
        
        self.state_sim()
        self.ideal_initial_state = self.current_state
        if(self.verbose):
            print("DEBUG: ideal_initial_state = ", self.ideal_initial_state)
        del self.qec_flag_base_ckt
        
        for j in range(len(self.p_phys)):
    
            # This print is just to check if the simulation is progressing
            print("Simulating for p_phys = ", self.p_phys[j])
            self.logical_error_counts[j] = 0
            
            # Error correction rounds
            # In the current implementation, for every round, the circuit gets
            # created and initialized anew
            for i in range(self.rounds):
                if(i % 500 == 0):
                    print("round = ", i, "#####")

                self.create_circuit()
                self.init_state(self.p_phys[j])
                self.encoding_circuit()
                self.syndrome_extraction(p_err=self.p_phys[j])
                # This function also applies the recovery/correction operation.
                self.syndrome_decoding()
                self.logical_error_tracking(j)
                
                self.cleanup()

    ########################################################################### 
    def p_phys_sweep_simulation_mpi(self):
        
        # This part is just to get the initial state vector after encoding, to
        # use it as a reference state for tracking logical errors, so there is
        # no need to run it in a loop, and no need to inject an error

        self.create_circuit()
            
        self.init_state(0)
            
        self.encoding_circuit()
        
        self.state_sim()
        self.ideal_initial_state = self.current_state
        if(self.verbose):
            print("DEBUG: ideal_initial_state = ", self.ideal_initial_state)
        
        del self.qec_flag_base_ckt

        batch_size = self.rounds // num_cores
        remainder = self.rounds % num_cores
        if my_rank < remainder:
            batch_size += 1
        if my_rank == 0:
            self.results_per_batch_per_p_phys = {}

        for j in range(len(self.p_phys)):
    
            # This print is just to check if the simulation is progressing
            print("NOTE: Simulating for p_phys = ", self.p_phys[j], " rank = ", my_rank, " batch_size = ", batch_size, " current time = ", datetime.now().time())

            self.logical_error_counts[j] = 0
            
            # Error correction rounds
            # In this implementation, for every round, the circuit gets
            # created and initialized anew
            for i in range(batch_size):
                if(i % 10**5 == 0):
                    print("NOTE: round = ", i, " rank = ", my_rank, "#####", " current time = ", datetime.now().time())

                self.create_circuit()

                self.init_state(self.p_phys[j])

                self.encoding_circuit()

                self.syndrome_extraction(p_err=self.p_phys[j])

                # This function also applies the recovery/correction operation.
                self.syndrome_decoding()

		            # Determine whether a logical error has occured using an
		            # additional error-free decoding step
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
            # Total samples = rounds * size of p_phys
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
