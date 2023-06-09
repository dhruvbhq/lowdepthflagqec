import qiskit
import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, execute, IBMQ
from qiskit.quantum_info import partial_trace, state_fidelity
import enum
from mpi4py import MPI

aer_sim = Aer.get_backend('aer_simulator')
statevector_sim = Aer.get_backend('statevector_simulator')

from qec_flag_base import qec_flag_base, syn_ex_status, error_spec

#######################################################################################

# For syndrome with flag, the order is (syndrome_bit, flag_bit)

#######################################################################################

# Lookup table for protocol without flag (the usual weight 1 corrections). This
# is used later as a sub-table in flag LUTs.
five_qubit_code_no_flag_LUT = {
    # usual weight-1 corrections, assuming no faults

    # X error on qubit 1
    '[0 0 0 1]': 'XIIII',
    # X error on qubit 2
    '[1 0 0 0]': 'IXIII',
    # X error on qubit 3
    '[1 1 0 0]': 'IIXII',
    # X error on qubit 4
    '[0 1 1 0]': 'IIIXI',
    # X error on qubit 5
    '[0 0 1 1]': 'IIIIX',
    # Z error on qubit 1
    '[1 0 1 0]': 'ZIIII',
    # Z error on qubit 2
    '[0 1 0 1]': 'IZIII',
    # Z error on qubit 3
    '[0 0 1 0]': 'IIZII',
    # Z error on qubit 4
    '[1 0 0 1]': 'IIIZI',
    # Z error on qubit 5
    '[0 1 0 0]': 'IIIIZ',
    # Y error on qubit 1
    '[1 0 1 1]': 'YIIII',
    # Y error on qubit 2
    '[1 1 0 1]': 'IYIII',
    # Y error on qubit 3
    '[1 1 1 0]': 'IIYII',
    # Y error on qubit 4
    '[1 1 1 1]': 'IIIYI',
    # Y error on qubit 5
    '[0 1 1 1]': 'IIIIY'
    }

#######################################################################################

# Lookup table with possibly high-weight corrections
five_qubit_code_flag_high_wt_LUT = {
        # Flag raised during 1st generator (XZZXI) measurement
        '[[0 1] [None None] [None None] [None None]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[0 1 0 0]': 'IIZXI', 
            # 1st bad gate (CNOT) failed with XZ
            '[1 1 0 0]': 'IXZXI',
            # 1st bad gate (CNOT) failed with YZ
            '[1 0 0 1]': 'IYZXI',
            # 1st bad gate (CNOT) failed with ZZ
            '[0 0 0 1]': 'IZZXI',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 1 1 0]': 'IIIXI',
            # 2nd bad gate (CNOT) failed with XZ
            '[1 0 1 0]': 'IIXXI',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 0 0 0]': 'IIYXI'
            },
        # Syndrome and Flag raised during 1st generator (XZZXI) measurement
        # This can happen due to Y (~X.Z) error on ancilla, of which Z will propagate
        '[[1 1] [None None] [None None] [None None]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[0 1 0 0]': 'IIZXI', 
            # 1st bad gate (CNOT) failed with XZ
            '[1 1 0 0]': 'IXZXI',
            # 1st bad gate (CNOT) failed with YZ
            '[1 0 0 1]': 'IYZXI',
            # 1st bad gate (CNOT) failed with ZZ
            '[0 0 0 1]': 'IZZXI',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 1 1 0]': 'IIIXI',
            # 2nd bad gate (CNOT) failed with XZ
            '[1 0 1 0]': 'IIXXI',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 0 0 0]': 'IIYXI'
            },
        # Syndrome measured as 1 and Flag not raised during 1st generator
        # (XZZXI) measurement
        '[[1 0] [None None] [None None] [None None]]':
        # usual weight-1 corrections, assuming no faults
        five_qubit_code_no_flag_LUT,

        # Flag not raised during 1st generator (XZZXI) measurement, syndrome
        # during 1st generator measurement is 0, but flag raised during 2nd
        # generator (IXZZX) measurement
        '[[0 0] [0 1] [None None] [None None]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[1 0 1 0]': 'IIIZX',
            # 1st bad gate (CNOT) failed with XZ
            '[0 1 1 0]': 'IIXZX',
            # 1st bad gate (CNOT) failed with YZ
            '[0 1 0 0]': 'IIYZX',
            # 1st bad gate (CNOT) failed with ZZ
            '[1 0 0 0]': 'IIZZX',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 0 1 1]': 'IIIIX',
            # 2nd bad gate (CNOT) failed with XZ
            '[0 1 0 1]': 'IIIXX',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 1 0 0]': 'IIIYX'
            },
        # Flag not raised during 1st generator (XZZXI) measurement, syndrome
        # during 1st generator measurement is 1, and flag raised during 2nd
        # generator (IXZZX) measurement
        # This can happen due to Y (~X.Z) error on ancilla, of which Z will propagate
        '[[0 0] [1 1] [None None] [None None]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[1 0 1 0]': 'IIIZX',
            # 1st bad gate (CNOT) failed with XZ
            '[0 1 1 0]': 'IIXZX',
            # 1st bad gate (CNOT) failed with YZ
            '[0 1 0 0]': 'IIYZX',
            # 1st bad gate (CNOT) failed with ZZ
            '[1 0 0 0]': 'IIZZX',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 0 1 1]': 'IIIIX',
            # 2nd bad gate (CNOT) failed with XZ
            '[0 1 0 1]': 'IIIXX',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 1 0 0]': 'IIIYX'
            },
        # Syndrome measured as 0 and Flag not raised during 1st generator
        # (XZZXI) measurement, but syndrome measured as 1 and flag not raised
        # during 2nd generator (IXZZX) measurement
        '[[0 0] [1 0] [None None] [None None]]':
        # usual weight-1 corrections, assuming no faults
        five_qubit_code_no_flag_LUT,

        # Flag not raised during 1st generator (XZZXI) measurement, flag not
        # raised during 2nd generator (IXZZX) measurement, syndromes during 1st
        # and 2nd generator measurements are 0, but flag raised during 3rd
        # generator (XIXZZ) measurement
        '[[0 0] [0 0] [0 1] [None None]]':
        {
            # 1st bad gate (XNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[1 1 0 1]': 'IIIZZ',
            # 1st bad gate (XNOT) failed with XZ
            '[0 0 0 1]': 'IIXZZ',
            # 1st bad gate (XNOT) failed with YZ
            '[0 0 1 1]': 'IIYZZ',
            # 1st bad gate (XNOT) failed with ZZ
            '[1 1 1 1]': 'IIZZZ',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 1 0 0]': 'IIIIZ',
            # 2nd bad gate (CNOT) failed with XZ
            '[0 0 1 0]': 'IIIXZ',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 0 1 1]': 'IIIYZ'
            },
        # Flag not raised during 1st generator (XZZXI) measurement, flag not
        # raised during 2nd generator (IXZZX) measurement, syndrome during 1st
        # generator measurement is 0, syndrome during 2nd generator measurement
        # is 1, and flag raised during 3rd generator (XIXZZ) measurement
        # This can happen due to Y (~X.Z) error on ancilla, of which Z will propagate
        '[[0 0] [0 0] [1 1] [None None]]':
        {
            # 1st bad gate (XNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[1 1 0 1]': 'IIIZZ',
            # 1st bad gate (XNOT) failed with XZ
            '[0 0 0 1]': 'IIXZZ',
            # 1st bad gate (XNOT) failed with YZ
            '[0 0 1 1]': 'IIYZZ',
            # 1st bad gate (XNOT) failed with ZZ
            '[1 1 1 1]': 'IIZZZ',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 1 0 0]': 'IIIIZ',
            # 2nd bad gate (CNOT) failed with XZ
            '[0 0 1 0]': 'IIIXZ',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 0 1 1]': 'IIIYZ'
            },
        # Syndrome measured as 0 and Flag not raised during 1st generator
        # (XZZXI) and 2nd generator (IXZZX) measurement, but syndrome measured
        # as 1 and flag not raised during 3rd generator (XIXZZ) measurement
        '[[0 0] [0 0] [1 0] [None None]]':
        # usual weight-1 corrections, assuming no faults
        five_qubit_code_no_flag_LUT,

        # Flag not raised during 1st generator (XZZXI) measurement, flag not
        # raised during 2nd generator (IXZZX) measurement, flag not raised
        # during 3rd generator (XIXZZ) measurement, syndromes during 1st, 2nd
        # and 3rd generator measurements are 0, but flag raised during 4th
        # generator (ZXIXZ) measurement
        '[[0 0] [0 0] [0 0] [0 1]]':
        {
            # 1st bad gate (XNOT) failed with IZ, or 2nd bad gate (XNOT) failed with XZ
            '[0 0 1 0]': 'IIIXZ',
            # 1st bad gate (XNOT) failed with XZ
            '[1 0 1 0]': 'IXIXZ',
            # 1st bad gate (XNOT) failed with YZ
            '[1 1 1 1]': 'IYIXZ',
            # 1st bad gate (XNOT) failed with ZZ
            '[0 1 1 1]': 'IZIXZ',
            # 2nd bad gate (XNOT) failed with IZ
            '[0 1 0 0]': 'IIIIZ',
            # 2nd bad gate (XNOT) failed with YZ
            '[1 0 1 1]': 'IIIYZ',
            # 2nd bad gate (XNOT) failed with ZZ
            '[1 1 0 1]': 'IIIZZ'
            },
        # Flag not raised during 1st generator (XZZXI) measurement, flag not
        # raised during 2nd generator (IXZZX) measurement, flag not raised
        # during 3rd generator (XIXZZ) measurement, syndromes during 1st and
        # 2nd generator measurements are 0, syndrome during 3rd generator
        # measurement is 1, and flag raised during 4th generator (ZXIXZ)
        # measurement
        # This can happen due to Y (~X.Z) error on ancilla, of which Z will propagate
        '[[0 0] [0 0] [0 0] [1 1]]':
        {
            # 1st bad gate (XNOT) failed with IZ, or 2nd bad gate (XNOT) failed with XZ
            '[0 0 1 0]': 'IIIXZ',
            # 1st bad gate (XNOT) failed with XZ
            '[1 0 1 0]': 'IXIXZ',
            # 1st bad gate (XNOT) failed with YZ
            '[1 1 1 1]': 'IYIXZ',
            # 1st bad gate (XNOT) failed with ZZ
            '[0 1 1 1]': 'IZIXZ',
            # 2nd bad gate (XNOT) failed with IZ
            '[0 1 0 0]': 'IIIIZ',
            # 2nd bad gate (XNOT) failed with YZ
            '[1 0 1 1]': 'IIIYZ',
            # 2nd bad gate (XNOT) failed with ZZ
            '[1 1 0 1]': 'IIIZZ'
            },
        # Syndrome measured as 0 and Flag not raised during 1st generator
        # (XZZXI), 2nd generator (IXZZX), and 3rd generator (XIXZZ) measurement, but syndrome measured
        # as 1 and flag not raised during 4th generator (ZXIXZ) measurement
        '[[0 0] [0 0] [0 0] [1 0]]':
        # usual weight-1 corrections, assuming no faults
        five_qubit_code_no_flag_LUT
        }

#######################################################################################

# Lookup table with minimal weight corrections.
# There are multiple minimal weight equivalents possible for a given Pauli
# string. One is chosen.
five_qubit_code_flag_min_wt_LUT = {
        # Flag raised during 1st generator (XZZXI) measurement
        '[[0 1] [None None] [None None] [None None]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[0 1 0 0]': 'IIZXI', 
            # 1st bad gate (CNOT) failed with XZ
            '[1 1 0 0]': 'XYIII',
            # 1st bad gate (CNOT) failed with YZ
            '[1 0 0 1]': 'XXIII',
            # 1st bad gate (CNOT) failed with ZZ
            '[0 0 0 1]': 'XIIII',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 1 1 0]': 'IIIXI',
            # 2nd bad gate (CNOT) failed with XZ
            '[1 0 1 0]': 'IIXXI',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 0 0 0]': 'IIYXI'
            },
        # Syndrome and Flag raised during 1st generator (XZZXI) measurement
        # This can happen due to Y (~X.Z) error on ancilla, of which Z will propagate
        '[[1 1] [None None] [None None] [None None]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[0 1 0 0]': 'IIZXI', 
            # 1st bad gate (CNOT) failed with XZ
            '[1 1 0 0]': 'XYIII',
            # 1st bad gate (CNOT) failed with YZ
            '[1 0 0 1]': 'XXIII',
            # 1st bad gate (CNOT) failed with ZZ
            '[0 0 0 1]': 'XIIII',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 1 1 0]': 'IIIXI',
            # 2nd bad gate (CNOT) failed with XZ
            '[1 0 1 0]': 'IIXXI',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 0 0 0]': 'IIYXI'
            },
        # Syndrome measured as 1 and Flag not raised during 1st generator
        # (XZZXI) measurement
        '[[1 0] [None None] [None None] [None None]]':
        # usual weight-1 corrections, assuming no faults
        five_qubit_code_no_flag_LUT,

        # Flag not raised during 1st generator (XZZXI) measurement, syndrome
        # during 1st generator measurement is 0, but flag raised during 2nd
        # generator (IXZZX) measurement
        '[[0 0] [0 1] [None None] [None None]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[1 0 1 0]': 'IIIZX',
            # 1st bad gate (CNOT) failed with XZ
            '[0 1 1 0]': 'XIIIY',
            # 1st bad gate (CNOT) failed with YZ
            '[0 1 0 0]': 'IXXII',
            # 1st bad gate (CNOT) failed with ZZ
            '[1 0 0 0]': 'IXIII',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 0 1 1]': 'IIIIX',
            # 2nd bad gate (CNOT) failed with XZ
            '[0 1 0 1]': 'IIIXX',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 1 0 0]': 'IIIYX'
            },
        # Flag not raised during 1st generator (XZZXI) measurement, syndrome
        # during 1st generator measurement is 1, and flag raised during 2nd
        # generator (IXZZX) measurement
        # This can happen due to Y (~X.Z) error on ancilla, of which Z will propagate
        '[[0 0] [1 1] [None None] [None None]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[1 0 1 0]': 'IIIZX',
            # 1st bad gate (CNOT) failed with XZ
            '[0 1 1 0]': 'XIIIY',
            # 1st bad gate (CNOT) failed with YZ
            '[0 1 0 0]': 'IXXII',
            # 1st bad gate (CNOT) failed with ZZ
            '[1 0 0 0]': 'IXIII',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 0 1 1]': 'IIIIX',
            # 2nd bad gate (CNOT) failed with XZ
            '[0 1 0 1]': 'IIIXX',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 1 0 0]': 'IIIYX'
            },
        # Syndrome measured as 0 and Flag not raised during 1st generator
        # (XZZXI) measurement, but syndrome measured as 1 and flag not raised
        # during 2nd generator (IXZZX) measurement
        '[[0 0] [1 0] [None None] [None None]]':
        # usual weight-1 corrections, assuming no faults
        five_qubit_code_no_flag_LUT,

        # Flag not raised during 1st generator (XZZXI) measurement, flag not
        # raised during 2nd generator (IXZZX) measurement, syndromes during 1st
        # and 2nd generator measurements are 0, but flag raised during 3rd
        # generator (XIXZZ) measurement
        '[[0 0] [0 0] [0 1] [None None]]':
        {
            # 1st bad gate (XNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[1 1 0 1]': 'IIIZZ',
            # 1st bad gate (XNOT) failed with XZ
            '[0 0 0 1]': 'XIIII',
            # 1st bad gate (XNOT) failed with YZ
            '[0 0 1 1]': 'XIZII',
            # 1st bad gate (XNOT) failed with ZZ
            '[1 1 1 1]': 'IXIIY',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 1 0 0]': 'IIIIZ',
            # 2nd bad gate (CNOT) failed with XZ
            '[0 0 1 0]': 'IIIXZ',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 0 1 1]': 'IIIYZ'
            },
        # Flag not raised during 1st generator (XZZXI) measurement, flag not
        # raised during 2nd generator (IXZZX) measurement, syndrome during 1st
        # generator measurement is 0, syndrome during 2nd generator measurement
        # is 1, and flag raised during 3rd generator (XIXZZ) measurement
        # This can happen due to Y (~X.Z) error on ancilla, of which Z will propagate
        '[[0 0] [0 0] [1 1] [None None]]':
        {
            # 1st bad gate (XNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[1 1 0 1]': 'IIIZZ',
            # 1st bad gate (XNOT) failed with XZ
            '[0 0 0 1]': 'XIIII',
            # 1st bad gate (XNOT) failed with YZ
            '[0 0 1 1]': 'XIZII',
            # 1st bad gate (XNOT) failed with ZZ
            '[1 1 1 1]': 'IXIIY',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 1 0 0]': 'IIIIZ',
            # 2nd bad gate (CNOT) failed with XZ
            '[0 0 1 0]': 'IIIXZ',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 0 1 1]': 'IIIYZ'
            },
        # Syndrome measured as 0 and Flag not raised during 1st generator
        # (XZZXI) and 2nd generator (IXZZX) measurement, but syndrome measured
        # as 1 and flag not raised during 3rd generator (XIXZZ) measurement
        '[[0 0] [0 0] [1 0] [None None]]':
        # usual weight-1 corrections, assuming no faults
        five_qubit_code_no_flag_LUT,

        # Flag not raised during 1st generator (XZZXI) measurement, flag not
        # raised during 2nd generator (IXZZX) measurement, flag not raised
        # during 3rd generator (XIXZZ) measurement, syndromes during 1st, 2nd
        # and 3rd generator measurements are 0, but flag raised during 4th
        # generator (ZXIXZ) measurement
        '[[0 0] [0 0] [0 0] [0 1]]':
        {
            # 1st bad gate (XNOT) failed with IZ, or 2nd bad gate (XNOT) failed with XZ
            '[0 0 1 0]': 'IIIXZ',
            # 1st bad gate (XNOT) failed with XZ
            '[1 0 1 0]': 'ZIIII',
            # 1st bad gate (XNOT) failed with YZ
            '[1 1 1 1]': 'ZZIII',
            # 1st bad gate (XNOT) failed with ZZ
            '[0 1 1 1]': 'ZYIII',
            # 2nd bad gate (XNOT) failed with IZ
            '[0 1 0 0]': 'IIIIZ',
            # 2nd bad gate (XNOT) failed with YZ
            '[1 0 1 1]': 'IIIYZ',
            # 2nd bad gate (XNOT) failed with ZZ
            '[1 1 0 1]': 'IIIZZ'
            },
        # Flag not raised during 1st generator (XZZXI) measurement, flag not
        # raised during 2nd generator (IXZZX) measurement, flag not raised
        # during 3rd generator (XIXZZ) measurement, syndromes during 1st and
        # 2nd generator measurements are 0, syndrome during 3rd generator
        # measurement is 1, and flag raised during 4th generator (ZXIXZ)
        # measurement
        # This can happen due to Y (~X.Z) error on ancilla, of which Z will propagate
        '[[0 0] [0 0] [0 0] [1 1]]':
        {
            # 1st bad gate (XNOT) failed with IZ, or 2nd bad gate (XNOT) failed with XZ
            '[0 0 1 0]': 'IIIXZ',
            # 1st bad gate (XNOT) failed with XZ
            '[1 0 1 0]': 'ZIIII',
            # 1st bad gate (XNOT) failed with YZ
            '[1 1 1 1]': 'ZZIII',
            # 1st bad gate (XNOT) failed with ZZ
            '[0 1 1 1]': 'ZYIII',
            # 2nd bad gate (XNOT) failed with IZ
            '[0 1 0 0]': 'IIIIZ',
            # 2nd bad gate (XNOT) failed with YZ
            '[1 0 1 1]': 'IIIYZ',
            # 2nd bad gate (XNOT) failed with ZZ
            '[1 1 0 1]': 'IIIZZ'
            },
        # Syndrome measured as 0 and Flag not raised during 1st generator
        # (XZZXI), 2nd generator (IXZZX), and 3rd generator (XIXZZ) measurement, but syndrome measured
        # as 1 and flag not raised during 4th generator (ZXIXZ) measurement
        '[[0 0] [0 0] [0 0] [1 0]]':
        # usual weight-1 corrections, assuming no faults
        five_qubit_code_no_flag_LUT
        }

#######################################################################################
# Lookup table with possibly high-weight corrections, based on the exact
# circuit used in Chao's code
five_qubit_code_flag_chao_ckt_my_high_wt_LUT = {
        # Flag raised during 1st generator (XZZXI) measurement
        '[[0 1] [None None] [None None] [None None]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[0 1 0 0]': 'IIZXI', 
            # 1st bad gate (CNOT) failed with XZ
            '[1 1 0 0]': 'IXZXI',
            # 1st bad gate (CNOT) failed with YZ
            '[1 0 0 1]': 'IYZXI',
            # 1st bad gate (CNOT) failed with ZZ
            '[0 0 0 1]': 'IZZXI',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 1 1 0]': 'IIIXI',
            # 2nd bad gate (CNOT) failed with XZ
            '[1 0 1 0]': 'IIXXI',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 0 0 0]': 'IIYXI'
            },
        # Syndrome and Flag raised during 1st generator (XZZXI) measurement
        # This can happen due to Y (~X.Z) error on ancilla, of which Z will propagate
        '[[1 1] [None None] [None None] [None None]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[0 1 0 0]': 'IIZXI', 
            # 1st bad gate (CNOT) failed with XZ
            '[1 1 0 0]': 'IXZXI',
            # 1st bad gate (CNOT) failed with YZ
            '[1 0 0 1]': 'IYZXI',
            # 1st bad gate (CNOT) failed with ZZ
            '[0 0 0 1]': 'IZZXI',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 1 1 0]': 'IIIXI',
            # 2nd bad gate (CNOT) failed with XZ
            '[1 0 1 0]': 'IIXXI',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 0 0 0]': 'IIYXI'
            },
        # Syndrome measured as 1 and Flag not raised during 1st generator
        # (XZZXI) measurement
        '[[1 0] [None None] [None None] [None None]]':
        # usual weight-1 corrections, assuming no faults
        five_qubit_code_no_flag_LUT,

        # Flag not raised during 1st generator (XZZXI) measurement, syndrome
        # during 1st generator measurement is 0, but flag raised during 2nd
        # generator (IXZZX) measurement
        '[[0 0] [0 1] [None None] [None None]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[1 0 1 0]': 'IIIZX',
            # 1st bad gate (CNOT) failed with XZ
            '[0 1 1 0]': 'IIXZX',
            # 1st bad gate (CNOT) failed with YZ
            '[0 1 0 0]': 'IIYZX',
            # 1st bad gate (CNOT) failed with ZZ
            '[1 0 0 0]': 'IIZZX',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 0 1 1]': 'IIIIX',
            # 2nd bad gate (CNOT) failed with XZ
            '[0 1 0 1]': 'IIIXX',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 1 0 0]': 'IIIYX'
            },
        # Flag not raised during 1st generator (XZZXI) measurement, syndrome
        # during 1st generator measurement is 1, and flag raised during 2nd
        # generator (IXZZX) measurement
        # This can happen due to Y (~X.Z) error on ancilla, of which Z will propagate
        '[[0 0] [1 1] [None None] [None None]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[1 0 1 0]': 'IIIZX',
            # 1st bad gate (CNOT) failed with XZ
            '[0 1 1 0]': 'IIXZX',
            # 1st bad gate (CNOT) failed with YZ
            '[0 1 0 0]': 'IIYZX',
            # 1st bad gate (CNOT) failed with ZZ
            '[1 0 0 0]': 'IIZZX',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 0 1 1]': 'IIIIX',
            # 2nd bad gate (CNOT) failed with XZ
            '[0 1 0 1]': 'IIIXX',
            # 2nd bad gate (CNOT) failed with YZ
            '[1 1 0 0]': 'IIIYX'
            },
        # Syndrome measured as 0 and Flag not raised during 1st generator
        # (XZZXI) measurement, but syndrome measured as 1 and flag not raised
        # during 2nd generator (IXZZX) measurement
        '[[0 0] [1 0] [None None] [None None]]':
        # usual weight-1 corrections, assuming no faults
        five_qubit_code_no_flag_LUT,

        # Flag not raised during 1st generator (XZZXI) measurement, flag not
        # raised during 2nd generator (IXZZX) measurement, syndromes during 1st
        # and 2nd generator measurements are 0, but flag raised during 3rd
        # generator (XIXZZ) measurement
        '[[0 0] [0 0] [0 1] [None None]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[0 1 0 1]': 'XIIIZ',
            # 1st bad gate (CNOT) failed with XZ
            '[0 0 1 1]': 'XIIXZ',
            # 1st bad gate (CNOT) failed with YZ
            '[1 0 1 0]': 'XIIYZ',
            # 1st bad gate (CNOT) failed with ZZ
            '[1 1 0 0]': 'XIIZZ',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 0 0 1]': 'XIIII',
            # 2nd bad gate (CNOT) failed with XZ
            '[0 0 1 0]': 'XIIIX',
            # 2nd bad gate (CNOT) failed with YZ
            '[0 1 1 0]': 'XIIIY'
            },
        # Flag not raised during 1st generator (XZZXI) measurement, flag not
        # raised during 2nd generator (IXZZX) measurement, syndrome during 1st
        # generator measurement is 0, syndrome during 2nd generator measurement
        # is 1, and flag raised during 3rd generator (XIXZZ) measurement
        # This can happen due to Y (~X.Z) error on ancilla, of which Z will propagate
        '[[0 0] [0 0] [1 1] [None None]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with ZZ
            '[0 1 0 1]': 'XIIIZ',
            # 1st bad gate (CNOT) failed with XZ
            '[0 0 1 1]': 'XIIXZ',
            # 1st bad gate (CNOT) failed with YZ
            '[1 0 1 0]': 'XIIYZ',
            # 1st bad gate (CNOT) failed with ZZ
            '[1 1 0 0]': 'XIIZZ',
            # 2nd bad gate (CNOT) failed with IZ
            '[0 0 0 1]': 'XIIII',
            # 2nd bad gate (CNOT) failed with XZ
            '[0 0 1 0]': 'XIIIX',
            # 2nd bad gate (CNOT) failed with YZ
            '[0 1 1 0]': 'XIIIY'
            },
        # Syndrome measured as 0 and Flag not raised during 1st generator
        # (XZZXI) and 2nd generator (IXZZX) measurement, but syndrome measured
        # as 1 and flag not raised during 3rd generator (XIXZZ) measurement
        '[[0 0] [0 0] [1 0] [None None]]':
        # usual weight-1 corrections, assuming no faults
        five_qubit_code_no_flag_LUT,

        # Flag not raised during 1st generator (XZZXI) measurement, flag not
        # raised during 2nd generator (IXZZX) measurement, flag not raised
        # during 3rd generator (XIXZZ) measurement, syndromes during 1st, 2nd
        # and 3rd generator measurements are 0, but flag raised during 4th
        # generator (ZXIXZ) measurement
        '[[0 0] [0 0] [0 0] [0 1]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with XZ
            '[0 0 1 0]': 'ZXIII',
            # 1st bad gate (CNOT) failed with XZ
            '[0 0 0 1]': 'ZXIIX',
            # 1st bad gate (CNOT) failed with YZ
            '[0 1 0 1]': 'ZXIIY',
            # 1st bad gate (CNOT) failed with ZZ
            '[0 1 1 0]': 'ZXIIZ',
            # 2nd bad gate (CNOT) failed with IZ
            '[1 0 0 0]': 'IXIII',
            # 2nd bad gate (CNOT) failed with XZ
            '[1 0 0 1]': 'XXIII',
            # 2nd bad gate (CNOT) failed with YZ
            '[0 0 1 1]': 'YXIII'
            },
        # Flag not raised during 1st generator (XZZXI) measurement, flag not
        # raised during 2nd generator (IXZZX) measurement, flag not raised
        # during 3rd generator (XIXZZ) measurement, syndromes during 1st and
        # 2nd generator measurements are 0, syndrome during 3rd generator
        # measurement is 1, and flag raised during 4th generator (ZXIXZ)
        # measurement
        # This can happen due to Y (~X.Z) error on ancilla, of which Z will propagate
        '[[0 0] [0 0] [0 0] [1 1]]':
        {
            # 1st bad gate (CNOT) failed with IZ, or 2nd bad gate (CNOT) failed with XZ
            '[0 0 1 0]': 'ZXIII',
            # 1st bad gate (CNOT) failed with XZ
            '[0 0 0 1]': 'ZXIIX',
            # 1st bad gate (CNOT) failed with YZ
            '[0 1 0 1]': 'ZXIIY',
            # 1st bad gate (CNOT) failed with ZZ
            '[0 1 1 0]': 'ZXIIZ',
            # 2nd bad gate (CNOT) failed with IZ
            '[1 0 0 0]': 'IXIII',
            # 2nd bad gate (CNOT) failed with XZ
            '[1 0 0 1]': 'XXIII',
            # 2nd bad gate (CNOT) failed with YZ
            '[0 0 1 1]': 'YXIII'
            },
        # Syndrome measured as 0 and Flag not raised during 1st generator
        # (XZZXI), 2nd generator (IXZZX), and 3rd generator (XIXZZ) measurement, but syndrome measured
        # as 1 and flag not raised during 4th generator (ZXIXZ) measurement
        '[[0 0] [0 0] [0 0] [1 0]]':
        # usual weight-1 corrections, assuming no faults
        five_qubit_code_no_flag_LUT
        }
#######################################################################################

class five_qubit_code_flag_protocol(qec_flag_base):
    def __init__(self,
                 num_data_qubits=5,
                 num_anc_qubits=1,
                 num_flag_qubits=1,
                 syndrome_lookup_table=five_qubit_code_flag_min_wt_LUT,
                 syndrome_lookup_table_no_flag=five_qubit_code_no_flag_LUT,
                 p_phys=np.array([10**(-4), 5*10**(-4), 10**(-3)]),
                 rounds=10**3,
                 seed_simulator=None,
                 seed_error_injection=None,
                 verbose=False,
                 debug=False,
                 barrier=True):

        self.syndrome_n_flag_1st_subround = None
        self.syndrome_2nd_subround = None

        super().__init__(num_data_qubits,
                num_anc_qubits,                
                num_flag_qubits,
                syndrome_lookup_table,
                syndrome_lookup_table_no_flag,
                p_phys,
                rounds,
                seed_simulator,
                seed_error_injection,
                verbose,
                debug,
                barrier)
    
    ########################################################################### 
    def init_state(self, p_err=0):
        # Preparation errors on ancilla and flag only
        for i in range(self.num_anc_qubits):
            # Error
            self.single_qubit_X_error(self.anc_qubits[i], self.error_scale_factor_prep*p_err)
        # Initialize flag qubits in |+> state
        for i in range(self.num_flag_qubits):
            # Error
            self.single_qubit_X_error(self.flag_qubits[i], self.error_scale_factor_prep*p_err)
            self.qec_flag_base_ckt.h(self.flag_qubits[i])

        # arb initial state - to be encoded (assuming noiseless/FT encoding)
        self.qec_flag_base_ckt.ry(np.pi/3,self.data_qubits[4])

        if(self.barrier):
            self.qec_flag_base_ckt.barrier()
    
    ########################################################################### 
    def encoding_circuit(self):
        # We are assuming that the encoding is noiseless/Fault tolerant
        self.qec_flag_base_ckt.h(self.data_qubits[0])
        self.qec_flag_base_ckt.z(self.data_qubits[0])
        
        self.qec_flag_base_ckt.cz(self.data_qubits[0],self.data_qubits[4])
        self.qec_flag_base_ckt.cx(self.data_qubits[0],self.data_qubits[4])

        self.qec_flag_base_ckt.cz(self.data_qubits[0],self.data_qubits[1])
        self.qec_flag_base_ckt.cz(self.data_qubits[0],self.data_qubits[3])
       
        self.qec_flag_base_ckt.h(self.data_qubits[1])
        
        self.qec_flag_base_ckt.cx(self.data_qubits[1],self.data_qubits[4])
        self.qec_flag_base_ckt.cz(self.data_qubits[1],self.data_qubits[2])
        self.qec_flag_base_ckt.cz(self.data_qubits[1],self.data_qubits[3])
        
        self.qec_flag_base_ckt.h(self.data_qubits[2])
        
        self.qec_flag_base_ckt.cx(self.data_qubits[2],self.data_qubits[4])
        self.qec_flag_base_ckt.cz(self.data_qubits[2],self.data_qubits[0])
        self.qec_flag_base_ckt.cz(self.data_qubits[2],self.data_qubits[1])
        
        self.qec_flag_base_ckt.h(self.data_qubits[3])
        self.qec_flag_base_ckt.z(self.data_qubits[3])

        self.qec_flag_base_ckt.cz(self.data_qubits[3],self.data_qubits[4])
        self.qec_flag_base_ckt.cx(self.data_qubits[3],self.data_qubits[4])
        
        self.qec_flag_base_ckt.cz(self.data_qubits[3],self.data_qubits[0])
        self.qec_flag_base_ckt.cz(self.data_qubits[3],self.data_qubits[2])

        if(self.barrier):
            self.qec_flag_base_ckt.barrier()
        return
    
    ########################################################################### 
    def measure_full_syndrome_without_flags(self, test_config:"error_spec"=None, p_err=0):
        """
        Helper method for syndrome_extraction. Measures all 4 stabilizer
        generators via circuits without flag qubits. This step might be needed
        many times in the protocol.
        """
        assert self.syndrome_ex_status == syn_ex_status.MEAS_GEN_WITHOUT_FLAG,\
            "Incorrect syndrome extraction status before measurement without flags."

        # Measure the 1st stabilizer generator (XZZXI) with a circuit without flag
        # Error: As of now, the locations in this function are is unreachable
        # by test_config. This only affects manual testing and not depol error.
        # if test_config is None, ie user overriding has to be absent. If
        # test_config is not None, override depol error; here the
        # implementation is no error is to be injected. This is added because,
        # during testing, I only want to add faults which lead to a flag being
        # raised, and disable the standard depolarizing error during this
        # testing.

        # XNOT
        self.xnot_subckt_err(self.data_qubits[0], self.anc_qubits[0], p_err, test_config, 100)
        # CNOT
        self.cnot_subckt_err(self.data_qubits[1], self.anc_qubits[0], p_err, test_config, 101)
        # CNOT
        self.cnot_subckt_err(self.data_qubits[2], self.anc_qubits[0], p_err, test_config, 102)
        # XNOT
        self.xnot_subckt_err(self.data_qubits[3], self.anc_qubits[0], p_err, test_config, 103)

        self.measure_ancilla_and_flag(with_flag=False, p_err=p_err)
        self.syndrome_2nd_subround = self.current_syndrome_n_flag
        # After measuring the ancilla, reset it to |0> for possible future use.
        self.reset_ancilla(p_err)
        if(self.barrier):
            self.qec_flag_base_ckt.barrier()
        
        # Measure the 2nd stabilizer generator (IXZZX) with a circuit without flag
        # XNOT
        self.xnot_subckt_err(self.data_qubits[1], self.anc_qubits[0], p_err, test_config, 104)
        # CNOT
        self.cnot_subckt_err(self.data_qubits[2], self.anc_qubits[0], p_err, test_config, 105)
        # CNOT
        self.cnot_subckt_err(self.data_qubits[3], self.anc_qubits[0], p_err, test_config, 106)
        # XNOT
        self.xnot_subckt_err(self.data_qubits[4], self.anc_qubits[0], p_err, test_config, 107)

        self.measure_ancilla_and_flag(with_flag=False, p_err=p_err)
        self.syndrome_2nd_subround = np.append(self.syndrome_2nd_subround, 
                                               self.current_syndrome_n_flag)
        # After measuring the ancilla, reset it to |0> for possible future use.
        self.reset_ancilla(p_err)
        if(self.barrier):
            self.qec_flag_base_ckt.barrier()

        # Measure the 3rd stabilizer generator (XIXZZ) with a circuit without flag
        # XNOT
        self.xnot_subckt_err(self.data_qubits[0], self.anc_qubits[0], p_err, test_config, 108)
        # XNOT
        self.xnot_subckt_err(self.data_qubits[2], self.anc_qubits[0], p_err, test_config, 109)
        # CNOT
        self.cnot_subckt_err(self.data_qubits[3], self.anc_qubits[0], p_err, test_config, 110)
        # CNOT
        self.cnot_subckt_err(self.data_qubits[4], self.anc_qubits[0], p_err, test_config, 111)

        self.measure_ancilla_and_flag(with_flag=False, p_err=p_err)
        self.syndrome_2nd_subround = np.append(self.syndrome_2nd_subround,
                                               self.current_syndrome_n_flag)
        # After measuring the ancilla, reset it to |0> for possible future use.
        self.reset_ancilla(p_err)
        if(self.barrier):
            self.qec_flag_base_ckt.barrier()

        # Measure the 4th stabilizer generator (XIXZZ) with a circuit without flag
        # CNOT
        self.cnot_subckt_err(self.data_qubits[0], self.anc_qubits[0], p_err, test_config, 112)
        # XNOT
        self.xnot_subckt_err(self.data_qubits[1], self.anc_qubits[0], p_err, test_config, 113)
        # XNOT
        self.xnot_subckt_err(self.data_qubits[3], self.anc_qubits[0], p_err, test_config, 114)
        # CNOT
        self.cnot_subckt_err(self.data_qubits[4], self.anc_qubits[0], p_err, test_config, 115)

        self.measure_ancilla_and_flag(with_flag=False, p_err=p_err)
        self.syndrome_2nd_subround = np.append(self.syndrome_2nd_subround,
                                               self.current_syndrome_n_flag)
        # After measuring the ancilla, reset it to |0> for possible future use.
        self.reset_ancilla(p_err)
        if(self.barrier):
            self.qec_flag_base_ckt.barrier()

        return

    ########################################################################### 
    def syndrome_extraction(self, test_config:"error_spec"=None, p_err=0):
        """
        The flag protocol for extracting syndrome as well as flag qubits.
        """

        # This is expected to be the place where the final syndrome will be decided.

        # Check syndrome extraction status, it should be IDLE.
        assert self.syndrome_ex_status == syn_ex_status.IDLE,\
            "Syndrome extraction status should be IDLE at the beginning."
        # Reset these so that final error-free decoding round finds these variables clean
        self.syndrome_n_flag_1st_subround = None
        self.syndrome_2nd_subround = None
        self.current_syndrome_n_flag = None

        # If syndrome extraction status is IDLE, set it to MEAS_GEN_WITH_FLAG
        self.syndrome_ex_status = syn_ex_status.MEAS_GEN_WITH_FLAG

        # Only for testing - not for actual simulation
        if((test_config is not None) and (test_config.inject_error) and (test_config.error_loc == 0)):
            if(self.barrier):
                self.qec_flag_base_ckt.barrier()
            self.two_qubit_pauli_error(test_config.pauli_idx1,
                                       test_config.pauli_idx2,
                                       test_config.qubit_idx1,
                                       test_config.qubit_idx2)
            if(self.barrier):
                self.qec_flag_base_ckt.barrier()

        # Measure the 1st stabilizer generator with a circuit with flag
        # XNOT
        self.xnot_subckt_err(self.data_qubits[0], self.anc_qubits[0], p_err, test_config, 1)
        # Flag CNOT
        self.cnot_subckt_err(self.flag_qubits[0], self.anc_qubits[0], p_err, test_config, 2)
        # CNOT
        self.cnot_subckt_err(self.data_qubits[1], self.anc_qubits[0], p_err, test_config, 3)
        # CNOT
        self.cnot_subckt_err(self.data_qubits[2], self.anc_qubits[0], p_err, test_config, 4)
        # Flag CNOT
        self.cnot_subckt_err(self.flag_qubits[0], self.anc_qubits[0], p_err, test_config, 5)
        # XNOT
        self.xnot_subckt_err(self.data_qubits[3], self.anc_qubits[0], p_err, test_config, 6)

        if(self.barrier):
            self.qec_flag_base_ckt.barrier()
        self.measure_ancilla_and_flag(with_flag=True, p_err=p_err)
        self.syndrome_n_flag_1st_subround = self.current_syndrome_n_flag
        if(self.barrier):
            self.qec_flag_base_ckt.barrier()
        # Whenever we are measuring both the flag and the ancilla, we reset the
        # ancilla to |0> and reinitialize the flag to |+> for possible future
        # use. (Note that measurement of flag is ultimately happening in the Z
        # basis, so it gets set to |0> or |1> after that).
        self.reset_ancilla(p_err)
        self.reset_flag(p_err)
        if(self.barrier):
            self.qec_flag_base_ckt.barrier()

        # update status for further decision-making
        # If flag is measured as 1 (i.e. |->), change status to DET_RAISED_FLAG
        # Else, if syndrome bit is nonzero, change status to DET_NONZERO_SYNDROME 
        # Else, if both flag and syndrome are 0, change status to
        # DET_UNRAISED_FLAG_AND_ZERO_SYNDROME
        self.update_syn_ex_status()

        # If status is DET_RAISED_FLAG or DET_NONZERO_SYNDROME, append Nones to
        # first subround syndome, change status to MEAS_GEN_WITHOUT_FLAG,
        # and measure all 4 syndrome bits with circuit without flags
        if((self.syndrome_ex_status == syn_ex_status.DET_RAISED_FLAG) or 
            (self.syndrome_ex_status == syn_ex_status.DET_NONZERO_SYNDROME)):
            self.syndrome_n_flag_1st_subround = np.append(self.syndrome_n_flag_1st_subround,
                np.array([[None,None],[None,None],[None,None]]), axis=0)
            self.syndrome_ex_status = syn_ex_status.MEAS_GEN_WITHOUT_FLAG
            self.measure_full_syndrome_without_flags(test_config, p_err)

            # Change status to IDLE and return from this function
            self.syndrome_ex_status = syn_ex_status.IDLE
            self.syndrome_n_flag_1st_subround = \
                np.array2string(self.syndrome_n_flag_1st_subround).replace('\n', '')
            self.syndrome_2nd_subround = np.array2string(self.syndrome_2nd_subround)
            return

        # Else, if status is DET_UNRAISED_FLAG_AND_ZERO_SYNDROME, change status
        # to MEAS_GEN_WITH_FLAG, reset ancilla and flag, and measure 2nd
        # stabilizer generator with a circuit with flag.
        elif(self.syndrome_ex_status == syn_ex_status.DET_UNRAISED_FLAG_AND_ZERO_SYNDROME):
            self.syndrome_ex_status = syn_ex_status.MEAS_GEN_WITH_FLAG
        else:
            assert False, "Invalid syndrome_ex_status"

        if(self.syndrome_ex_status == syn_ex_status.MEAS_GEN_WITH_FLAG):
            # Measure the 2nd stabilizer generator (IXZZX) with a circuit with flag
            # XNOT
            self.xnot_subckt_err(self.data_qubits[1], self.anc_qubits[0], p_err, test_config, 7)
            # Flag CNOT
            self.cnot_subckt_err(self.flag_qubits[0], self.anc_qubits[0], p_err, test_config, 8)
            # CNOT
            self.cnot_subckt_err(self.data_qubits[2], self.anc_qubits[0], p_err, test_config, 9)
            # CNOT
            self.cnot_subckt_err(self.data_qubits[3], self.anc_qubits[0], p_err, test_config, 10)
            # Flag CNOT
            self.cnot_subckt_err(self.flag_qubits[0], self.anc_qubits[0], p_err, test_config, 11)
            # XNOT
            self.xnot_subckt_err(self.data_qubits[4], self.anc_qubits[0], p_err, test_config, 12)

            if(self.barrier):
                self.qec_flag_base_ckt.barrier()
            self.measure_ancilla_and_flag(with_flag=True, p_err=p_err)
            self.syndrome_n_flag_1st_subround = np.append(self.syndrome_n_flag_1st_subround,
                                                          self.current_syndrome_n_flag,
                                                          axis=0)
            if(self.barrier):
                self.qec_flag_base_ckt.barrier()
            # Whenever we are measuring both the flag and the ancilla, we reset the
            # ancilla to |0> and reinitialize the flag to |+> for possible future
            # use. (Note that measurement of flag is ultimately happening in the Z
            # basis, so it gets set to |0> or |1> after that).
            self.reset_ancilla(p_err)
            self.reset_flag(p_err)
            if(self.barrier):
                self.qec_flag_base_ckt.barrier()

        # update status for further decision-making
        # If flag is measured as 1 (i.e. |->), change status to DET_RAISED_FLAG
        # Else, if syndrome bit is nonzero, change status to DET_NONZERO_SYNDROME 
        # Else, if both flag and syndrome are 0, change status to
        # DET_UNRAISED_FLAG_AND_ZERO_SYNDROME
        self.update_syn_ex_status()

        # If status is DET_RAISED_FLAG or DET_NONZERO_SYNDROME, append Nones to
        # first subround syndome, change status to MEAS_GEN_WITHOUT_FLAG, reset
        # ancilla, and measure all 4 syndrome bits with circuit without flags
        if((self.syndrome_ex_status == syn_ex_status.DET_RAISED_FLAG) or 
            (self.syndrome_ex_status == syn_ex_status.DET_NONZERO_SYNDROME)):
            self.syndrome_n_flag_1st_subround = np.append(self.syndrome_n_flag_1st_subround,
                    np.array([[None,None],[None,None]]), axis=0)
            self.syndrome_ex_status = syn_ex_status.MEAS_GEN_WITHOUT_FLAG
            self.measure_full_syndrome_without_flags(test_config, p_err)

            # Change status to IDLE and return from this function
            self.syndrome_ex_status = syn_ex_status.IDLE
            self.syndrome_n_flag_1st_subround = \
                np.array2string(self.syndrome_n_flag_1st_subround).replace('\n', '')
            self.syndrome_2nd_subround = np.array2string(self.syndrome_2nd_subround)
            return

        # Else, if status is DET_UNRAISED_FLAG_AND_ZERO_SYNDROME, change status
        # to MEAS_GEN_WITH_FLAG, reset ancilla and flag, and measure 3rd
        # stabilizer generator with a circuit with flag.
        elif(self.syndrome_ex_status == syn_ex_status.DET_UNRAISED_FLAG_AND_ZERO_SYNDROME):
            self.syndrome_ex_status = syn_ex_status.MEAS_GEN_WITH_FLAG
        else:
            assert False, "Invalid syndrome_ex_status"

        if(self.syndrome_ex_status == syn_ex_status.MEAS_GEN_WITH_FLAG):
            # Measure the 3rd stabilizer generator (XIXZZ) with a circuit with flag
            # XNOT
            self.xnot_subckt_err(self.data_qubits[0], self.anc_qubits[0], p_err, test_config, 13)
            # Flag CNOT
            self.cnot_subckt_err(self.flag_qubits[0], self.anc_qubits[0], p_err, test_config, 14)
            # XNOT
            self.xnot_subckt_err(self.data_qubits[2], self.anc_qubits[0], p_err, test_config, 15)
            # CNOT
            self.cnot_subckt_err(self.data_qubits[3], self.anc_qubits[0], p_err, test_config, 16)
            # Flag CNOT
            self.cnot_subckt_err(self.flag_qubits[0], self.anc_qubits[0], p_err, test_config, 17)
            # CNOT
            self.cnot_subckt_err(self.data_qubits[4], self.anc_qubits[0], p_err, test_config, 18)

            if(self.barrier):
                self.qec_flag_base_ckt.barrier()
            self.measure_ancilla_and_flag(with_flag=True, p_err=p_err)
            self.syndrome_n_flag_1st_subround = np.append(self.syndrome_n_flag_1st_subround,
                                                          self.current_syndrome_n_flag,
                                                          axis=0)
            if(self.barrier):
                self.qec_flag_base_ckt.barrier()
            # Whenever we are measuring both the flag and the ancilla, we reset the
            # ancilla to |0> and reinitialize the flag to |+> for possible future
            # use. (Note that measurement of flag is ultimately happening in the Z
            # basis, so it gets set to |0> or |1> after that).
            self.reset_ancilla(p_err)
            self.reset_flag(p_err)
            if(self.barrier):
                self.qec_flag_base_ckt.barrier()

        # update status for further decision-making
        # If flag is measured as 1 (i.e. |->), change status to DET_RAISED_FLAG
        # Else, if syndrome bit is nonzero, change status to DET_NONZERO_SYNDROME 
        # Else, if both flag and syndrome are 0, change status to
        # DET_UNRAISED_FLAG_AND_ZERO_SYNDROME
        self.update_syn_ex_status()

        # If status is DET_RAISED_FLAG or DET_NONZERO_SYNDROME, append Nones to
        # first subround syndome, change status to MEAS_GEN_WITHOUT_FLAG, reset
        # ancilla, and measure all 4 syndrome bits with circuit without flags
        if((self.syndrome_ex_status == syn_ex_status.DET_RAISED_FLAG) or 
            (self.syndrome_ex_status == syn_ex_status.DET_NONZERO_SYNDROME)):
            self.syndrome_n_flag_1st_subround = np.append(self.syndrome_n_flag_1st_subround,
                np.array([[None,None]]), axis=0)
            self.syndrome_ex_status = syn_ex_status.MEAS_GEN_WITHOUT_FLAG
            self.measure_full_syndrome_without_flags(test_config, p_err)

            # Change status to IDLE and return from this function
            self.syndrome_ex_status = syn_ex_status.IDLE
            self.syndrome_n_flag_1st_subround = \
                np.array2string(self.syndrome_n_flag_1st_subround).replace('\n', '')
            self.syndrome_2nd_subround = np.array2string(self.syndrome_2nd_subround)
            return

        # Else, if status is DET_UNRAISED_FLAG_AND_ZERO_SYNDROME, change status
        # to MEAS_GEN_WITH_FLAG, reset ancilla and flag, and measure 4th
        # stabilizer generator with a circuit with flag.
        elif(self.syndrome_ex_status == syn_ex_status.DET_UNRAISED_FLAG_AND_ZERO_SYNDROME):
            self.syndrome_ex_status = syn_ex_status.MEAS_GEN_WITH_FLAG
        else:
            assert False, "Invalid syndrome_ex_status"

        if(self.syndrome_ex_status == syn_ex_status.MEAS_GEN_WITH_FLAG):
            # Measure the 4th stabilizer generator (XIXZZ) with a circuit with flag
            # CNOT
            self.cnot_subckt_err(self.data_qubits[0], self.anc_qubits[0], p_err, test_config, 19)
            # Flag CNOT
            self.cnot_subckt_err(self.flag_qubits[0], self.anc_qubits[0], p_err, test_config, 20)
            # XNOT
            self.xnot_subckt_err(self.data_qubits[1], self.anc_qubits[0], p_err, test_config, 21)
            # XNOT
            self.xnot_subckt_err(self.data_qubits[3], self.anc_qubits[0], p_err, test_config, 22)
            # Flag CNOT
            self.cnot_subckt_err(self.flag_qubits[0], self.anc_qubits[0], p_err, test_config, 23)
            # CNOT
            self.cnot_subckt_err(self.data_qubits[4], self.anc_qubits[0], p_err, test_config, 24)

            if(self.barrier):
                self.qec_flag_base_ckt.barrier()
            self.measure_ancilla_and_flag(with_flag=True, p_err=p_err)
            self.syndrome_n_flag_1st_subround = np.append(self.syndrome_n_flag_1st_subround,
                                                          self.current_syndrome_n_flag,
                                                          axis=0)
            if(self.barrier):
                self.qec_flag_base_ckt.barrier()
            # Whenever we are measuring both the flag and the ancilla, we reset the
            # ancilla to |0> and reinitialize the flag to |+> for possible future
            # use. (Note that measurement of flag is ultimately happening in the Z
            # basis, so it gets set to |0> or |1> after that).
            self.reset_ancilla(p_err)
            self.reset_flag(p_err)
            if(self.barrier):
                self.qec_flag_base_ckt.barrier()

        # update status for further decision-making
        # If flag is measured as 1 (i.e. |->), change status to DET_RAISED_FLAG
        # Else, if syndrome bit is nonzero, change status to DET_NONZERO_SYNDROME 
        # Else, if both flag and syndrome are 0, change status to
        # DET_UNRAISED_FLAG_AND_ZERO_SYNDROME
        self.update_syn_ex_status()

        # If status is DET_RAISED_FLAG or DET_NONZERO_SYNDROME, change status
        # to MEAS_GEN_WITHOUT_FLAG, reset ancilla, and measure all 4 syndrome
        # bits with circuit without flags
        if((self.syndrome_ex_status == syn_ex_status.DET_RAISED_FLAG) or 
            (self.syndrome_ex_status == syn_ex_status.DET_NONZERO_SYNDROME)):
            self.syndrome_ex_status = syn_ex_status.MEAS_GEN_WITHOUT_FLAG
            self.measure_full_syndrome_without_flags(test_config, p_err)

            # Change status to IDLE and return from this function
            self.syndrome_ex_status = syn_ex_status.IDLE
            self.syndrome_n_flag_1st_subround = \
                np.array2string(self.syndrome_n_flag_1st_subround).replace('\n', '')
            self.syndrome_2nd_subround = np.array2string(self.syndrome_2nd_subround)
            return

        # Else, if status is DET_UNRAISED_FLAG_AND_ZERO_SYNDROME, there is
        # nothing to be done, except perhaps for some post-processing before
        # decoding.
        # Change status to IDLE and return from this function
        self.syndrome_ex_status = syn_ex_status.IDLE
        self.syndrome_n_flag_1st_subround = \
            np.array2string(self.syndrome_n_flag_1st_subround).replace('\n', '')
        # without final error-free decoding, the next block will never be executed
        if(self.syndrome_2nd_subround is not None):
            self.syndrome_2nd_subround = np.array2string(self.syndrome_2nd_subround)

        return

#############################################################
if __name__=="__main__":

    # To run for a larger number of samples, use MPI function
    ckt = five_qubit_code_flag_protocol(p_phys=[0.001,0.0012589254117941675,0.001584893192461114,0.001995262314968879,0.0025118864315095794,0.0031622776601683794,0.003981071705534973,0.005011872336272725,0.00630957344480193,0.007943282347242814,0.01], rounds=10**5)
    ckt.p_phys_sweep_simulation()
    ckt.logical_error_rate_reporting()
