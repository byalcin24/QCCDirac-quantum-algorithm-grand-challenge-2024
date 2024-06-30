import sys
from typing import Any
from time import time

sys.path.append("../")
from typing import Any
import numpy as np
from openfermion.transforms import jordan_wigner

from quri_parts.algo.ansatz import SymmetryPreservingReal
from quri_parts.algo.optimizer import SPSA, OptimizerStatus, Adam
from quri_parts.circuit import LinearMappedUnboundParametricQuantumCircuit
from quri_parts.circuit.parameter import Parameter
from quri_parts.core.estimator.gradient import parameter_shift_gradient_estimates
from quri_parts.core.measurement import bitwise_commuting_pauli_measurement, CachedMeasurementFactory
from quri_parts.core.sampling.shots_allocator import (
    create_equipartition_shots_allocator,
)
from quri_parts.core.state import ParametricCircuitQuantumState, ComputationalBasisState
from quri_parts.openfermion.operator import operator_from_openfermion_op
from utils.challenge_2024 import ChallengeSampling, problem_hamiltonian,ExceededError
from quri_parts.qiskit.circuit import circuit_from_qiskit

challenge_sampling = ChallengeSampling()


####################################
#The VQE template was obtained from the example_vqe.py file
#The idea for the ansatz was taken from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.105.032419
#The decomposition of the XXPlusYY gate and the ZZ gates were taken from the Qiskit Documentation https://github.com/Qiskit
import numpy as np
'''from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.mappers import JordanWignerMapper,BravyiKitaevMapper
from qiskit.circuit import QuantumCircuit, ParameterVector, QuantumRegister
from qiskit.circuit.library import SwapGate, XXPlusYYGate
from qiskit.quantum_info.operators import Operator'''



def eZZ(circ,i,j,t):
    circ.add_CNOT_gate(i,j)
    circ.add_ParametricRZ_gate(j,t)
    circ.add_CNOT_gate(i,j)

def fij(circ,i,j,t):
    #xxplusyy
    circ.add_RZ_gate(j,-np.pi/2)
    circ.add_SqrtX_gate(j)
    circ.add_RZ_gate(j,np.pi/2)
    circ.add_S_gate(i)
    circ.add_CNOT_gate(j,i)
    circ.add_ParametricRY_gate(i,t)
    circ.add_ParametricRY_gate(j,t)
    circ.add_CNOT_gate(j,i)
    circ.add_Sdag_gate(i)
    circ.add_RZ_gate(j,-np.pi/2)
    circ.add_SqrtXdag_gate(j)
    circ.add_RZ_gate(j,np.pi/2)
    #cp
    circ.add_ParametricRZ_gate(i,t)
    circ.add_CNOT_gate(i,j)
    circ.add_ParametricRZ_gate(j,{t:-1})
    circ.add_CNOT_gate(i,j)
    circ.add_ParametricRZ_gate(j,t)
    


def efSwap(circ,i,j,t):
    for k in range(j-1,i,-1):
        circ.add_CZ_gate(j,k)
    fij(circ,i,j,t)
    for k in range(i+1,j,1):
        circ.add_CZ_gate(j,k)

def hopping(circ,r):
    n = circ.qubit_count
    for i in range(n):
        nm = "t_h{r}{i}".format(r=r,i=i)
        t_s = circ.add_parameter(nm)
        efSwap(circ,i,(i+2)%n,t_s)

def interaction(circ,r):
    n = circ.qubit_count
    for i in range(0,n,2):
        nm = "t_i{r}{i}".format(r=r,i=i)
        t_s = circ.add_parameter(nm)
        eZZ(circ,i,i+1,t_s)

def start2(circ):
    n = circ.qubit_count
    li = [i for i in range(0,n//2,2)]+[n-1-i for i in range(0,n//2,2)]
    for i in li:
        circ.add_X_gate(i)

def hopint(circ,reps = 3):
    for r in range(reps):
        hopping(circ,r)
        interaction(circ,r)



def cost_fn(hamiltonian, parametric_state, param_values, estimator):
    estimate = estimator(hamiltonian, parametric_state, [param_values])
    return estimate[0].value.real


def vqe(hamiltonian, parametric_state, estimator, init_params, optimizer, maxiter = 100):
    opt_state = optimizer.get_init_state(init_params)
    energy_history = []

    def c_fn(param_values):
        return cost_fn(hamiltonian, parametric_state, param_values, estimator)

    def g_fn(param_values):
        grad = parameter_shift_gradient_estimates(
            hamiltonian, parametric_state, param_values, estimator
        )
        return np.asarray([i.real for i in grad.values])
    
    while True:
        try:
            opt_state = optimizer.step(opt_state, c_fn, g_fn)
            energy_history.append(opt_state.cost)

        except ExceededError as e:
            print(str(e))
            print(opt_state.cost)
            return opt_state, energy_history
        if opt_state.status == OptimizerStatus.FAILED:
            print("Optimizer failed")
            break
        if opt_state.status == OptimizerStatus.CONVERGED:
            print("Optimizer converged")
            break
    return opt_state, energy_history


'''
def iter_vqe(hamiltonian, circ, estimator, init_params, optimizer, n_qubits, ans_list = [],ans_lim = 6):
    opt_state = optimizer.get_init_state(init_params)
    energy_history = []
    parametric_state = ParametricCircuitQuantumState(n_qubits, circ)

    def c_fn(param_values):
        return cost_fn(hamiltonian, parametric_state, param_values, estimator)

    def g_fn(param_values):
        grad = parameter_shift_gradient_estimates(
            hamiltonian, parametric_state, param_values, estimator
        )
        return np.asarray([i.real for i in grad.values])
    iter = 0
    while True:
        try:
            print(iter)
            if iter < ans_lim:
                if iter%2 == 0:
                    div = 1
                else:
                    div = 2
                ans_list[iter%2](circ,iter//2+1)
                parametric_state = ParametricCircuitQuantumState(n_qubits, circ)
                #print("---------------------")
                #print(opt_state)
                #print(type(opt_state.params))
                #print("---------------------")
                new_params = np.append(opt_state.params,np.random.rand(n_qubits//div) * 2 * np.pi)
                opt_state = OptimizerStateSPSA(params = new_params,cost = opt_state.cost,status = opt_state.status, niter = opt_state.niter, funcalls = opt_state.funcalls, gradcalls = opt_state.gradcalls, rng = opt_state.rng)
            opt_state = optimizer.step(opt_state, c_fn, g_fn)
            energy_history.append(opt_state.cost)
            print(opt_state.cost)
            iter +=1
        except ExceededError as e:
            print(str(e))
            print(opt_state.cost)
            return opt_state, energy_history
        if opt_state.status == OptimizerStatus.FAILED:
            print("Optimizer failed")
            break
        if opt_state.status == OptimizerStatus.CONVERGED:
            print("Optimizer converged")
            break
    return opt_state, energy_history
'''
####################################





class RunAlgorithm:
    def __init__(self) -> None:
        challenge_sampling.reset()

    def result_for_evaluation(self, seed: int, hamiltonian_directory: str) -> tuple[Any, float]:
        energy_final = self.get_result(seed, hamiltonian_directory)
        total_shots = challenge_sampling.total_shots
        return energy_final, total_shots

    def get_result(self, seed: int, hamiltonian_directory: str) -> float:
        """
            param seed: the last letter in the Hamiltonian data file, taking one of the values 0,1,2,3,4
            param hamiltonian_directory: directory where hamiltonian data file exists
            return: calculated energy.
        """
        n_qubits = 28
        ham = problem_hamiltonian(n_qubits, seed, hamiltonian_directory)

        ####################################
        circ = LinearMappedUnboundParametricQuantumCircuit(n_qubits)
        start2(circ)
        hopint(circ,reps = 3)
        
        total_shots = 10**6
        jw_hamiltonian = jordan_wigner(ham)

        hamiltonian = operator_from_openfermion_op(jw_hamiltonian)

        shots_allocator = create_equipartition_shots_allocator()
        cached_measurement_factory = CachedMeasurementFactory(
            bitwise_commuting_pauli_measurement
        )

        #ans = circuit_from_qiskit(circ)

        parametric_state = ParametricCircuitQuantumState(n_qubits, circ)
        #print("Done with the hard part")

        sampling_estimator = (
            challenge_sampling.create_concurrent_parametric_sampling_estimator(
                total_shots, cached_measurement_factory, shots_allocator
            )
        )

        optimizer = SPSA(ftol=10e-5)

        init_param = np.random.rand(circ.parameter_count) * 2 * np.pi

        result, energy_history = vqe(
            hamiltonian,
            parametric_state,
            sampling_estimator,
            init_param,
            optimizer,
        )
        print(f"iteration used: {result.niter}")
        ####################################


        return min(energy_history)


if __name__ == "__main__":
    run_algorithm = RunAlgorithm()
    print(run_algorithm.get_result(seed=0, hamiltonian_directory="../hamiltonian"))
