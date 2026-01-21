import requests
import qiskit
from qiskit.result import Result
#from qiskit.result.counts import Counts
import numpy as np
import uuid
import datetime

def _get_qindex(circ, name, index):
    """
    Find the qubit index.

    Args:
        circ: The Qiskit QuantumCircuit in question
        name: The name of the quantum register
        index: The qubit's relative index inside the register

    Returns:
        The qubit's absolute index if all registers are concatenated.
    """
    ret = 0
    for reg in circ.qregs:
        if name != reg.name:
            ret += reg.size
        else:
            return ret + index
    return ret + index
    
class AutomatskiKomencoQiskit:
    
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.gateMap = {}
        
        self.gateMap["cu1"]="cp"
        self.gateMap["cnot"]="cx"
        self.gateMap["phase"] = "p"
        self.gateMap["cphase"] = "cp"
        self.gateMap["ccphase"] = "ccp"
        self.gateMap["mcphase"] = "mcp"
        self.gateMap["toffoli"] = "ccx"
        self.gateMap["fredkin"] = "cswap"
        self.gateMap["cxp"] = "cnotp"
        self.gateMap["ccxp"] = "ccnotp"
        
    def run(self, circuit, repetitions=1000, topK=20, silent=False):
        self.silent = silent
        tstart = datetime.datetime.now()
        
        body = self.serialize_circuit(circuit, repetitions, topK)
        response = requests.post(f'http://{self.host}:{self.port}/api/komenco', json=body, timeout=None)
        response.close() #bug fix - too many connections open
        struct = response.json()
        
        tend = datetime.datetime.now()
        execution_time = (tend - tstart).microseconds
        if not self.silent: print(f"Time Taken {(tend - tstart)}")
        
        cannotProceed = None
        try:
            if struct["error"] :
                print(struct["error"])
                cannotProceed = True
                raise Exception(struct["error"])
        except Exception as e:
            if cannotProceed:
                raise e            
            pass 
        
        
        
        return self.deserialize_result(struct, repetitions, execution_time)

    def serialize_circuit(self, circuit, repetitions, topK):
        # Extract the number of qubits
        num_qubits = circuit.num_qubits
        
        operations = []
        measurements = []

        for instr, qargs, cargs in circuit.data:
            gate = instr.name.lower() #the server uses lowercase gate names 
            
            #we need to map some gate naming conventions to make sure there are no errors
            if gate in self.gateMap:
                gate = self.gateMap[gate]
                
            params = [param for param in instr.params]
            #print([q for q in qargs])
            qubits = []
            # Get qbit arguments
            # ref: myqlm converters [interop]
            for qarg in qargs:
                qubits.append(
                    _get_qindex(circuit, qarg._register.name, qarg._index))
                    
            if gate == 'measure':
                for q in qubits:
                    measurements.append(q)
            elif gate in ['barrier']:
                continue 
            elif gate in ['delay','initialize','reset','store']:
                raise(Exception(f"gate or operation: '{gate}' is not supported by Komenco yet"))                
            else:
                operations.append({"gate": gate, "params": params, "qubits": qubits})
        
        if not self.silent: print("Executing Quantum Circuit With...")
        if not self.silent: print(f"{num_qubits} Qubits And ...")
        if not self.silent: print(f"{len(operations)} Gates")
        
        if len(measurements) == 0:
            raise(Exception("There are no measurements done at the end of the circuit."))        
                
        return { "num_qubits": num_qubits, "operations": operations, "measurements": measurements, "shots": repetitions, "topK": topK}    

    @staticmethod
    def deserialize_result(response_data, repetitions, execution_time):
        measurementsStrings = response_data['measurements']
        # Build counts and probabilities
        counts = {}
        probabilities = {}
        for key_original, value in measurementsStrings.items():
            #combo = np.array( [int(char) for char in key] , dtype=np.int32)
            count = round(value*repetitions)
            if count == 0:
                continue            
            #key = hex(int(key_original, 2)) # key_original[::-1]
            key = key_original 
            counts[key] = count
            probabilities[key] = value
        
        job_result = [
            {
                "data": {
                    "counts": counts,
                    "probabilities": probabilities,
                    # Qiskit/experiments relies on this being present in this location in the
                    # ExperimentData class.
                    "metadata": {},
                },
                "shots": repetitions,
                "header": {},
                "success": True,
            }
        ]
        
        
        return Result.from_dict(
            {
                "results": job_result,
                "job_id": str(uuid.uuid4()),
                "backend_name": "Automatski Yocto",
                "backend_version": "V1.0b",
                "qobj_id": str(uuid.uuid4()),
                "success": True,
                "time_taken": execution_time,
            }
        ) 