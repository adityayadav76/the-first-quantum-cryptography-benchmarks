# the-first-quantum-cryptography-benchmarks
The First Quantum Cryptography Benchmarks ( RSA, ECDSA, ... )

## About

TFQCB is a Quantum Cryptography Benchmark created by [Automatski](https://automatski.com).
It is a part of a larger suite of benchmarks used by Automatski to benchmark its Quantum Computers, which have not been released publicly till now.
These Benchmarks are used to verify whether Automatski' Quantum Computers work perfectly after every engineering cycle, changes and upgrades.

### Intellectual Property
All Rights Reserved By Automatski for its parts of the code.
Rights to the original work is retained by the original vendors/authors who created the code.

### Warning!
These are classified projects.
And these benchmarks could have been delay-released by "upto" 5 years.

## Installation

TFQCB requires Python v3.11+ to run.

Install the dependencies and devDependencies and start the server.

```sh
pip install requests numpy qiskit==1.4.2 qiskit-aer qiskit-algorithms
```

Run the Benchmarks
```sh
cd Benchmarks\RSA\Shors
python RSA-Shors-Benchmark-Main.py
```
