# The First Quantum Cryptography Benchmarks
The First Quantum Cryptography Benchmarks ( RSA, ECDSA, ... )

![](https://automatski.com/wp-content/uploads/2025/05/Automatski-New-Logo.svg)

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

Install the dependencies.

```sh
pip install requests numpy qiskit==1.4.2 qiskit-aer qiskit-algorithms
```

Run the Benchmarks
```sh
cd Benchmarks\RSA\Shors
python RSA-Shors-Benchmark-Main.py
```

## Results

### N=67297 70 Logical Qubits 1.15m Gates

![](https://raw.githubusercontent.com/adityayadav76/the-first-quantum-cryptography-benchmarks/refs/heads/main/Runs/RSA/Shors/67297.png)

### N=1398488603 126 Logical Qubits 10m Gates

![](https://raw.githubusercontent.com/adityayadav76/the-first-quantum-cryptography-benchmarks/refs/heads/main/Runs/RSA/Shors/1398488603.png)



### N=85143280699972919909
**In Progress**

## The Shors Algorithm

The process of factorization involves finding the prime numbers that multiply together to create a given number. This seemingly simple task becomes exponentially more complex as the size of the number increases. Traditional computers struggle to factorize large numbers efficiently, leading to the need for more advanced algorithms.

Factorization plays a crucial role in various fields, such as cryptography and number theory. It is the foundation of many encryption algorithms, where the difficulty of factorizing large numbers ensures the security of sensitive information. Thus, developing efficient factorization methods has been a long-standing challenge in the field of computer science.

In the context of quantum computing, factorization holds special significance due to its connection with Shor’s algorithm.

Shor’s algorithm represents a groundbreaking accomplishment in the realm of factorization, as it leverages the power of quantum computing to factorize numbers significantly faster than classical computers.Classical computers rely on brute force methods to factorize large numbers, which are computationally expensive and time-consuming. In contrast, Shor’s algorithm exploits the unique properties of quantum systems, such as superposition and entanglement, to perform factorization more efficiently.

The fundamental idea behind Shor’s algorithm is to use quantum Fourier transform and quantum phase estimation to find the period of a modular function. By finding the period, Shor’s algorithm can deduce the factors of a composite number. This algorithm revolutionizes factorization by reducing the complexity from exponential to polynomial, making it much faster than classical methods. 

If large-scale, fault-tolerant quantum computers can run Shor’s algorithm, the impact on RSA and ECDSA would be catastrophic and immediate for today’s digital security.

RSA and ECDSA would be fundamentally broken
- RSA security relies on the difficulty of factoring large integers.
- ECDSA security relies on the difficulty of the elliptic-curve discrete logarithm problem.
- Shor’s algorithm solves both problems in polynomial time on a sufficiently powerful quantum computer.

➡️ This means private keys can be derived from public keys. And all the systems which use RSA and ECDSA cryptography to secure themselves can be immediately opened and compromised.

## The History

Shor’s algorithm was created in 1994 by mathematician Peter Shor while he was working at AT&T Bell Laboratories, at a time when quantum computing was largely a theoretical curiosity rather than a practical technology. Shor’s key insight was to show that a quantum computer could exploit quantum superposition and interference to solve certain mathematical problems exponentially faster than any known classical algorithm. By reducing the problem of integer factorization to the problem of period-finding and then solving that period-finding task efficiently using the quantum Fourier transform, Shor demonstrated for the first time that quantum mechanics could directly undermine the security assumptions of widely used cryptographic systems such as RSA.

The significance of Shor’s algorithm went far beyond factoring large numbers. It provided the first concrete example of a quantum algorithm with an exponential speedup over classical methods for a problem of real-world importance, transforming quantum computing from an abstract idea into a strategically urgent field of research. Almost overnight, governments, academia, and industry realized that scalable quantum computers could one day break public-key cryptography, reshaping global cybersecurity and national security priorities. Shor’s work effectively launched modern quantum algorithms research and remains the foundational reason why quantum computing is now viewed as a disruptive and potentially world-altering technology.
