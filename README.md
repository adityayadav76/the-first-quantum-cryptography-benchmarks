# The First Quantum Cryptography Benchmarks
The First Quantum Cryptography Benchmarks ( RSA, ECDSA, ... )

![](https://automatski.com/wp-content/uploads/2025/05/Automatski-New-Logo.svg)

## About

TFQCB is a Quantum Cryptography Benchmark created by [Automatski](https://automatski.com). It is part of a larger suite of benchmarks used by Automatski to evaluate its quantum computers, which have not yet been released publicly. These benchmarks are used to validate correct operation after each engineering cycle, including changes and upgrades.

### Intellectual Property
All rights are reserved by Automatski for Automatski-authored components of this codebase. Rights to third-party or upstream components remain with their respective original authors and licensors.

### Notice!
This repository contains material that may have been held back from release for up to five years.

## Installation

TFQCB requires Python v3.11+ to run.
Install dependencies:

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

**Whats Happening Here?**

The benchmark program generates a quantum circuit, transpiles it, and submits it to Automatski’s quantum computers. The system performs circuit optimization, control-pulse generation/optimization, execution planning, and runs the circuit in a single execution. Results are returned to the Python program for post-processing, including continued-fraction calculations and extraction of the period and factors from raw device output.

End-to-end execution can take hours. Large-scale demonstrations of Shor’s algorithm are rare, and practical runtime is often understated in non-technical discussions. In particular, popular narratives can create the impression that RSA keys would be broken “in milliseconds,” which is not realistic for production-scale implementations.

Factoring RSA-2048 would likely require sustained effort even in optimistic scenarios. For example, circuit generation and transpilation alone may take months and require substantial memory resources; post-processing (e.g., continued fractions) is also non-trivial. In addition, success may require multiple runs depending on noise, sampling requirements, and algorithmic parameters. As a result, such an effort should be treated as a multi-month to multi-year undertaking.


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

## FAQs

### Is This A World Record?

Yes! Each one of them (every instance) is an incremental world record. As per [Wikipedia](https://en.wikipedia.org/wiki/Integer_factorization_records#Records_for_efforts_by_quantum_computers) - The largest number reliably factored by Shor's algorithm is 21 which was factored in 2012. 15 had previously been factored by several labs.

### Has Quantum Advantage Been Achieved?

"Quantum advantage is the point where a quantum computer demonstrably solves a specific problem significantly faster, more accurately, or more cost-effectively than the best available classical computer, making previously impossible or impractical tasks feasible." The largest RSA Number factored classically using the general number field sieve is [RSA-250](https://en.wikipedia.org/wiki/RSA_Factoring_Challenge) which is a 829 Bit number. Thats way larger than any of these benchmarks as of now so we have to say that Quantum Advantage has not been achieved yet. Or in other words Quantum Computers have not surpassed Classical Supercomputers yet.

### Does This Imply That Fault Tolerant Quantum Computers Already Exist And That Automatski' Quantum Computers Are The Worlds First Production Grade FTQC Quantum Computers?

The answer to this question is constrained by the current Geo-Political Situation. You are free to infer whatever you feel like. ** No Comments **
