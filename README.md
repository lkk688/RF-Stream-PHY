# RF-Stream-PHY
**RF-Stream-PHY** is an open research and development framework for building modern wireless communication physical layers from scratch using software-defined radios and modern Python-based ML tools such as **NumPy**, **SciPy**, **PyTorch**, and related open-source packages.
The project is designed for researchers, students, and engineers who want to prototype, analyze, and improve communication PHY systems in a flexible software environment, while also connecting those designs to **real hardware experiments**. Instead of treating the physical layer as a fixed black box, RF-Stream-PHY treats it as a programmable, learnable, and data-driven system.

## Vision
RF-Stream-PHY aims to make PHY-layer communication research more open, reproducible, and extensible by combining:
- **Classical communication system design**  
  such as synchronization, channel estimation, equalization, modulation, coding, and packet detection
- **Modern scientific computing in Python**  
  using tools like NumPy, SciPy, matplotlib, and Jupyter for fast iteration and visualization
- **Deep learning and differentiable modeling**  
  using PyTorch to develop neural receivers, learned demappers, signal detectors, and hardware-adaptive models
- **Hardware-in-the-loop experimentation**  
  using SDR platforms to bridge the gap between simulation and real radio deployment

## What This Project Covers
RF-Stream-PHY focuses on end-to-end PHY development, including:
- waveform generation and packet construction
- synchronization and frame detection
- OFDM and single-carrier receiver pipelines
- channel estimation and equalization
- modulation and demodulation
- soft information generation
- dataset collection from real radios
- self-supervised and weakly supervised labeling
- neural front-end and receiver model development
- benchmarking in both simulation and real hardware environments
The long-term goal is to support both **traditional communication pipelines** and **learning-based PHY architectures** within one unified framework.
## Design Philosophy
This project is built around several core ideas:
### 1. PHY should be programmable
Modern communication research benefits from software-first design. By building PHY blocks in Python and PyTorch, we can iterate faster, inspect intermediate signals more easily, and connect classical algorithms with machine learning methods.
### 2. Real hardware matters
Simulation is valuable, but real-world RF behavior often differs from idealized channel models. RF-Stream-PHY is designed to support hardware-in-the-loop workflows so that models and algorithms can be evaluated under realistic conditions.
### 3. Data collection should be part of the PHY workflow
A modern PHY system should not only transmit and receive packets, but also generate useful research datasets. Logging, labeling, metadata collection, and hard-negative mining are treated as first-class capabilities.
### 4. Learning should augment, not replace, the PHY
The project supports both classical DSP pipelines and neural components. This allows direct comparison, hybrid designs, and gradual integration of learned modules where they add practical value.
## Subprojects
### AutoDataPHY
**AutoDataPHY** is one of the key subprojects within RF-Stream-PHY.
AutoDataPHY focuses on **automated dataset construction for communication PHY research**. It is designed to collect, annotate, and organize real-radio captures with minimal human effort. The goal is to create a scalable data engine for training and evaluating deep learning models on real communication signals.
AutoDataPHY includes ideas such as:
- automated packet capture and logging
- PHY-derived labels such as CRC success/failure
- metadata collection for receiver diagnostics
- hard-negative mining
- structured dataset export for offline training
- iterative retraining loops from newly collected data
In short, **RF-Stream-PHY** is the broader project, and **AutoDataPHY** is the data-centric subproject that supports learning-based PHY development.
## Why RF-Stream-PHY
Many existing communication codebases fall into one of two categories:
- highly specialized DSP implementations that are difficult to extend
- ML-only research code that ignores real radio constraints
RF-Stream-PHY is intended to bridge these worlds. It provides a place to explore questions such as:
- How should we design a communication PHY from scratch in modern Python?
- When do learned receiver components outperform classical methods?
- How can we collect useful labeled RF data directly from a running receiver?
- How can we continuously improve a real-radio PHY through iterative retraining?
## Intended Use Cases
RF-Stream-PHY is suitable for:
- communication systems research
- SDR-based prototyping
- deep learning for wireless PHY
- dataset generation for RF ML
- classroom and lab instruction
- rapid experimentation with new receiver ideas
- reproducible hardware-in-the-loop evaluation
## Planned Technical Directions
The project is intended to grow over time toward support for:
- OFDM PHY implementations
- packet detection and neural gating
- learned soft demappers
- differentiable communication blocks
- self-supervised receiver training
- hardware-aware model adaptation
- support for multiple SDR platforms
- integration with simulation and autonomous sensing workloads

## Status

RF-Stream-PHY is an active research-oriented project under development. The repository will continue to expand with:

* reusable PHY modules
* experiment scripts
* dataset tools
* training pipelines
* hardware integration examples
* documentation for reproducible research workflows

## Contributing

Contributions, discussions, and collaboration are welcome. This project is intended to grow into an open platform for communication PHY research, real-radio experimentation, and learning-based wireless system design.

## Citation

If you use this project in research, please cite the repository and any related papers when available.
