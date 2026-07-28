# TEBD-TNMC
This is the ITensor-based implementation of tensor-network Monte Calro (TNMC) approach based on time-evolving block decimation (TEBD) proposed in arXiv:2608.XXXX.

# Requirements
The codes heavily depends on the C++ version of ITensor library (https://github.com/ITensor/ITensor) that requries C++17. Consequently, a compiler supporting C++17 is required for compiling the codes. The codes are developed with ITensor v3.2.0.

For the input/output of simulations, the codes use JSON for Modern C++ (https://github.com/nlohmann/json). Since this is header-only library, please put the header file in your include path.

To run the Python scripts for analyzing the obtained samples, following non-standard libralies
- numpy (https://pypi.org/project/numpy/)
- msgpack (https://pypi.org/project/msgpack/)

are requried.

# Programs
The codes contain three main programs: /tebd_projector/Heisenebrg_chain/tevol.cc, /tebd_projector/kicked_ising_chain/tevol.cc, and /random_projector/tevol.cc. For TNMC with TEBD projectors, the programs for the Henseiberg and the kicked-Ising chains are available.
 For TNMC with random projectors, only the program for the kicked-Ising chain is provided.

 # Complie
 For compiling the main programs, please put Makefile from /tutorial/project_template/ in ITensor repository, and modify it to fit your enviornment. The vaiable APP in Makefile should be set to "tevol". The variable HEADERS should be set to blank (or appropriate header files).

 # How to run
 Please run tevol in a directory that contiains setting.json. The template setting files are available in the directories containing the main programs. Sampled quantities are accumulated in an output file sample_(Random seed).mpac.

 For analyzing the obtained samples, please use the Python script statistics_Heisenberg.py or statistics_kicked.py. Please run the script from a directory containing the output file.
