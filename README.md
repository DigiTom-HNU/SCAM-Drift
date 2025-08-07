# SCAM-Drift
SCAM-Drift for drift correction in List-FISH. Simulation results show that the correction precision is 65 nm for synthetic data with complete trajectories, and 216 nm for synthetic data with 50% trajectory loss. The test results from biological experiments show that SCAM-Drift achieved 23% improvement in cell recognition via corrected spots (quantified by ClusterMap). We believe that SCAM-Drift helps to promote the development and maturation of List-FISH for intact brain tissue sections.


# Brief introduction
For simulation data, download DriftEstimation_Simulite directly and run main.m

For the experimental data, first, download DriftEstimation_FISH and run calDrift.m. Then use SAITS-PSO imputation. Finally, use ReconstructData.m to correct the data. Before using SAITS-PSO, users need to run the command "Conda env create -f environment.yml" through the Anaconda command line (CondaShell/AnacondaPrompt) to create the Conda virtual environment required by the project.
