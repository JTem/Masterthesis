
# Master Thesis: Exploiting Redundancies with Predictive Differential Kinematics in Dual-Quaternion-Space for Configuration-optimized and Singularity-robust Trajectory Tracking

This master thesis was submitted to the Institute of Mechanism Theory, Machine Dynamics and Robotics at RWTH Aachen University. It explores advanced kinematic techniques in the context of robotic trajectory tracking, emphasizing on configuration optimization and robustness against singularities. The proposed method leverages the usage of unit dual quaternions for increased efficiency and singularity robustness.

## Directory Structure

- `DualQuaternionQuinticBlends/`: Provided code package of interpolation in dual-quaternion space with quintic blends.
- `Resources/`: Supplementary resources that support the thesis, which include images and videos.
- `Simulation/`: Simulation setup to test and validate the proposed methods.
- `styles/`: Styling files for the Jupyter notebooks when exported as HTML or PDF.

### Notebooks

- `0.0_cover_and_content.ipynb`: The digital cover page and table of contents.
- `1.0_introduction.ipynb`: Introduction to the thesis and its scope.
- `2.0_state_of_the_art.ipynb`: A literature review of the application of dual quaternions in kinematics and interpolation, common motion controller methods and their applications.
- `3.x_*.ipynb`: Series of notebooks detailing the theoretical foundation and proposed models.
- `4.0_application_robot_welding.ipynb`: Application of the proposed methods to a robot welding case study.
- `5.0_results.ipynb`: Presentation and discussion of the simulation and experimental results.
- `6.0_conclusion.ipynb`: Conclusion, including a summary of findings and suggestions for future work.

### Additional Files

- `helper.py`: Python script containing helper functions used across notebooks.
- `README.md`: Guide for navigating and understanding the repository.

## Installation

The Thesis requires several packages to run the Notebooks.

[neura_robotics_toolbox](https://github.com/JTem/neura_robotics_toolbox) extends the [Robotics Toolbox Python](https://github.com/petercorke/robotics-toolbox-python). Installation may require some special care.
This will also install other requirements such as Swift and Spatialmath-Python.

```bash
pip install neura-roboticstoolbox
```

The quternion and dual quaternion maths can be found in the for the thesis developed package [neura_dual_quaternions](https://github.com/JTem/neura_dual_quaternions)

```bash
pip install neura-dual-quaternions
```

For interactive plots, the packages jupyter-matplotlib and jupyter-widgets are needed

```bash
pip install ipympl
```

```bash
pip install ipywidgets
```

Other site packages, which are needed are:

```bash
pip install numpy
```

```bash
pip install matplotlib
```

## Usage

Guidelines on how to run the notebooks and reproduce the results. Provide commands for starting JupyterLab and executing the notebooks.

## Contact

For questions, please contact Jens Temminghoff at [jens.temminghoff@neura-robotics.com].

## Thesis Details

- Author: Jens Temminghoff, B.Sc.
- Supervisors: Dr.-Ing. Markus Schmitz, Prof. Dr.-Ing. Marcel Hupytch, Dr.-Ing. Jörn Malzahn
- Examiners: Univ.-Prof. Dr.-Ing. Dr. h.c. Burkhard Corves, Prof. Dr.-Ing. Mathias Hüsing
- Submission Date: 16th January 2024
