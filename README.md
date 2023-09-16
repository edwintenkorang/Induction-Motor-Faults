# Motor Fault Detection using Machine Learning

## Project Members
- Edwin Tenkorang
- Elvis Twumasi (Senior Lecturer, K.N.U.S.T)

This repository is dedicated to the detection of two common motor faults: broken rotor bars and supply voltage imbalances in induction motors. Leveraging machine learning and signal processing techniques, we aim to fast fault detection and reduce unplanned downtime.

## Repository Structure

The project is organized into two main folders, each addressing a specific motor fault:

### 1. Broken Rotor Bars

Folder 1 focuses on detecting broken rotor bars, and it contains the following components:

- **Motor.mlx**: MATLAB code for simulating and analyzing motor faults, with a particular emphasis on broken rotor bars.
- **extraction.pyu.py**: Python script for feature extraction, including Principal Component Analysis (PCA) and deep learning techniques. Input features encompass three-phase stator and rotor currents, mechanical and load torque, time, rotor speed, and the number of poles.

### 2. Supply Voltage Imbalances

Folder 2 is dedicated to detecting supply voltage imbalances and includes MATLAB and Simulink files:

- **.slx and .mlx files**: MATLAB files that leverage Support Vector Machines (SVMs) in the classification learner. These files initialize fault parameters and process current waveforms using Discrete Wavelet Transform (DWT) at nine levels before feeding the resulting coefficients to the SVM for prediction.
- **preprocessor.py**: Python file to concatenate the different datasets for each fault into a single CSV file to feed into the classification learner.



3. Explore the broken rotor bars and supply voltage imbalances detection techniques to enhance motor reliability and performance.

## Conclusion

This project combines the expertise of Edwin Tenkorang and Elvis Twumasi to address common motor faults effectively. By detecting broken rotor bars and supply voltage imbalances, we contribute to the proactive maintenance and reliability of induction motors, ultimately reducing operational disruptions and associated costs.
