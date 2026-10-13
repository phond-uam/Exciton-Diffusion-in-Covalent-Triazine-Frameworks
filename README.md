# Exciton-Diffusion-in-Covalent-Triazine-Frameworks

Analysis code for 'Exciton Diffusion in Covalent Triazine Frameworks and its Role in Photocatalytic Redox Transformations'.

trpl_main.py, trpl_functions.py: transient photoluminescence microscopy (TPLM) analysis, giving the exciton diffusivity, lifetime and diffusion length (Figure 3 of the main text).
pow_dep_main.py, pow_dep_functions.py: excitation power dependent photoluminescence analysis (Figure S25 of the Supporting Information).

Requirements
python3
numpy
scipy
matplotlib
pandas
Usage

Download the datasets from [DOI link to be added] and unpack them to a folder named data at the same level as the code.

In trpl_main.py and pow_dep_main.py, set the folder of the functions file, the data folder and the folder where the results are saved. In trpl_main.py, also set arraySelection to the indices of the scans to analyse (the indices are printed when the data is loaded). Then run the scripts cell by cell in an IDE, or from the command window:
python trpl_main.py
python pow_dep_main.py
