# Furby
Mock FRB generator (requires python2.7 or above)

### gen_furby.py

Code that generates Mock FRBs in a given range of DM, width, and SNR. Has a few options for frequency structure. Intrachannel dm smearing can be toggled on or off.

Enabling the dmsmear makes the computation memory and compute intensive. Having high bandwidth and/or high DM might give "Memory Error" or take too long to generate a furby, when run with -dmsmear True

If required, modify the resolution in time and frequency defined in the first few lines in gen_furby.py, to make the code run faster but at the cost of accuracy of temporal/freq structure.

### Furby_reader.py

A Class to read the dada files output by the furby generator

### check_furby.py

A quick and handy tool to read and visualize a furby file

### check_type.py

A helper code to enable parsing configuration files

### parse_cfg.py

A helper code to help with parsing configuration files

### params.cfg

A config file containing the parameters of a receiving system which will be matched in the generated furby