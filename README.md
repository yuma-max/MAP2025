# MAP2025

### 5/27/2025 - 7/31/2025

***Note***
All jupyter notebook just run the code using the function defined in the python files. There is no need to look at them.

### Purpose of this research
To trace all the star particles in the surviiving and disrupted galaxies and find the relationships between the final distributions of star distirbution and progenitor's infall properties.

###Python Files###
disrupted_traceCopy_verJuly14(1).py:
  The main code to analyze the simulation.
  What they do:
    1. Trace the halo id of disrupted and surviving galaxies
    2. Trace the star particles after disruption
    3. Use the main function which is defined in the star_trace_Yuma(2).py and make the csv file which stores the information of the halo particles' position, velocity
    4. Load the csv file and make the plot of the xy and xz distribution of the star particles and combine them and make mp4 file
    5. Calculate the COM of the main halo and surviving halo

making_COM_cord.py:
  The code to trace the movement of the COM of the main halo. This is used to calculate the relative coordinates. Note that the main halo is moving and since stars are moving around the main halo, then it is important to use relative coordinates.

R_dist_paper1.py:
  The code to analyze radial distribution.
  What plots they make:
  1. Cumulative mass distribution
  2. Differential mass distribution
  3. Fraction of the mass from a progenitor to the total mass in each enclosed sphere

R_dist_paper2.py:
  The code to analyze radial distribution.
  What plots they make:
  1. differential mass fraction in each shell

analysis1_COM_motion.py:
  The code to visualise the movement of the COM of all the satellites and progenitors in xy + xz plane or in the radial distance v.s. time plot.

Analysis2_infall_property_vs_R.py:
  The code to make a scatter plot of final distribution vs infall time/mass.


  


    
    
