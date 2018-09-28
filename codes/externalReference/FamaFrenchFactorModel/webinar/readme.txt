Using MATLAB to Develop Asset Pricing Models
--------------------------------------------

The .pdf and MATLAB .mat and .m files in this folder are the materials used in the MathWorks Webinar "Using MATLAB to Develop Asset Pricing Models." The MATLAB files can be used to demonstrate and implement the Fama & French three-factor model. 

IMPORTANT: The MATLAB code will only work with MATLAB versions 2006b and above.

Requirements
------------
MATLAB
Financial Toolbox
Statistics Toolbox (comes bundled with Financial Toolbox)

Files
-----
FFwebinar.m - main script
FFestimateCAPM.m - a script to compute CAPM model parameters
FFestimateFF.m - a script to compute Fama & French model parameters
FFUniverse.mat - a "mat" file that contains all the data for the analysis

Instructions
------------
To work through the example, open the script FFwebinar.m the editor in MATLAB. It is set up in "cell" mode with 13 "steps." You might notice that the main script looks somewhat different from the script in the recorded webinar - the only change is the addition of lots of comments to help you work through the example.

Run each step in sequence since some steps depend upon successful execution of prior steps. To run each step in "cell-mode," click on a step in the MATLAB editor and it will be highlighted (you do not need to highlight the code by dragging over it, you just need to click on a section). The highlighted code is called a cell. There are several ways to execute the code in a cell. You can find the "Cell" dropdown menu on the top toolbar and click on the "Evaluate Current Cell" command. You can use the keyboard shortcut <ctrl>-<enter>, you can right-click on the cell and select "Evaluate Current Cell," or, finally, you can click on the "Evaluate Cell" icon on the third toolbar.

If you work step-by-step through FFwebinar, you can see how to build and test a factor model. In addition, additional files are created that are needed for subsequent steps in FFwebinar.

Files Created by the Scripts
----------------------------
CAPMResults.mat - rolling estimates for the CAPM
FFResults.mat - rolling estimates for the Fama & French three-factor model
FFUniverseCAPM.mat - the universe of stocks and CAPM factor returns
FFUniverseFF.mat - the universe of stocks and Fama & French factor returns
XResults.mat - rolling estimates for the Fama & French model with Alpha = 0
