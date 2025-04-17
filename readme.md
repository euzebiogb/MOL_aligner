# User Guide: Flexible Molecular Aligner ([LS-align](https://academic.oup.com/bioinformatics/article/34/13/2209/4860363) based)

Please cite: 10.1093/bioinformatics/bty081 - LS-align: an atom-level, flexible ligand structural alignment algorithm for high-throughput virtual screening 
Zi Liu, Dong-Jun Yu, and Yang Zhang,
School of Computer Science and Engineering, Nanjing University of Science and Technology, Nanjing 210094, China
Department of Computational Medicine and Bioinformatics, University of Michigan, Ann Arbor, MI 48109-2218, USA


## 1. Introduction

This Python script (`aligner_flex.py`) performs 3D molecular alignment using a method inspired by LS-align, utilizing the PC-score for evaluating alignment quality. It can perform both rigid alignment and a comprehensive flexible alignment.

The flexible alignment mode generates an ensemble of conformers for the query molecule, including sampling of ring conformations and **all combinations** of pyramidal nitrogen inversions (excluding amides and aromatic nitrogens). It then aligns every successfully generated and optimized conformer against the reference molecule and selects the best alignment based purely on the highest PC-score, implementing a "brute force" search over the generated conformational space.

## 2. Dependencies

This script requires the following Python libraries:

* **RDKit:** For cheminformatics functionalities (molecule handling, conformer generation, etc.).
* **NumPy:** For numerical operations.
* **SciPy:** For optimization routines (specifically `linear_sum_assignment`).

You can typically install these using Conda (recommended for RDKit):

`conda install -c conda-forge rdkit numpy scipy`

Or, if you manage RDKit installation separately, you might use pip for NumPy and SciPy:

`pip install numpy scipy`

(Optional for Visualization):

py3Dmol: To automatically generate an HTML file visualizing the alignment.

pip install py3Dmol

## 3. Usage
The script is run from the command line.

Basic Syntax:
`python aligner_flex.py -f <reference_file> -i <query_file> -o <output_file> [options]`

Arguments:
-f, --reference <file>: Required. Path to the reference (template) molecule file. Supported formats: SDF, MOL, PDB.

-i, --query <file>: Required. Path to the query molecule file. Supported formats: SDF, MOL, PDB.

-o, --output <file>: Required. Path for the output SDF file, which will contain the best-aligned conformer of the query molecule.

--flexible: Optional. Activates the flexible alignment mode. If omitted, a rigid alignment of the input query conformer is performed.

--num_conformers <N>: Optional. (Default: 20) Specifies the number of initial conformers to generate using RDKit's ETKDG algorithm before combinatorial nitrogen inversion is applied in flexible mode. Increasing this number can improve sampling of the initial conformational space (especially for rings) but also increases computation time significantly, as each initial conformer seeds the generation of all nitrogen inversion combinations. This argument has no effect in rigid mode.

## 4. Alignment Modes
a) Rigid Alignment (Default)
If the --flexible flag is not provided, the script performs a rigid alignment. It takes the input conformer from the query file and aligns it to the reference molecule's conformer.

Example:

`python aligner_flex.py -f template.sdf -i query.mol -o aligned_rigid.sdf`

b) Flexible Alignment (--flexible)
If the --flexible flag is provided, the script performs a comprehensive flexible alignment:

Initial Conformer Generation: Generates an initial pool of conformers for the query molecule using RDKit ETKDG (number controlled by --num_conformers, multiplied internally for better sampling). Ring conformations are implicitly sampled here.

Combinatorial Nitrogen Inversion: Identifies all non-amide, non-aromatic, pyramidal nitrogen atoms. For each initial conformer, it generates new conformers representing all possible combinations of inversions at these nitrogen centers.

Optimization: Attempts to energy-minimize all generated conformers (initial + all inversion combinations) using the MMFF94s force field (or UFF as fallback). Conformers that fail optimization are discarded.

Alignment: Aligns every successfully optimized conformer to the reference molecule using the iterative LS-align/PC-score algorithm.

Selection: Selects the single query conformer that yielded the highest PC-score during alignment.

Warning: This mode can be computationally very expensive, especially for molecules with multiple invertible nitrogens, as the number of conformers grows exponentially (2^k factor, where k is the number of invertible nitrogens) before the alignment step.

Example:

`python aligner_flex.py -f template.pdb -i query.sdf -o aligned_flexible.sdf --flexible --num_conformers 30`

(This will generate 30 initial ETKDG conformers, then create all N-inversion combinations for each, optimize them, align all successful ones, and save the best.)

## 5. Input Files
Provide reference and query molecules in standard formats like SDF, MOL, or PDB.

The script expects 3D coordinates. It will attempt to add hydrogens and generate an initial 3D conformer if one is missing, but providing good starting 3D structures is recommended.

## 6. Output Files
a) Main Output SDF (-o <file>)
An SDF file containing the single best-aligned conformer of the query molecule (based on the highest PC-score achieved).

The molecule entry will have the following properties added:

LSalign_PC_Score: The PC-score of the alignment (higher is better).

LSalign_RMSD: The RMSD calculated between the aligned query atoms and the corresponding reference atoms based on the final alignment mapping.

## 7. Example Workflow
Prepare Inputs: Ensure you have reference.sdf and query.sdf with 3D coordinates.

Install Dependencies: Make sure RDKit, NumPy, and SciPy are installed.

Run Alignment (Flexible):

`python aligner_flex.py -f reference.sdf -i query.sdf -o query_aligned.sdf --flexible`

Check Output:

Examine the console output for progress, PC-scores, RMSD values, and total time.

Inspect the generated query_aligned.sdf file in a molecular viewer. Check the added properties.

I usually use [UCSF Chimera](https://www.cgl.ucsf.edu/chimera/) to see the results. 

Examine the console output for progress, PC-scores, RMSD values, and total time.

Inspect the generated query_aligned.sdf file in a molecular viewer. Check the added properties.


