# -*- coding: utf-8 -*-
"""
Script Python para alinhamento molecular flexível como ferramenta de linha de comando,
IMPLEMENTANDO A LÓGICA DO LS-align (Rigid-LS-align com PC-score e opção Flexível)
Baseado em: Hu et al., Bioinformatics, 34(13), 2018, 2209-2218
DOI: 10.1093/bioinformatics/bty081

Este script usa RDKit para manipulação básica de moléculas e implementa
o alinhamento iterativo e o PC-score do LS-align. A opção --flexible
permite gerar múltiplos confôrmeros da query e alinhar cada um. A opção
--hybrid ativa um método de geração de confôrmeros híbrido AI-Física.

Uso na linha de comando:
python aligner_script.py -f reference.sdf -i query.sdf -o aligned_query.sdf [--flexible | --hybrid] [--num_conformers N]

Argumentos:
  -f, --reference       Arquivo da molécula de referência/template (.sdf, .mol, .pdb)
  -i, --query           Arquivo da molécula query (.sdf, .mol, .pdb)
  -o, --output          Arquivo de saída para a molécula query alinhada (.sdf)
  --flexible            Ativa o modo de alinhamento flexível (gera confôrmeros RDKit para a query).
  --hybrid              Ativa o modo de alinhamento flexível com geração HÍBRIDA AI-Física de confôrmeros.
  --num_conformers N    Número de confôrmeros a gerar/testar no modo flexível/híbrido (default: 20).
  --ai_weight           (Modo Híbrido) Proporção de confôrmeros gerados por AI (0-1, default: 0.7).
  --mmff_iter           (Modo Híbrido) Iterações de otimização MMFF (default: 200).
  -n, --n_confs         (Ignorado) Número de conformações a salvar na saída.

Dependências principais:
- RDKit: (conda install -c conda-forge rdkit)
- NumPy: (pip install numpy)
- SciPy: (pip install scipy)
- PyTorch: (pip install torch) - Necessário para o modo --hybrid
"""

import os
import argparse
import copy # Para copiar objetos de molécula
import time # Para medir o tempo
import numpy as np
from scipy.optimize import linear_sum_assignment # Para encontrar o melhor mapeamento
from scipy.spatial.transform import Rotation # Para Kabsch
from scipy.spatial.distance import cdist # Para filtragem RMSD no modo híbrido

# --- RDKit Imports ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign, rdchem, Descriptors
    from rdkit.Chem.rdMolTransforms import TransformConformer
except ImportError:
    print("ERRO: RDKit não encontrado. Instale com 'conda install -c conda-forge rdkit'")
    exit(1)

# --- PyTorch Import (Opcional, apenas para modo híbrido) ---
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Não sai imediatamente, mas avisa se --hybrid for usado

# --- Constantes e Parâmetros do LS-align ---

# Pesos para PC-score (do paper, Eq. 3)
PC_SCORE_WEIGHT_DISTANCE = 0.45
PC_SCORE_WEIGHT_MASS = 0.10
PC_SCORE_WEIGHT_BOND = 0.45

# Escala para diferença de massa (Eq. 3 - valor não especificado, usando 1.0 como placeholder)
MASS_DIFF_SCALE_M0 = 1.0 # Unidade de massa atômica

# Pesos para tipos de ligação no BWJS (Eq. 4 - valores de Supp. Table S1 não disponíveis, usando placeholders)
BOND_WEIGHTS = {
    rdchem.BondType.SINGLE: 1.0,
    rdchem.BondType.DOUBLE: 1.5,
    rdchem.BondType.TRIPLE: 2.0,
    rdchem.BondType.AROMATIC: 1.2,
    rdchem.BondType.UNSPECIFIED: 1.0, # Placeholder
}
DEFAULT_BOND_WEIGHT = 1.0

# Máximo de iterações para o refinamento do alinhamento rígido
MAX_ALIGNMENT_ITERATIONS = 50

# Parâmetros para geração de confôrmeros no modo flexível RDKit
DEFAULT_NUM_CONFORMERS = 20 # Número de confôrmeros a gerar/testar
CONFORMER_RMSD_THRESHOLD = 0.5 # Para remover confôrmeros redundantes RDKit

# Parâmetros para geração híbrida
DEFAULT_AI_WEIGHT = 0.7
DEFAULT_MMFF_ITER = 200
DEFAULT_HYBRID_MAX_ENERGY = 100.0 # kcal/mol? Unidade depende do MMFF
DEFAULT_HYBRID_RMSD_THRESH = 1.0 # Angstrom

# --- AI Sampler Core (Inspired by Str2Str) ---
# NOTE: This requires PyTorch to be installed.
class ScoreBasedSampler:
    """
    Mockup of a score-based generative sampler for conformers.
    Inspired by diffusion models like Str2Str.
    Requires a pre-trained score network model.
    """
    def __init__(self, device=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ScoreBasedSampler. Please install it.")

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"INFO (AI Sampler): Using device: {self.device}")
        self.diffusion_steps = 100 # Number of noise/denoise steps
        self.noise_schedule = torch.linspace(1e-4, 2e-2, self.diffusion_steps).to(self.device)

        # !!! CRITICAL: Load a pre-trained score network model here !!!
        # self.model = load_pretrained_model('path/to/your/model.pt')
        self.model = self._create_mockup_score_network() # Using mockup for demonstration
        self.model.eval() # Set model to evaluation mode

    def _create_mockup_score_network(self):
        """
        !!! MOCKUP !!! Creates a placeholder score network.
        Replace this with loading your actual pre-trained model.
        The network should predict the score (gradient of log probability)
        of the noisy coordinates. Input/output dimensions depend on atom features used.
        This simple example assumes input is just 3D coords per atom.
        """
        # Example: Simple MLP. A real model would be much more complex (e.g., E(3) Equivariant GNN)
        # Input dimension should match the number of features per atom (here, just 3 for x,y,z)
        # Output dimension should also be 3 (score vector for x,y,z)
        num_atoms = 3 # Placeholder - this needs to be dynamic or handled differently
        # A real model would likely operate on the graph structure, not fixed size linear layers per atom.
        # This mockup is highly simplified and likely non-functional for real tasks.
        print("WARNING: Using a MOCKUP score network. Replace with a pre-trained model.")
        # This structure is likely incorrect for a real score model.
        # A real model would need to handle variable numbers of atoms and graph structure.
        # Example: A simple linear layer per atom (won't work well)
        # return torch.nn.Linear(3, 3).to(self.device)

        # A slightly more complex (but still likely wrong) sequential model:
        # This assumes the input tensor is flattened (num_atoms * 3) which loses spatial info.
        # A graph neural network (GNN) would be more appropriate.
        # Let's assume the model processes each atom's coordinate independently (incorrect but simple):
        return torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        ).to(self.device)


    def generate_conformers(self, mol, num_confs):
        """
        Generates conformers using score-based diffusion sampling.
        Starts from an initial RDKit embedding and refines using the score model.
        """
        if not TORCH_AVAILABLE:
            print("ERRO: PyTorch não está disponível, não é possível usar o ScoreBasedSampler.")
            return None

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            print("ERRO (AI Sampler): Molécula sem átomos.")
            return None

        # Get initial coordinates (e.g., from RDKit embedding)
        initial_coords_np = self._get_initial_coords(mol)
        if initial_coords_np is None:
             print("ERRO (AI Sampler): Falha ao obter coordenadas iniciais.")
             return None
        initial_coords = torch.tensor(initial_coords_np, dtype=torch.float32).to(self.device)

        conformers_list = []
        print(f"INFO (AI Sampler): Generating {num_confs} conformers using score-based sampling...")

        with torch.no_grad(): # Disable gradient calculation during inference
            for i in range(num_confs):
                # Start with initial coords for each sample (could also start from noise)
                coords = initial_coords.clone() # Shape: (num_atoms, 3)

                # Reverse diffusion process (sampling)
                for t in reversed(range(self.diffusion_steps)):
                    time_step = torch.tensor([t / self.diffusion_steps], device=self.device) # Example time encoding
                    noise_level = self.noise_schedule[t]
                    # Generate random noise (Gaussian)
                    z = torch.randn_like(coords)

                    # Predict score (gradient of log p(x_t))
                    # A real model might need more inputs (atom types, graph, time_step)
                    # This mockup assumes model(coords) returns the score directly.
                    # The model needs to handle variable number of atoms. This mockup won't.
                    # We'll process atom by atom for this mockup (inefficient and likely wrong)
                    score_pred = torch.zeros_like(coords)
                    try:
                        # This loop is inefficient and likely incorrect for real models
                        for atom_idx in range(num_atoms):
                            score_pred[atom_idx] = self.model(coords[atom_idx].unsqueeze(0)).squeeze(0)
                    except Exception as e:
                         print(f"ERRO (AI Sampler): Falha na predição do modelo (mockup): {e}")
                         # Skip this conformer if model fails
                         score_pred = None # Mark as failed
                         break

                    if score_pred is None: continue # Skip to next conformer if score prediction failed

                    # Update coordinates using Langevin dynamics step (simplified)
                    # x_{t-1} = x_t + alpha * score + sqrt(beta) * z
                    # Exact update rule depends on the specific diffusion formulation
                    # This is a simplified Euler-Maruyama step
                    coords += (noise_level / 2) * score_pred # Drift term based on score
                    if t > 0:
                         coords += torch.sqrt(noise_level) * z # Noise term

                if score_pred is not None: # Check if prediction succeeded for this conformer
                    conformers_list.append(coords.cpu().numpy()) # Store final coordinates

        print(f"INFO (AI Sampler): Generated {len(conformers_list)} raw conformers.")
        if not conformers_list:
            return None

        # Convert numpy arrays back to RDKit conformers in a new Mol object
        return self._process_conformers(mol, conformers_list)

    def _get_initial_coords(self, mol):
        """Get initial coordinates using RDKit embedding."""
        mol_copy = copy.deepcopy(mol) # Work on a copy
        try:
            # Use a robust embedding method like ETKDG
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xf00d
            if AllChem.EmbedMolecule(mol_copy, params) == -1:
                print("AVISO (AI Sampler): ETKDG falhou, tentando incorporação aleatória.")
                params.useRandomCoords = True
                if AllChem.EmbedMolecule(mol_copy, params) == -1:
                     print("ERRO (AI Sampler): Falha na incorporação inicial.")
                     return None
            # Optional: Initial optimization
            AllChem.UFFOptimizeMolecule(mol_copy)
            return mol_copy.GetConformer().GetPositions()
        except Exception as e:
            print(f"ERRO (AI Sampler): Exceção durante a incorporação inicial: {e}")
            return None

    def _process_conformers(self, mol, conformers_np):
        """Convert numpy arrays to RDKit conformers in a new Mol object."""
        if not conformers_np:
            return None
        # Create a new molecule object to hold the conformers
        new_mol = Chem.Mol(mol) # Copy structure from original molecule
        new_mol.RemoveAllConformers() # Ensure it's clean

        for i, conf_coords in enumerate(conformers_np):
            if conf_coords.shape != (mol.GetNumAtoms(), 3):
                 print(f"AVISO (AI Sampler): Confôrmero {i} tem formato inesperado {conf_coords.shape}, pulando.")
                 continue
            rd_conf = Chem.Conformer(mol.GetNumAtoms())
            rd_conf.SetId(i)
            for j, pos in enumerate(conf_coords):
                # RDKit expects Geometry.Point3D, create it from numpy array
                rd_conf.SetAtomPosition(j, pos.tolist())
            new_mol.AddConformer(rd_conf, assignId=True)

        print(f"INFO (AI Sampler): Processados {new_mol.GetNumConformers()} confôrmeros AI em objeto RDKit.")
        return new_mol

# --- Hybrid Conformer Generator ---

def generate_physics_conformers(mol, num_confs, random_seed=0xf00d):
    """
    Generates conformers using traditional physics-based RDKit embedding (ETKDG).
    Slightly simplified version for the hybrid approach.
    """
    print(f"INFO (Physics Sampler): Gerando {num_confs} confôrmeros com ETKDG...")
    mol_copy = copy.deepcopy(mol)
    mol_copy.RemoveAllConformers()

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.pruneRmsThresh = -1 # Disable pruning here, filter later
    cids = AllChem.EmbedMultipleConfs(mol_copy, numConfs=num_confs, params=params)

    if not cids:
        print("AVISO (Physics Sampler): Falha ao gerar confôrmeros com EmbedMultipleConfs.")
        return None # Return None if generation fails

    print(f"INFO (Physics Sampler): Gerados {len(cids)} confôrmeros físicos brutos.")
    # Optimization will happen *after* combining with AI conformers
    return mol_copy


def combine_conformers(mol1, mol2):
    """Merges conformers from mol2 into mol1."""
    if mol1 is None and mol2 is None: return None
    if mol1 is None: return mol2
    if mol2 is None: return mol1

    combined_mol = Chem.Mol(mol1) # Start with mol1
    start_id = combined_mol.GetNumConformers()
    for i, conf in enumerate(mol2.GetConformers()):
        conf.SetId(start_id + i) # Assign unique IDs
        combined_mol.AddConformer(conf, assignId=False) # assignId=False because we set it manually
    print(f"INFO (Combine): Combinados {mol1.GetNumConformers()} + {mol2.GetNumConformers()} -> {combined_mol.GetNumConformers()} confôrmeros.")
    return combined_mol

def optimize_conformers(mol, max_iter=200, mmffVariant='MMFF94s'):
    """
    Optimizes all conformers in a molecule using MMFF.
    Returns the molecule and a list of energies (or None for failed optimizations).
    """
    if mol is None or mol.GetNumConformers() == 0:
        return mol, []

    print(f"INFO (Optimize): Otimizando {mol.GetNumConformers()} confôrmeros com {mmffVariant} (max_iter={max_iter})...")
    energies = []
    try:
        # MMFFOptimizeMoleculeConfs modifies the molecule in-place
        results = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iter, mmffVariant=mmffVariant)
        # results is a list of tuples (not_converged, energy)
        num_converged = 0
        for i, res in enumerate(results):
             if res[0] == 0: # Converged
                 energies.append(res[1])
                 num_converged += 1
             else:
                 # Assign a high energy or NaN to non-converged conformers
                 energies.append(np.inf) # Or np.nan
                 print(f"AVISO (Optimize): Confôrmero ID {i} não convergiu na otimização.")
        print(f"INFO (Optimize): Otimização concluída. {num_converged}/{len(results)} confôrmeros convergiram.")
    except Exception as e:
        print(f"ERRO (Optimize): Falha durante a otimização MMFF: {e}. Retornando molécula não otimizada.")
        # Return original molecule and empty energies if optimization fails globally
        return mol, [np.inf] * mol.GetNumConformers()

    return mol, energies


def filter_conformers(mol, energies, max_energy=100.0, rmsd_thresh=1.0):
    """
    Filters conformers based on energy and RMSD diversity.
    Uses scipy.spatial.distance.cdist for RMSD calculation between coordinate sets.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return mol

    num_initial_confs = mol.GetNumConformers()
    if len(energies) != num_initial_confs:
        print("AVISO (Filter): Disparidade entre número de confôrmeros e energias. Pulando filtragem.")
        return mol

    # 1. Energy filtering
    energy_threshold = min(energies) + max_energy if energies else 0 # Relative energy window
    valid_ids_energy = [
        conf.GetId() for conf, energy in zip(mol.GetConformers(), energies)
        if energy <= energy_threshold and np.isfinite(energy) # Check for finite energy within window
    ]

    if not valid_ids_energy:
        print("AVISO (Filter): Nenhum confôrmero passou na filtragem de energia.")
        # Return an empty molecule or the one with the lowest energy? Return empty for now.
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        return new_mol

    print(f"INFO (Filter): {len(valid_ids_energy)}/{num_initial_confs} confôrmeros passaram no filtro de energia (E <= min(E) + {max_energy:.2f}).")

    # Sort by energy to keep lower energy conformers first during RMSD check
    valid_confs_energy = sorted(
        [(mol.GetConformer(cid), energies[cid]) for cid in valid_ids_energy],
        key=lambda x: x[1] # Sort by energy
    )

    # 2. RMSD diversity filtering
    heavy_atom_indices = get_heavy_atom_indices(mol) # Use heavy atoms for RMSD
    if not heavy_atom_indices:
         print("AVISO (Filter): Nenhum átomo pesado encontrado para cálculo de RMSD. Usando todos os átomos.")
         heavy_atom_indices = list(range(mol.GetNumAtoms())) # Fallback to all atoms

    filtered_confs_final = []
    filtered_coords_list = [] # Store coordinate arrays of kept conformers

    print(f"INFO (Filter): Aplicando filtro de diversidade RMSD (threshold={rmsd_thresh:.2f} Å)...")
    for conf, energy in valid_confs_energy:
        coords = conf.GetPositions()[heavy_atom_indices] # Get heavy atom coordinates

        is_diverse = True
        if filtered_coords_list: # Only check if we already have kept conformers
            # Calculate RMSD against all previously kept conformers
            # cdist computes pairwise distances, not RMSD directly.
            # We need pairwise RMSD. RDKit's AlignMol is better suited but slower for many pairs.
            # Let's use a simplified check with cdist (average distance) or implement pairwise RMSD.

            # Option A: Simplified check using average distance (less accurate than RMSD)
            # distances = cdist([coords.flatten()], [f.flatten() for f in filtered_coords_list])
            # min_dist = distances.min() if distances.size > 0 else np.inf
            # if min_dist < some_threshold: # Threshold needs tuning
            #     is_diverse = False

            # Option B: Calculate pairwise RMSD properly (more accurate but slower)
            min_rmsd = np.inf
            temp_mol_conf = Chem.Mol(mol) # Create temp mol for alignment
            temp_mol_conf.RemoveAllConformers()
            temp_mol_conf.AddConformer(conf, assignId=True)

            for kept_conf in filtered_confs_final:
                temp_mol_kept = Chem.Mol(mol)
                temp_mol_kept.RemoveAllConformers()
                temp_mol_kept.AddConformer(kept_conf, assignId=True)
                try:
                    # Align and get RMSD using heavy atoms
                    atom_map = list(zip(heavy_atom_indices, heavy_atom_indices))
                    rmsd = rdMolAlign.AlignMol(temp_mol_conf, temp_mol_kept, atomMap=atom_map)
                    min_rmsd = min(min_rmsd, rmsd)
                except Exception as e:
                    print(f"AVISO (Filter): Falha no cálculo de RMSD pairwise: {e}")
                    # If RMSD fails, maybe keep the conformer? Or discard? Let's be conservative and assume not diverse.
                    # is_diverse = False # Or keep it? Let's keep it if RMSD fails.
                    pass # Keep is_diverse = True

            if min_rmsd < rmsd_thresh:
                is_diverse = False

        if is_diverse:
            filtered_confs_final.append(conf)
            filtered_coords_list.append(coords) # Store coords for next comparisons (if using cdist)

    print(f"INFO (Filter): {len(filtered_confs_final)} confôrmeros mantidos após filtro de diversidade.")

    # Create final molecule with filtered conformers
    final_mol = Chem.Mol(mol)
    final_mol.RemoveAllConformers()
    for i, conf in enumerate(filtered_confs_final):
        conf.SetId(i) # Reset IDs sequentially
        final_mol.AddConformer(conf, assignId=False)

    return final_mol


def generate_conformers_hybrid(mol, num_confs=20,
                               ai_weight=DEFAULT_AI_WEIGHT,
                               mmff_iter=DEFAULT_MMFF_ITER,
                               max_energy=DEFAULT_HYBRID_MAX_ENERGY,
                               rmsd_thresh=DEFAULT_HYBRID_RMSD_THRESH):
    """
    Generates conformers using a hybrid AI-Physics approach.

    1. Generates a portion of conformers using ScoreBasedSampler (AI).
    2. Generates the remaining portion using RDKit ETKDG (Physics).
    3. Combines the conformer sets.
    4. Optimizes all conformers using MMFF.
    5. Filters the optimized conformers by energy and RMSD diversity.

    Args:
        mol: RDKit molecule object.
        num_confs: Total number of conformers desired *before* final filtering.
        ai_weight: Ratio of initial conformers to generate using AI (0-1).
        mmff_iter: MMFF optimization iterations.
        max_energy: Energy window (relative to lowest) for filtering (kcal/mol).
        rmsd_thresh: RMSD threshold for diversity filtering (Angstrom).

    Returns:
        An RDKit molecule object containing the final filtered conformers,
        or None if generation fails.
    """
    if not TORCH_AVAILABLE:
        print("ERRO: PyTorch não está instalado. Geração Híbrida requer PyTorch.")
        return None

    print("\n--- Iniciando Geração Híbrida de Confôrmeros ---")
    print(f"Total desejado (antes filtro): {num_confs}, Peso AI: {ai_weight:.2f}, Iter MMFF: {mmff_iter}")

    # Determine number of conformers for each method
    ai_confs_target = int(num_confs * ai_weight)
    phys_confs_target = num_confs - ai_confs_target

    # --- 1. AI-based generation ---
    ai_mol = None
    if ai_confs_target > 0:
        try:
            # Initialize sampler here to manage device context if needed
            ai_sampler = ScoreBasedSampler()
            ai_mol = ai_sampler.generate_conformers(mol, ai_confs_target)
            del ai_sampler # Release resources if possible
            if ai_mol is None or ai_mol.GetNumConformers() == 0:
                 print("AVISO: Geração AI não produziu confôrmeros.")
                 ai_mol = None # Ensure it's None if failed
            else:
                 print(f"INFO: AI gerou {ai_mol.GetNumConformers()} confôrmeros.")
        except Exception as e:
            print(f"ERRO: Falha na geração de confôrmeros AI: {e}")
            ai_mol = None
    else:
        print("INFO: Nenhuma conformação AI solicitada (ai_weight=0 ou num_confs baixo).")


    # --- 2. Physics-based generation ---
    phys_mol = None
    if phys_confs_target > 0:
        try:
            phys_mol = generate_physics_conformers(mol, phys_confs_target)
            if phys_mol is None or phys_mol.GetNumConformers() == 0:
                 print("AVISO: Geração Física não produziu confôrmeros.")
                 phys_mol = None # Ensure it's None if failed
            else:
                 print(f"INFO: Física gerou {phys_mol.GetNumConformers()} confôrmeros.")
        except Exception as e:
            print(f"ERRO: Falha na geração de confôrmeros Físicos: {e}")
            phys_mol = None
    else:
         print("INFO: Nenhuma conformação Física solicitada (ai_weight=1 ou num_confs baixo).")


    # --- 3. Combine ---
    combined_mol = combine_conformers(ai_mol, phys_mol)

    if combined_mol is None or combined_mol.GetNumConformers() == 0:
        print("ERRO: Nenhum confôrmero gerado por AI ou Física. Impossível continuar.")
        return None

    # --- 4. Optimize ---
    optimized_mol, energies = optimize_conformers(combined_mol, mmff_iter)

    # --- 5. Filter ---
    final_mol = filter_conformers(optimized_mol, energies, max_energy, rmsd_thresh)

    if final_mol is None or final_mol.GetNumConformers() == 0:
         print("AVISO: Nenhum confôrmero restou após a filtragem final.")
         # Return an empty mol object instead of None if filtering removed everything
         if final_mol is None:
             final_mol = Chem.Mol(mol)
             final_mol.RemoveAllConformers()
         return final_mol # Return empty molecule

    print(f"--- Geração Híbrida Concluída: {final_mol.GetNumConformers()} confôrmeros finais ---")
    return final_mol


# --- Funções Auxiliares (Já existentes no script original) ---

def get_heavy_atoms(mol):
    """Retorna uma lista de átomos pesados (não hidrogênio)."""
    return [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]

def get_heavy_atom_indices(mol):
    """Retorna uma lista de índices de átomos pesados."""
    return [atom.GetIdx() for atom in get_heavy_atoms(mol)]

def calculate_d0(n_min):
    """Calcula a escala d0 dependente do tamanho (Eq. 2 do paper LS-align)."""
    if n_min <= 9:
        return 0.15
    # Use np.cbrt for cube root
    return 0.55 * np.cbrt(n_min - 9) + 0.15

def calculate_bwjs(atom_i, atom_j):
    """
    Calcula o Bond Weighted Jaccard Score (BWJS) entre dois átomos (Eq. 4).
    Compara os tipos de ligações conectadas a cada átomo.
    """
    bonds_i = atom_i.GetBonds()
    bonds_j = atom_j.GetBonds()

    bond_types_i = {}
    g_si = 0.0
    for bond in bonds_i:
        b_type = bond.GetBondType()
        weight = BOND_WEIGHTS.get(b_type, DEFAULT_BOND_WEIGHT)
        # Use bond type and count as key for intersection calculation
        bond_key = b_type # Simpler key
        bond_types_i[bond_key] = bond_types_i.get(bond_key, 0) + 1
        g_si += weight

    bond_types_j = {}
    g_sj = 0.0
    for bond in bonds_j:
        b_type = bond.GetBondType()
        weight = BOND_WEIGHTS.get(b_type, DEFAULT_BOND_WEIGHT)
        bond_key = b_type
        bond_types_j[bond_key] = bond_types_j.get(bond_key, 0) + 1
        g_sj += weight

    g_intersection = 0.0
    # Iterate through bond types present in atom i
    for b_key, count_i in bond_types_i.items():
        if b_key in bond_types_j:
            count_j = bond_types_j[b_key]
            intersection_count = min(count_i, count_j)
            # Get the weight associated with this bond type
            weight = BOND_WEIGHTS.get(b_key, DEFAULT_BOND_WEIGHT)
            g_intersection += intersection_count * weight

    denominator = g_si + g_sj - g_intersection
    # Add small epsilon to avoid division by zero if g_si and g_sj are zero
    return g_intersection / (denominator + 1e-9) if denominator > -1e-9 else 0.0


def calculate_pc_score_terms(atom_i, atom_j, pos_i, pos_j, d0_val):
    """
    Calcula os três termos do PC-score para um par de átomos i, j
    APÓS a superposição (usa as posições pos_i, pos_j).
    Retorna (termo_distancia, termo_massa, termo_bwjs)
    """
    # Ensure inputs are numpy arrays
    pos_i_np = np.array(pos_i)
    pos_j_np = np.array(pos_j)

    dist_sq = np.sum((pos_i_np - pos_j_np)**2)
    term_dist = PC_SCORE_WEIGHT_DISTANCE / (1.0 + dist_sq / (d0_val**2))

    mass_i = atom_i.GetMass()
    mass_j = atom_j.GetMass()
    mass_diff_sq = (mass_i - mass_j)**2
    term_mass = PC_SCORE_WEIGHT_MASS / (1.0 + mass_diff_sq / (MASS_DIFF_SCALE_M0**2))

    bwjs_val = calculate_bwjs(atom_i, atom_j)
    term_bond = PC_SCORE_WEIGHT_BOND * bwjs_val

    return term_dist, term_mass, term_bond

def get_kabsch_transformation(coords_ref, coords_mov):
    """
    Calcula a rotação e translação ótimas para superpor coords_mov em coords_ref
    usando o algoritmo de Kabsch.
    Retorna: matriz de rotação (3x3), vetor de translação (3,), RMSD.
    """
    # Ensure numpy arrays
    coords_ref = np.asarray(coords_ref)
    coords_mov = np.asarray(coords_mov)

    if coords_ref.shape != coords_mov.shape or coords_ref.shape[0] < 3: # Need at least 3 points for stable Kabsch
         print(f"AVISO: Coordenadas inválidas ou insuficientes para Kabsch (Shape Ref: {coords_ref.shape}, Shape Mov: {coords_mov.shape}).")
         # Return identity transform and infinite RMSD
         return np.identity(3), np.zeros(3), np.inf

    center_ref = np.mean(coords_ref, axis=0)
    center_mov = np.mean(coords_mov, axis=0)
    coords_ref_cen = coords_ref - center_ref
    coords_mov_cen = coords_mov - center_mov

    # Covariance matrix H = P^T Q
    cov_matrix = coords_mov_cen.T @ coords_ref_cen

    try:
        U, S, Vt = np.linalg.svd(cov_matrix)
    except np.linalg.LinAlgError:
         print("ERRO: SVD falhou na matriz de covariância.")
         return np.identity(3), np.zeros(3), np.inf

    # Ensure a right-handed coordinate system (reflection check)
    d = np.linalg.det(Vt.T @ U.T)
    if d < 0:
        # print("Corrigindo reflexão na rotação Kabsch.")
        Vt[-1, :] *= -1 # Flip the sign of the last row of Vt

    # Calculate rotation matrix R = V U^T
    rotation_matrix = Vt.T @ U.T
    # Calculate translation vector t = center_ref - R @ center_mov
    translation_vector = center_ref - rotation_matrix @ center_mov

    # Calculate RMSD after applying the transformation
    coords_mov_aligned = (rotation_matrix @ coords_mov.T).T + translation_vector
    rmsd = np.sqrt(np.mean(np.sum((coords_mov_aligned - coords_ref)**2, axis=1)))

    return rotation_matrix, translation_vector, rmsd


def load_molecule(filepath):
    """
    Carrega uma molécula de um arquivo (ex: SDF, MOL, PDB).
    Adiciona hidrogênios e tenta gerar coordenadas 3D se não existirem.
    """
    if not os.path.exists(filepath):
        print(f"ERRO: Arquivo não encontrado: {filepath}")
        return None

    _, ext = os.path.splitext(filepath)
    mol = None
    suppl = None # Initialize suppl
    try:
        if ext.lower() == '.sdf':
            # Keep Hs that are explicitly defined in the SDF
            suppl = Chem.SDMolSupplier(filepath, removeHs=False, sanitize=True)
            mol = next(iter(suppl), None)
        elif ext.lower() == '.mol':
            mol = Chem.MolFromMolFile(filepath, removeHs=False, sanitize=True)
        elif ext.lower() == '.pdb':
            mol = Chem.MolFromPDBFile(filepath, removeHs=False, sanitize=True)
        else:
            print(f"Formato de arquivo não suportado ou não reconhecido: {ext}, tentando como SDF.")
            # Try loading as SDF as a fallback
            suppl = Chem.SDMolSupplier(filepath, removeHs=False, sanitize=True)
            mol = next(iter(suppl), None)
    except Exception as e:
         print(f"ERRO ao ler o arquivo {filepath}: {e}")
         return None
    finally:
        # Ensure the supplier is closed if it was opened
        if suppl is not None and hasattr(suppl, 'close'):
             suppl.close()


    if mol is None:
        print(f"Não foi possível carregar a molécula de {filepath}")
        return None

    # Store original name if present
    original_name = None
    if mol.HasProp('_Name'):
        original_name = mol.GetProp('_Name')

    # Add hydrogens if they seem missing and generate coordinates if needed
    needs_h_add = not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms())
    needs_coords = mol.GetNumConformers() == 0

    if needs_h_add:
        print("INFO: Adicionando hidrogênios...")
        try:
            # Add Hs, but only add coords if conformers already exist
            mol = Chem.AddHs(mol, addCoords=(not needs_coords))
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"AVISO: Problema durante adição de H ou sanitização: {e}. Continuando com a molécula como está.")
            # Attempt to sanitize the original mol again just in case
            try: Chem.SanitizeMol(mol)
            except: pass

    if needs_coords:
        print("INFO: Molécula sem confôrmeros. Gerando confôrmero 3D inicial...")
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xf00d
            embed_result = AllChem.EmbedMolecule(mol, params)
            if embed_result == -1:
                 print("AVISO: Falha na incorporação inicial (EmbedMolecule), tentando UFF.")
                 # Ensure Hs are present for UFF
                 if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
                      mol = Chem.AddHs(mol, addCoords=False) # Add Hs without coords first
                 AllChem.EmbedMolecule(mol, params) # Try embedding again after adding Hs

                 ff_params = AllChem.UFFGetMoleculeForceField(mol)
                 if ff_params:
                     ff_params.Initialize()
                     ff_params.Minimize(maxIts=200)
                 else:
                     print("ERRO: Não foi possível obter parâmetros UFF após falha de ETKDG.")
                     # Return None if we absolutely cannot generate coords
                     return None
            else:
                # Optimize the embedded conformer
                AllChem.UFFOptimizeMolecule(mol)
        except Exception as e:
            print(f"ERRO: Falha crítica ao gerar/otimizar confôrmero inicial: {e}")
            return None

    # Restore original name if it was present
    if original_name:
        mol.SetProp('_Name', original_name)

    # Final check for conformer
    if mol.GetNumConformers() == 0:
         print("ERRO: Falha final ao garantir que a molécula tenha um confôrmero.")
         return None

    try:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol)) if mol else "N/A"
        print(f"Molécula carregada: {smiles} com {mol.GetNumAtoms()} átomos (incl. H).")
    except:
         print(f"Molécula carregada com {mol.GetNumAtoms()} átomos (incl. H). (Falha ao gerar SMILES)")

    return mol


def generate_conformers_rdkit(mol, num_confs=20, random_seed=0xf00d):
    """
    Gera múltiplos confôrmeros para uma molécula usando RDKit ETKDG.
    Retorna uma nova molécula com múltiplos confôrmeros otimizados.
    """
    print(f"INFO: Gerando até {num_confs} confôrmeros para a query usando RDKit ETKDGv3...")
    mol_copy = copy.deepcopy(mol) # Trabalha com cópia
    mol_copy.RemoveAllConformers() # Remove confôrmeros existentes

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.useRandomCoords = False # ETKDGv3 generally better without this
    params.pruneRmsThresh = CONFORMER_RMSD_THRESHOLD # Prune similar conformers during generation
    # Try to generate slightly more conformers initially, as pruning might remove some
    cids = AllChem.EmbedMultipleConfs(mol_copy, numConfs=int(num_confs * 1.5) + 5, params=params)

    if not cids:
        print("AVISO: Falha ao gerar múltiplos confôrmeros com EmbedMultipleConfs. Tentando gerar um único.")
        # Fallback to generating a single conformer
        params.pruneRmsThresh = -1 # Disable pruning for single conformer generation
        if AllChem.EmbedMolecule(mol_copy, params) == -1:
             print("ERRO: Falha ao gerar até mesmo um único confôrmero de fallback.")
             return None # Return None if even single conformer fails
        else:
             cids = [0] # Use the single generated conformer ID

    print(f"INFO: {len(cids)} confôrmeros brutos gerados (antes da otimização).")

    # Optimize the generated conformers using MMFF or UFF
    print("INFO: Otimizando confôrmeros gerados (pode levar tempo)...")
    try:
        # Use MMFF94s if available and applicable
        results = AllChem.MMFFOptimizeMoleculeConfs(mol_copy, maxIters=500, mmffVariant='MMFF94s')
        energies = [(cid, res[1]) for cid, res in zip(cids, results) if res[0] == 0] # Keep (id, energy) for converged
        print(f"INFO: Otimização MMFF concluída. {len(energies)} confôrmeros convergiram.")
    except Exception as e_mmff:
        print(f"AVISO: Falha na otimização MMFF ({e_mmff}). Tentando UFF...")
        try:
            results_uff = AllChem.UFFOptimizeMoleculeConfs(mol_copy, maxIters=500)
            energies = [(cid, res[1]) for cid, res in zip(cids, results_uff) if res[0] == 0]
            print(f"INFO: Otimização UFF concluída. {len(energies)} confôrmeros convergiram.")
        except Exception as e_uff:
            print(f"AVISO: Falha na otimização UFF também ({e_uff}). Usando confôrmeros não otimizados.")
            # Keep all originally generated cids, assign dummy high energy
            energies = [(cid, np.inf) for cid in cids]

    # Filter out non-converged conformers and sort by energy
    converged_cids = [e[0] for e in energies]
    mol_optimized = Chem.Mol(mol_copy) # Create a new molecule for the results
    mol_optimized.RemoveAllConformers()

    # Sort converged conformers by energy
    energies.sort(key=lambda x: x[1])
    final_cids_to_keep = [e[0] for e in energies]

    # Add sorted, converged conformers to the new molecule, up to num_confs
    added_count = 0
    for cid in final_cids_to_keep:
        if added_count >= num_confs:
            break
        try:
            conf = mol_copy.GetConformer(cid)
            conf.SetId(added_count) # Reset ID sequentially
            mol_optimized.AddConformer(conf, assignId=False)
            added_count += 1
        except ValueError:
             print(f"AVISO: Confôrmero ID {cid} não encontrado na molécula original após otimização.")

    if added_count == 0:
         print("ERRO: Nenhum confôrmero válido restou após otimização e filtragem.")
         return None

    print(f"INFO: Mantendo {mol_optimized.GetNumConformers()} confôrmeros otimizados e filtrados para alinhamento.")
    return mol_optimized


# --- Função Principal de Alinhamento (LS-align Rígido) ---

def align_single_conformer_ls_align(query_mol, template_mol, query_conf_id=-1):
    """
    Alinha UM ÚNICO confôrmero da query na template usando Rigid-LS-align com PC-score.
    Retorna a molécula query com o confôrmero alinhado, PC-score, RMSD.
    """
    # Garante que as moléculas são válidas e têm conformeros
    if not query_mol or not template_mol:
        print("ERRO (Align): Molécula query ou template inválida.")
        return None, np.nan, np.nan
    if query_mol.GetNumConformers() == 0:
        print("ERRO (Align): Molécula query não tem confôrmeros.")
        return None, np.nan, np.nan
    if template_mol.GetNumConformers() == 0:
        print("ERRO (Align): Molécula template não tem confôrmeros.")
        return None, np.nan, np.nan

    # Trabalhar apenas com átomos pesados para o alinhamento LS-align
    q_heavy_indices = get_heavy_atom_indices(query_mol)
    t_heavy_indices = get_heavy_atom_indices(template_mol)
    q_atoms = [query_mol.GetAtomWithIdx(i) for i in q_heavy_indices]
    t_atoms = [template_mol.GetAtomWithIdx(i) for i in t_heavy_indices]
    n_q = len(q_atoms)
    n_t = len(t_atoms)

    if n_q == 0 or n_t == 0:
        print("ERRO (Align): Molécula query ou template não tem átomos pesados.")
        return None, np.nan, np.nan

    # Obter coordenadas do confôrmero especificado da query e do template (assume 1º conf)
    try:
        # Use query_conf_id=-1 to get the default conformer if multiple exist but only one is needed.
        # If query_conf_id is specified (e.g., during flexible alignment), use that ID.
        q_conf = query_mol.GetConformer(query_conf_id if query_conf_id >= 0 else 0)
        t_conf = template_mol.GetConformer(0) # Assume template tem 1 conf (ou usa o primeiro)
    except ValueError as e:
         print(f"ERRO (Align): Confôrmero ID {query_conf_id} inválido para a query ou template não tem confôrmero. Detalhes: {e}")
         return None, np.nan, np.nan

    # Get all atom coordinates first
    q_coords_all = q_conf.GetPositions()
    t_coords_all = t_conf.GetPositions()
    # Then select heavy atom coordinates
    q_coords_heavy = q_coords_all[q_heavy_indices]
    t_coords_heavy = t_coords_all[t_heavy_indices]

    # Calcular d0 baseado no número de átomos pesados
    n_min = min(n_q, n_t)
    d0_val = calculate_d0(n_min)

    # --- Alinhamento Iterativo ---
    # Initial guess for alignment: map first min(n_q, n_t) heavy atoms
    current_alignment_pairs = [(i, i) for i in range(n_min)]
    last_alignment_set = set() # Para verificar convergência

    best_pc_score_iter = -np.inf
    best_alignment_iter_pairs = []
    best_rotation_iter = np.identity(3)
    best_translation_iter = np.zeros(3)
    converged = False

    for iteration in range(MAX_ALIGNMENT_ITERATIONS):
        if not current_alignment_pairs:
            # print(f"AVISO (Align): Alinhamento vazio na iteração {iteration + 1}.")
            break # Stop if no alignment pairs exist

        # Get coordinates corresponding to the current alignment for Kabsch
        q_indices_in_alignment = [pair[0] for pair in current_alignment_pairs] # Indices relative to heavy atom list
        t_indices_in_alignment = [pair[1] for pair in current_alignment_pairs] # Indices relative to heavy atom list
        coords_q_kabsch = q_coords_heavy[q_indices_in_alignment]
        coords_t_kabsch = t_coords_heavy[t_indices_in_alignment]

        if len(coords_q_kabsch) < 3:
             # print(f"AVISO (Align): Menos de 3 pontos no alinhamento atual (Iter {iteration + 1}). Parando.")
             break # Kabsch needs at least 3 points

        # Calculate transformation based on current alignment
        rotation, translation, _ = get_kabsch_transformation(coords_t_kabsch, coords_q_kabsch)

        # Apply transformation to *all* heavy atoms of the query for scoring
        q_coords_heavy_transformed = (rotation @ q_coords_heavy.T).T + translation

        # --- Calculate PC Score Matrix ---
        # Matrix where rows are query heavy atoms, columns are template heavy atoms
        score_matrix = np.zeros((n_q, n_t))
        for i in range(n_q): # Iterate through query heavy atoms
            for j in range(n_t): # Iterate through template heavy atoms
                term_dist, term_mass, term_bond = calculate_pc_score_terms(
                    q_atoms[i], t_atoms[j],                      # The atoms themselves
                    q_coords_heavy_transformed[i], t_coords_heavy[j], # Their transformed/original coords
                    d0_val                                     # d0 scale factor
                )
                score_matrix[i, j] = term_dist + term_mass + term_bond

        # --- Find Optimal Assignment using Hungarian Algorithm ---
        # We want to maximize the score, so we minimize the negative score
        row_ind, col_ind = linear_sum_assignment(-score_matrix)
        # new_alignment_pairs gives the optimal (query_idx, template_idx) based on score_matrix
        new_alignment_pairs = list(zip(row_ind, col_ind))
        # Calculate the total PC score for this new alignment
        current_total_pc_score = score_matrix[row_ind, col_ind].sum()
        # Normalize PC score by the number of template heavy atoms (as per LS-align paper)
        current_normalized_pc_score = current_total_pc_score / n_t if n_t > 0 else 0.0

        # --- Store Best Result Found So Far ---
        # LS-align aims to maximize PC-score
        if current_normalized_pc_score > best_pc_score_iter:
            best_pc_score_iter = current_normalized_pc_score
            best_rotation_iter = rotation
            best_translation_iter = translation
            best_alignment_iter_pairs = new_alignment_pairs
            # print(f"Iter {iteration + 1}: New best PC-score = {best_pc_score_iter:.6f}")


        # --- Check for Convergence ---
        new_alignment_set = set(new_alignment_pairs)
        if new_alignment_set == last_alignment_set:
            # print(f"INFO (Align): Alinhamento convergiu na iteração {iteration + 1}.")
            converged = True
            break # Exit loop if alignment hasn't changed

        current_alignment_pairs = new_alignment_pairs
        last_alignment_set = new_alignment_set
    # --- End of Iteration Loop ---

    if not converged and iteration == MAX_ALIGNMENT_ITERATIONS - 1:
         print(f"AVISO (Align): Alinhamento não convergiu após {MAX_ALIGNMENT_ITERATIONS} iterações.")

    if not best_alignment_iter_pairs:
         print("ERRO (Align): Nenhum alinhamento válido encontrado após iterações.")
         return None, np.nan, np.nan

    # --- Apply the BEST transformation found during iterations ---
    # Create a copy of the original query molecule to store the aligned conformer
    final_query_mol = copy.deepcopy(query_mol)
    # Get the specific conformer we were aligning (by its original ID)
    try:
        final_conf = final_query_mol.GetConformer(query_conf_id if query_conf_id >= 0 else 0)
    except ValueError:
         # This shouldn't happen if the initial check passed, but handle defensively
         print(f"ERRO (Align): Não foi possível encontrar o confôrmero final ID {query_conf_id} na cópia.")
         return None, np.nan, np.nan

    # Create the 4x4 transformation matrix
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = best_rotation_iter
    transform_matrix[:3, 3] = best_translation_iter
    # Apply the transformation IN-PLACE to the conformer in the copied molecule
    TransformConformer(final_conf, transform_matrix)

    # --- Calculate Final RMSD using the BEST alignment map ---
    # Get the *original* atom indices corresponding to the best heavy atom alignment
    q_indices_final_map = [q_heavy_indices[pair[0]] for pair in best_alignment_iter_pairs]
    t_indices_final_map = [t_heavy_indices[pair[1]] for pair in best_alignment_iter_pairs]

    final_rmsd = np.nan
    if len(q_indices_final_map) > 0:
        try:
            # Use RDKit's AlignMol to calculate RMSD based on the specific atom map
            # Provide the map explicitly for accuracy based on the LS-align result
            atom_map = list(zip(q_indices_final_map, t_indices_final_map))
            # AlignMol calculates RMSD *without* further alignment if map is given
            # We want RMSD between the already transformed query and the original template
            final_rmsd = rdMolAlign.CalcRMS(final_query_mol, template_mol,
                                            confId=final_conf.GetId(), refConfId=0,
                                            map=atom_map)
            # Note: rdMolAlign.AlignMol re-aligns, CalcRMS just calculates based on map.
        except Exception as e_rmsd:
            print(f"AVISO (Align): Falha no cálculo do RMSD final com rdMolAlign: {e_rmsd}. Calculando manualmente.")
            # Manual calculation as fallback
            q_coords_final_aligned = final_conf.GetPositions()[q_indices_final_map]
            t_coords_original_aligned = t_coords_all[t_indices_final_map] # Use original template coords
            if len(q_coords_final_aligned) == len(t_coords_original_aligned):
                 diff = q_coords_final_aligned - t_coords_original_aligned
                 final_rmsd = np.sqrt(np.sum(diff * diff) / len(q_coords_final_aligned))

    # Return the molecule containing the single transformed conformer,
    # the best PC-score achieved, and the final RMSD.
    return final_query_mol, best_pc_score_iter, final_rmsd


# --- Execução Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Alinha uma molécula query a uma molécula template usando LS-align (Rígido, Flexível RDKit, ou Flexível Híbrido AI-Física) com PC-score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-f", "--reference", required=True, help="Arquivo da molécula de referência/template (.sdf, .mol, .pdb)")
    parser.add_argument("-i", "--query", required=True, help="Arquivo da molécula query (.sdf, .mol, .pdb)")
    parser.add_argument("-o", "--output", required=True, help="Arquivo de saída para a molécula query alinhada (.sdf)")

    # Conformer generation mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--flexible", action='store_true', help="Modo flexível: Gera confôrmeros RDKit para query.")
    mode_group.add_argument("--hybrid", action='store_true', help="Modo flexível HÍBRIDO: Gera confôrmeros AI+Física (requer PyTorch).")

    # Parameters for flexible/hybrid modes
    parser.add_argument("--num_conformers", type=int, default=DEFAULT_NUM_CONFORMERS, help="Número de confôrmeros a gerar/testar nos modos flexível/híbrido.")

    # Parameters specific to hybrid mode
    parser.add_argument("--ai_weight", type=float, default=DEFAULT_AI_WEIGHT, help="(Modo Híbrido) Proporção de confôrmeros gerados por AI (0-1).")
    parser.add_argument("--mmff_iter", type=int, default=DEFAULT_MMFF_ITER, help="(Modo Híbrido) Iterações de otimização MMFF pós-combinação.")
    parser.add_argument("--max_energy", type=float, default=DEFAULT_HYBRID_MAX_ENERGY, help="(Modo Híbrido) Janela de energia para filtragem final (kcal/mol relativo ao mínimo).")
    parser.add_argument("--rmsd_thresh", type=float, default=DEFAULT_HYBRID_RMSD_THRESH, help="(Modo Híbrido) Threshold RMSD para filtragem de diversidade final (Angstrom).")


    # Ignored argument (for compatibility maybe?)
    parser.add_argument("-n", "--n_confs", type=int, default=1, help="(Ignorado) Script salva apenas o melhor confôrmero alinhado.")

    args = parser.parse_args()

    # --- Input Validation ---
    if args.hybrid and not TORCH_AVAILABLE:
        print("ERRO: O modo --hybrid foi solicitado, mas PyTorch não está instalado.")
        print("Instale PyTorch (https://pytorch.org/) ou use o modo --flexible ou rígido.")
        exit(1)

    if args.ai_weight < 0 or args.ai_weight > 1:
        print("ERRO: --ai_weight deve estar entre 0 e 1.")
        exit(1)

    start_time = time.time()

    # --- Carregar Moléculas ---
    print(f"Carregando referência: {args.reference}")
    template_mol = load_molecule(args.reference)
    print(f"\nCarregando query: {args.query}")
    query_mol_original = load_molecule(args.query) # Guarda a original

    if not query_mol_original or not template_mol:
        print("\nERRO CRÍTICO: Não foi possível carregar as moléculas. Verifique os arquivos de entrada e mensagens de erro.")
        exit(1)

    # --- Variáveis para guardar o melhor resultado ---
    best_aligned_mol_final = None # Molécula com o *único* melhor confôrmero alinhado
    best_pc_score_final = -np.inf
    best_rmsd_final = np.inf
    conf_count_tested = 0
    generation_mode = "Rígido" # Default

    # --- Geração de Confôrmeros (se aplicável) ---
    query_mol_to_align = query_mol_original # Por padrão, usa a query original (modo rígido)

    if args.flexible:
        generation_mode = "Flexível (RDKit)"
        print(f"\n--- Iniciando Modo {generation_mode} ---")
        query_mol_conformers = generate_conformers_rdkit(query_mol_original, args.num_conformers)
        if query_mol_conformers and query_mol_conformers.GetNumConformers() > 0:
            query_mol_to_align = query_mol_conformers
        else:
            print("ERRO: Falha ao gerar confôrmeros RDKit. Saindo.")
            exit(1)

    elif args.hybrid:
        generation_mode = "Flexível (Híbrido AI-Física)"
        print(f"\n--- Iniciando Modo {generation_mode} ---")
        query_mol_conformers = generate_conformers_hybrid(
            query_mol_original,
            num_confs=args.num_conformers,
            ai_weight=args.ai_weight,
            mmff_iter=args.mmff_iter,
            max_energy=args.max_energy,
            rmsd_thresh=args.rmsd_thresh
        )
        if query_mol_conformers and query_mol_conformers.GetNumConformers() > 0:
             query_mol_to_align = query_mol_conformers
        else:
             print("ERRO: Falha ao gerar confôrmeros Híbridos ou nenhum restou após filtragem. Saindo.")
             # Se a geração híbrida falhar ou retornar vazio, não há o que alinhar.
             exit(1)

    # --- Alinhamento (Itera sobre confôrmeros se flexível/híbrido) ---
    num_conformers_in_mol = query_mol_to_align.GetNumConformers()
    conf_ids_to_test = [conf.GetId() for conf in query_mol_to_align.GetConformers()]
    conf_count_tested = len(conf_ids_to_test)

    if conf_count_tested == 0:
         print("ERRO: Nenhuma conformação para alinhar encontrada na molécula query preparada.")
         exit(1)

    print(f"\n--- Iniciando Alinhamento LS-align ---")
    print(f"Modo de Geração: {generation_mode}")
    print(f"Alinhando {conf_count_tested} confôrmero(s) da query contra o template...")

    for i, conf_id in enumerate(conf_ids_to_test):
        if conf_count_tested > 1:
             print(f"\n--- Alinhando Confôrmero {i+1}/{conf_count_tested} (ID Original: {conf_id}) ---")
        else:
             print(f"\n--- Alinhando Confôrmero Único ---")

        # Realiza o alinhamento LS-align para este confôrmero específico
        # Passa a molécula que contém o(s) confôrmero(s) e o ID específico a alinhar
        aligned_mol_single_conf, pc_score_conf, rmsd_conf = align_single_conformer_ls_align(
            query_mol_to_align, template_mol, query_conf_id=conf_id
        )

        # Verifica se o alinhamento foi bem-sucedido
        if aligned_mol_single_conf is not None and not np.isnan(pc_score_conf):
            print(f"Resultado Conf. {i+1}: PC-score={pc_score_conf:.6f}, RMSD={rmsd_conf:.4f} Å")
            # Compara com o melhor resultado global encontrado até agora
            # LS-align maximiza PC-score
            if pc_score_conf > best_pc_score_final:
                best_pc_score_final = pc_score_conf
                best_rmsd_final = rmsd_conf
                # Guarda a molécula que contém APENAS este melhor confôrmero alinhado
                # A função align_single_conformer_ls_align já retorna uma cópia
                # com apenas o confôrmero alinhado.
                best_aligned_mol_final = aligned_mol_single_conf
                print(f"  * Novo melhor resultado global encontrado!")
        else:
            print(f"Falha ao alinhar confôrmero {i+1} (ID: {conf_id}).")


    # --- Resultados Finais ---
    end_time = time.time()
    print(f"\n--- Processo Concluído ---")
    print(f"Modo de Geração Usado: {generation_mode}")
    print(f"Confôrmeros testados no alinhamento: {conf_count_tested}")
    print(f"Tempo total: {end_time - start_time:.2f} segundos")

    if best_aligned_mol_final:
        # Salvar a melhor molécula alinhada encontrada
        try:
            writer = Chem.SDWriter(args.output)
            # Formata os scores para string, tratando NaN
            pc_score_str = f"{best_pc_score_final:.6f}" if not np.isnan(best_pc_score_final) else "N/A"
            rmsd_str = f"{best_rmsd_final:.4f}" if not np.isnan(best_rmsd_final) else "N/A"

            # Adiciona propriedades à molécula de saída
            if not np.isnan(best_pc_score_final):
                 best_aligned_mol_final.SetDoubleProp("LSalign_PC_Score", best_pc_score_final)
            if not np.isnan(best_rmsd_final):
                 best_aligned_mol_final.SetDoubleProp("LSalign_RMSD", best_rmsd_final)

            # Tenta pegar o nome original da query para incluir no nome da molécula salva
            original_name = query_mol_original.GetProp('_Name') if query_mol_original.HasProp('_Name') else 'Query'
            output_mol_name = f"LSaligned_{original_name}_Mode_{generation_mode.split('(')[0].strip()}_PC_{pc_score_str}_RMSD_{rmsd_str}"
            best_aligned_mol_final.SetProp("_Name", output_mol_name)

            # Escreve a molécula (que contém apenas o melhor confôrmero alinhado)
            writer.write(best_aligned_mol_final)
            writer.close()
            print(f"\nMelhor molécula query alinhada salva em: {args.output}")
            print(f"  Melhor PC-score: {pc_score_str}")
            print(f"  RMSD correspondente: {rmsd_str} Å")

        except Exception as e:
            print(f"ERRO CRÍTICO ao salvar o arquivo de saída {args.output}: {e}")
            exit(1) # Sai se não conseguir salvar o resultado principal

        # Bloco de visualização opcional com py3Dmol
        try:
             import py3Dmol
             view = py3Dmol.view(width=600, height=400)
             # Adiciona Template (Ciano)
             view.addModel(Chem.MolToMolBlock(template_mol), 'mol')
             view.setStyle({'model': 0}, {'stick': {'colorscheme': 'cyanCarbon'}})
             # Adiciona Query Alinhada (Magenta)
             view.addModel(Chem.MolToMolBlock(best_aligned_mol_final), 'mol')
             view.setStyle({'model': 1}, {'stick': {'colorscheme': 'magentaCarbon'}})
             view.zoomTo()
             # Gera nome do arquivo HTML
             html_file = os.path.splitext(args.output)[0] + "_view.html"
             view.write_html(html_file)
             print(f"\nVisualização 3D salva em: {html_file}")
             print("(Template=ciano, Query Alinhada=magenta)")
        except ImportError:
             print("\n(Opcional) Para visualização 3D automática, instale py3Dmol: pip install py3Dmol")
        except Exception as e_vis:
             print(f"\nAVISO: Falha ao gerar visualização 3D: {e_vis}")

    else:
         print("\nERRO: Alinhamento LS-align falhou ou nenhum confôrmero válido foi encontrado/gerado.")
         print("Nenhuma molécula de saída foi gerada.")
         exit(1) # Sai com erro se nenhum resultado foi obtido
