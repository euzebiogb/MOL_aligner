# -*- coding: utf-8 -*-
"""
Script Python para alinhamento molecular flexível como ferramenta de linha de comando,
IMPLEMENTANDO A LÓGICA DO LS-align (Rigid-LS-align com PC-score e opção Flexível)
Baseado em: Hu et al., Bioinformatics, 34(13), 2018, 2209-2218
DOI: 10.1093/bioinformatics/bty081

Este script usa RDKit para manipulação básica de moléculas e implementa
o alinhamento iterativo e o PC-score do LS-align. A opção --flexible
permite gerar múltiplos confôrmeros da query e alinhar cada um.

Uso na linha de comando:
python aligner_script.py -f reference.sdf -i query.sdf -o aligned_query.sdf [--flexible] [--num_conformers N]

Argumentos:
  -f, --reference       Arquivo da molécula de referência/template (.sdf, .mol, .pdb)
  -i, --query           Arquivo da molécula query (.sdf, .mol, .pdb)
  -o, --output          Arquivo de saída para a molécula query alinhada (.sdf)
  --flexible            Ativa o modo de alinhamento flexível (gera confôrmeros para a query).
  --num_conformers N    Número de confôrmeros a gerar/testar no modo flexível (default: 20).
  -n, --n_confs         (Ignorado) Número de conformações a salvar na saída.

Dependências principais:
- RDKit: (conda install -c conda-forge rdkit)
- NumPy: (pip install numpy)
- SciPy: (pip install scipy)
"""

import os
import argparse
import copy # Para copiar objetos de molécula
import time # Para medir o tempo
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdchem, Descriptors
from rdkit.Chem.rdMolTransforms import TransformConformer
import numpy as np
from scipy.optimize import linear_sum_assignment # Para encontrar o melhor mapeamento
from scipy.spatial.transform import Rotation # Para Kabsch

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

# Parâmetros para geração de confôrmeros no modo flexível
DEFAULT_NUM_CONFORMERS = 20 # Número de confôrmeros a gerar/testar
CONFORMER_RMSD_THRESHOLD = 0.5 # Para remover confôrmeros redundantes


# --- Funções Auxiliares ---

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
        bond_types_i[b_type] = bond_types_i.get(b_type, 0) + 1
        g_si += weight

    bond_types_j = {}
    g_sj = 0.0
    for bond in bonds_j:
        b_type = bond.GetBondType()
        weight = BOND_WEIGHTS.get(b_type, DEFAULT_BOND_WEIGHT)
        bond_types_j[b_type] = bond_types_j.get(b_type, 0) + 1
        g_sj += weight

    g_intersection = 0.0
    for b_type, count_i in bond_types_i.items():
        if b_type in bond_types_j:
            count_j = bond_types_j[b_type]
            intersection_count = min(count_i, count_j)
            weight = BOND_WEIGHTS.get(b_type, DEFAULT_BOND_WEIGHT)
            g_intersection += intersection_count * weight

    denominator = g_si + g_sj - g_intersection
    return g_intersection / denominator if denominator > 1e-6 else 0.0

def calculate_pc_score_terms(atom_i, atom_j, pos_i, pos_j, d0_val):
    """
    Calcula os três termos do PC-score para um par de átomos i, j
    APÓS a superposição (usa as posições pos_i, pos_j).
    Retorna (termo_distancia, termo_massa, termo_bwjs)
    """
    dist_sq = np.sum((pos_i - pos_j)**2)
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
    if coords_ref.shape != coords_mov.shape or coords_ref.shape[0] == 0:
         print("ERRO: Coordenadas inválidas para Kabsch.")
         return np.identity(3), np.zeros(3), np.inf

    center_ref = np.mean(coords_ref, axis=0)
    center_mov = np.mean(coords_mov, axis=0)
    coords_ref_cen = coords_ref - center_ref
    coords_mov_cen = coords_mov - center_mov

    cov_matrix = coords_mov_cen.T @ coords_ref_cen

    try:
        U, S, Vt = np.linalg.svd(cov_matrix)
    except np.linalg.LinAlgError:
         print("ERRO: SVD falhou na matriz de covariância.")
         return np.identity(3), np.zeros(3), np.inf

    d = np.linalg.det(Vt.T @ U.T)
    if d < 0:
        Vt[-1, :] *= -1

    rotation_matrix = Vt.T @ U.T
    translation_vector = center_ref - rotation_matrix @ center_mov

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
    if ext.lower() == '.sdf':
        suppl = Chem.SDMolSupplier(filepath, removeHs=False, sanitize=True)
        mol = next(iter(suppl), None)
    elif ext.lower() == '.mol':
        mol = Chem.MolFromMolFile(filepath, removeHs=False, sanitize=True)
    elif ext.lower() == '.pdb':
        mol = Chem.MolFromPDBFile(filepath, removeHs=False, sanitize=True)
    else:
        print(f"Formato de arquivo não suportado ou não reconhecido: {ext}, tentando como SDF.")
        try:
            suppl = Chem.SDMolSupplier(filepath, removeHs=False, sanitize=True)
            mol = next(iter(suppl), None)
        except: pass

    if mol is None:
        print(f"Não foi possível carregar a molécula de {filepath}")
        return None

    try:
        mol_with_hs = Chem.AddHs(mol, addCoords=(mol.GetNumConformers() > 0))
        Chem.SanitizeMol(mol_with_hs)
        mol = mol_with_hs # Usa a versão com H se a sanitização funcionar
    except Exception as e:
        print(f"AVISO: Problema durante adição de H ou sanitização: {e}. Usando molécula original.")
        try:
             Chem.SanitizeMol(mol) # Tenta sanitizar a original
        except Exception as e2:
             print(f"AVISO: Falha ao sanitizar molécula original também: {e2}")

    if mol.GetNumConformers() == 0:
        print("Gerando conformero 3D inicial...")
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xf00d
            embed_result = AllChem.EmbedMolecule(mol, params)
            if embed_result == -1:
                 print("AVISO: Falha na incorporação inicial (EmbedMolecule), tentando UFF.")
                 ff_params = AllChem.UFFGetMoleculeForceField(mol)
                 if ff_params:
                     ff_params.Initialize()
                     ff_params.Minimize(maxIts=200)
                 else:
                     print("ERRO: Não foi possível obter parâmetros UFF.")
                     return None
            else:
                AllChem.UFFOptimizeMolecule(mol)
        except Exception as e:
            print(f"ERRO: Falha crítica ao gerar/otimizar conformero inicial: {e}")
            return None

    print(f"Molécula carregada: {Chem.MolToSmiles(Chem.RemoveHs(mol))} com {mol.GetNumAtoms()} átomos (incl. H).")
    return mol

def generate_conformers_rdkit(mol, num_confs=20, random_seed=0xf00d):
    """
    Gera múltiplos confôrmeros para uma molécula usando RDKit ETKDG.
    Retorna uma nova molécula com múltiplos confôrmeros.
    """
    print(f"INFO: Gerando até {num_confs} confôrmeros para a query...")
    mol_copy = copy.deepcopy(mol) # Trabalha com cópia
    mol_copy.RemoveAllConformers() # Remove confôrmeros existentes

    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.useRandomCoords = True # Adiciona mais aleatoriedade
    params.pruneRmsThresh = CONFORMER_RMSD_THRESHOLD # Remove confôrmeros muito similares
    # Tenta gerar mais confôrmeros do que o necessário, pois alguns podem falhar
    cids = AllChem.EmbedMultipleConfs(mol_copy, numConfs=num_confs * 2, params=params)

    if not cids:
        print("ERRO: Falha ao gerar confôrmeros com EmbedMultipleConfs.")
        # Tenta gerar um único como fallback
        if AllChem.EmbedMolecule(mol_copy, params) == -1:
             print("ERRO: Falha ao gerar até mesmo um único confôrmero.")
             return None
        else:
             cids = [0] # Usa o único gerado

    print(f"INFO: {len(cids)} confôrmeros brutos gerados.")

    # Otimizar os confôrmeros gerados (ex: MMFF)
    print("INFO: Otimizando confôrmeros gerados (pode levar tempo)...")
    try:
        results = AllChem.MMFFOptimizeMoleculeConfs(mol_copy, maxIters=500, mmffVariant='MMFF94s')
        # results é uma lista de tuplas (not_converged, energy)
        energies = [res[1] for res in results if res[0] == 0] # Pega energias dos convergidos
        print(f"INFO: Otimização concluída. {len(energies)} confôrmeros convergiram.")
    except Exception as e:
        print(f"AVISO: Falha na otimização MMFF dos confôrmeros: {e}. Usando confôrmeros não otimizados.")
        # Pode ser necessário usar UFF ou pular otimização se MMFF falhar

    # Manter apenas o número desejado de confôrmeros (os primeiros após EmbedMultipleConfs)
    # Ou poderia ordenar por energia se a otimização funcionou
    valid_cids = list(mol_copy.GetConformers()) # Pega os objetos Conformer
    if len(valid_cids) > num_confs:
         # Remove os extras (EmbedMultipleConfs já removeu os muito similares)
         extra_ids = [conf.GetId() for conf in valid_cids[num_confs:]]
         for cid in extra_ids:
             mol_copy.RemoveConformer(cid)

    print(f"INFO: Mantendo {mol_copy.GetNumConformers()} confôrmeros para alinhamento.")
    return mol_copy


# --- Função Principal de Alinhamento (LS-align Rígido) ---

def align_single_conformer_ls_align(query_mol, template_mol, query_conf_id=-1):
    """
    Alinha UM ÚNICO confôrmero da query na template usando Rigid-LS-align com PC-score.
    Retorna a molécula query com o confôrmero alinhado, PC-score, RMSD.
    """
    # Garante que as moléculas são válidas
    if not query_mol or not template_mol: return None, np.nan, np.nan
    if query_mol.GetNumConformers() == 0 or template_mol.GetNumConformers() == 0: return None, np.nan, np.nan

    # Trabalhar apenas com átomos pesados
    q_heavy_indices = get_heavy_atom_indices(query_mol)
    t_heavy_indices = get_heavy_atom_indices(template_mol)
    q_atoms = [query_mol.GetAtomWithIdx(i) for i in q_heavy_indices]
    t_atoms = [template_mol.GetAtomWithIdx(i) for i in t_heavy_indices]
    n_q = len(q_atoms)
    n_t = len(t_atoms)

    if n_q == 0 or n_t == 0: return None, np.nan, np.nan

    # Obter coordenadas do confôrmero especificado da query e do template
    try:
        q_conf = query_mol.GetConformer(query_conf_id)
        t_conf = template_mol.GetConformer() # Assume que template tem 1 conf
    except ValueError:
         print(f"ERRO: Confôrmero ID {query_conf_id} inválido para a query.")
         return None, np.nan, np.nan

    q_coords_all = q_conf.GetPositions()
    t_coords_all = t_conf.GetPositions()
    q_coords = q_coords_all[q_heavy_indices]
    t_coords = t_coords_all[t_heavy_indices]

    # Calcular d0
    n_min = min(n_q, n_t)
    d0_val = calculate_d0(n_min)

    # --- Alinhamento Iterativo ---
    current_alignment = [(i, i) for i in range(n_min)] # Inicial simples
    last_alignment_set = set() # Para verificar convergência

    best_pc_score_iter = -np.inf
    best_alignment_iter = []
    best_rotation_iter = np.identity(3)
    best_translation_iter = np.zeros(3)

    for iteration in range(MAX_ALIGNMENT_ITERATIONS):
        if not current_alignment: break

        q_indices_in_alignment = [q_heavy_indices[pair[0]] for pair in current_alignment]
        t_indices_in_alignment = [t_heavy_indices[pair[1]] for pair in current_alignment]
        coords_q_aligned = q_coords_all[q_indices_in_alignment]
        coords_t_aligned = t_coords_all[t_indices_in_alignment]

        if len(coords_q_aligned) < 3: break

        rotation, translation, _ = get_kabsch_transformation(coords_t_aligned, coords_q_aligned)
        q_coords_transformed = (rotation @ q_coords.T).T + translation

        score_matrix = np.zeros((n_q, n_t))
        for i in range(n_q):
            for j in range(n_t):
                term_dist, term_mass, term_bond = calculate_pc_score_terms(
                    q_atoms[i], t_atoms[j], q_coords_transformed[i], t_coords[j], d0_val
                )
                score_matrix[i, j] = term_dist + term_mass + term_bond

        row_ind, col_ind = linear_sum_assignment(-score_matrix)
        new_alignment = list(zip(row_ind, col_ind))
        current_pc_score = score_matrix[row_ind, col_ind].sum() / n_t # Normaliza pelo template

        # Guarda a melhor transformação encontrada *nesta iteração*
        best_rotation_iter = rotation
        best_translation_iter = translation
        best_alignment_iter = new_alignment
        best_pc_score_iter = current_pc_score # Guarda o score da iteração atual

        new_alignment_set = set(new_alignment)
        if new_alignment_set == last_alignment_set:
            # print(f"INFO: Alinhamento convergiu na iteração {iteration + 1}.")
            break # Convergiu

        current_alignment = new_alignment
        last_alignment_set = new_alignment_set
    # Fim do loop de iteração

    if not best_alignment_iter:
         print("ERRO: Nenhum alinhamento encontrado para este confôrmero.")
         return None, np.nan, np.nan

    # --- Aplica a MELHOR transformação encontrada ao confôrmero da query ---
    final_query_mol = copy.deepcopy(query_mol) # Copia a molécula original
    final_conf = final_query_mol.GetConformer(query_conf_id) # Pega o confôrmero certo

    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = best_rotation_iter
    transform_matrix[:3, 3] = best_translation_iter
    TransformConformer(final_conf, transform_matrix) # Modifica o confôrmero na cópia

    # Calcular RMSD final usando os átomos pesados do MELHOR alinhamento
    q_indices_final = [q_heavy_indices[pair[0]] for pair in best_alignment_iter]
    t_indices_final = [t_heavy_indices[pair[1]] for pair in best_alignment_iter]

    final_rmsd = np.nan
    if len(q_indices_final) > 0:
        try:
            atom_map = list(zip(q_indices_final, t_indices_final))
            # Calcula RMSD entre o confôrmero transformado e o template original
            final_rmsd = rdMolAlign.AlignMol(final_query_mol, template_mol, confId=query_conf_id, refConfId=-1, atomMap=atom_map)
        except Exception:
            # Tenta cálculo manual se AlignMol falhar
            q_coords_final_aligned = final_conf.GetPositions()[q_indices_final]
            t_coords_original_aligned = t_coords_all[t_indices_final]
            if len(q_coords_final_aligned) == len(t_coords_original_aligned):
                 diff = q_coords_final_aligned - t_coords_original_aligned
                 final_rmsd = np.sqrt(np.sum(diff * diff) / len(q_coords_final_aligned))

    # Retorna a molécula com o confôrmero transformado, o PC-score e o RMSD
    return final_query_mol, best_pc_score_iter, final_rmsd


# --- Execução Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Alinha uma molécula query a uma molécula template usando LS-align (Rígido ou Flexível) com PC-score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-f", "--reference", required=True, help="Arquivo da molécula de referência/template (.sdf, .mol, .pdb)")
    parser.add_argument("-i", "--query", required=True, help="Arquivo da molécula query (.sdf, .mol, .pdb)")
    parser.add_argument("-o", "--output", required=True, help="Arquivo de saída para a molécula query alinhada (.sdf)")
    parser.add_argument("--flexible", action='store_true', help="Ativar modo de alinhamento flexível (gera confôrmeros para query)")
    parser.add_argument("--num_conformers", type=int, default=DEFAULT_NUM_CONFORMERS, help="Número de confôrmeros a gerar/testar no modo flexível")
    parser.add_argument("-n", "--n_confs", type=int, default=1, help="(Ignorado) Número de conformações a salvar na saída")

    args = parser.parse_args()

    start_time = time.time()

    # Carregar moléculas
    print(f"Carregando referência: {args.reference}")
    template_mol = load_molecule(args.reference)
    print(f"Carregando query: {args.query}")
    query_mol_original = load_molecule(args.query) # Guarda a original

    if not query_mol_original or not template_mol:
        print("\nNão foi possível carregar as moléculas. Verifique os arquivos de entrada.")
        exit(1)

    best_aligned_mol = None
    best_pc_score = -np.inf
    best_rmsd = np.inf
    conf_count = 0

    if args.flexible:
        print("\n--- Iniciando Modo de Alinhamento Flexível ---")
        query_mol_conformers = generate_conformers_rdkit(query_mol_original, args.num_conformers)

        if not query_mol_conformers or query_mol_conformers.GetNumConformers() == 0:
            print("ERRO: Falha ao gerar confôrmeros para alinhamento flexível. Saindo.")
            exit(1)

        conf_ids = [conf.GetId() for conf in query_mol_conformers.GetConformers()]
        conf_count = len(conf_ids)
        print(f"INFO: Iniciando alinhamento para {conf_count} confôrmeros...")

        for i, conf_id in enumerate(conf_ids):
            print(f"\n--- Alinhando Confôrmero {i+1}/{conf_count} (ID: {conf_id}) ---")
            # Passa a molécula com múltiplos confôrmeros, mas especifica qual alinhar
            aligned_mol_conf, pc_score_conf, rmsd_conf = align_single_conformer_ls_align(
                query_mol_conformers, template_mol, query_conf_id=conf_id
            )

            if aligned_mol_conf is not None and not np.isnan(pc_score_conf):
                print(f"Resultado Conf. {i+1}: PC-score={pc_score_conf:.6f}, RMSD={rmsd_conf:.4f}")
                if pc_score_conf > best_pc_score:
                    best_pc_score = pc_score_conf
                    best_rmsd = rmsd_conf
                    # Guarda a molécula inteira, mas saberemos qual confôrmero foi o melhor
                    # Precisamos extrair/salvar APENAS o melhor confôrmero alinhado
                    best_aligned_mol = copy.deepcopy(aligned_mol_conf) # Copia a molécula com o confôrmero já alinhado
                    # Opcional: remover outros confôrmeros se existirem na cópia
                    all_cids = [c.GetId() for c in best_aligned_mol.GetConformers()]
                    for cid in all_cids:
                         if cid != conf_id:
                             best_aligned_mol.RemoveConformer(cid)

                    print(f"  * Novo melhor resultado global encontrado!")
            else:
                print(f"Falha ao alinhar confôrmero {i+1}.")

    else:
        # Modo Rígido (comportamento anterior)
        print("\n--- Iniciando Modo de Alinhamento Rígido ---")
        conf_count = 1
        aligned_mol, pc_score, rmsd = align_single_conformer_ls_align(
            query_mol_original, template_mol
        )
        if aligned_mol is not None:
            best_aligned_mol = aligned_mol
            best_pc_score = pc_score
            best_rmsd = rmsd

    # --- Resultados Finais ---
    end_time = time.time()
    print(f"\n--- Processo Concluído ---")
    print(f"Modo: {'Flexível' if args.flexible else 'Rígido'}")
    print(f"Confôrmeros testados: {conf_count}")
    print(f"Tempo total: {end_time - start_time:.2f} segundos")

    if best_aligned_mol:
        # Salvar a melhor molécula alinhada encontrada
        try:
            writer = Chem.SDWriter(args.output)
            pc_score_str = f"{best_pc_score:.6f}" if not np.isnan(best_pc_score) else "N/A"
            rmsd_str = f"{best_rmsd:.4f}" if not np.isnan(best_rmsd) else "N/A"

            best_aligned_mol.SetDoubleProp("LSalign_PC_Score", best_pc_score)
            best_aligned_mol.SetDoubleProp("LSalign_RMSD", best_rmsd)
            # Tenta pegar o nome original da query
            original_name = query_mol_original.GetProp('_Name') if query_mol_original.HasProp('_Name') else 'Query'
            best_aligned_mol.SetProp("_Name", f"LSaligned_{original_name}_PC_{pc_score_str}_RMSD_{rmsd_str}")

            writer.write(best_aligned_mol) # Salva a molécula com o único melhor confôrmero
            writer.close()
            print(f"\nMelhor molécula query alinhada salva em: {args.output}")
            print(f"  Melhor PC-score: {pc_score_str}")
            print(f"  RMSD correspondente: {rmsd_str} Å")

        except Exception as e:
            print(f"ERRO ao salvar o arquivo de saída {args.output}: {e}")

        # Bloco de visualização (opcional)
        try:
             import py3Dmol
             view = py3Dmol.view(width=600, height=400)
             view.addModel(Chem.MolToMolBlock(template_mol), 'mol')
             view.setStyle({'model': 0}, {'stick': {'colorscheme': 'cyanCarbon'}})
             view.addModel(Chem.MolToMolBlock(best_aligned_mol), 'mol')
             view.setStyle({'model': 1}, {'stick': {'colorscheme': 'magentaCarbon'}})
             view.zoomTo()
             html_file = os.path.splitext(args.output)[0] + "_view.html"
             view.write_html(html_file)
             print(f"\nVisualização 3D salva em: {html_file} (Template=ciano, Query Alinhada=magenta)")
        except ImportError:
             print("\n(Opcional) Para visualização 3D automática, instale py3Dmol: pip install py3Dmol")
        except Exception as e_vis:
             print(f"\nAVISO: Falha ao gerar visualização 3D: {e_vis}")

    else:
         print("\nAlinhamento LS-align falhou. Nenhuma molécula de saída foi gerada.")
         exit(1)

