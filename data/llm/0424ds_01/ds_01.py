import pandas as pd
import re
from collections import Counter


def calculate_physicochemical_properties(seq):
    """Calculate key physicochemical properties for CPP evaluation"""
    seq = seq.upper()
    aa_counts = Counter(seq)

    # Basic counts
    arg_count = aa_counts.get('R', 0)
    lys_count = aa_counts.get('K', 0)
    his_count = aa_counts.get('H', 0)
    asp_count = aa_counts.get('D', 0)
    glu_count = aa_counts.get('E', 0)
    trp_count = aa_counts.get('W', 0)
    phe_count = aa_counts.get('F', 0)
    tyr_count = aa_counts.get('Y', 0)

    # Derived properties
    length = len(seq)
    net_charge = (arg_count + lys_count + his_count * 0.1) - (asp_count + glu_count)
    hydrophobic_count = sum(aa_counts.get(aa, 0) for aa in "AILMFWYV")
    hydrophobicity = hydrophobic_count / length
    aromatic_count = trp_count + phe_count + tyr_count
    cationic_count = arg_count + lys_count + his_count
    anionic_count = asp_count + glu_count

    # Secondary structure indicators
    pro_count = aa_counts.get('P', 0)
    gly_count = aa_counts.get('G', 0)

    return {
        'length': length,
        'arg_count': arg_count,
        'lys_count': lys_count,
        'his_count': his_count,
        'net_charge': net_charge,
        'hydrophobicity': hydrophobicity,
        'aromatic_count': aromatic_count,
        'cationic_count': cationic_count,
        'anionic_count': anionic_count,
        'pro_count': pro_count,
        'gly_count': gly_count,
        'sequence': seq
    }


def evaluate_cpp_rules(properties):
    """Evaluate the peptide against 20 CPP rules"""
    p = properties
    hydrophobic_count = int(p['hydrophobicity'] * p['length']) if 'hydrophobicity' in p else 0



    rules = [
        # Sequence Characteristics (Rules 1-10)
        1 if p['arg_count'] >= 4 else 0,  # Rule 1
        1 if p['lys_count'] > 0 else 0,  # Rule 2
        1 if p['hydrophobicity'] < 0.4 else 0,  # Rule 3
        1 if (0.3 < p['hydrophobicity'] < 0.6) else 0,  # Rule 4 (simplified amphipathicity)
        1 if p['aromatic_count'] == 0 else 0,  # Rule 5
        1 if 8 <= p['length'] <= 30 else 0,  # Rule 6
        1 if (p['pro_count'] > 0 or p['gly_count'] > 0) else 0,  # Rule 7
        1 if (0.8 <= p['cationic_count'] / max(1, hydrophobic_count) <= 1.2) else 0,  # Rule 8 (fixed)
        1 if p['anionic_count'] == 0 else 0,  # Rule 9
        1 if p['his_count'] > 0 else 0,  # Rule 10

        # Physicochemical Properties (Rules 11-15)
        1 if p['net_charge'] >= 4 else 0,  # Rule 11
        1 if (0.2 <= p['hydrophobicity'] <= 0.4) else 0,  # Rule 12
        1 if (p['arg_count'] + p['lys_count']) >= 4 else 0,  # Rule 13 (simplified pI)
        1 if (len(p['sequence']) * 110) < 3000 else 0,  # Rule 14 (MW < 3kDa)
        1 if re.search(r'[RK][FILVWY]', p['sequence']) else 0,  # Rule 15 (PSA proxy)

        # Structural Characteristics (Rules 16-20)
        1 if re.search(r'[RK]{3,}', p['sequence']) else 0,  # Rule 16 (helix proxy)
        1 if not re.search(r'[ED]{3,}', p['sequence']) else 0,  # Rule 17 (β-sheet avoidance)
        1 if re.search(r'P.{1,2}P', p['sequence']) else 0,  # Rule 18 (turn motifs)
        1 if p['sequence'].startswith('A') else 0,  # Rule 19 (N-term mod proxy)
        1 if re.search(r'C.{3,7}C', p['sequence']) else 0  # Rule 20 (cyclization)
    ]

    return ''.join(map(str, rules))


def process_cpp_dataset(input_file, output_file):
    """Process an Excel file of peptide sequences"""
    df = pd.read_excel(input_file)

    # Calculate properties and apply rules
    df['Properties'] = df['Sequence'].apply(calculate_physicochemical_properties)
    df['CPP_Code'] = df['Properties'].apply(evaluate_cpp_rules)

    # Save results
    df.to_excel(output_file, index=False)
    print(f"✅ CPP encoding complete. Results saved to {output_file}")


# Example usage
if __name__ == "__main__":
    input_path = "./CPP924.xlsx"
    output_path = "./CPP_encoded.xlsx"
    process_cpp_dataset(input_path, output_path)