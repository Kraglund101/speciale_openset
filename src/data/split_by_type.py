"""
Split raw JSON annotations by anomaly type across all datasets.

Reads all dataset_info.json files from AnomVerse_data, groups anomaly instances
by SEMANTIC category (not just exact name), and writes per-concept JSON files.

Semantic grouping ensures all variations of a concept (e.g., scratch, scratches,
scratch_neck) are grouped together for learning a single concept token.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# SEMANTIC GROUPING CONFIGURATION
# =============================================================================

# Map original defect names to semantic concept groups.
# A valid "concept" must be a GENERALIZABLE VISUAL PATTERN that can appear
# across different objects/materials (e.g., "scratch" appears on metal, plastic, fabric).
# Domain-specific defects that don't generalize are EXCLUDED.

# Each concept has: description, members list
SEMANTIC_GROUPS: Dict[str, Dict[str, Any]] = {
    'scratch': {
        'description': 'Linear surface damage - scratches, scuffs, abrasion marks',
        'members': ['scratch', 'scratches', 'scratch_head', 'scratch_neck'],
    },
    'crack': {
        'description': 'Fracture lines in material - cracks, fissures, splits in surface',
        'members': ['crack', 'weft_crack'],
    },
    'broken': {
        'description': 'Structural breakage - fractured, snapped, or shattered parts',
        'members': [
            'broken', 'break', 'broken_bean', 'broken_end', 'broken_inside',
            'broken_large', 'broken_outside', 'broken_pick', 'broken_small',
            'broken_teeth', 'broken_yarn'
        ],
    },
    'cut': {
        'description': 'Sharp incisions - cuts, slices, severed material',
        'members': ['cut', 'cut_inner_insulation', 'cut_lead', 'cut_outer_insulation', 'cut_selvage'],
    },
    'bent': {
        'description': 'Deformation - bent, warped, or twisted shapes',
        'members': ['bent', 'bent_lead', 'bent_wire', 'deformation'],
    },
    'contamination': {
        'description': 'Foreign material on surface - particles, debris, unwanted substances',
        'members': ['contamination', 'metal_contamination', 'foreign_body'],
    },
    'stain': {
        'description': 'Discoloration - spots, marks, color anomalies, darkening',
        'members': [
            'stain', 'spot', 'color',
            'black_bean', 'black_embryo',
            'defective_painting',
            'pill_type',  # miscoloring
        ],
    },
    'rust': {
        'description': 'Oxidation/corrosion - rust spots, metal degradation',
        'members': ['major_rust', 'total_rust'],
    },
    'hole': {
        'description': 'Punctures/perforations - holes, pits, cavities in material',
        'members': ['hole', 'perforation', 'poke', 'poke_insulation', 'blowhole'],
    },
    'thread': {
        'description': 'Loose/exposed threads - fraying, unraveling, thread protrusion',
        'members': ['thread', 'thread_side', 'thread_top', 'fray'],
    },
    'mold': {
        'description': 'Organic growth - mold, mildew, fungal contamination',
        'members': ['mold', 'moldy_bean'],
    },
    'glue': {
        'description': 'Adhesive issues - excess glue, glue residue, adhesive marks',
        'members': ['glue', 'glue_strip'],
    },
    'dent': {
        'description': 'Compression damage - dents, indentations, squeeze marks',
        'members': ['squeeze', 'squeezed_teeth', 'compression', 'indentation'],
    },
    'print_defect': {
        'description': 'Printing/marking errors - smudges, misprints, ink issues',
        'members': ['print', 'misprint', 'faulty_imprint', 'gray_stroke'],
    },
    'split': {
        'description': 'Tearing/separation - splits, tears, material separation',
        'members': ['split', 'split_teeth'],
    },
    'crease': {
        'description': 'Folds/wrinkles - creases, folding marks, surface waves',
        'members': ['crease', 'fold', 'wrinkling', 'weft_curling'],
    },
    'rough': {
        'description': 'Surface texture anomalies - roughness, bumpy texture, pilling',
        'members': ['rough', 'uneven', 'nep', 'fuzzyball', 'knots', 'warp_ball'],
    },
    'protrusion': {
        'description': 'Bumps/raised areas - protrusions, bulges, growths',
        'members': ['protrusion', 'sprouting'],
    },
    'liquid': {
        'description': 'Liquid contamination - spills, wet spots, oil marks',
        'members': ['liquid', 'oil'],
    },
    'shrink': {
        'description': 'Shrinkage/shriveling - dried out, shrunken, peeling material',
        'members': ['shriveling', 'peeling'],
    },
    'fragment': {
        'description': 'Fragmentation - broken into pieces, chips, debris',
        'members': ['fragmentation'],
    },
    'missing': {
        'description': 'Missing parts - absent components, gaps, incomplete structures',
        'members': ['missing_cable', 'missing_wire'],
    },
    'damage': {
        'description': 'Physical damage - heat damage, impact damage, degradation',
        'members': ['damaged_case', 'heat_damage', 'insect_damage'],
    },
    'misalignment': {
        'description': 'Positional errors - misplaced parts, wrong orientation, swapped components',
        'members': ['parts_mismatch', 'bend_and_parts_mismatch', 'misplaced', 'cable_swap', 'flip', 'overlap'],
    },
}

# Defect types to EXCLUDE:
# - Too generic (defect, anomalous)
# - Combinations (combined, multi_anomalies)
# - Domain-specific that don't generalize visually
EXCLUDE_TYPES: Set[str] = {
    # Too generic
    'defect',
    'defective',
    'anomalous',
    'substandard',
    # Combinations
    'combined',
    'multi_anomalies',
    # Not defects
    'free',
    # Domain-specific / not generalizable
    'double_cap',
    'take-all',
    'peaberry',
    'label',
    'open',
    'manipulated_front',
    'fabric_border',
    'fabric_interior',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sanitize_filename(name: str) -> str:
    """Convert defect name to safe filename."""
    safe = re.sub(r'[^\w\-]', '_', name.lower())
    safe = re.sub(r'_+', '_', safe)
    return safe.strip('_')


def build_reverse_mapping() -> Dict[str, str]:
    """Build mapping from original defect name to semantic group."""
    reverse_map = {}
    for group_name, group_info in SEMANTIC_GROUPS.items():
        for member in group_info['members']:
            # Normalize to lowercase for matching
            reverse_map[member.lower()] = group_name
            reverse_map[sanitize_filename(member)] = group_name
    return reverse_map


def get_semantic_group(defect_name: str, reverse_map: Dict[str, str]) -> Optional[str]:
    """Get semantic group for a defect name, or None if excluded/unmapped."""
    normalized = sanitize_filename(defect_name)

    # Check exclusion first
    if normalized in EXCLUDE_TYPES or defect_name.lower() in EXCLUDE_TYPES:
        return None

    # Look up in reverse map
    if normalized in reverse_map:
        return reverse_map[normalized]
    if defect_name.lower() in reverse_map:
        return reverse_map[defect_name.lower()]

    # Not mapped - return None (will be reported as unmapped)
    return None


def find_dataset_jsons(data_root: Path) -> List[Path]:
    """Find all dataset_info.json files in the data directory."""
    return sorted(data_root.rglob('dataset_info.json'))


def load_dataset(json_path: Path) -> Dict[str, Any]:
    """Load a dataset JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_resolved_path(relative_path: str, dataset_root: Path) -> str:
    """Resolve relative path against dataset root, keeping result relative to project root."""
    return str(dataset_root / relative_path)


# =============================================================================
# MAIN SPLITTING LOGIC
# =============================================================================

def split_by_semantic_group(
    data_root: Path,
    output_dir: Path,
    resolve_paths: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Split all datasets by semantic anomaly groups.

    Returns:
        Dictionary with stats: group counts, excluded counts, unmapped types
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    reverse_map = build_reverse_mapping()

    # Collect images by semantic group
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # Track original defect names per group (for metadata)
    group_original_names: Dict[str, Set[str]] = defaultdict(set)
    group_sources: Dict[str, Set[str]] = defaultdict(set)

    # Track excluded and unmapped
    excluded_count = 0
    unmapped_types: Dict[str, int] = defaultdict(int)

    # Find and process all dataset JSONs
    json_files = find_dataset_jsons(data_root)
    print(f"Found {len(json_files)} dataset JSON files\n")

    for json_path in json_files:
        dataset_root = json_path.parent
        dataset = load_dataset(json_path)
        dataset_name = dataset.get('name', dataset_root.name)

        print(f"  Processing: {dataset_name} ({len(dataset.get('images', []))} images)")

        for img in dataset.get('images', []):
            original_defect = img.get('defect_name', '')
            if not original_defect:
                continue

            # Get semantic group
            group = get_semantic_group(original_defect, reverse_map)

            if group is None:
                # Check if excluded or unmapped
                if sanitize_filename(original_defect) in EXCLUDE_TYPES:
                    excluded_count += 1
                else:
                    unmapped_types[original_defect] += 1
                continue

            # Skip entries where image and mask are the same file
            # (e.g. MTD stores .jpg=image and .png=mask in same folder,
            # but dataset_info.json has duplicate entries using .png for both)
            if img['image_path'] == img.get('mask_path'):
                excluded_count += 1
                continue

            # Create enriched entry
            entry = {
                'image_path': img['image_path'],
                'mask_path': img.get('mask_path'),
                'original_defect_name': original_defect,
                'original_defect_code': img.get('defect_code'),
                'category': img.get('category'),
                'split': img.get('split'),
                'analysis_files': img.get('analysis_files'),
                'analysis_files_short': img.get('analysis_files_short'),
                'source_dataset': dataset_name,
                'dataset_root': str(dataset_root),
            }

            if resolve_paths:
                entry['image_path_full'] = make_resolved_path(img['image_path'], dataset_root)
                if img.get('mask_path'):
                    entry['mask_path_full'] = make_resolved_path(img['mask_path'], dataset_root)
                if img.get('analysis_files'):
                    entry['analysis_files_full'] = make_resolved_path(img['analysis_files'], dataset_root)
                if img.get('analysis_files_short'):
                    entry['analysis_files_short_full'] = make_resolved_path(img['analysis_files_short'], dataset_root)

            by_group[group].append(entry)
            group_original_names[group].add(original_defect)
            group_sources[group].add(dataset_name)

    # Write per-group JSON files
    print(f"\nWriting {len(by_group)} semantic group files...")

    group_stats = {}
    for group_name, images in sorted(by_group.items()):
        output_path = output_dir / f"{group_name}.json"

        output_data = {
            'concept_name': group_name,
            'description': SEMANTIC_GROUPS[group_name]['description'],
            'total_instances': len(images),
            'original_defect_names': sorted(group_original_names[group_name]),
            'source_datasets': sorted(group_sources[group_name]),
            'images': images
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        group_stats[group_name] = {
            'count': len(images),
            'original_names': sorted(group_original_names[group_name]),
            'datasets': sorted(group_sources[group_name])
        }

    return {
        'groups': group_stats,
        'excluded_count': excluded_count,
        'unmapped': dict(unmapped_types)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Split anomaly datasets by semantic concept groups'
    )
    parser.add_argument(
        '--input_dir',
        type=Path,
        default=Path('data/AnomVerse_data'),
        help='Root directory containing dataset folders'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('data/concepts'),
        help='Output directory for per-concept JSON files'
    )
    parser.add_argument(
        '--no_resolve',
        action='store_true',
        help='Do not resolve paths against dataset root (keep original relative paths only)'
    )
    args = parser.parse_args()

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    stats = split_by_semantic_group(
        data_root=args.input_dir,
        output_dir=args.output_dir,
        resolve_paths=not args.no_resolve
    )

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_instances = sum(g['count'] for g in stats['groups'].values())
    print(f"Total semantic concepts: {len(stats['groups'])}")
    print(f"Total instances: {total_instances}")
    print(f"Excluded (generic/combined): {stats['excluded_count']}")

    if stats['unmapped']:
        print(f"\nWARNING: {len(stats['unmapped'])} unmapped defect types:")
        for name, count in sorted(stats['unmapped'].items(), key=lambda x: -x[1]):
            print(f"  - {name}: {count} instances")

    print()
    print("Instances per concept (grouped):")
    print("-" * 70)
    for group_name, info in sorted(stats['groups'].items(), key=lambda x: -x[1]['count']):
        original = ', '.join(info['original_names'][:3])
        if len(info['original_names']) > 3:
            original += f" (+{len(info['original_names'])-3} more)"
        print(f"  {group_name:25s} {info['count']:5d}  <- {original}")


if __name__ == '__main__':
    main()
