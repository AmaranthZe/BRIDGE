"""
Command Line Interface for Cross-Species Gene Mapping.

Commands:
- preprocess: Build sparse matrices and unified graph
- train: Train the model
- infer: Run inference on gene IDs
- disease-report: Generate disease relevance report for query genes
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cmd_preprocess(args):
    """Preprocess data: load CSVs, build sparse matrices, build graph."""
    from .io_schema import load_all_data, save_indexers
    from .build_sparse import build_all_sparse_matrices, save_sparse_matrices
    from .unified_graph import build_unified_graph, save_unified_graph

    config = load_config(args.config)
    proc_dir = Path(config['paths']['proc_dir'])
    proc_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and validating data...")
    dataframes, indexers = load_all_data(config)

    logger.info("Building sparse matrices...")
    sparse_matrices = build_all_sparse_matrices(dataframes, indexers, config)

    logger.info("Building unified graph...")
    graph_data = build_unified_graph(sparse_matrices, indexers)

    logger.info("Saving preprocessed data...")
    save_indexers(indexers, proc_dir / 'indexers.json')
    save_sparse_matrices(sparse_matrices, proc_dir)
    save_unified_graph(graph_data, proc_dir / 'unified_graph.pt')

    logger.info("Preprocessing complete!")

    # Print statistics
    print("\n=== Data Statistics ===")
    print(f"Human genes: {len(indexers['human_genes'])}")
    print(f"Mouse genes: {len(indexers['mouse_genes'])}")
    print(f"BP terms: {len(indexers['bp_terms'])}")
    print(f"MF terms: {len(indexers['mf_terms'])}")
    print(f"Diseases: {len(indexers['diseases'])}")
    print(f"\nSparse matrices saved to: {proc_dir}")


def cmd_train(args):
    """Train the model."""
    import torch
    from .train import train_model

    config = load_config(args.config)

    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    logger.info(f"Using device: {device}")

    # Override config with command line args
    if args.epochs:
        config['train']['epochs'] = args.epochs
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
    if args.lr:
        config['train']['lr'] = args.lr

    # Train
    trainer = train_model(config, device=device)

    logger.info("Training complete!")
    print(f"\nBest model saved to: {trainer.log_dir / 'best_model.pt'}")


def cmd_infer(args):
    """Run inference on gene IDs."""
    import torch
    from .infer import run_inference

    config = load_config(args.config)

    # Determine device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Use best model from default log dir
        log_dir = Path(config.get('logging', {}).get('log_dir', 'runs/default'))
        checkpoint_path = log_dir / 'best_model.pt'

    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Override config with command line args
    if args.topk_genes:
        config['infer']['topk_genes'] = args.topk_genes
    if args.topk_bp:
        config['infer']['topk_bp'] = args.topk_bp
    if args.topk_mf:
        config['infer']['topk_mf'] = args.topk_mf
    if args.topk_disease:
        config['infer']['topk_disease'] = args.topk_disease

    # Run inference
    result = run_inference(
        query=args.ids,
        checkpoint_path=str(checkpoint_path),
        config=config,
        direction=args.direction,
        device=device,
    )

    # Output result
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Results saved to: {args.output}")
    else:
        print(json.dumps(result, indent=2))


def cmd_disease_report(args):
    """Generate disease relevance report for query genes."""
    from .disease_report import run_disease_report

    config = load_config(args.config)

    # Override config with CLI args
    if args.topk_genes:
        config.setdefault('disease_report', {})['topk_func_genes'] = args.topk_genes

    resolve_names = not args.no_resolve_names

    # Determine output path
    output = args.output
    if output is None:
        # Default: <first_gene_id>_disease_report.xlsx
        first_id = args.ids.split(';')[0].strip()
        output = f'{first_id}_disease_report.xlsx'

    logger.info(f"Generating disease report for: {args.ids}")

    result = run_disease_report(
        query=args.ids,
        config=config,
        output=output,
        resolve_names=resolve_names,
    )

    # Print summary to stdout
    for gene_result in result['per_gene']:
        if 'error' in gene_result:
            print(f"ERROR: {gene_result['error']}")
            continue
        q = gene_result['query']
        s = gene_result['summary']
        orth = gene_result['ortholog']
        print(f"\n{'=' * 70}")
        print(f"  Disease Report: {q['species']} {q['symbol']} ({q['gene_id']})")
        print(f"  Ortholog:       {orth['species']} {orth['symbol']} ({orth['gene_id']})")
        print(f"{'=' * 70}")
        print(f"  Total diseases:          {s['total_diseases']}")
        print(f"  Direct associations:     {s['direct_count']}")
        print(f"  Ortholog associations:   {s['ortholog_count']}")
        print(f"  Enrichment diseases:     {s['enrichment_count']}")
        print(f"  Multi-source supported:  {s['multi_source_count']}")
        print(f"{'=' * 70}")

        # Top 15 diseases
        diseases = gene_result['diseases'][:15]
        if diseases:
            print(f"\n  {'Rank':<5}{'Disease':<45}{'Score':<9}{'Src':>3}")
            print(f"  {'-' * 62}")
            for i, d in enumerate(diseases, 1):
                name = d['Disease_Name'] or d['Disease_ID']
                if len(name) > 43:
                    name = name[:40] + '...'
                print(f"  {i:<5}{name:<45}{d['Combined_Relevance']:<9.4f}"
                      f"{d['Num_Sources']:>3}")

    print(f"\nOutput saved to: {output}")


def main():
    parser = argparse.ArgumentParser(
        description='Cross-Species Gene Mapping System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Preprocess data and build graph',
    )
    preprocess_parser.add_argument(
        '--config', '-c',
        default='configs/default.yaml',
        help='Path to configuration file',
    )

    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train the model',
    )
    train_parser.add_argument(
        '--config', '-c',
        default='configs/default.yaml',
        help='Path to configuration file',
    )
    train_parser.add_argument(
        '--device',
        help='Device to use (cuda or cpu)',
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs',
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        dest='batch_size',
        help='Batch size',
    )
    train_parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
    )

    # Infer command
    infer_parser = subparsers.add_parser(
        'infer',
        help='Run inference on gene IDs',
    )
    infer_parser.add_argument(
        '--config', '-c',
        default='configs/default.yaml',
        help='Path to configuration file',
    )
    infer_parser.add_argument(
        '--ids',
        required=True,
        help='Semicolon-separated gene IDs (e.g., "ENSG00000141510;ENSMUSG00000064341")',
    )
    infer_parser.add_argument(
        '--checkpoint',
        help='Path to model checkpoint',
    )
    infer_parser.add_argument(
        '--direction',
        choices=['auto', 'human_to_mouse', 'mouse_to_human', 'bidirectional'],
        default='auto',
        help='Mapping direction',
    )
    infer_parser.add_argument(
        '--device',
        help='Device to use (cuda or cpu)',
    )
    infer_parser.add_argument(
        '--output', '-o',
        help='Output file path (default: stdout)',
    )
    infer_parser.add_argument(
        '--topk-genes',
        type=int,
        dest='topk_genes',
        help='Top-k matched genes to return',
    )
    infer_parser.add_argument(
        '--topk-bp',
        type=int,
        dest='topk_bp',
        help='Top-k BP terms to return',
    )
    infer_parser.add_argument(
        '--topk-mf',
        type=int,
        dest='topk_mf',
        help='Top-k MF terms to return',
    )
    infer_parser.add_argument(
        '--topk-disease',
        type=int,
        dest='topk_disease',
        help='Top-k diseases to return',
    )

    # Disease-report command
    disease_parser = subparsers.add_parser(
        'disease-report',
        help='Generate disease relevance report for query genes',
    )
    disease_parser.add_argument(
        '--config', '-c',
        default='configs/default.yaml',
        help='Path to configuration file',
    )
    disease_parser.add_argument(
        '--ids',
        required=True,
        help='Semicolon-separated Ensembl gene IDs '
             '(e.g., "ENSMUSG00000022100" or "ENSG00000130227")',
    )
    disease_parser.add_argument(
        '--output', '-o',
        help='Output file path (.xlsx or .json). '
             'Default: <gene_symbol>_disease_report.xlsx',
    )
    disease_parser.add_argument(
        '--topk-genes',
        type=int,
        dest='topk_genes',
        help='Top-K functionally similar genes for enrichment '
             '(default: from config, typically 10)',
    )
    disease_parser.add_argument(
        '--no-resolve-names',
        action='store_true',
        dest='no_resolve_names',
        help='Skip resolving disease names via Open Targets API',
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'preprocess':
        cmd_preprocess(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'infer':
        cmd_infer(args)
    elif args.command == 'disease-report':
        cmd_disease_report(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
