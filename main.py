"""
ResearchForge-ECRM - Main CLI Entrypoint

Usage:
  python main.py run           # Run research loop on synthetic task
  python main.py benchmark     # Run RDE-Bench
  python main.py api           # Start REST API server
  python main.py demo          # Quick demo with visualization
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from config.settings import settings

console = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("researchforge")


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug: bool) -> None:
    """ResearchForge-ECRM: Autonomous Model Evolution & Algorithm Discovery"""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    settings.ensure_dirs()


@cli.command()
@click.option("--problem", default="Improve image classification accuracy on CIFAR-10",
              help="Research problem description")
@click.option("--iterations", default=10, help="Number of research iterations")
@click.option("--mock/--no-mock", default=True, help="Use mock experiments (fast demo)")
def run(problem: str, iterations: int, mock: bool) -> None:
    """Run the ResearchForge-ECRM research loop."""
    from agents.controller_agent import ResearchController
    from ecrm.memory_store import ECRMMemoryStore
    from rdg.graph import ResearchDevelopmentGraph
    from rdg.nodes import RDGNode
    from rdg.edges import EdgeRelation
    from agents.manuscript_agent import ManuscriptAgent

    console.print(Panel(
        f"[bold cyan]ResearchForge-ECRM[/bold cyan]\n"
        f"Problem: [yellow]{problem}[/yellow]\n"
        f"Iterations: {iterations} | Mock: {mock}",
        title="🔬 Starting Research",
        border_style="cyan",
    ))

    rdg = ResearchDevelopmentGraph()
    memory = ECRMMemoryStore()

    # Seed
    problem_node = RDGNode.problem(content=problem)
    rdg.add_node(problem_node)
    gap_node = RDGNode.gap(content=f"Gap: Current approaches underperform. How to improve?")
    rdg.add_node(gap_node)
    rdg.connect(problem_node.id, gap_node.id, EdgeRelation.IDENTIFIES, validate=False)

    controller = ResearchController(
        rdg=rdg,
        memory=memory,
        problem_description=problem,
        use_mock_experiments=mock,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running research iterations...", total=iterations)
        
        original_step = controller._research_step

        def tracked_step():
            result = original_step()
            progress.advance(task)
            console.print(
                f"  Iter {result['iteration']:2d}: "
                f"score=[green]{result['score']:.4f}[/green] "
                f"best=[bold]{result['best_score']:.4f}[/bold] "
                f"{'✓' if result['success'] else '✗'} "
                f"[dim]{result.get('failure_category', '')}[/dim]"
            )
            return result

        controller._research_step = tracked_step
        summary = controller.run(n_iterations=iterations)

    # Results table
    table = Table(title="Research Summary", border_style="green")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")
    table.add_row("Best Score", f"{summary['best_score']:.4f}")
    table.add_row("Total Experiments", str(summary['total_experiments']))
    table.add_row("Success Rate", f"{summary['success_rate']:.1%}")
    table.add_row("RDG Nodes", str(summary['rdg_stats']['total_nodes']))
    table.add_row("Memory Records", str(summary['memory_stats']['total_records']))
    table.add_row("Global NTR", f"{summary['memory_stats']['global_ntr']:.3f}")
    console.print(table)

    # Generate mini manuscript
    manuscript_agent = ManuscriptAgent()
    manuscript = manuscript_agent.write_summary(rdg, memory, problem, summary["best_score"])
    summary_path = Path("./research_summary.md")
    summary_path.write_text(manuscript)
    console.print(f"\n[green]✓ Research summary saved to {summary_path}[/green]")


@cli.command()
@click.option("--tasks", default="digits,synthetic", help="Comma-separated task names")
@click.option("--iterations", default=15, help="Iterations per task")
@click.option("--mock/--no-mock", default=True)
def benchmark(tasks: str, iterations: int, mock: bool) -> None:
    """Run the RDE-Bench benchmark suite."""
    from benchmarks.evaluator import BenchmarkEvaluator

    task_list = [t.strip() for t in tasks.split(",")]
    console.print(Panel(
        f"Tasks: [yellow]{', '.join(task_list)}[/yellow] | Iterations: {iterations}",
        title="📊 RDE-Bench",
        border_style="yellow",
    ))
    evaluator = BenchmarkEvaluator(tasks=task_list, n_iterations=iterations, mock=mock)
    evaluator.run()


@cli.command()
@click.option("--host", default="0.0.0.0", help="API host")
@click.option("--port", default=8000, help="API port")
def api(host: str, port: int) -> None:
    """Start the ResearchForge REST API server."""
    try:
        import uvicorn
        from api.main import app
        console.print(Panel(
            f"API running at [cyan]http://{host}:{port}[/cyan]\n"
            f"Docs: http://{host}:{port}/docs",
            title="🚀 ResearchForge API",
            border_style="blue",
        ))
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install uvicorn[/red]")


@cli.command()
def demo() -> None:
    """Quick end-to-end demo using synthetic task."""
    console.print(Panel(
        "[bold]ResearchForge-ECRM Quick Demo[/bold]\n"
        "Running 5 iterations on synthetic time-series task.",
        title="🎯 Demo",
        border_style="magenta",
    ))

    from agents.controller_agent import ResearchController
    from ecrm.memory_store import ECRMMemoryStore
    from rdg.graph import ResearchDevelopmentGraph
    from rdg.nodes import RDGNode
    from rdg.edges import EdgeRelation
    from benchmarks.tasks.synthetic_task import SyntheticTimeSeriesTask
    from benchmarks.metrics import compute_all_metrics

    task = SyntheticTimeSeriesTask(mock=True)
    rdg = ResearchDevelopmentGraph()
    memory = ECRMMemoryStore()

    problem_node = RDGNode.problem(content=task.description())
    rdg.add_node(problem_node)
    gap_node = RDGNode.gap(content="Gap: Current model achieves 50% score. Improve it.")
    rdg.add_node(gap_node)
    rdg.connect(problem_node.id, gap_node.id, EdgeRelation.IDENTIFIES, validate=False)

    ctrl = ResearchController(rdg=rdg, memory=memory, use_mock_experiments=True)
    summary = ctrl.run(n_iterations=5)

    perf = [h["score"] for h in ctrl.history]
    metrics = compute_all_metrics(
        performance_history=perf,
        compute_costs=[1.0] * len(perf),
        failure_log=[{"type": h["failure_category"], "context_hash": h["hypothesis"][:20]}
                     for h in ctrl.history if not h["success"]],
        memory_uses=[(True, s - task.baseline_score) for s in perf],
        claims=[{"supported_by": [n.id]} for n in rdg.claims],
    )

    console.print("\n[bold green]Demo Results:[/bold green]")
    for k, v in metrics.items():
        console.print(f"  {k}: {v:.4f}")
    console.print(f"\n  Best Score: [bold]{summary['best_score']:.4f}[/bold]")
    console.print("[green]✓ Demo complete![/green]")


@cli.command("real-demo")
@click.option("--iterations", default=6, show_default=True, help="Real train/evaluate iterations")
def real_demo(iterations: int) -> None:
    """Run a reproducible, network-free training demonstration on sklearn digits."""
    from agents.controller_agent import ResearchController
    from benchmarks.tasks import DigitsTask
    from ecrm.memory_store import ECRMMemoryStore
    from rdg.edges import EdgeRelation
    from rdg.graph import ResearchDevelopmentGraph
    from rdg.nodes import RDGNode

    task = DigitsTask(seed=42)
    rdg, memory = ResearchDevelopmentGraph(), ECRMMemoryStore()
    problem = RDGNode.problem(task.description())
    gap = RDGNode.gap("Gap: establish a reliable small-data classifier.")
    rdg.add_node(problem)
    rdg.add_node(gap)
    rdg.connect(problem.id, gap.id, EdgeRelation.IDENTIFIES, validate=False)
    result = ResearchController(rdg, memory, problem_description=task.description(),
                                use_mock_experiments=False, task=task).run(iterations)
    console.print(Panel(
        f"Best validation accuracy: [bold green]{result['best_score']:.4f}[/bold green]\n"
        f"Experiments: {result['total_experiments']} | Memory records: {result['memory_stats']['total_records']}",
        title="Real offline training complete", border_style="green"))


if __name__ == "__main__":
    cli()
