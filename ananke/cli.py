"""Command line interface for timeseries_jarvis."""

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from ananke.core.configs.data.config import (
    create_example_config,
    get_data_configurations,
)
from ananke.core.configs.model.config import (
    create_example_configs,
    get_model_configurations,
)

app = typer.Typer(
    help="timeseries_jarvis - A comprehensive timeseries experimentation framework"
)
console = Console()

# Setup logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("tsj")


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Name of the project to initialize"),
    output_dir: Path = typer.Option(
        Path.cwd(), help="Output directory for the project"
    ),
):
    """Initialize a new TSJ project with example configurations."""
    project_path = output_dir / project_name
    project_path.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    (project_path / "configs" / "data").mkdir(parents=True, exist_ok=True)
    (project_path / "configs" / "model").mkdir(parents=True, exist_ok=True)
    (project_path / "experiments").mkdir(parents=True, exist_ok=True)
    (project_path / "data").mkdir(parents=True, exist_ok=True)
    (project_path / "results").mkdir(parents=True, exist_ok=True)

    # Create example configurations
    create_example_config(str(project_path / "configs" / "data"))
    create_example_configs(str(project_path / "configs" / "model"))

    # Create example experiment script
    experiment_script = project_path / "experiments" / "example_experiment.py"
    experiment_script.write_text("""#!/usr/bin/env python3
\"\"\"Example timeseries experiment using TSJ.\"\"\"

from pathlib import Path
from ananke.core.configs.data.config import get_data_configurations
from ananke.core.configs.model.config import get_model_configurations
from ananke.core.runners.sklearn_runner import SklearnRunner, SklearnRunConfig

def main():
    # Load configurations
    project_root = Path(__file__).parent.parent
    data_configs = get_data_configurations(str(project_root / "configs" / "data"))
    model_configs = get_model_configurations(str(project_root / "configs" / "model"))

    # Create run configuration
    run_config = SklearnRunConfig(
        name="example_experiment",
        output_dir=project_root / "results",
        log_to_mlflow=True,
    )

    # Run experiment
    for data_config in data_configs:
        for model_config in model_configs:
            if model_config.type == "sklearn":
                runner = SklearnRunner(run_config, data_config, model_config)
                results = runner.run()
                print(f"Experiment completed: {results}")

if __name__ == "__main__":
    main()
""")

    # Create README
    readme = project_path / "README.md"
    readme.write_text(f"""# {project_name}

A timeseries project created with Timeseries Jarvis (TSJ).

## Structure

- `configs/`: Configuration files for data and models
- `experiments/`: Experiment scripts
- `data/`: Data files
- `results/`: Experiment results and outputs

## Getting Started

1. Modify the configurations in `configs/` to match your data and models
2. Run the example experiment:
   ```bash
   python experiments/example_experiment.py
   ```

## Configuration

### Data Configuration
Edit `configs/data/default.yaml` to configure your data processing pipeline.

### Model Configuration
Edit files in `configs/model/` to configure your models.

## Running Experiments

Use the TSJ CLI or create custom experiment scripts in the `experiments/` directory.
""")

    console.print(
        f"✅ Project '{project_name}' initialized successfully at {project_path}"
    )
    console.print("📁 Directory structure created")
    console.print("⚙️  Example configurations generated")
    console.print("🧪 Example experiment script created")
    console.print("Project initialized successfully")


@app.command()
def list_configs(
    config_dir: Path = typer.Option(Path("configs"), help="Configuration directory"),
    config_type: str = typer.Option(
        "all", help="Type of configs to list (data, model, all)"
    ),
):
    """List available configurations."""
    if config_type in ["data", "all"]:
        data_dir = config_dir / "data"
        if data_dir.exists():
            console.print("📊 Data Configurations:")
            data_configs = get_data_configurations(str(data_dir))
            for config in data_configs:
                console.print(f"  - {config.name}")
        else:
            console.print("❌ No data configuration directory found")

    if config_type in ["model", "all"]:
        model_dir = config_dir / "model"
        if model_dir.exists():
            console.print("🤖 Model Configurations:")
            model_configs = get_model_configurations(str(model_dir))
            for config in model_configs:
                console.print(f"  - {config.name} ({config.type})")
        else:
            console.print("❌ No model configuration directory found")


@app.command()
def run_experiment(
    experiment_script: Path = typer.Argument(..., help="Path to experiment script"),
    config_dir: Path = typer.Option(Path("configs"), help="Configuration directory"),
    output_dir: Path = typer.Option(Path("results"), help="Output directory"),
):
    """Run a timeseries experiment."""
    if not experiment_script.exists():
        console.print(f"❌ Experiment script not found: {experiment_script}")
        raise typer.Exit(1)

    console.print(f"🚀 Running experiment: {experiment_script}")

    # Execute the experiment script
    import subprocess
    import sys

    try:
        result = subprocess.run(
            [sys.executable, str(experiment_script)], capture_output=True, text=True
        )

        if result.returncode == 0:
            console.print("✅ Experiment completed successfully")
            if result.stdout:
                console.print("📋 Output:")
                console.print(result.stdout)
        else:
            console.print("❌ Experiment failed")
            if result.stderr:
                console.print("🚨 Error:")
                console.print(result.stderr)
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"❌ Error running experiment: {e}")
        raise typer.Exit(1)


# Create validate subcommand group
validate_app = typer.Typer(help="Validate configuration files")
app.add_typer(validate_app, name="validate")

@validate_app.command("data-config")
def validate_data_config(
    config_path: Path = typer.Argument(..., help="Path to data configuration file"),
):
    """Validate a data configuration file."""
    if not config_path.exists():
        console.print(f"❌ Configuration file not found: {config_path}")
        raise typer.Exit(1)

    try:
        import yaml
        from ananke.core.configs.data.config import DataConfig

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        config = DataConfig(**config_data)
        console.print(f"✅ Data configuration is valid: {config.name}")

    except Exception as e:
        console.print(f"❌ Validation failed: {e}")
        raise typer.Exit(1)

@validate_app.command("model-config")
def validate_model_config(
    config_path: Path = typer.Argument(..., help="Path to model configuration file"),
):
    """Validate a model configuration file."""
    if not config_path.exists():
        console.print(f"❌ Configuration file not found: {config_path}")
        raise typer.Exit(1)

    try:
        import yaml
        from ananke.core.configs.model.config import ModelConfig

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        config = ModelConfig(**config_data)
        console.print(f"✅ Model configuration is valid: {config.name}")

    except Exception as e:
        console.print(f"❌ Validation failed: {e}")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    try:
        from ananke import __version__
    except ImportError:
        __version__ = "0.1.0"

    console.print(f"timeseries_jarvis (TSJ) version {__version__}")


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
