"""Command-line interface for PaperTuner."""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from papertuner.config import Config, DEFAULT_MODEL_NAME, DEFAULT_LORA_RANK, DEFAULT_MAX_SEQ_LENGTH
from papertuner.dataset import ResearchPaperProcessor, DatasetManager
from papertuner.train import ResearchAssistantTrainer

# Create the Typer app
app = typer.Typer(
    name="papertuner",
    help="Create research assistant models using academic papers.",
    add_completion=False,
)

# Create console for rich output
console = Console()

# Create subcommands
dataset_app = typer.Typer(help="Manage paper datasets")
train_app = typer.Typer(help="Train and manage models")
app.add_typer(dataset_app, name="dataset")
app.add_typer(train_app, name="train")

def format_duration(seconds):
    """Format seconds into readable time duration."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

@app.command("version")
def version():
    """Display the current PaperTuner version."""
    import papertuner
    console.print(f"[bold green]PaperTuner[/bold green] version: [bold]{papertuner.__version__}[/bold]")

@app.command("setup")
def setup(
    output_dir: str = typer.Option(
        str(Path.home() / ".papertuner"),
        help="Directory to store PaperTuner data"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        help="API key for LLM service (OpenAI, Anthropic, etc.)"
    ),
    api_type: str = typer.Option(
        "openai",
        help="Type of API to use (openai, anthropic, gemini)"
    ),
    hf_token: Optional[str] = typer.Option(
        None,
        help="Hugging Face token for uploading datasets and models"
    ),
):
    """Set up PaperTuner environment."""
    # Create config
    config_dir = Path(output_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Create config object
    config = Config()
    
    # Update with provided values
    config.set("data_dir", str(config_dir / "data"))
    
    # Setup API keys based on type
    if api_key:
        if api_type.lower() == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
            console.print("[bold green]Set OpenAI API key[/bold green]")
        elif api_type.lower() == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = api_key
            console.print("[bold green]Set Anthropic API key[/bold green]")
        elif api_type.lower() == "gemini":
            os.environ["GEMINI_API_KEY"] = api_key
            console.print("[bold green]Set Gemini API key[/bold green]")
    
    # Setup Hugging Face token
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        console.print("[bold green]Set Hugging Face token[/bold green]")
    
    # Save config
    config_path = config_dir / "config.json"
    config.save(str(config_path))
    
    console.print(f"[bold green]PaperTuner setup complete![/bold green]")
    console.print(f"Configuration saved to: [bold]{config_path}[/bold]")
    console.print(f"Data directory: [bold]{config.get('data_dir')}[/bold]")

@dataset_app.command("create")
def create_dataset(
    search_query: str = typer.Argument(..., help="Search query for papers"),
    max_papers: int = typer.Option(20, help="Maximum number of papers to process"),
    output_name: str = typer.Option("research_dataset", help="Name of the output dataset"),
    push_to_hub: bool = typer.Option(False, help="Push dataset to Hugging Face Hub"),
    hub_repo_id: Optional[str] = typer.Option(None, help="Hugging Face Hub repository ID"),
    config_file: Optional[str] = typer.Option(None, help="Path to configuration file"),
    llm_provider: str = typer.Option("openai", help="LLM provider (openai, anthropic, gemini)"),
    domain: str = typer.Option("machine_learning", help="Research domain"),
    qa_pairs_per_paper: int = typer.Option(5, help="Number of QA pairs per paper"),
):
    """Create a dataset from research papers."""
    start_time = time.time()
    
    # Load config if provided
    config = Config(config_file) if config_file else Config()
    
    console.print(Panel(f"[bold]Creating dataset from research papers[/bold]\n\nSearch query: {search_query}\nMax papers: {max_papers}", 
                       title="[bold green]PaperTuner Dataset Creator[/bold green]"))
    
    # Configure LLM
    llm_config = config.get("llm_config", {})
    llm_config["provider"] = llm_provider
    
    # Create processor
    processor = ResearchPaperProcessor(
        api_key=os.environ.get(f"{llm_provider.upper()}_API_KEY"),
        hf_token=os.environ.get("HF_TOKEN"),
        hf_repo_id=hub_repo_id or f"user/{output_name}",
        domain_config=config.get("domain_configs", {}).get(domain)
    )
    
    # Process papers
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Processing papers...[/bold green]"),
        console=console
    ) as progress:
        progress.add_task("Processing", total=None)
        results = processor.process_papers(
            max_papers=max_papers,
            search_query=search_query,
            num_qa_pairs=qa_pairs_per_paper
        )
    
    # Show statistics
    stats = processor.generate_statistics()
    
    table = Table(title="Dataset Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Papers", str(stats.get("total_papers", 0)))
    table.add_row("Total QA Pairs", str(stats.get("total_qa_pairs", 0)))
    table.add_row("Average QA Pairs per Paper", f"{stats.get('avg_qa_pairs_per_paper', 0):.1f}")
    table.add_row("Processing Time", format_duration(time.time() - start_time))
    
    console.print(table)
    
    # Push to Hub if requested
    if push_to_hub:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Pushing to Hugging Face Hub...[/bold blue]"),
            console=console
        ) as progress:
            progress.add_task("Pushing", total=None)
            processor.push_to_hf()
        
        console.print(f"[bold green]Dataset pushed to Hugging Face Hub:[/bold green] {processor.hf_repo_id}")
    
    console.print(f"[bold green]Dataset creation complete![/bold green]")

@dataset_app.command("stats")
def dataset_stats(
    dataset_path: str = typer.Argument(..., help="Path to dataset or Hugging Face dataset ID"),
    split: str = typer.Option("train", help="Dataset split to analyze"),
):
    """Display statistics for a dataset."""
    # Create dataset manager
    manager = DatasetManager()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Loading dataset...[/bold green]"),
        console=console
    ) as progress:
        progress.add_task("Loading", total=None)
        stats = manager.analyze_dataset(dataset_path, split)
    
    table = Table(title=f"Dataset Statistics: {dataset_path}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats.items():
        if isinstance(value, (int, str)):
            table.add_row(key.replace("_", " ").title(), str(value))
        elif isinstance(value, float):
            table.add_row(key.replace("_", " ").title(), f"{value:.2f}")
    
    console.print(table)
    
    # Show sample QA pair
    if stats.get("sample_qa_pair"):
        sample = stats["sample_qa_pair"]
        console.print("\n[bold cyan]Sample QA Pair:[/bold cyan]")
        console.print(Panel(f"[bold]Question:[/bold] {sample['question']}\n\n[bold]Answer:[/bold] {sample['answer'][:500]}...", 
                           title="Sample", border_style="green"))

@train_app.command("model")
def train_model(
    dataset: str = typer.Argument(..., help="Dataset to train on (Hugging Face dataset ID)"),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, help="Base model to fine-tune"),
    output_dir: str = typer.Option("./outputs", help="Directory to save model outputs"),
    lora_rank: int = typer.Option(DEFAULT_LORA_RANK, help="Rank for LoRA adapters"),
    max_steps: int = typer.Option(500, help="Maximum training steps"),
    config_file: Optional[str] = typer.Option(None, help="Path to configuration file"),
    checkpoint: Optional[str] = typer.Option(None, help="Path to a checkpoint to resume training from"),
    push_to_hub: bool = typer.Option(False, help="Push model to Hugging Face Hub after training"),
    hub_model_id: Optional[str] = typer.Option(None, help="Model ID for Hugging Face Hub"),
    filter_category: Optional[str] = typer.Option(None, help="Filter dataset by question category"),
    create_demo: bool = typer.Option(False, help="Create a Gradio demo after training"),
    share_demo: bool = typer.Option(False, help="Share the Gradio demo publicly"),
):
    """Train a research assistant model."""
    start_time = time.time()
    
    console.print(Panel(f"[bold]Training Research Assistant Model[/bold]\n\nDataset: {dataset}\nBase Model: {model_name}\nLoRA Rank: {lora_rank}", 
                        title="[bold green]PaperTuner Trainer[/bold green]"))
    
    # Check if push_to_hub is set but no hub_model_id
    if push_to_hub and not hub_model_id:
        console.print("[bold red]Error: --hub-model-id is required when --push-to-hub is set[/bold red]")
        sys.exit(1)
    
    # Create trainer
    trainer = ResearchAssistantTrainer(
        model_name=model_name,
        lora_rank=lora_rank,
        output_dir=output_dir,
        max_steps=max_steps,
        config_file=config_file
    )
    
    # Train the model
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Training model... This may take a while...[/bold green]"),
        console=console
    ) as progress:
        progress.add_task("Training", total=None)
        results = trainer.train(
            dataset_name=dataset,
            checkpoint_path=checkpoint,
            filter_by_category=filter_category
        )
    
    # Display training time
    console.print(f"[bold green]Training completed in {format_duration(time.time() - start_time)}![/bold green]")
    console.print(f"Model saved to: [bold]{results['lora_path']}[/bold]")
    
    # Push to Hub if requested
    if push_to_hub:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Pushing to Hugging Face Hub...[/bold blue]"),
            console=console
        ) as progress:
            progress.add_task("Pushing", total=None)
            trainer.push_to_hf(
                model=results["model"],
                tokenizer=results["tokenizer"],
                repo_id=hub_model_id,
                lora_path=results["lora_path"]
            )
        
        console.print(f"[bold green]Model pushed to Hugging Face Hub:[/bold green] {hub_model_id}")
    
    # Create demo if requested
    if create_demo:
        console.print("[bold blue]Creating Gradio demo...[/bold blue]")
        demo_path = f"{output_dir}/demo"
        trainer.create_gradio_demo(
            model=results["model"],
            tokenizer=results["tokenizer"],
            lora_path=results["lora_path"],
            output_dir=demo_path,
            share=share_demo
        )
        
        console.print(f"[bold green]Demo created at:[/bold green] {demo_path}")
        if share_demo:
            console.print("[bold yellow]Note: The public link will only be active while this program is running.[/bold yellow]")
    
    # Display evaluation results if available
    if results.get("evaluation"):
        eval_results = results["evaluation"]
        
        console.print("\n[bold cyan]Evaluation Results:[/bold cyan]")
        
        # Display semantic similarity
        if "semantic_similarity" in eval_results:
            sem_sim = eval_results["semantic_similarity"]
            console.print(f"Semantic Similarity: Mean = [bold]{sem_sim.get('mean', 0):.4f}[/bold], "
                          f"Min = {sem_sim.get('min', 0):.4f}, Max = {sem_sim.get('max', 0):.4f}")
        
        # Display ROUGE scores
        if "rouge" in eval_results:
            rouge = eval_results["rouge"]
            console.print(f"ROUGE-1: [bold]{rouge.get('rouge1', 0):.4f}[/bold]")
            console.print(f"ROUGE-2: [bold]{rouge.get('rouge2', 0):.4f}[/bold]")
            console.print(f"ROUGE-L: [bold]{rouge.get('rougeL', 0):.4f}[/bold]")
        
        # Display BERTScore if available
        if "bertscore" in eval_results:
            bertscore = eval_results["bertscore"]
            console.print(f"BERTScore F1: [bold]{bertscore.get('f1', 0):.4f}[/bold]")
    
    console.print(f"[bold green]Training workflow complete![/bold green]")

@app.command("demo")
def create_demo(
    model_path: str = typer.Argument(..., help="Path to the model or adapter"),
    base_model: str = typer.Option(DEFAULT_MODEL_NAME, help="Base model name if using adapter"),
    share: bool = typer.Option(False, help="Share the demo with a public URL"),
    port: int = typer.Option(7860, help="Port to run the demo on"),
):
    """Create a demo for a trained model."""
    console.print(Panel(f"[bold]Creating Demo for Model[/bold]\n\nModel: {model_path}", 
                        title="[bold green]PaperTuner Demo Creator[/bold green]"))
    
    # Create trainer
    trainer = ResearchAssistantTrainer(model_name=base_model)
    
    # Load model
    model, tokenizer, peft_model = trainer.load_model()
    
    console.print("[bold blue]Starting Gradio demo...[/bold blue]")
    console.print("The demo will keep running until you press Ctrl+C")
    
    # Create and launch demo
    trainer.create_gradio_demo(
        model=peft_model,
        tokenizer=tokenizer,
        lora_path=model_path,
        share=share
    )
    
    console.print("[bold green]Demo is running![/bold green]")
    
    # Keep the process alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("[bold yellow]Demo stopped.[/bold yellow]")

if __name__ == "__main__":
    app() 