#!/usr/bin/env python3
"""CLI utility for managing and testing the Research Citations MCP Server."""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from src.config import get_settings
from src.rag_engine import RAGEngine
from src.pdf_processor import PDFProcessor

console = Console()


@click.group()
def cli():
    """Research Citations MCP Server CLI."""
    pass


@cli.command()
def status():
    """Show server configuration and status."""
    try:
        settings = get_settings()
        
        table = Table(title="Server Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Papers Directory", str(settings.papers_directory))
        table.add_row("Vector DB Path", str(settings.vector_db_path))
        table.add_row("Collection Name", settings.collection_name)
        table.add_row("Embedding Model", settings.embedding_model)
        table.add_row("LLM Model", settings.llm_model)
        table.add_row("Chunk Size", str(settings.chunk_size))
        table.add_row("Chunk Overlap", str(settings.chunk_overlap))
        table.add_row("Server Host", settings.host)
        table.add_row("Server Port", str(settings.port))
        
        console.print(table)
        
        # Check if papers exist
        pdf_files = list(settings.papers_directory.glob("**/*.pdf"))
        console.print(f"\n📚 Found {len(pdf_files)} PDF files")
        
        # Check if vector store exists
        if settings.vector_db_path.exists():
            console.print(f"✅ Vector store exists at {settings.vector_db_path}")
        else:
            console.print(f"⚠️  Vector store not yet created")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--force", is_flag=True, help="Force rebuild even if index exists")
def build_index(force):
    """Build or rebuild the vector store index."""
    try:
        settings = get_settings()
        console.print(f"[bold]Building vector store index...[/bold]")
        console.print(f"Papers directory: {settings.papers_directory}")
        console.print(f"Force rebuild: {force}")
        
        with console.status("[bold green]Processing PDFs..."):
            rag_engine = RAGEngine()
            rag_engine.initialize_vector_store(force_rebuild=force)
        
        papers = rag_engine.list_papers()
        console.print(f"\n[green]✅ Index built successfully![/green]")
        console.print(f"Total papers indexed: {len(papers)}")
        
        # Show papers
        if papers:
            table = Table(title="Indexed Papers")
            table.add_column("#", style="cyan", width=4)
            table.add_column("Filename", style="green")
            
            for i, paper in enumerate(papers, 1):
                table.add_row(str(i), paper)
            
            console.print(table)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
def list_papers():
    """List all indexed papers."""
    try:
        rag_engine = RAGEngine()
        rag_engine.initialize_vector_store(force_rebuild=False)
        
        papers = rag_engine.list_papers()
        
        if not papers:
            console.print("[yellow]No papers indexed yet. Run 'build-index' first.[/yellow]")
            return
        
        table = Table(title=f"Indexed Papers ({len(papers)})")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Filename", style="green")
        
        for i, paper in enumerate(papers, 1):
            table.add_row(str(i), paper)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option("-k", "--num-results", default=5, help="Number of results")
def search(query, num_results):
    """Search for passages in research papers."""
    try:
        rag_engine = RAGEngine()
        rag_engine.initialize_vector_store(force_rebuild=False)
        
        console.print(f"[bold]Searching for: [cyan]{query}[/cyan][/bold]\n")
        
        with console.status("[bold green]Searching..."):
            results = rag_engine.search_with_scores(query, k=num_results)
        
        if not results:
            console.print("[yellow]No results found.[/yellow]")
            return
        
        for i, result in enumerate(results, 1):
            panel = Panel(
                f"[green]{result['content'][:500]}...[/green]\n\n"
                f"[cyan]File:[/cyan] {result['metadata']['filename']}\n"
                f"[cyan]Page:[/cyan] {result['metadata']['page']}\n"
                f"[cyan]Score:[/cyan] {result['score']:.4f}",
                title=f"Result {i}",
                border_style="blue"
            )
            console.print(panel)
            console.print()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("topic")
@click.option("-k", "--num-citations", default=3, help="Number of citations")
def cite(topic, num_citations):
    """Find citations for a topic."""
    try:
        rag_engine = RAGEngine()
        rag_engine.initialize_vector_store(force_rebuild=False)
        
        console.print(f"[bold]Finding citations for: [cyan]{topic}[/cyan][/bold]\n")
        
        with console.status("[bold green]Searching..."):
            citations = rag_engine.find_citation(topic, k=num_citations)
        
        console.print(f"[green]Found {citations['total_sources']} source(s)[/green]\n")
        
        for filename, cites in citations['citations'].items():
            panel_content = f"[bold]{filename}[/bold]\n\n"
            for cite in cites:
                panel_content += (
                    f"[cyan]Page {cite['page']}[/cyan] (score: {cite['relevance_score']:.4f})\n"
                    f"{cite['excerpt']}\n\n"
                )
            
            console.print(Panel(panel_content, border_style="green"))
            console.print()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.argument("filename")
@click.option("-k", "--num-results", default=5, help="Number of results")
def search_paper(query, filename, num_results):
    """Search within a specific paper only."""
    try:
        rag_engine = RAGEngine()
        rag_engine.initialize_vector_store(force_rebuild=False)
        
        console.print(f"[bold]Searching in: [cyan]{filename}[/cyan][/bold]")
        console.print(f"[bold]Query: [yellow]{query}[/yellow][/bold]\n")
        
        with console.status("[bold green]Searching..."):
            results = rag_engine.search_in_paper(query, filename, k=num_results)
        
        if not results:
            console.print(f"[yellow]No results found in {filename}[/yellow]")
            return
        
        for i, result in enumerate(results, 1):
            panel = Panel(
                f"[green]{result['content'][:500]}...[/green]\n\n"
                f"[cyan]Page:[/cyan] {result['metadata']['page']}\n"
                f"[cyan]Score:[/cyan] {result['score']:.4f}",
                title=f"Result {i}",
                border_style="blue"
            )
            console.print(panel)
            console.print()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("question")
def ask(question):
    """Ask a question about your research papers."""
    try:
        rag_engine = RAGEngine()
        rag_engine.initialize_vector_store(force_rebuild=False)
        
        console.print(f"[bold]Question: [cyan]{question}[/cyan][/bold]\n")
        
        with console.status("[bold green]Thinking..."):
            response = rag_engine.answer_question(question, return_sources=True)
        
        # Print answer
        console.print(Panel(
            response['answer'],
            title="Answer",
            border_style="green"
        ))
        console.print()
        
        # Print sources
        if response.get('sources'):
            console.print("[bold]Sources:[/bold]")
            for i, source in enumerate(response['sources'], 1):
                console.print(f"\n{i}. [cyan]{source['filename']}[/cyan] (Page {source['page']})")
                console.print(f"   {source['excerpt']}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
