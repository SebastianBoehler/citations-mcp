#!/usr/bin/env python3
"""Verify the Research Citations MCP Server setup."""

import sys
from pathlib import Path

def check_env_file():
    """Check if .env file exists."""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("   Run: cp .env.example .env")
        print("   Then edit .env with your settings")
        return False
    print("✅ .env file exists")
    return True


def check_environment():
    """Check environment configuration."""
    try:
        from src.config import get_settings
        settings = get_settings()
        
        # Check papers directory
        if not settings.papers_directory.exists():
            print(f"❌ Papers directory not found: {settings.papers_directory}")
            print(f"   Create it with: mkdir -p {settings.papers_directory}")
            return False
        print(f"✅ Papers directory exists: {settings.papers_directory}")
        
        # Check for PDFs
        pdf_files = list(settings.papers_directory.glob("**/*.pdf"))
        if not pdf_files:
            print(f"⚠️  No PDF files found in {settings.papers_directory}")
            print("   Add some PDF research papers to get started")
        else:
            print(f"✅ Found {len(pdf_files)} PDF file(s)")
        
        # Check OpenAI API key
        if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
            print("❌ OpenAI API key not configured")
            print("   Set OPENAI_API_KEY in .env file")
            return False
        print("✅ OpenAI API key configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking environment: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    required = [
        "fastapi",
        "uvicorn",
        "mcp",
        "langchain",
        "langchain_community",
        "langchain_openai",
        "langchain_chroma",
        "chromadb",
        "pypdf",
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("   Install with: uv sync")
        print("   Or: pip install -e .")
        return False
    
    return True


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Research Citations MCP Server - Setup Verification")
    print("=" * 60)
    
    print("\n📦 Checking Dependencies:")
    deps_ok = check_dependencies()
    
    print("\n📁 Checking Environment Files:")
    env_ok = check_env_file()
    
    if env_ok:
        print("\n⚙️  Checking Configuration:")
        config_ok = check_environment()
    else:
        config_ok = False
    
    print("\n" + "=" * 60)
    
    if deps_ok and env_ok and config_ok:
        print("✅ All checks passed! You're ready to go!")
        print("\nNext steps:")
        print("1. Build the index: uv run python cli.py build-index")
        print("2. Test search: uv run python cli.py search 'your query'")
        print("3. Start server: uv run uvicorn src.main:app --reload")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
