#!/usr/bin/env python3
"""
Quick start script for GraphRAG system
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import streamlit
        import langchain
        import neo4j
        print("✅ All requirements are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        return False

def check_environment():
    """Check if environment variables are set"""
    required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GROQ_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set them in your environment or create a .env file")
        return False
    else:
        print("✅ All environment variables are set")
        return True

def install_requirements():
    """Install requirements from requirements.txt"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def run_streamlit():
    """Run the Streamlit application"""
    print("🚀 Starting Streamlit application...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

def main():
    """Main function"""
    print("🔗 GraphRAG System - Quick Start")
    print("=" * 40)
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        sys.exit(1)
    
    # Check and install requirements
    if not check_requirements():
        print("Installing missing requirements...")
        if not install_requirements():
            sys.exit(1)
    
    # Check environment variables
    if not check_environment():
        print("\n📝 Create a .env file with the following variables:")
        print("NEO4J_URI=your-neo4j-uri")
        print("NEO4J_USERNAME=your-username") 
        print("NEO4J_PASSWORD=your-password")
        print("GROQ_API_KEY=your-groq-api-key")
        print("\nOr set them as environment variables")
        sys.exit(1)
    
    # Run the application
    print("\n🎉 Everything looks good! Starting the application...")
    run_streamlit()

if __name__ == "__main__":
    main()