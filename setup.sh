#!/bin/bash

set -e  # Exit on error

echo "🚀 Setting up Terminal Bench Trainer..."

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
MIN_PYTHON="3.11"
TARGET_PYTHON="3.12"

# 1. Install uv if not already installed
echo -e "\n${GREEN}[1/6] Checking for uv...${NC}"
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH - try common locations
    if [ -f "$HOME/.cargo/bin/uv" ]; then
        export PATH="$HOME/.cargo/bin:$PATH"
    elif [ -f "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    
    # Verify uv is now available
    if ! command -v uv &> /dev/null; then
        echo "Error: uv was installed but not found in PATH"
        echo "Please add uv to your PATH manually and re-run this script"
        exit 1
    fi
    echo "✓ uv installed successfully"
else
    echo "✓ uv is already installed"
fi

# 2. Check and install Python 3.12+
echo -e "\n${GREEN}[2/6] Checking Python version (>=${MIN_PYTHON}, prefer ${TARGET_PYTHON})...${NC}"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' || echo "unknown")
    echo "Current Python version: $PYTHON_VERSION"
    echo "This project requires Python ${MIN_PYTHON} or higher (Python ${TARGET_PYTHON} recommended)"
    echo "Installing Python ${TARGET_PYTHON} using uv..."
    uv python install "${TARGET_PYTHON}"
    echo "✓ Python ${TARGET_PYTHON} installed via uv"
else
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✓ Python version $PYTHON_VERSION meets requirements"
fi

if ! uv python find "${TARGET_PYTHON}" >/dev/null 2>&1; then
    echo "Ensuring uv-managed Python ${TARGET_PYTHON} is available..."
    uv python install "${TARGET_PYTHON}"
fi
PYTHON_BIN=$(uv python find "${TARGET_PYTHON}" 2>/dev/null || command -v python3)
echo "Using Python interpreter: ${PYTHON_BIN}"

# 3. Create virtual environment and install dependencies
echo -e "\n${GREEN}[3/6] Setting up virtual environment and installing dependencies...${NC}"
uv venv --python "$PYTHON_BIN"
source .venv/bin/activate
uv pip install -e .

# 4. Create .env file if it doesn't exist
echo -e "\n${GREEN}[4/6] Setting up .env file...${NC}"
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
TINKER_API_KEY="your-api-key-here"
WANDB_API_KEY="your-wandb-api-key-here"
# Tokenizer settings
TOKENIZERS_PARALLELISM=false
EOF
    echo "✓ Created .env file"
    echo -e "${YELLOW}⚠️  Please edit .env and add your API keys!${NC}"
else
    echo "✓ .env file already exists"
fi

# 5. Check for Docker
echo -e "\n${GREEN}[5/6] Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠️  Docker is not installed. Please install Docker to continue.${NC}"
else
    echo "✓ Docker is installed"
    if docker info &> /dev/null; then
        echo "✓ Docker daemon is running"
    else
        echo -e "${YELLOW}⚠️  Docker daemon is not running. Please start Docker.${NC}"
    fi
fi

# 6. Docker login prompt
echo -e "\n${GREEN}[6/6] Docker login...${NC}"
echo "You may need to login to Docker Hub to pull prebuilt images."
read -p "Do you want to run 'docker login' now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker login
fi

# Final instructions
echo -e "\n${GREEN}✅ Setup complete!${NC}"
echo -e "\nNext steps:"
echo -e "  1. Edit .env and add your TINKER_API_KEY and WANDB_API_KEY"
echo -e "  2. Activate the virtual environment: ${GREEN}source .venv/bin/activate${NC}"
echo -e "  3. Add tasks under terminal-bench-2/ (Harbor task format)"
echo -e "  4. Run training (edit flags as needed):"
echo -e "     ${GREEN}python -m src.train \\"
echo -e "       model_name=Qwen/Qwen3-235B-A22B-Instruct-2507 \\"
echo -e "       tasks_dir=./terminal-bench-2 \\"
echo -e "       wandb_project=train-qwen \\"
echo -e "       wandb_name=qwen-run${NC}"
echo -e "\nFor more information, see README.md"

