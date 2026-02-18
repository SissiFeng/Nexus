#!/bin/bash
set -e

echo "🚀 Setting up Nexus development environment..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect workspace directory
if [ -d "/workspaces/Nexus" ]; then
    WORKSPACE_DIR="/workspaces/Nexus"
elif [ -d "/workspaces/optimization-copilot" ]; then
    WORKSPACE_DIR="/workspaces/optimization-copilot"
else
    WORKSPACE_DIR="$(pwd)"
fi

cd "$WORKSPACE_DIR"
echo -e "${BLUE}📁 Working directory: $(pwd)${NC}"

echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
pip install --user -e ".[dev]"

echo -e "${BLUE}📦 Installing Node.js dependencies...${NC}"
cd optimization_copilot/web
npm install
cd ../..

echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo -e "${YELLOW}📝 Nexus will automatically start when the Codespace is ready.${NC}"
echo ""
echo -e "${YELLOW}📚 Quick Commands:${NC}"
echo "   nexus server start --host 0.0.0.0 --port 8000       # Start backend"
echo "   cd optimization_copilot/web && npm run dev -- --host 0.0.0.0  # Start frontend"
echo "   bash start-codespace.sh                             # Start both"
echo "   pytest                                              # Run tests"
echo ""
echo -e "${YELLOW}🌐 After starting, access:${NC}"
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
