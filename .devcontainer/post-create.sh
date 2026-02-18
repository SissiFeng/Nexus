#!/bin/bash
set -e

echo "🚀 Setting up Nexus development environment..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
pip install --user -e ".[dev]"

echo -e "${BLUE}📦 Installing Node.js dependencies...${NC}"
cd optimization_copilot/web
npm install
cd ../..

echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo -e "${YELLOW}📝 To start Nexus:${NC}"
echo "   - Backend will auto-start on port 8000"
echo "   - Frontend will auto-start on port 5173"
echo "   - Or run: bash .devcontainer/post-start.sh"
echo ""
echo -e "${YELLOW}📚 Quick Commands:${NC}"
echo "   nexus server start       # Start backend manually"
echo "   cd optimization_copilot/web && npm run dev  # Start frontend manually"
echo "   pytest                   # Run tests"
