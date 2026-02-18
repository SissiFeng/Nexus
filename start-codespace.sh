#!/bin/bash
# Quick start script for GitHub Codespaces
# Run this if services didn't auto-start

set -e

echo "🚀 Starting Nexus in Codespace..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to check if port is in use
check_port() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

# Check current directory and adjust paths
if [ -d "/workspaces/Nexus" ]; then
    cd /workspaces/Nexus
elif [ -d "/workspaces/optimization-copilot" ]; then
    cd /workspaces/optimization-copilot
fi

echo -e "${BLUE}Working directory: $(pwd)${NC}"

# Check/install Python dependencies
echo -e "${BLUE}Checking Python dependencies...${NC}"
if ! python -c "import optimization_copilot" 2>/dev/null; then
    echo "Installing Python package..."
    pip install -e ".[dev]"
fi

# Check/install Node dependencies
echo -e "${BLUE}Checking Node.js dependencies...${NC}"
if [ ! -d "optimization_copilot/web/node_modules" ]; then
    echo "Installing npm packages..."
    cd optimization_copilot/web && npm install && cd ../..
fi

# Start Backend
echo ""
echo -e "${BLUE}🔧 Starting Backend (port 8000)...${NC}"
if check_port 8000; then
    echo -e "${YELLOW}Port 8000 already in use, skipping backend start${NC}"
else
    nexus server start --host 0.0.0.0 --port 8000 &
    echo "Backend starting in background (PID: $!)"
fi

# Wait for backend
echo -n "Waiting for backend"
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1 || \
       curl -s http://localhost:8000/api/health >/dev/null 2>&1; then
        echo -e "\n${GREEN}✅ Backend ready at http://localhost:8000${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# Start Frontend
echo ""
echo -e "${BLUE}🎨 Starting Frontend (port 5173)...${NC}"
if check_port 5173; then
    echo -e "${YELLOW}Port 5173 already in use, skipping frontend start${NC}"
else
    cd optimization_copilot/web
    npm run dev -- --host 0.0.0.0 --port 5173 &
    echo "Frontend starting in background (PID: $!)"
    cd ../..
fi

# Wait for frontend
echo -n "Waiting for frontend"
for i in {1..30}; do
    if curl -s http://localhost:5173 >/dev/null 2>&1; then
        echo -e "\n${GREEN}✅ Frontend ready at http://localhost:5173${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

echo ""
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}  🎉 Nexus is running!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo ""
echo -e "  Frontend:  http://localhost:5173"
echo -e "  Backend:   http://localhost:8000"
echo -e "  API Docs:  http://localhost:8000/docs"
echo ""
echo -e "  View logs: tail -f /tmp/nexus-*.log"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
