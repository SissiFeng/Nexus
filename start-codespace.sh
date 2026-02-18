#!/bin/bash
# Quick start script for GitHub Codespaces
# Run this if services didn't auto-start

set -e

echo "🚀 Starting Nexus in Codespace..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Function to check if port is in use
check_port() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 || \
    netstat -tuln 2>/dev/null | grep -q ":$1 " || \
    ss -tuln 2>/dev/null | grep -q ":$1 "
}

# Detect workspace directory
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
    pip install --user -e ".[dev]"
fi

# Check/install Node dependencies
echo -e "${BLUE}Checking Node.js dependencies...${NC}"
if [ ! -d "optimization_copilot/web/node_modules" ]; then
    echo "Installing npm packages..."
    (cd optimization_copilot/web && npm install)
fi

# Kill existing processes if any
if check_port 8000; then
    echo -e "${YELLOW}Port 8000 in use, stopping...${NC}"
    lsof -t -i:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 2
fi

if check_port 5173; then
    echo -e "${YELLOW}Port 5173 in use, stopping...${NC}"
    lsof -t -i:5173 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Start Backend
echo ""
echo -e "${BLUE}🔧 Starting Backend (port 8000)...${NC}"
if command -v nexus &> /dev/null; then
    nexus server start --host 0.0.0.0 --port 8000 > /tmp/nexus-backend.log 2>&1 &
else
    python -m optimization_copilot.cli_app.main server start --host 0.0.0.0 --port 8000 > /tmp/nexus-backend.log 2>&1 &
fi
BACKEND_PID=$!

# Wait for backend
echo -n "Waiting for backend"
for i in {1..60}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1 || \
       curl -s http://localhost:8000/api/health >/dev/null 2>&1; then
        echo -e "\n${GREEN}✅ Backend ready at http://localhost:8000${NC}"
        break
    fi
    if ! ps -p $BACKEND_PID > /dev/null 2>&1; then
        echo -e "\n${RED}❌ Backend failed. Check /tmp/nexus-backend.log${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# Start Frontend
echo ""
echo -e "${BLUE}🎨 Starting Frontend (port 5173)...${NC}"
cd optimization_copilot/web
npm run dev -- --host 0.0.0.0 --port 5173 > /tmp/nexus-frontend.log 2>&1 &
FRONTEND_PID=$!
cd ../..

# Wait for frontend
echo -n "Waiting for frontend"
for i in {1..60}; do
    if curl -s http://localhost:5173 >/dev/null 2>&1; then
        echo -e "\n${GREEN}✅ Frontend ready at http://localhost:5173${NC}"
        break
    fi
    if ! ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo -e "\n${RED}❌ Frontend failed. Check /tmp/nexus-frontend.log${NC}"
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
