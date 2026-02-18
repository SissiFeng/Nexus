#!/bin/bash
set -e

echo "🚀 Starting Nexus services..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

# Function to check if a port is in use
check_port() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 || netstat -tuln 2>/dev/null | grep -q ":$1 " || ss -tuln 2>/dev/null | grep -q ":$1 "
}

# Kill existing processes on the ports if any
if check_port 8000; then
    echo -e "${YELLOW}⚠️  Port 8000 is in use, stopping existing process...${NC}"
    lsof -Pi :8000 -sTCP:LISTEN -t 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 2
fi

if check_port 5173; then
    echo -e "${YELLOW}⚠️  Port 5173 is in use, stopping existing process...${NC}"
    lsof -Pi :5173 -sTCP:LISTEN -t 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Check if Python dependencies are installed
echo -e "${BLUE}🔍 Checking Python dependencies...${NC}"
if ! python -c "import optimization_copilot" 2>/dev/null; then
    echo -e "${YELLOW}📦 Installing Python package...${NC}"
    pip install --user -e ".[dev]"
fi

# Check if Node dependencies are installed
echo -e "${BLUE}🔍 Checking Node.js dependencies...${NC}"
if [ ! -d "optimization_copilot/web/node_modules" ]; then
    echo -e "${YELLOW}📦 Installing npm packages...${NC}"
    (cd optimization_copilot/web && npm install)
fi

# Start Backend
echo ""
echo -e "${BLUE}🔧 Starting Backend Server on port 8000...${NC}"

# Try different methods to start backend
if command -v nexus &> /dev/null; then
    nexus server start --host 0.0.0.0 --port 8000 > /tmp/nexus-backend.log 2>&1 &
else
    python -m optimization_copilot.cli_app.main server start --host 0.0.0.0 --port 8000 > /tmp/nexus-backend.log 2>&1 &
fi

BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait for backend to start
echo -n "Waiting for backend"
BACKEND_READY=false
for i in {1..60}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1 || \
       curl -s http://localhost:8000/api/health >/dev/null 2>&1; then
        echo -e "\n${GREEN}✅ Backend is running on http://localhost:8000${NC}"
        BACKEND_READY=true
        break
    fi
    if ! ps -p $BACKEND_PID > /dev/null 2>&1; then
        echo -e "\n${RED}❌ Backend process died. Check /tmp/nexus-backend.log${NC}"
        cat /tmp/nexus-backend.log
        break
    fi
    echo -n "."
    sleep 1
done

if [ "$BACKEND_READY" = false ]; then
    echo -e "\n${YELLOW}⚠️  Backend may not be fully ready yet. Continuing...${NC}"
fi

# Start Frontend
echo ""
echo -e "${BLUE}🎨 Starting Frontend Dev Server on port 5173...${NC}"

cd optimization_copilot/web
npm run dev -- --host 0.0.0.0 --port 5173 > /tmp/nexus-frontend.log 2>&1 &
FRONTEND_PID=$!
cd ../..

echo "Frontend started with PID: $FRONTEND_PID"

# Wait for frontend to start
echo -n "Waiting for frontend"
FRONTEND_READY=false
for i in {1..60}; do
    if curl -s http://localhost:5173 >/dev/null 2>&1; then
        echo -e "\n${GREEN}✅ Frontend is running on http://localhost:5173${NC}"
        FRONTEND_READY=true
        break
    fi
    if ! ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo -e "\n${RED}❌ Frontend process died. Check /tmp/nexus-frontend.log${NC}"
        cat /tmp/nexus-frontend.log
        break
    fi
    echo -n "."
    sleep 1
done

if [ "$FRONTEND_READY" = false ]; then
    echo -e "\n${YELLOW}⚠️  Frontend may not be fully ready yet.${NC}"
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  🎉 Nexus services started!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}  📊 Backend API:${NC}  http://localhost:8000"
echo -e "${BLUE}  🌐 Frontend:${NC}    http://localhost:5173"
echo -e "${BLUE}  📚 API Docs:${NC}    http://localhost:8000/docs"
echo ""
echo -e "${YELLOW}  📝 View logs:${NC}"
echo "     tail -f /tmp/nexus-backend.log"
echo "     tail -f /tmp/nexus-frontend.log"
echo ""
echo -e "${YELLOW}  🛑 Stop services:${NC}"
echo "     kill \$(lsof -t -i:8000) \$(lsof -t -i:5173) 2>/dev/null || true"
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"

# Keep script running for a bit to show output
sleep 5
