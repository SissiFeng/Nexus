#!/bin/bash
set -e

echo "🚀 Starting Nexus services..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a port is in use
check_port() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

# Kill existing processes on the ports if any
if check_port 8000; then
    echo -e "${YELLOW}⚠️  Port 8000 is in use, stopping existing process...${NC}"
    lsof -Pi :8000 -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
fi

if check_port 5173; then
    echo -e "${YELLOW}⚠️  Port 5173 is in use, stopping existing process...${NC}"
    lsof -Pi :5173 -sTCP:LISTEN -t | xargs kill -9 2>/dev/null || true
fi

sleep 1

echo -e "${BLUE}🔧 Starting Backend Server on port 8000...${NC}"

# Start backend in background
(cd /workspaces/Nexus && nexus server start --host 0.0.0.0 --port 8000 > /tmp/nexus-backend.log 2>&1 &) || \
(cd /workspaces/optimization-copilot && nexus server start --host 0.0.0.0 --port 8000 > /tmp/nexus-backend.log 2>&1 &) || \
python -m optimization_copilot.cli_app.main server start --host 0.0.0.0 --port 8000 > /tmp/nexus-backend.log 2>&1 &

# Wait for backend to start
echo -n "Waiting for backend"
for i in {1..30}; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1 || curl -s http://localhost:8000/api/health >/dev/null 2>&1; then
        echo -e "\n${GREEN}✅ Backend is running on http://localhost:8000${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

# Check if backend started
if ! curl -s http://localhost:8000/health >/dev/null 2>&1 && ! curl -s http://localhost:8000/api/health >/dev/null 2>&1; then
    echo -e "\n${RED}❌ Backend failed to start. Check /tmp/nexus-backend.log${NC}"
    echo "Trying alternative startup method..."
    
    # Try alternative method - direct Python
    cd /workspaces/Nexus 2>/dev/null || cd /workspaces/optimization-copilot
    python -c "from optimization_copilot.api.app import create_app; app = create_app(); import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')" > /tmp/nexus-backend.log 2>&1 &
    
    sleep 5
fi

echo ""
echo -e "${BLUE}🎨 Starting Frontend Dev Server on port 5173...${NC}"

# Find the web directory
WEB_DIR=""
if [ -d "/workspaces/Nexus/optimization_copilot/web" ]; then
    WEB_DIR="/workspaces/Nexus/optimization_copilot/web"
elif [ -d "/workspaces/optimization-copilot/optimization_copilot/web" ]; then
    WEB_DIR="/workspaces/optimization-copilot/optimization_copilot/web"
elif [ -d "optimization_copilot/web" ]; then
    WEB_DIR="optimization_copilot/web"
fi

if [ -n "$WEB_DIR" ]; then
    cd "$WEB_DIR"
    npm run dev -- --host 0.0.0.0 --port 5173 > /tmp/nexus-frontend.log 2>&1 &
    
    # Wait for frontend to start
    echo -n "Waiting for frontend"
    for i in {1..30}; do
        if curl -s http://localhost:5173 >/dev/null 2>&1; then
            echo -e "\n${GREEN}✅ Frontend is running on http://localhost:5173${NC}"
            break
        fi
        echo -n "."
        sleep 1
    done
else
    echo -e "${RED}❌ Could not find web directory${NC}"
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  🎉 Nexus is now running!${NC}"
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
echo "     kill \$(lsof -t -i:8000) \$(lsof -t -i:5173)"
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"

# Keep script running to keep Codespace alive
sleep 5
