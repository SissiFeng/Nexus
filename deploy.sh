#!/bin/bash
#
# Nexus - One-Click Deployment Script
# 
# This script automatically sets up and starts the Nexus optimization platform.
# 
# Usage:
#   ./deploy.sh              # Interactive setup
#   ./deploy.sh --docker     # Run with Docker (recommended for production)
#   ./deploy.sh --local      # Run locally with Python/Node
#   ./deploy.sh --stop       # Stop all running services
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BACKEND_PORT=${BACKEND_PORT:-8000}
FRONTEND_PORT=${FRONTEND_PORT:-5173}
DOCKER_COMPOSE_FILE="docker-compose.yml"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing=()
    
    if ! command -v git &> /dev/null; then
        missing+=("git")
    fi
    
    if [ "$1" == "docker" ]; then
        if ! command -v docker &> /dev/null; then
            missing+=("docker")
        fi
        if ! command -v docker-compose &> /dev/null; then
            missing+=("docker-compose")
        fi
    else
        if ! command -v python3 &> /dev/null; then
            missing+=("python3")
        fi
        if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
            missing+=("pip")
        fi
        if ! command -v node &> /dev/null; then
            missing+=("node")
        fi
        if ! command -v npm &> /dev/null; then
            missing+=("npm")
        fi
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        log_info "Please install the missing tools and try again."
        exit 1
    fi
    
    log_success "All prerequisites satisfied"
}

# Clone repository if not already in it
clone_or_update() {
    if [ ! -f "pyproject.toml" ]; then
        log_info "Cloning Nexus repository..."
        git clone https://github.com/sissifeng/Nexus.git
        cd Nexus
        log_success "Repository cloned"
    else
        log_info "Already in Nexus directory"
        log_info "Checking for updates..."
        git pull origin main || log_warn "Could not check for updates"
    fi
}

# Setup Python environment
setup_python() {
    log_info "Setting up Python environment..."
    
    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    REQUIRED_VERSION="3.10"
    
    if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
        log_error "Python $REQUIRED_VERSION+ required, found $PYTHON_VERSION"
        exit 1
    fi
    
    # Install package
    log_info "Installing Python package (this may take a few minutes)..."
    pip install -e ".[dev]" --quiet
    
    log_success "Python environment ready"
}

# Setup Node.js environment
setup_node() {
    log_info "Setting up Node.js environment..."
    
    cd optimization_copilot/web
    
    if [ ! -d "node_modules" ]; then
        log_info "Installing npm dependencies (this may take a few minutes)..."
        npm install --silent
    else
        log_info "Node modules already installed"
    fi
    
    cd ../..
    log_success "Node.js environment ready"
}

# Create environment file
setup_env() {
    if [ ! -f ".env" ]; then
        log_info "Creating environment configuration..."
        cp .env.example .env
        log_warn "Please edit .env to add your API keys (optional)"
    fi
}

# Start services locally
start_local() {
    log_info "Starting Nexus in local mode..."
    
    # Check if ports are available
    if lsof -Pi :$BACKEND_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_error "Port $BACKEND_PORT is already in use"
        exit 1
    fi
    
    # Start backend in background
    log_info "Starting backend server on port $BACKEND_PORT..."
    nexus server start --host 0.0.0.0 --port $BACKEND_PORT &
    BACKEND_PID=$!
    
    # Wait for backend to be ready
    log_info "Waiting for backend to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:$BACKEND_PORT/api/health &>/dev/null; then
            break
        fi
        sleep 1
    done
    
    if ! curl -s http://localhost:$BACKEND_PORT/api/health &>/dev/null; then
        log_error "Backend failed to start"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
    
    log_success "Backend running at http://localhost:$BACKEND_PORT"
    
    # Start frontend
    log_info "Starting frontend dev server on port $FRONTEND_PORT..."
    cd optimization_copilot/web
    npm run dev -- --host --port $FRONTEND_PORT &
    FRONTEND_PID=$!
    cd ../..
    
    # Save PIDs for cleanup
    echo "$BACKEND_PID $FRONTEND_PID" > .nexus.pids
    
    log_success "Nexus is running!"
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  🚀 Nexus Platform Ready${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  📊 Frontend: ${BLUE}http://localhost:$FRONTEND_PORT${NC}"
    echo -e "  🔌 Backend:  ${BLUE}http://localhost:$BACKEND_PORT${NC}"
    echo -e "  📚 API Docs: ${BLUE}http://localhost:$BACKEND_PORT/docs${NC}"
    echo ""
    echo -e "  Press ${YELLOW}Ctrl+C${NC} to stop all services"
    echo ""
    
    # Wait for interrupt
    wait $BACKEND_PID $FRONTEND_PID
}

# Start with Docker
start_docker() {
    log_info "Starting Nexus with Docker..."
    
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        log_error "docker-compose.yml not found"
        exit 1
    fi
    
    docker-compose up --build -d
    
    log_success "Nexus Docker containers started!"
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  🚀 Nexus Platform Ready (Docker)${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  📊 Frontend: ${BLUE}http://localhost:$FRONTEND_PORT${NC}"
    echo -e "  🔌 Backend:  ${BLUE}http://localhost:$BACKEND_PORT${NC}"
    echo -e "  📚 API Docs: ${BLUE}http://localhost:$BACKEND_PORT/docs${NC}"
    echo ""
    echo -e "  Run ${YELLOW}./deploy.sh --stop${NC} to stop services"
    echo ""
}

# Stop services
stop_services() {
    log_info "Stopping Nexus services..."
    
    # Stop local processes
    if [ -f ".nexus.pids" ]; then
        read -r BACKEND_PID FRONTEND_PID < .nexus.pids
        kill $FRONTEND_PID 2>/dev/null || true
        kill $BACKEND_PID 2>/dev/null || true
        rm .nexus.pids
    fi
    
    # Stop Docker containers
    if command -v docker-compose &> /dev/null && [ -f "$DOCKER_COMPOSE_FILE" ]; then
        docker-compose down 2>/dev/null || true
    fi
    
    log_success "All services stopped"
}

# Show help
show_help() {
    cat << EOF
Nexus Deployment Script

Usage: ./deploy.sh [OPTION]

Options:
  --docker     Deploy using Docker (recommended for production)
  --local      Deploy locally with Python/Node (development)
  --stop       Stop all running services
  --help       Show this help message

Examples:
  ./deploy.sh              # Interactive mode
  ./deploy.sh --docker     # Production deployment
  ./deploy.sh --local      # Development deployment
  ./deploy.sh --stop       # Stop all services

Environment Variables:
  BACKEND_PORT   Backend server port (default: 8000)
  FRONTEND_PORT  Frontend dev server port (default: 5173)

For more information, visit: https://sissifeng.github.io/Nexus
EOF
}

# Cleanup on exit
cleanup() {
    if [ -f ".nexus.pids" ]; then
        stop_services
    fi
}
trap cleanup EXIT INT TERM

# Main logic
main() {
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}     🔬 Nexus Deployment${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    case "${1:-}" in
        --docker)
            check_prerequisites docker
            clone_or_update
            start_docker
            ;;
        --local)
            check_prerequisites local
            clone_or_update
            setup_python
            setup_node
            setup_env
            start_local
            ;;
        --stop)
            stop_services
            ;;
        --help|-h)
            show_help
            ;;
        "")
            # Interactive mode
            echo "Choose deployment method:"
            echo ""
            echo "  1) Docker (recommended for production)"
            echo "  2) Local (development with hot-reload)"
            echo ""
            read -p "Enter choice [1-2]: " choice
            
            case $choice in
                1)
                    check_prerequisites docker
                    clone_or_update
                    start_docker
                    ;;
                2)
                    check_prerequisites local
                    clone_or_update
                    setup_python
                    setup_node
                    setup_env
                    start_local
                    ;;
                *)
                    log_error "Invalid choice"
                    exit 1
                    ;;
            esac
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
