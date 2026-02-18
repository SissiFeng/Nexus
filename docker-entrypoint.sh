#!/bin/bash
set -e

# Nexus Docker Entrypoint Script

# Initialize workspace if needed
if [ ! -d "$NEXUS_WORKSPACE" ]; then
    mkdir -p "$NEXUS_WORKSPACE"
fi

# Check if we should serve frontend
if [ "$1" = "server" ] && [ "$2" = "start" ]; then
    echo "🔬 Starting Nexus Optimization Platform..."
    echo ""
    echo "Workspace: $NEXUS_WORKSPACE"
    echo "Frontend:  $NEXUS_FRONTEND_PATH"
    echo ""
    
    # Start the server
    exec nexus "$@"
else
    exec "$@"
fi
