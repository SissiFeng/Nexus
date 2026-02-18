# Nexus Optimization Platform - Production Dockerfile
# Multi-stage build for optimized image size

# Stage 1: Python Backend
FROM python:3.11-slim as backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Python package files
COPY pyproject.toml ./
COPY optimization_copilot/ ./optimization_copilot/

# Install Python package
RUN pip install --no-cache-dir -e ".[dev]"

# Create workspace directory
RUN mkdir -p /app/workspace

# Stage 2: Node Frontend
FROM node:20-alpine as frontend-builder

WORKDIR /app

# Copy frontend package files
COPY optimization_copilot/web/package.json ./
RUN npm install

# Copy frontend source
COPY optimization_copilot/web/ ./

# Build production frontend
RUN npm run build

# Stage 3: Final Production Image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python backend from stage 1
COPY --from=backend /app /app
COPY --from=backend /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=backend /usr/local/bin /usr/local/bin

# Copy built frontend from stage 2
COPY --from=frontend-builder /app/dist /app/frontend_dist

# Copy startup script
COPY docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Create non-root user
RUN useradd -m -u 1000 nexus && \
    chown -R nexus:nexus /app/workspace
USER nexus

# Environment variables
ENV PYTHONPATH=/app
ENV NEXUS_WORKSPACE=/app/workspace
ENV NEXUS_FRONTEND_PATH=/app/frontend_dist
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Expose port
EXPOSE 8000

# Start command
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["server", "start", "--host", "0.0.0.0", "--port", "8000"]
