FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir numpy matplotlib

# Copy project files
COPY edge_server.py lyapunov_engine.py demo.py dashboard.py dashboard.html run_experiment.py ./

# Expose ports
EXPOSE 8080 9999 10000 10001

# Default: run the dashboard
CMD ["python", "dashboard.py"]
