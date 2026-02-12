# Production Deployment Guide

This guide covers best practices for deploying NLU Trading Indicators to production environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Security Hardening](#security-hardening)
- [Monitoring & Logging](#monitoring--logging)
- [Performance Optimization](#performance-optimization)
- [High Availability](#high-availability)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **Operating System**: Linux (Ubuntu 20.04+ / Debian 11+ / CentOS 8+), macOS, Windows Server
- **Memory**: Minimum 2GB RAM, recommended 4GB+ for multi-pair analysis
- **CPU**: 2+ cores recommended
- **Disk Space**: 10GB+ for logs and cache
- **Network**: Stable internet connection for Binance API access

### Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.9 python3-pip python3-venv git

# CentOS/RHEL
sudo yum install -y python39 python39-pip git

# macOS
brew install python@3.9 git
```

## Installation

### 1. Clone Repository

```bash
# Production server
cd /opt  # Or your preferred application directory
git clone https://github.com/simonpeter880/nlu-trading-indicator.git
cd nlu-trading-indicator
```

### 2. Create Virtual Environment

```bash
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Install the package
pip install -e .
```

### 4. Verify Installation

```bash
python -c "from nlu_analyzer.indicators import EMARibbonEngine; print('Success')"
pytest --version
```

## Configuration

### 1. Environment Variables

Create production `.env` file:

```bash
cp .env.example .env
chmod 600 .env  # Restrict file permissions
```

Edit `.env` with your production settings:

```bash
# Production Configuration
BINANCE_API_KEY=your_production_api_key
BINANCE_API_SECRET=your_production_api_secret

DATA_SOURCE=binance_live
LOG_LEVEL=INFO
LOG_FILE=/var/log/nlu-trading/app.log
LOG_JSON=true  # Structured logging for production
LOG_CONSOLE=false  # Disable console output in production

DEFAULT_TIMEFRAME=1h
MAX_CANDLES=500

# Retry configuration
API_MAX_RETRIES=3
API_RETRY_BASE_DELAY=1.0
API_RETRY_MAX_DELAY=30.0
API_TIMEOUT_TOTAL=30.0
API_TIMEOUT_CONNECT=10.0

# Analysis settings
ENABLE_VOLUME_ANALYSIS=true
ENABLE_FUNDING_ANALYSIS=true
ENABLE_OI_ANALYSIS=true
ENABLE_ORDERBOOK_ANALYSIS=true

# Production settings
DEBUG_MODE=false
CACHE_ENABLED=true
CACHE_TTL=300
```

### 2. Create Log Directory

```bash
sudo mkdir -p /var/log/nlu-trading
sudo chown $USER:$USER /var/log/nlu-trading
```

### 3. Load Configuration

```python
# In your application code
from indicator.logging_config import configure_default_logging

# Initialize logging on application startup
configure_default_logging()
```

## Security Hardening

### 1. API Key Security

**Create Read-Only API Keys:**

1. Log in to Binance: https://www.binance.com/en/my/settings/api-management
2. Create new API key with:
   - ✅ **Enable Reading** (data access)
   - ❌ **Disable Spot & Margin Trading**
   - ❌ **Disable Futures**
   - ❌ **Disable Withdrawals**

3. Enable IP whitelist:
   ```
   # Add your server's public IP
   xx.xx.xx.xx
   ```

**Store Keys Securely:**

```bash
# Option 1: Environment variables (recommended)
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"

# Option 2: Secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.)
# Option 3: Kubernetes Secrets
kubectl create secret generic nlu-binance-keys \
  --from-literal=api-key='your_key' \
  --from-literal=api-secret='your_secret'
```

### 2. File Permissions

```bash
# Restrict access to sensitive files
chmod 600 .env
chmod 700 venv/
chmod 755 indicator/
chmod 644 indicator/*.py

# Set ownership (replace 'appuser' with your application user)
sudo chown -R appuser:appuser /opt/nlu-trading-indicator
```

### 3. Firewall Configuration

```bash
# Ubuntu/Debian (ufw)
sudo ufw allow from YOUR_IP to any port 22  # SSH
sudo ufw enable

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --reload
```

### 4. Network Security

```bash
# Verify Binance API endpoints are accessible
curl -I https://api.binance.com/api/v3/ping
curl -I https://fapi.binance.com/fapi/v1/ping

# Expected: HTTP/1.1 200 OK
```

## Monitoring & Logging

### 1. Log Management

**Centralized Logging (Recommended):**

```bash
# Option 1: Use systemd journal
sudo journalctl -u nlu-trading -f

# Option 2: Use logrotate
sudo nano /etc/logrotate.d/nlu-trading
```

**Logrotate Configuration:**

```
/var/log/nlu-trading/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 appuser appuser
    sharedscripts
    postrotate
        systemctl reload nlu-trading >/dev/null 2>&1 || true
    endscript
}
```

### 2. Application Monitoring

**Health Check Endpoint (example):**

```python
# health_check.py
import asyncio
from indicator.engines.data_fetcher import BinanceIndicatorFetcher

async def health_check():
    """Check if application can connect to Binance API."""
    try:
        async with BinanceIndicatorFetcher() as fetcher:
            valid = await fetcher.validate_symbol("BTCUSDT", futures=True)
            if valid:
                print("OK: Binance API accessible")
                return 0
            else:
                print("ERROR: Symbol validation failed")
                return 1
    except Exception as e:
        print(f"ERROR: Health check failed - {e}")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(health_check()))
```

**Monitoring with cron:**

```bash
# Add to crontab
crontab -e

# Health check every 5 minutes
*/5 * * * * /opt/nlu-trading-indicator/venv/bin/python /opt/nlu-trading-indicator/health_check.py >> /var/log/nlu-trading/health.log 2>&1
```

### 3. Alerting

**Email Alerts on Errors:**

```python
# In your application
import smtplib
from email.mime.text import MIMEText

def send_alert(subject, body):
    """Send email alert on critical errors."""
    msg = MIMEText(body)
    msg['Subject'] = f"[NLU Trading Alert] {subject}"
    msg['From'] = "alerts@yourdomain.com"
    msg['To'] = "admin@yourdomain.com"

    with smtplib.SMTP('localhost') as smtp:
        smtp.send_message(msg)
```

## Performance Optimization

### 1. Connection Pooling

```python
# Reuse aiohttp session for better performance
import aiohttp
from indicator.engines.data_fetcher import BinanceIndicatorFetcher

async def main():
    # Create shared session with connection pooling
    connector = aiohttp.TCPConnector(
        limit=100,  # Max connections
        limit_per_host=30,  # Max per host
        ttl_dns_cache=300  # DNS cache TTL
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        fetcher = BinanceIndicatorFetcher(session=session)
        # Use fetcher for multiple requests
        # ...
```

### 2. Caching

```python
# Cache expensive calculations
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_indicator(symbol: str, timeframe: str):
    """Cached indicator calculation."""
    # Implementation
    pass
```

### 3. Parallel Processing

```python
# Analyze multiple pairs concurrently
import asyncio

async def analyze_pairs(pairs: list):
    """Analyze multiple trading pairs in parallel."""
    tasks = [analyze_single_pair(pair) for pair in pairs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

## High Availability

### 1. Systemd Service

Create systemd service file:

```bash
sudo nano /etc/systemd/system/nlu-trading.service
```

```ini
[Unit]
Description=NLU Trading Indicators Service
After=network.target

[Service]
Type=simple
User=appuser
Group=appuser
WorkingDirectory=/opt/nlu-trading-indicator
Environment="PATH=/opt/nlu-trading-indicator/venv/bin"
EnvironmentFile=/opt/nlu-trading-indicator/.env
ExecStart=/opt/nlu-trading-indicator/venv/bin/python -m indicator.apps.continuous_runner
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/nlu-trading

[Install]
WantedBy=multi-user.target
```

**Enable and start service:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable nlu-trading
sudo systemctl start nlu-trading

# Check status
sudo systemctl status nlu-trading

# View logs
sudo journalctl -u nlu-trading -f
```

### 2. Docker Deployment

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN pip install --no-cache-dir -e .

# Create log directory
RUN mkdir -p /var/log/nlu-trading

# Non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /var/log/nlu-trading
USER appuser

# Health check
HEALTHCHECK --interval=5m --timeout=10s --retries=3 \
  CMD python health_check.py || exit 1

# Run application
CMD ["python", "-m", "indicator.apps.continuous_runner"]
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  nlu-trading:
    build: .
    container_name: nlu-trading-app
    restart: unless-stopped
    env_file:
      - .env
    volumes:
      - ./logs:/var/log/nlu-trading
    networks:
      - nlu-network
    healthcheck:
      test: ["CMD", "python", "health_check.py"]
      interval: 5m
      timeout: 10s
      retries: 3

networks:
  nlu-network:
    driver: bridge
```

**Deploy with Docker:**

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Stop
docker-compose down
```

### 3. Kubernetes Deployment

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nlu-trading
  labels:
    app: nlu-trading
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nlu-trading
  template:
    metadata:
      labels:
        app: nlu-trading
    spec:
      containers:
      - name: nlu-trading
        image: your-registry/nlu-trading:latest
        env:
        - name: BINANCE_API_KEY
          valueFrom:
            secretKeyRef:
              name: nlu-binance-keys
              key: api-key
        - name: BINANCE_API_SECRET
          valueFrom:
            secretKeyRef:
              name: nlu-binance-keys
              key: api-secret
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          exec:
            command:
            - python
            - health_check.py
          initialDelaySeconds: 30
          periodSeconds: 300
        readinessProbe:
          exec:
            command:
            - python
            - health_check.py
          initialDelaySeconds: 10
          periodSeconds: 60
```

## Troubleshooting

### Common Issues

**1. API Rate Limiting**

```
Error: BinanceRateLimitError: Rate limit exceeded
```

**Solution:**
- Reduce request frequency
- Implement request queuing
- Use weight-aware request management

**2. Connection Timeouts**

```
Error: BinanceTimeoutError: Request timed out after 30.0s
```

**Solution:**
- Check network connectivity
- Increase timeout values in `.env`
- Verify Binance API status: https://www.binance.com/en/support/announcement

**3. Memory Issues**

```
Error: MemoryError or OOM killed
```

**Solution:**
- Reduce `MAX_CANDLES` value
- Limit concurrent analysis pairs
- Enable result caching with `CACHE_ENABLED=true`

**4. Permission Errors**

```
Error: PermissionError: [Errno 13] Permission denied: '/var/log/nlu-trading/app.log'
```

**Solution:**
```bash
sudo mkdir -p /var/log/nlu-trading
sudo chown -R $USER:$USER /var/log/nlu-trading
chmod 755 /var/log/nlu-trading
```

### Debug Mode

Enable debug logging:

```bash
# Temporary
export LOG_LEVEL=DEBUG

# In .env
DEBUG_MODE=true
LOG_LEVEL=DEBUG
```

### Performance Profiling

```bash
# Profile application
python -m cProfile -o profile.stats your_script.py

# Analyze profile
python -m pstats profile.stats
>>> sort cumulative
>>> stats 20
```

## Backup & Recovery

### Backup Configuration

```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backup/nlu-trading"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup .env and logs
tar -czf "$BACKUP_DIR/nlu-trading-$DATE.tar.gz" \
    /opt/nlu-trading-indicator/.env \
    /var/log/nlu-trading/*.log

# Keep last 7 days
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +7 -delete
```

### Recovery

```bash
# Restore from backup
tar -xzf /backup/nlu-trading/nlu-trading-YYYYMMDD_HHMMSS.tar.gz -C /

# Restart service
sudo systemctl restart nlu-trading
```

## Scaling Considerations

### Horizontal Scaling

- Deploy multiple instances for different trading pairs
- Use message queue (RabbitMQ, Kafka) for task distribution
- Implement distributed caching (Redis)

### Vertical Scaling

- Increase CPU/RAM for resource-intensive analysis
- Use faster storage (SSD) for logs and cache

## Support

- **Documentation**: [README.md](README.md), [DEVELOPMENT.md](DEVELOPMENT.md)
- **Issues**: https://github.com/simonpeter880/nlu-trading-indicator/issues
- **Discussions**: https://github.com/simonpeter880/nlu-trading-indicator/discussions

---

**Production Checklist:**

- [ ] Read-only API keys created and tested
- [ ] IP whitelist configured on Binance
- [ ] `.env` file created with production settings
- [ ] File permissions restricted (`.env`: 600)
- [ ] Log directory created with proper permissions
- [ ] Systemd service configured and enabled
- [ ] Log rotation configured
- [ ] Health check monitoring set up
- [ ] Firewall rules configured
- [ ] Backup strategy implemented
- [ ] Alerting configured for critical errors
- [ ] Performance optimizations applied
- [ ] Documentation reviewed

Deploy with confidence! 🚀
