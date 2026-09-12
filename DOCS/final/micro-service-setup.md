
# 🔬 **SockShop LTTng Tracing: Complete Setup & Collection Guide**

## **1. Environment Setup**

### **Infrastructure**
```
GCP VM: 12 CPU, 40GB RAM, 100GB SSD
Ubuntu 24.04 + LTTng 2.15 + Docker 27.0
```

### **SockShop Deployment**
```
git clone https://github.com/microservices-demo/microservices-demo
cd microservices-demo/deploy
docker-compose up -d
```
**Architecture**:
```
Frontend (port 80) → Apache → [carts, orders, catalogue, payment, shipping, user]
                          ↓
                      MongoDB (carts/orders)
```

**Traffic**: `load_generator.py` simulates 100-200 users → **1000+ req/s** (register/cart/checkout).

## **2. Tracing Pipeline**

```
App (Spring Boot) ── OpenTelemetry ── Python Relay ── LTTng ── CTF Traces
     spans^1                    ^2              ^3         ^4
```

### **1. OpenTelemetry in Apps**
Spring Boot auto-instruments **HTTP endpoints + Mongo queries**:
```
GET /carts/{id}/items  ← SERVER span
├── CartRepository.findByCustomerId  ← INTERNAL
└── find data.cart      ← Mongo CLIENT
```

### **2. Python Relay** (`/home/sehgaluv17/agents/otel-to-lttng.py`)
```
tail -f /tmp/otel-relay.log
[LTTng] op=GET /carts/123/items trace_id=abc span_id=xyz
```
**Forwards OTel log lines → LTTng Python domain**.

### **3. LTTng Sessions** (`collect_trace.sh`)
**Dual sessions**:
```
lttng create sockshop-ust --output=traces/*/ust    # User-space (OTel)
lttng enable-event --python otel.spans

sudo lttng create sockshop-kernel --output=traces/*/kernel  # Kernel  
sudo lttng enable-event -k --all '*'
```

### **4. CTF Output**
```
traces/normal/run01/
├── kernel/kernel/channel0_*.idx    # Syscalls, scheduling (2M+ events)
└── ust/ust/uid/1002/64-bit/        # OTel spans (~5K)
    └── lttng_python_channel_*.idx
```

## **3. Data Collection Workflow**

```
./normal.sh run_01  # or cpu_stress.sh, disk_stress.sh
1. Start LTTng (kernel+UST)
2. CPU/Disk stress-ng (anomaly runs)  
3. load_generator.py (100-200 users)
4. Collect Prometheus metrics
```

**Output**:
```
experiments/normal/run01/
├── load_results.csv       # 20K requests (scenario, latency, status)
└── metrics/*.json         # 33 files (QPS, P95 per service)
```

## **4. What We Collected**

### **A. Load Data** (`load_results.csv`)
```
timestamp,userid,scenario,method,endpoint,status,latency,success,error
2026-03-02T12:35:06,user-9,add_to_cart,POST,/cart,201,38ms,true,
```

**Scenarios**: register, catalogue, cart/add/delete, checkout, orders.

### **B. Application Traces** (`ust/ust/uid/1002/64-bit`)
**OTel spans** (Spring Boot + Mongo):
```
op=GET /carts/{id}/items trace_id=abc span_id=xyz
op=find data.cart (Mongo)
op=CartRepository.findByCustomerId (Spring)
```
**~5K spans/run**, **40% business purity**.

### **C. Kernel Traces** (`kernel/kernel`)
**System context** for app spans:
```
syscall_entry_epoll_ctl  # Network polling
syscall_entry_read       # Socket I/O
net_dev_queue            # Packet TX/RX
block_rq_issue           # Disk I/O
```
**~2M events/run** — shows **why** spans slow down.

### **D. System Metrics** (`metrics/*.json`)
**Prometheus time-series**:
```
catalogue_qps.json       # Requests/sec
orders_p95_latency.json  # 95th percentile latency
vm_cpu.json              # Host CPU usage
```

## **5. Anomaly Experiments**

| Stress | `stress-ng` Command | Expected |
|--------|-------------------|----------|
| CPU | `--cpu 12 --cpu-load 100` | 200ms+ latency |
| Disk | `--hdd 200 --hdd-bytes 50G` | Mongo timeouts |
| Memory | `--vm 12 --vm-bytes 2G` | GC pauses |

## **6. Analysis Commands**

```bash
# App spans
babeltrace2 ust/ust/uid/1002/64-bit | grep otel.spans | grep cart

# Kernel context
babeltrace2 kernel/kernel | grep epoll_ctl | head -20

# Correlate
babeltrace2 kernel/kernel | grep -A10 "block_rq_issue"  # Disk waits

# Stats
tail -n +2 experiments/*/load_results.csv | wc -l  # Total requests
find experiments -name "*.json" | wc -l            # Metrics files
```

## **7. Scale**
```
10+ normal/anomaly runs
50MB traces + 2MB experiments
200K+ requests total
50K+ business spans
```

**End-to-end distributed tracing dataset** — **app spans + kernel context + load + metrics** under realistic stress. Ready for **ML anomaly detection**! 🚀