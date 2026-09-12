# ✅ STEP-BY-STEP CLEAN RESET

## 1. Go to compose directory

```bash
cd ~/microservices-demo/deploy/docker-compose
```

---

## 2. Stop and remove stack (including volumes)

```bash
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml down -v --remove-orphans
```

👉 This removes:

* all Sock Shop containers
* all associated volumes
* networks created by compose

---

## 3. Remove ALL leftover containers (safety)

```bash
docker rm -f $(docker ps -aq) 2>/dev/null || true
```

---

## 4. Remove ALL unused volumes

```bash
docker volume prune -f
```

👉 This ensures:

* no leftover DB state
* clean start

---

## 5. Remove unused networks

```bash
docker network prune -f
```

---

## 6. (Optional but recommended) Remove images

If you want a **completely fresh pull**:

```bash
docker image prune -a -f
```

👉 This will re-pull all images next time

---

## 7. Verify everything is clean

```bash
docker ps
docker volume ls
docker network ls
```

👉 You should see:

* no running containers
* only default networks

---

# 🚀 REDEPLOY FROM SCRATCH

## 8. Start stack again

```bash
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

---

## 9. Wait for services to stabilize

```bash
sleep 20
```

---

## 10. Verify services

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

---

## 11. Verify Prometheus is working

```bash
curl http://localhost:9090/-/ready
```

---

## 12. Verify OTel logs exist

```bash
docker logs docker-compose_carts_1 2>&1 | grep LoggingSpanExporter | head
```

---

# 🔁 OPTIONAL (if using cgroupns=private)

Check again:

```bash
docker inspect -f '{{.Name}} {{.HostConfig.CgroupnsMode}}' $(docker ps -q)
```

---

# 🧠 When should you do this?

Do a full reset when:

* switching anomaly configs
* changing tracing setup
* rebuilding dataset from scratch
* debugging weird behavior
* ensuring reproducibility

---

# ⚡ One-line “hard reset” (advanced)

If you want everything in one go:

```bash
docker-compose down -v --remove-orphans && \
docker rm -f $(docker ps -aq) 2>/dev/null || true && \
docker volume prune -f && \
docker network prune -f
```

---

# 🚀 Final note

After reset, always:

```bash
sleep 20
```

before running:

```bash
./normal.sh run01 100
```

---

If you want, I can give you a **single script (`reset_and_start.sh`)** so you can reset + redeploy in one command 👍
