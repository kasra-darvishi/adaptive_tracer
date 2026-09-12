#!/usr/bin/env python3
"""
Sock Shop (Weaveworks microservices-demo) Traffic Generator
===========================================================
Sends realistic HTTP traffic to the Sock Shop front-end, exercising
all major user-facing flows and therefore all downstream microservices:

  front-end  -> catalogue, carts, orders, payment, shipping, user

Services covered
----------------
  catalogue   GET  /catalogue, /catalogue/{id}, /catalogue/size, /tags
  carts       GET/POST/DELETE  /cart, /cart/items, /cart/items/{itemId}
  orders      GET/POST  /orders
  user        POST /login, /register, GET /address, /card, /customers
  front-end   GET  /, /category, /detail/{id}, /basket, /orders, /health

Usage
-----
  python load_generator.py --host http://<VM_IP>:30001 [options]

  --host          Base URL of the front-end (default: http://localhost:30001)
  --users         Number of concurrent virtual users  (default: 5)
  --duration      Total run duration in seconds       (default: 300)
  --think-min     Minimum think-time between requests (default: 0.5)
  --think-max     Maximum think-time between requests (default: 2.0)
  --log-level     Logging level: DEBUG|INFO|WARNING   (default: INFO)
  --output        Path to CSV results file            (default: results.csv)

Requirements
------------
  pip install requests
"""

import argparse
import csv
import logging
import random
import string
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import requests
from requests import Session

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sock Shop traffic generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",       default="http://localhost:30001",
                   help="Base URL of the Sock Shop front-end")
    p.add_argument("--users",      type=int, default=5,
                   help="Number of concurrent virtual users")
    p.add_argument("--duration",   type=int, default=300,
                   help="Total run duration in seconds")
    p.add_argument("--think-min",  type=float, default=0.5,
                   help="Minimum think-time (seconds) between requests")
    p.add_argument("--think-max",  type=float, default=2.0,
                   help="Maximum think-time (seconds) between requests")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"],
                   help="Logging verbosity")
    p.add_argument("--output",     default="results.csv",
                   help="Path to CSV file to write per-request results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared result store (thread-safe)
# ---------------------------------------------------------------------------
@dataclass
class RequestResult:
    timestamp: str
    user_id: int
    scenario: str
    method: str
    endpoint: str
    status_code: int
    latency_ms: float
    success: bool
    error: str = ""


_results_lock = threading.Lock()
_results: list[RequestResult] = []


def record(result: RequestResult) -> None:
    with _results_lock:
        _results.append(result)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("sockshop-traffic")


# ---------------------------------------------------------------------------
# Helper: random string generators
# ---------------------------------------------------------------------------
def rand_str(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=n))


def rand_email() -> str:
    return f"{rand_str(6)}@{rand_str(4)}.com"


def rand_password() -> str:
    return rand_str(12)


# ---------------------------------------------------------------------------
# Virtual User
# ---------------------------------------------------------------------------
class VirtualUser:
    """
    Simulates one user session against the Sock Shop front-end.
    All requests go through the front-end proxy (port 30001) which
    internally fans out to the backend microservices.
    """

    # Catalogue item IDs that are seeded in the default dataset
    DEFAULT_ITEM_IDS = [
        "03fef6ac-1896-4ce8-bd69-b798f85c6e0b",
        "3395a43e-2d88-40de-b95f-e00e1502085b",
        "510a0d7e-8e83-4193-b483-e27e09ddc34d",
        "819e1fbf-8b7e-4f6d-811f-693534916a8b",
        "837ab141-399e-4c1f-9abc-bace40296bac",
        "a0a4f044-b040-410d-8ead-4de0446aec7e",
        "d3588630-ad8e-49df-bbd7-3167f7efb246",
        "zzz4f044-b040-410d-8ead-4de0446aec7e",
    ]

    def __init__(self, user_id: int, base_url: str, think_min: float, think_max: float):
        self.user_id = user_id
        self.base_url = base_url.rstrip("/")
        self.think_min = think_min
        self.think_max = think_max
        self.session = Session()
        self.session.headers.update({
            "User-Agent": f"SockShopTrafficGen/1.0 (user-{user_id})",
        })
        # Credentials — filled in after registration
        self.username: Optional[str] = None
        self.password: Optional[str] = None
        self.logged_in: bool = False
        self.customer_id: Optional[str] = None
        self.cart_item_ids: list[str] = []

    # ------------------------------------------------------------------
    # Core HTTP helpers
    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        scenario: str,
        *,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        timeout: int = 10,
    ) -> Optional[requests.Response]:
        url = self.base_url + path
        start = time.time()
        resp = None
        error = ""
        try:
            resp = self.session.request(method, url, json=json, params=params,
                                        timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            success = True
        except requests.HTTPError as e:
            success = False
            error = str(e)
        except Exception as e:
            success = False
            error = str(e)
        finally:
            latency_ms = (time.time() - start) * 1000
            status_code = resp.status_code if resp is not None else 0
            result = RequestResult(
                timestamp=datetime.now(timezone.utc).isoformat(),
                user_id=self.user_id,
                scenario=scenario,
                method=method.upper(),
                endpoint=path,
                status_code=status_code,
                latency_ms=round(latency_ms, 2),
                success=success,
                error=error,
            )
            record(result)
            log.debug(
                "[user-%d] %s %s -> %d (%.0f ms) %s",
                self.user_id, method.upper(), path, status_code,
                latency_ms, "" if success else f"ERROR: {error}",
            )
        return resp if success else None

    def _think(self) -> None:
        time.sleep(random.uniform(self.think_min, self.think_max))

    # ------------------------------------------------------------------
    # Scenario helpers
    # ------------------------------------------------------------------

    def scenario_browse_homepage(self) -> None:
        """GET / (front-end homepage, triggers catalogue fetch internally)"""
        self._request("GET", "/", "browse_homepage")
        self._think()

    def scenario_browse_catalogue(self) -> None:
        """Browse all socks with optional tag/page filters (catalogue service)"""
        tags = random.choice([None, "sku", "formal", "casual", "blue", "red"])
        params: dict = {"size": random.randint(3, 12)}
        if tags:
            params["tags"] = tags

        # Catalogue page
        self._request("GET", "/catalogue", "browse_catalogue", params=params)
        self._think()

        # Fetch catalogue size
        self._request("GET", "/catalogue/size", "catalogue_size", params=({"tags": tags} if tags else {}))
        self._think()

        # Tags list
        self._request("GET", "/tags", "get_tags")
        self._think()

    def scenario_view_item(self) -> Optional[str]:
        """View detail page for a random catalogue item"""
        item_id = random.choice(self.DEFAULT_ITEM_IDS)
        resp = self._request("GET", f"/catalogue/{item_id}", "view_item")
        self._think()
        return item_id

    def scenario_add_to_cart(self, item_id: Optional[str] = None) -> None:
        """Add an item to the cart (carts service)"""
        if item_id is None:
            item_id = random.choice(self.DEFAULT_ITEM_IDS)

        payload = {
            "id": item_id,
            "quantity": random.randint(1, 3),
        }
        resp = self._request("POST", "/cart", "add_to_cart", json=payload)
        if resp:
            self.cart_item_ids.append(item_id)
        self._think()

    def scenario_view_cart(self) -> None:
        """View cart (carts service)"""
        self._request("GET", "/cart", "view_cart")
        self._think()

    def scenario_update_cart_item(self) -> None:
        """
        Simulate a cart update by viewing the cart, then re-adding an item
        with a new quantity. The Sock Shop front-end has no PATCH /cart/update
        endpoint; quantity changes go through delete + re-add.
        """
        resp = self._request("GET", "/cart", "update_cart_item")
        self._think()
        if resp is None:
            return
        try:
            cart = resp.json()
            items = cart.get("items") or []
            if not items:
                return
            item = random.choice(items)
            item_id = item.get("itemId") or item.get("id", "")
            if not item_id:
                return
            # Remove then re-add with new quantity
            self._request("DELETE", f"/cart/{item_id}", "update_cart_item")
            self._think()
            self._request("POST", "/cart",
                          "update_cart_item",
                          json={"id": item_id, "quantity": random.randint(1, 4)})
            self._think()
        except Exception:
            pass

    def scenario_delete_cart_item(self) -> None:
        """
        Remove an item from the cart.
        Front-end exposes: DELETE /cart/{itemId}  (not /cart/items/{itemId})
        We GET /cart first to obtain the real itemId assigned by the carts service.
        """
        resp = self._request("GET", "/cart", "delete_cart_item")
        self._think()
        if resp is None:
            return
        try:
            cart = resp.json()
            items = cart.get("items") or []
            if not items:
                return
            item = random.choice(items)
            item_id = item.get("itemId") or item.get("id", "")
            if item_id:
                self._request("DELETE", f"/cart/{item_id}", "delete_cart_item")
                self._think()
        except Exception:
            pass

    def scenario_login(self) -> bool:
        """
        GET /login with HTTP Basic Auth — the Sock Shop front-end validates
        credentials via the user service and sets a session cookie on success.
        If no account exists yet for this virtual user, register one first.
        """
        if not self.username:
            self.username = rand_str(8)
            self.password = rand_password()
            self.scenario_register(self.username, self.password)
            return self.logged_in

        start = time.time()
        try:
            raw = self.session.get(
                self.base_url + "/login",
                auth=(self.username, self.password),
                timeout=10,
                allow_redirects=True,
            )
            latency_ms = (time.time() - start) * 1000
            success = raw.ok
            record(RequestResult(
                timestamp=datetime.now(timezone.utc).isoformat(),
                user_id=self.user_id,
                scenario="login",
                method="GET",
                endpoint="/login",
                status_code=raw.status_code,
                latency_ms=round(latency_ms, 2),
                success=success,
                error="" if success else f"{raw.status_code} {raw.reason}",
            ))
            self.logged_in = success
            if success:
                log.info("[user-%d] Logged in as %s", self.user_id, self.username)
                try:
                    data = raw.json()
                    self.customer_id = data.get("id") or self.customer_id
                except Exception:
                    pass
            return success
        except Exception as e:
            log.warning("[user-%d] Login request failed: %s", self.user_id, e)
            return False

    def scenario_register(self, username: Optional[str] = None,
                          password: Optional[str] = None) -> bool:
        """POST /register (user service via front-end)"""
        uname = username or rand_str(8)
        pwd   = password or rand_password()

        payload = {
            "username": uname,
            "password": pwd,
            "email":    rand_email(),
            "firstName": rand_str(5).capitalize(),
            "lastName":  rand_str(7).capitalize(),
        }
        resp = self._request("POST", "/register", "register", json=payload)
        if resp is not None:
            self.username = uname
            self.password = pwd
            self.logged_in = True
            try:
                data = resp.json()
                self.customer_id = data.get("id")
            except Exception:
                pass
            log.info("[user-%d] Registered as %s", self.user_id, uname)
        self._think()
        return resp is not None

    def scenario_get_orders(self) -> None:
        """GET /orders (orders service via front-end)"""
        self._request("GET", "/orders", "get_orders")
        self._think()

    def scenario_place_order(self) -> None:
        """
        POST /orders — full checkout flow:
          1. browse catalogue
          2. add item to cart
          3. place order (triggers payment + shipping internally)
        """
        # 1. pick an item
        item_id = random.choice(self.DEFAULT_ITEM_IDS)
        self._request("GET", f"/catalogue/{item_id}", "checkout_view_item")
        self._think()

        # 2. add to cart
        payload = {"id": item_id, "quantity": 1}
        self._request("POST", "/cart", "checkout_add_to_cart", json=payload)
        self._think()

        # 3. place order (carts -> orders -> payment -> shipping)
        # POST /orders with no body — the front-end reads the session cart
        self._request("POST", "/orders", "place_order")
        self._think()

    def scenario_get_address(self) -> None:
        """
        GET /address — returns the logged-in user's addresses via the front-end.
        Note: /customers/{id}/addresses is an internal user-service API path
        not exposed by the edge-router; use the front-end proxy endpoint instead.
        """
        self._request("GET", "/address", "get_address")
        self._think()

    def scenario_get_card(self) -> None:
        """
        GET /card — returns the logged-in user's payment card via the front-end.
        """
        self._request("GET", "/card", "get_card")
        self._think()

    def scenario_detail_page(self) -> None:
        """
        View a product's detail by fetching it from the catalogue API directly.
        The SPA's /detail/{id} route is not served server-side by the edge-router;
        the underlying data comes from GET /catalogue/{id} which works fine.
        """
        item_id = random.choice(self.DEFAULT_ITEM_IDS)
        self._request("GET", f"/catalogue/{item_id}", "detail_page")
        self._think()

    def scenario_basket_page(self) -> None:
        """View the cart (basket) — uses the working GET /cart API endpoint."""
        self._request("GET", "/cart", "basket_page")
        self._think()

    # ------------------------------------------------------------------
    # Account setup helpers
    # ------------------------------------------------------------------
    def _setup_account(self) -> None:
        """
        Provision a delivery address and payment card for this user.

        The Sock Shop /orders endpoint checks that the logged-in customer
        has at least one address AND one payment card on file before it
        will accept an order (returns 406 otherwise). Newly registered
        users have neither, so we add them once immediately after login.

        The front-end proxies these as:
          POST /addresses  → user service
          POST /cards      → user service
        """
        streets = ["123 Main St", "456 Oak Ave", "789 Pine Rd", "10 Sock Lane"]
        cities  = ["London", "New York", "Berlin", "Paris", "Toronto"]
        postals = ["EC1A 1BB", "10001", "10115", "75001", "M5H 2N2"]
        countries = ["GB", "US", "DE", "FR", "CA"]

        idx = self.user_id % len(cities)
        address_payload = {
            "number":   str(random.randint(1, 999)),
            "street":   random.choice(streets),
            "city":     cities[idx],
            "postcode": postals[idx],
            "country":  countries[idx],
        }
        self._request("POST", "/addresses", "setup_address", json=address_payload)

        # Luhn-valid test card numbers safe for demo environments
        card_numbers = [
            "5454545454545454",
            "4111111111111111",
            "5105105105105100",
        ]
        exp_month = random.randint(1, 12)
        exp_year  = 2027 + (self.user_id % 3)
        card_payload = {
            "longNum": random.choice(card_numbers),
            "expires": f"{exp_month:02d}/{str(exp_year)[2:]}",
            "ccv":     str(random.randint(100, 999)),
        }
        self._request("POST", "/cards", "setup_card", json=card_payload)
        log.debug("[user-%d] Account provisioned (address + card)", self.user_id)

    # ------------------------------------------------------------------
    # Full user journey
    # ------------------------------------------------------------------
    def run_journey(self, stop_event: threading.Event) -> None:
        """
        Simulate a realistic user session until stop_event is set.
        Weighted random selection of scenarios matches typical eCommerce
        traffic patterns (heavy read, lighter write, rare purchase).
        """
        # Each user registers then immediately logs in so the 'logged_in'
        # session cookie is established before any authenticated endpoints
        # (e.g. POST /orders) are called. Then we provision address + card
        # so that order placement succeeds (the orders endpoint returns 406
        # if the customer has no address or card on file).
        self.scenario_register()
        self.scenario_login()
        self._setup_account()

        scenarios = [
            # (weight, callable)
            (15, self.scenario_browse_homepage),
            (14, self.scenario_browse_catalogue),
            (12, self.scenario_view_item),
            (10, self.scenario_detail_page),
            (9,  self.scenario_add_to_cart),
            (9,  self.scenario_basket_page),
            (3,  self.scenario_update_cart_item),
            (3,  self.scenario_delete_cart_item),
            (4,  self.scenario_get_orders),
            (5,  self.scenario_place_order),
            (3,  self.scenario_get_address),
            (3,  self.scenario_get_card),
            (5,  self.scenario_login),
        ]

        weights    = [w for w, _ in scenarios]
        callables  = [fn for _, fn in scenarios]

        while not stop_event.is_set():
            chosen = random.choices(callables, weights=weights, k=1)[0]
            try:
                chosen()
            except Exception as exc:
                log.warning("[user-%d] Unhandled exception in %s: %s",
                            self.user_id, chosen.__name__, exc)
            # Extra jitter between scenario executions
            time.sleep(random.uniform(self.think_min, self.think_max))


# ---------------------------------------------------------------------------
# Stats printer
# ---------------------------------------------------------------------------
def print_stats(results: list[RequestResult]) -> None:
    if not results:
        print("\nNo requests recorded.")
        return

    total   = len(results)
    passed  = sum(1 for r in results if r.success)
    failed  = total - passed
    avg_lat = sum(r.latency_ms for r in results) / total
    min_lat = min(r.latency_ms for r in results)
    max_lat = max(r.latency_ms for r in results)

    # Per-scenario breakdown
    scenario_stats: dict[str, dict] = {}
    for r in results:
        s = scenario_stats.setdefault(r.scenario, {"total": 0, "pass": 0, "lat": []})
        s["total"] += 1
        if r.success:
            s["pass"] += 1
        s["lat"].append(r.latency_ms)

    print("\n" + "=" * 70)
    print("  SOCK SHOP TRAFFIC GENERATOR — RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total requests : {total}")
    print(f"  Passed         : {passed}  ({100*passed/total:.1f}%)")
    print(f"  Failed         : {failed}  ({100*failed/total:.1f}%)")
    print(f"  Avg latency    : {avg_lat:.0f} ms")
    print(f"  Min latency    : {min_lat:.0f} ms")
    print(f"  Max latency    : {max_lat:.0f} ms")
    print()
    print(f"  {'Scenario':<30} {'Requests':>8} {'Pass%':>7} {'Avg(ms)':>9}")
    print(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*9}")
    for name, s in sorted(scenario_stats.items()):
        pct = 100 * s["pass"] / s["total"] if s["total"] else 0
        avg = sum(s["lat"]) / len(s["lat"]) if s["lat"] else 0
        print(f"  {name:<30} {s['total']:>8} {pct:>6.1f}% {avg:>9.0f}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------
def write_csv(results: list[RequestResult], path: str) -> None:
    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "user_id", "scenario",
                "method", "endpoint", "status_code",
                "latency_ms", "success", "error",
            ])
            for r in results:
                writer.writerow([
                    r.timestamp, r.user_id, r.scenario,
                    r.method, r.endpoint, r.status_code,
                    r.latency_ms, r.success, r.error,
                ])
        log.info("Results written to %s (%d rows)", path, len(results))
    except Exception as e:
        log.error("Failed to write CSV: %s", e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Target host  : %s", args.host)
    log.info("Virtual users: %d", args.users)
    log.info("Duration     : %d s", args.duration)
    log.info("Think time   : %.1f – %.1f s", args.think_min, args.think_max)
    log.info("Output CSV   : %s", args.output)

    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    for uid in range(1, args.users + 1):
        user = VirtualUser(
            user_id=uid,
            base_url=args.host,
            think_min=args.think_min,
            think_max=args.think_max,
        )
        t = threading.Thread(
            target=user.run_journey,
            args=(stop_event,),
            name=f"user-{uid}",
            daemon=True,
        )
        threads.append(t)

    log.info("Starting %d virtual users …", args.users)
    for t in threads:
        t.start()

    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        log.info("Interrupted by user — shutting down …")

    log.info("Stopping virtual users …")
    stop_event.set()

    for t in threads:
        t.join(timeout=15)

    with _results_lock:
        snapshot = list(_results)

    print_stats(snapshot)
    write_csv(snapshot, args.output)


if __name__ == "__main__":
    main()
