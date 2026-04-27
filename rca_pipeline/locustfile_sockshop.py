import logging
import random

from locust import HttpUser, between, task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SockShopUser(HttpUser):
    """
    Workload Sock-Shop (weaveworksdemo).
    Parcours: browse catalogue -> register/login -> add to cart -> checkout.
    Ratios approximatifs 60/25/15 (read-heavy, puis cart, puis checkout).
    """

    wait_time = between(1, 3)

    def on_start(self) -> None:
        self._username = f"u{random.randint(1_000_000, 9_999_999)}"
        self._password = "pw"
        # Register (idempotent enough pour une run courte)
        self.client.post(
            "/register",
            json={
                "username": self._username,
                "password": self._password,
                "email": f"{self._username}@rca.local",
                "firstName": "rca",
                "lastName": "bench",
            },
            name="/register",
            catch_response=True,
        )
        with self.client.get(
            "/login",
            auth=(self._username, self._password),
            name="/login",
            catch_response=True,
        ) as r:
            if r.status_code >= 400:
                r.failure(f"login {r.status_code}")

    @task(6)
    def browse(self) -> None:
        self.client.get("/", name="/")
        self.client.get("/catalogue", name="/catalogue")
        with self.client.get("/catalogue", name="/catalogue.list", catch_response=True) as r:
            try:
                items = r.json() or []
            except Exception:
                items = []
        if items:
            item = random.choice(items)
            item_id = item.get("id")
            if item_id:
                self.client.get(f"/catalogue/{item_id}", name="/catalogue/:id")

    @task(3)
    def add_to_cart(self) -> None:
        with self.client.get("/catalogue", name="/catalogue.list", catch_response=True) as r:
            try:
                items = r.json() or []
            except Exception:
                items = []
        if not items:
            return
        item = random.choice(items)
        item_id = item.get("id")
        if not item_id:
            return
        self.client.post(
            "/cart",
            json={"id": item_id, "quantity": random.randint(1, 3)},
            name="/cart.add",
            catch_response=True,
        )
        self.client.get("/cart", name="/cart.view", catch_response=True)

    @task(1)
    def checkout(self) -> None:
        # Best-effort: un utilisateur avec un panier déclenche un order.
        self.client.post("/orders", name="/orders", catch_response=True)
