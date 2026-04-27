import logging

from locust import HttpUser, between, task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RCAUser(HttpUser):
    """
    Simulates a user visiting the frontend with an 80/20 read/write ratio
    to solicit the database asynchronously.
    """
    wait_time = between(1, 4)

    @task(4)
    def read_path(self) -> None:
        """
        80% probability path:
        Simulate a user loading the frontend and fetching data.
        """
        # Load main page
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to load frontend: {response.status_code}")

        # Frontend JS simulates a call to auth then data
        with self.client.get("/auth/verify", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to verify auth: {response.status_code}")

        with self.client.get("/api/data", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to load data: {response.status_code}")

    @task(1)
    def write_path(self) -> None:
        """
        20% probability path:
        Simulate a user performing a write operation.
        """
        with self.client.post("/api/data", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to write data: {response.status_code}")
