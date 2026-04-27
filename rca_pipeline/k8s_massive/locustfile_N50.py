
from locust import HttpUser, between, task

# Target up to 10 edge nodes randomly as entrypoints
ENTRYPOINTS = ['core-000']

class MatrixUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task
    def invoke_api(self):
        # We assume `host` passed to locust is the base domain. 
        # In minikube/kind, we might port-forward just one, or we can use internal DNS if running inside.
        # But commonly we just target the first one or a gateway.
        # For simplicity, if running from outside via port-forward, we need a single entrypoint.
        # This script assumes Locust runs from inside the cluster OR we port forward one edge node.
        
        # We will hit the root data endpoint
        self.client.get("/api/data", name="/api/data")
