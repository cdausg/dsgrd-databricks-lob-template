# Model Serving Endpoint Deployment
# Deploys the champion model from Unity Catalog to a Databricks Model Serving endpoint
# For use cases requiring real-time API scoring

import argparse
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--validate-only", type=bool, default=False)
    return parser.parse_args()

def get_champion_version(catalog, schema, model_name):
    """Get the version number of the current champion model."""
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()
    model_uri = f"{catalog}.{schema}.{model_name}"

    champion = client.get_model_version_by_alias(model_uri, "champion")
    return champion.version

def deploy_endpoint(w, catalog, schema, model_name, model_version):
    """Create or update the model serving endpoint."""
    endpoint_name = f"{catalog}-{schema}-{model_name}"
    model_uri = f"{catalog}.{schema}.{model_name}"

    config = EndpointCoreConfigInput(
        served_models=[
            ServedModelInput(
                model_name=model_uri,
                model_version=model_version,
                workload_size=ServedModelInputWorkloadSize.SMALL,
                scale_to_zero_enabled=True
            )
        ]
    )

    # Check if endpoint already exists
    existing_endpoints = [e.name for e in w.serving_endpoints.list()]

    if endpoint_name in existing_endpoints:
        print(f"Updating existing endpoint: {endpoint_name}")
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_models=config.served_models
        )
    else:
        print(f"Creating new endpoint: {endpoint_name}")
        w.serving_endpoints.create(
            name=endpoint_name,
            config=config
        )

    return endpoint_name

def validate_endpoint(w, endpoint_name):
    """Validate that the endpoint is ready and responding."""
    import time
    max_retries = 20
    retry_interval = 30

    for i in range(max_retries):
        endpoint = w.serving_endpoints.get(name=endpoint_name)
        state = endpoint.state.ready.value if endpoint.state.ready else "NOT_READY"

        if state == "READY":
            print(f"Endpoint {endpoint_name} is ready")
            return True

        print(f"Waiting for endpoint to be ready... ({i+1}/{max_retries})")
        time.sleep(retry_interval)

    raise TimeoutError(f"Endpoint {endpoint_name} did not become ready in time")

def main():
    args = parse_args()
    w = WorkspaceClient()

    endpoint_name = f"{args.catalog}-{args.schema}-{args.model_name}"

    if args.validate_only:
        print(f"Validating endpoint: {endpoint_name}")
        validate_endpoint(w, endpoint_name)
        return

    # Get champion model version
    model_version = get_champion_version(args.catalog, args.schema, args.model_name)
    print(f"Deploying champion model version: {model_version}")

    # Deploy endpoint
    endpoint_name = deploy_endpoint(w, args.catalog, args.schema, args.model_name, model_version)

    # Validate endpoint is ready
    validate_endpoint(w, endpoint_name)

    print(f"Endpoint deployed successfully: {endpoint_name}")

if __name__ == "__main__":
    main()