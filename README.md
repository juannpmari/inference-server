AIBrix Replica: Hybrid Orchestration for LLM Serving

This project replicates the core control plane and data plane architecture of a modern LLM serving system like AIBrix. It is structured to separate orchestration logic (Control Plane) from serving logic (Data Plane), enabling scalable, model-aware, and LoRA-enabled inference.

Architecture Overview

The system is split into two primary layers:

Control Plane: External Kubernetes Controllers responsible for managing the desired state of the system (scaling, LoRA configuration).

Data Plane: The user-facing services and the LLM inference engine that executes requests and exposes metrics.

The entire system is designed to run on a Kubernetes cluster, where the Control Plane manipulates Custom Resources (CRDs) and standard Deployments to manage the Data Plane.

Directory Structure and Component Descriptions

1. /control-plane

These Python scripts run as dedicated Kubernetes Controllers. They are the "brains" of the system, implementing the core orchestration logic.

File

Role

Description

lora_manager.py

LoRA Controller

This controller watches the Kubernetes API for LoraAdapter Custom Resources (CRDs). When a new adapter is requested, it communicates with the Inference Sidecar of the target LLM deployment, issuing commands to load or unload the adapter from the LLM's memory.

autoscaler.py

Horizontal Pod Autoscaler

This script watches custom load metrics (e.g., token queue length) exposed by the Inference Sidecar. Based on observed load, it dynamically scales the number of LLM Inference Pods (replicas) up or down via the Kubernetes API to maintain target performance levels.

2. /data-plane

This directory holds the application code for the container images that execute inference and manage local state.

/data-plane/gateway (The Intelligent Router)

File

Role

Description

routing_service.py

Intelligent Router

This FastAPI service is the user-facing API endpoint. It provides the OpenAI-compatible interface (/v1/chat/completions). Its primary task is Model-Based Routing: it inspects the request body's model field and dynamically routes the request to the correct Kubernetes Service that fronts the target LLM Deployment (e.g., routing traffic for llama-70b to the llama-70b-svc).

/data-plane/inference (The LLM Execution Unit)

These files are bundled into the containers that run on the GPU nodes.

File

Role

Description

engine_api.py

Inference Engine Entrypoint

This is the main container's entrypoint. It launches the underlying LLM serving technology (e.g., vLLM or equivalent) with the required model. It directly exposes the OpenAI-compatible API consumed by the Gateway, handling the actual token generation.

sidecar_runtime.py

Local Control Agent (Sidecar)

This lightweight container runs alongside the Inference Engine. It serves two main functions: 1) It exposes a Prometheus-compatible metrics endpoint (queue length, GPU usage) for the Autoscaler. 2) It exposes a local management API used by the LoRA Manager to directly execute LoRA load/unload commands on the local vLLM process.

3. /docker

Contains the Dockerfiles used to build the final container images deployed in Kubernetes.

File

Role

Description

Dockerfile.controller

Control Plane Image

Builds the image containing Python and necessary libraries for the two Control Plane controllers (lora_manager.py and autoscaler.py).

Dockerfile.gateway

Gateway Image

Builds the lightweight image for the FastAPI Intelligent Router (routing_service.py).

Dockerfile.engine

Inference Engine Image

Builds the heavy image containing the LLM framework (e.g., vLLM), CUDA drivers, PyTorch, and the base model weights, used by engine_api.py.

Dockerfile.sidecar

Sidecar Image

Builds the lightweight image containing the minimal Python dependencies needed for local monitoring and command handling (sidecar_runtime.py).

4. /k8s

All Kubernetes manifests required for declarative deployment and management.

Directory/File

Role

Description

/k8s/crds/LoraAdapter.yaml

LoRA CRD

Defines the Custom Resource Schema (kind: LoraAdapter) that the lora_manager.py controller watches to receive commands from users or external systems.

/k8s/01-rbac-roles.yaml

RBAC

Defines the necessary ServiceAccounts, Roles, and RoleBindings for the controllers and application Pods to interact with the Kubernetes API securely.

/k8s/02-gateway-deploy.yaml

Gateway Deployment

Standard Kubernetes Deployment and Service definition for the standalone Intelligent Router (routing_service.py). This is the entry point for all external user traffic.

/k8s/03-controller-deploy.yaml

Controller Deployment

Deployment for the Control Plane applications (lora_manager.py and autoscaler.py).

/k8s/llm-inference-deploy.yaml

Inference Deployment (Conceptual)

[Assuming a single-node deployment] This YAML defines the main LLM Inference Pod. It uses the Multi-Container Pod pattern to run both the heavy Inference Engine and the lightweight Sidecar Runtime simultaneously on a single GPU node.