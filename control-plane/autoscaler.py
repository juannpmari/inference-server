# Script to handle LLM-aware load balancing
# Based on queue length, etc. it will scale up or down the number of pods for each model
# Cold-start mitigation is a feature of the autoscaler (e.g., proactive scaling or keeping minimum replicas)