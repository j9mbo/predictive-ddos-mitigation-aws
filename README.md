# Predictive–Adaptive Anti-DDoS Defense on AWS

This repository contains the experimental implementation of a **predictive–adaptive framework for DDoS mitigation** in cloud environments.  
The approach combines short-horizon traffic forecasting with real-time classification to proactively adjust **AWS WAF** policies at the **CloudFront edge**.

---

## Overview

Traditional DDoS defenses are mostly reactive: rules are triggered once malicious traffic has already degraded service performance.  
This project evaluates a **proactive defense model** that reduces the *vulnerability window* by predicting traffic surges several minutes in advance and adapting WAF actions automatically.

**Key features:**
- Short-term time series forecasting (ARIMA / Prophet) for RPS and unique IP counts.
- Supervised classification (LightGBM / Logistic Regression) to distinguish between benign flash crowds and adversarial floods.
- AWS Lambda–based control loop that updates WAF rules dynamically (rate limits, blocklists, CAPTCHA).
- Automatic rollback and cooldown logic to prevent over-blocking.
- Experimentation with synthetic traffic (load generators such as k6/Locust) to evaluate resilience.

---

## Architecture

Internet → CloudFront (WAF) → ALB → ECS Fargate (Juice Shop container)

↓

Logs to S3 (CloudFront, WAF)

↓

Athena / Glue queries

↓

Lambda controller (forecast + 
classify)

↓

Adaptive WAF policy updates via API


- **Application**: [OWASP Juice Shop](https://github.com/juice-shop/juice-shop) deployed on ECS Fargate.  
- **Edge protection**: AWS CloudFront with attached WebACL.  
- **Logging**: CloudFront & WAF logs streamed into S3.  
- **Analytics**: Amazon Athena (SQL queries over logs, 1-minute sliding windows).  
- **Models**: Prophet / ARIMA for forecasting, LightGBM / Logistic Regression for classification.  
- **Control**: Lambda triggered via EventBridge to evaluate metrics and update WAF in real time.  

---

## Deployment

The infrastructure is provisioned via **AWS CloudFormation**.

1. Deploy the stack in `us-east-1` (required for CloudFront WAF).
2. Parameters (defaults are suitable for initial experiments):
   - `ContainerImage`: `bkimminich/juice-shop:latest`
   - `Cpu`: `256`
   - `Memory`: `512`
   - `DesiredCount`: `1`
   - `StackNamePrefix`: `juice-ecs`

After creation:
- Access the application via the `CloudFrontDomain` output.
- Logs are stored in the S3 bucket `<account>-juice-ecs-logs-us-east-1`.
- ECS service and tasks can be monitored from the ECS console.

---

## Experiments

Two main types of traffic are generated for evaluation:
1. **Legitimate flash crowds** (simulated marketing spikes, high but diverse traffic).
2. **Adversarial floods** (many unique IPs, repetitive URIs, abnormal geodistribution).

Metrics:
- Lead time (minutes of early detection before peak).
- True positive / false positive classification.
- False block rate of legitimate users.
- Latency and error rates under stress.

---

## References

- National Institute of Standards and Technology (NIST), *Framework for Improving Critical Infrastructure Cybersecurity*, v1.1, 2018.  
- ENISA, *Threat Landscape 2023*, 2023.  
- AWS, *Best Practices for Anti-DDoS*, AWS WAF and Shield Developer Guide, 2025.  

---

## Disclaimer

This repository is for **academic and experimental purposes only**.  
Do not use generated attack traffic against systems you do not own or operate.  
Always restrict tests to controlled environments within your own AWS account.
