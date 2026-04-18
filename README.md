# AWS ML Pipeline Architectures

A series of progressively more complete MLOps pipeline designs on AWS, covering batch ETL orchestration, feature store integration, real-time streaming feature ingestion, and end-to-end CI/CD model serving. Each PoC builds on the previous, culminating in a production-pattern deployment workflow.

**Use case throughout:** Customer churn prediction on telecom data (AWS SageMaker built-in XGBoost).

---

## Tech Stack

**Orchestration:** AWS Step Functions, AWS Lambda

**Data & ETL:** AWS Glue (PySpark), AWS S3, Amazon Athena

**Streaming:** Amazon Kinesis Data Streams, AWS Lambda (stream consumer), Amazon DynamoDB

**Feature Store:** SageMaker Feature Store, AWS Lambda (feature engineering + S3 upload)

**Training & Inference:** Amazon SageMaker (training jobs, hosted endpoints)

**CI/CD & Serving:** AWS CodeCommit, AWS CodeBuild, Amazon ECR, AWS CloudFormation, Docker, Flask

**Monitoring:** AWS CloudWatch (custom metrics), AWS CodeGuru Profiler

**IaC:** CloudFormation (Lambda deployment from ECR)

---

## Architecture Overview

```
PoC1  →  Step Functions + Glue ETL + SageMaker training (batch pipeline)
PoC2  →  PoC1 + Feature Store integration (Lambda feature engineering)
PoC3  →  Streaming pipeline: Kinesis → Lambda → DynamoDB + Athena (online + offline features)
PoC4  →  CI/CD: CodeCommit → CodeBuild → ECR → Lambda inference + CloudWatch monitoring
```

---

## PoC1: Batch ML Pipeline with Step Functions

**What it does:** End-to-end orchestration of a batch ML training pipeline using AWS Step Functions as the state machine.

![PoC1 Architecture](https://github.com/TianqGuo/ML_Infra_POC/assets/52896247/05cb230b-7da0-41af-a17f-3f2f5237f23e)

**Pipeline stages:**
1. **S3 trigger** — raw customer churn CSV lands in S3
2. **AWS Glue ETL** (PySpark) — reads CSV, performs 70/30 train/val split, writes partitioned output back to S3
3. **SageMaker training job** — triggered by Step Functions with S3 data paths as parameters
4. **Lambda: training status poller** — queries `describe_training_job` via boto3; Step Functions uses a wait loop until `TrainingJobStatus` is `Completed`

**Design decisions:**
- Step Functions chosen over bare Lambda chaining because it provides built-in retry logic, visual workflow monitoring, and error handling states without custom coordination code
- Glue chosen for ETL over a Lambda script because PySpark on Glue scales horizontally for larger datasets without refactoring; the job accepts S3 source/destination as parameters for reuse across environments
- IAM roles scoped per service (Glue, Lambda, SageMaker each get least-privilege roles) — S3 bucket/key naming conventions enforce separation of data between pipeline stages

**Key files:** `PoC1/code/glue_etl.py`, `PoC1/code/query_training_status.py`

---

## PoC2: Batch Pipeline + Feature Store Integration

**What it does:** Extends PoC1 by adding a Lambda-based feature engineering and feature store loading step before SageMaker training.

![PoC2 Architecture](https://github.com/TianqGuo/ML_Infra_POC/assets/52896247/c324e661-4753-4044-a979-97a9c1b373c2)

**Additional stage over PoC1:**
- **Lambda: feature engineering** — reads raw churn data, drops leaky/irrelevant columns (`Phone`, derived charge columns), casts categorical types, applies one-hot encoding (`pd.get_dummies`), writes feature-engineered CSV to S3 feature store path
- **AWS CodeGuru Profiler** instrumented on the Lambda (`@with_lambda_profiler`) to measure CPU/memory hotspots in feature transformation code

**Design decisions:**
- Feature engineering moved from Glue into Lambda because the transformation is lightweight (single-file CSV) and Lambda avoids Glue's cold-start overhead for small datasets; Glue is retained only for the heavier train/val split
- Feature store loading requires **two Lambda layers** (boto3 + pandas) — a single layer exceeds the 50 MB unzipped limit; this is called out explicitly as a common deployment gotcha
- CodeGuru profiler added to Lambda to identify whether pandas operations become a bottleneck as data scales, without changing the function interface

**Key files:** `PoC2/code/feature_retrieval.py`, `PoC2/code/glue_etl.py`, `PoC2/code/query_training_status.py`

---

## PoC3: Streaming Feature Pipeline (Online + Offline)

**What it does:** Designs a unified feature store that handles both real-time streaming features and batch-aggregated offline features — a common requirement in production ML systems where some features (e.g., "number of clicks in last 5 minutes") must be computed in real time while others (e.g., "30-day average spend") require batch aggregation.

![PoC3 Architecture](https://github.com/TianqGuo/ML_Infra_POC/assets/52896247/b2d88baa-5e81-4703-8df5-214c6900fc6d)

**Pipeline stages:**
1. **Kinesis Data Streams** — ingests real-time events (e.g., `{event_time, id, children, payment}`) simulated by `send_kinesis.py`
2. **Lambda: stream consumer** (`process_online_feature.py`) — triggered by Kinesis records; decodes base64 payload, writes online features directly to **DynamoDB** for low-latency lookup
3. **Batch aggregation path** — offline features (requiring historical aggregation) flow through S3 → Glue → SageMaker Feature Store; queried via **Athena**
4. **Feature retrieval at inference time** — joins online (DynamoDB) + offline (Feature Store/Athena) features

**Design decisions:**
- Online features written to DynamoDB (not the Feature Store) because DynamoDB provides single-digit millisecond latency for key lookups — the SageMaker Feature Store online store has higher latency and cost for high-QPS inference
- Kinesis chosen over SQS for the streaming layer because Kinesis preserves ordering within a shard and supports multiple independent consumers (e.g., one for feature storage, one for monitoring) on the same stream without message deletion
- The split between online (DynamoDB) and offline (Feature Store + Athena) features reflects a common production pattern: not all features need real-time freshness, and batch features are cheaper to store and query at training time

**Note:** Event notifications, IAM policies, error handling (DLQ, retry), and Kinesis shard configuration are architectural decisions that would be required in production but are omitted from this PoC scope.

**Key files:** `PoC3/send_kinesis.py`, `PoC3/process_online_feature.py`, `PoC3/MLOpsFeature/lambda_function.py`

---

## PoC4: CI/CD Pipeline for Model Serving

**What it does:** Implements a CI/CD workflow for deploying model inference code changes — from source commit to live Lambda endpoint — with CloudWatch monitoring of inference results.

![PoC4 Architecture](https://github.com/TianqGuo/ML_Infra_POC/assets/52896247/98960230-220f-44da-ab50-167b7a78b213)

**Pipeline stages:**
1. **AWS CodeCommit** — source trigger on push to model serving code
2. **AWS CodeBuild** (`buildspec.yaml`) — builds Docker image, tags and pushes to **Amazon ECR**
3. **Lambda code update** — CodeBuild post-build step calls `aws lambda update-function-code` with the new ECR image URI, achieving zero-downtime deployment
4. **CloudFormation** — declares the Lambda function resource (`mlops-lambda.yaml`), sourcing the container image from ECR; enables reproducible infrastructure changes via IaC
5. **Lambda inference** (`lambda_function.py`) — invokes SageMaker endpoint (`CustomerChurn`), publishes prediction result as a **CloudWatch custom metric** under `MLApp` namespace for monitoring
6. **Flask serving app** — local Docker container for offline testing before deployment

**Design decisions:**
- Lambda packaged as a **container image** (not a zip deployment) because the inference dependencies (SageMaker SDK, model artifacts) exceed the 250 MB Lambda deployment package limit; ECR-based Lambda images support up to 10 GB
- CloudBuild chosen over Jenkins/CircleCI for this PoC because it has native IAM integration with ECR/Lambda — no credential management needed; in production this would swap to Jenkins or CircleCI for more flexible pipeline control
- CloudWatch custom metric published per inference call to enable alarming on prediction distribution drift without a separate monitoring service
- CloudFormation used for Lambda declaration (rather than manual console setup) so that environment promotion (dev → UAT → prod) only requires parameter changes, not manual recreation

**Production considerations noted:** Multiple environments (DEV/UAT/PRD), KMS encryption for secrets, dependency pinning, model size/latency SLAs, and tag-based resource management are all required additions beyond this PoC scope.

**Key files:** `PoC4/codecommit/buildspec.yaml`, `PoC4/codecommit/lambda_function.py`, `PoC4/cloudformation/mlops-lambda.yaml`, `PoC4/serving/webapp/app.py`

---

## Repository Structure

```
ML_Infra_POC/
├── PoC1/                          # Batch pipeline: Step Functions + Glue + SageMaker
│   ├── code/
│   │   ├── glue_etl.py            # PySpark ETL: S3 read, train/val split, S3 write
│   │   └── query_training_status.py  # Lambda: SageMaker training job status poller
│   └── ML_Workflow_Integration.ipynb
├── PoC2/                          # PoC1 + Feature Store integration
│   ├── code/
│   │   ├── glue_etl.py
│   │   ├── feature_retrieval.py   # Lambda: feature engineering + S3 upload (CodeGuru profiled)
│   │   └── query_training_status.py
│   └── MLOps - Workflow.ipynb
├── PoC3/                          # Streaming: Kinesis → Lambda → DynamoDB + Athena
│   ├── send_kinesis.py            # Event producer: publishes records to Kinesis stream
│   ├── process_online_feature.py  # Lambda consumer: Kinesis → DynamoDB (online features)
│   └── MLOpsFeature/
│       └── lambda_function.py     # Offline feature engineering → S3
├── PoC4/                          # CI/CD: CodeCommit → CodeBuild → ECR → Lambda
│   ├── codecommit/
│   │   ├── buildspec.yaml         # CodeBuild: Docker build → ECR push → Lambda update
│   │   └── lambda_function.py     # Inference Lambda: SageMaker endpoint + CloudWatch metrics
│   ├── cloudformation/
│   │   └── mlops-lambda.yaml      # IaC: Lambda function from ECR image
│   └── serving/
│       └── webapp/app.py          # Flask app for local inference testing
└── Transformer_Encoder_From_Draft/  # Supplementary: Transformer encoder from scratch (PyTorch)
```
