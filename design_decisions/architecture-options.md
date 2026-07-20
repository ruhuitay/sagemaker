# ML Inference Architecture Options

## Option 1: Lambda (Small model, bursty traffic)

```mermaid
graph LR
    Client -->|x-api-key| APIGateway[API Gateway]
    APIGateway -->|invoke| Lambda[Lambda Function<br/>onnxruntime + model.onnx]
    Lambda -->|prediction| APIGateway
    APIGateway -->|response| Client

    style Lambda fill:#f9f,stroke:#333
    style APIGateway fill:#ff9,stroke:#333
```

- Model loaded in Lambda memory
- No always-on cost
- Cold starts: 3-30 seconds
- Model size limit: ~250 MB

---

## Option 2: Lambda + Provisioned Concurrency (Small model, moderate traffic)

```mermaid
graph LR
    Client -->|x-api-key| APIGateway[API Gateway]
    APIGateway -->|invoke| Lambda[Lambda Function<br/>Provisioned Concurrency<br/>onnxruntime + model.onnx]
    Lambda -->|prediction| APIGateway
    APIGateway -->|response| Client

    style Lambda fill:#f9f,stroke:#333
    style APIGateway fill:#ff9,stroke:#333
```

- Pre-warmed Lambda instances (no cold starts)
- Paying for warm instances when idle
- Model size limit: ~250 MB
- Consistent latency

---

## Option 3: SageMaker Serverless Inference (Medium model, bursty traffic)

```mermaid
graph LR
    Client -->|x-api-key| APIGateway[API Gateway]
    APIGateway -->|invoke| ProxyLambda[Lambda<br/>Proxy]
    ProxyLambda -->|SigV4| SageMaker[SageMaker<br/>Serverless Endpoint<br/>Triton + ONNX]
    SageMaker -->|prediction| ProxyLambda
    ProxyLambda -->|response| APIGateway
    APIGateway -->|response| Client

    style SageMaker fill:#9ff,stroke:#333
    style ProxyLambda fill:#f9f,stroke:#333
    style APIGateway fill:#ff9,stroke:#333
```

- Scales to zero when idle ($0 idle cost)
- Cold starts: 1-2 minutes
- Model size: up to 4 GB
- Not suitable for sub-1-second latency requirement

---

## Option 4: SageMaker Real-Time Endpoint (Any model size, consistent latency)

```mermaid
graph LR
    Client -->|x-api-key| APIGateway[API Gateway]
    APIGateway -->|invoke| ProxyLambda[Lambda<br/>Proxy]
    ProxyLambda -->|SigV4| SageMaker[SageMaker<br/>Real-Time Endpoint<br/>Triton + ONNX<br/>Always Warm]
    SageMaker -->|prediction| ProxyLambda
    ProxyLambda -->|response| APIGateway
    APIGateway -->|response| Client

    style SageMaker fill:#9ff,stroke:#333
    style ProxyLambda fill:#f9f,stroke:#333
    style APIGateway fill:#ff9,stroke:#333
```

- Instance always running (model pre-loaded in memory)
- No cold starts after deployment
- Typical latency: 30-120 ms
- Supports GPU, dynamic batching, multi-model
- Always-on cost ($$$)
- **Best for sub-1-second latency requirement**

---

## Cost Breakdown

### Pricing Components Per Option

```mermaid
graph TD
    subgraph "Option 1: Lambda"
        L1[Invocation Cost<br/>$0.20 per 1M requests]
        L2[Compute Cost<br/>$0.0000166 per GB-sec]
        L3[API Gateway<br/>$3.50 per 1M requests]
        L4[Idle Cost: $0]
    end

    subgraph "Option 2: Lambda + Provisioned"
        P1[Provisioned Cost<br/>$0.0000041 per GB-sec<br/>24/7 whether used or not]
        P2[Invocation Cost<br/>$0.20 per 1M requests]
        P3[API Gateway<br/>$3.50 per 1M requests]
        P4[Idle Cost: $$]
    end

    subgraph "Option 3: SageMaker Serverless"
        S1[Compute Cost<br/>$0.0001 per sec of inference]
        S2[Lambda Proxy<br/>$0.20 per 1M requests]
        S3[API Gateway<br/>$3.50 per 1M requests]
        S4[Idle Cost: $0]
    end

    subgraph "Option 4: SageMaker Real-Time"
        R1[Instance Cost<br/>ml.c5.large: ~$0.102/hr<br/>= ~$74/month 24/7]
        R2[Lambda Proxy<br/>$0.20 per 1M requests]
        R3[API Gateway<br/>$3.50 per 1M requests]
        R4[Idle Cost: $$$]
    end

    style L4 fill:#9f9,stroke:#333
    style S4 fill:#9f9,stroke:#333
    style P4 fill:#ff9,stroke:#333
    style R4 fill:#f99,stroke:#333
```

### Monthly Cost Estimates (eu-central-1)

```mermaid
xychart-beta
    title "Monthly Cost by Request Volume"
    x-axis ["1K req/day", "10K req/day", "100K req/day", "1M req/day"]
    y-axis "Monthly Cost (USD)" 0 --> 300
    bar [1, 5, 45, 420]
    bar [35, 38, 65, 310]
    bar [3, 15, 130, 1200]
    bar [74, 75, 80, 110]
```

> Bars in order: Lambda, Lambda + Provisioned, SageMaker Serverless, SageMaker Real-Time

### Cost Comparison Table

| Option | Idle cost (0 requests) | 10K req/day | 100K req/day | 1M req/day | Break-even |
|--------|----------------------|-------------|--------------|------------|------------|
| **Lambda** | $0/mo | ~$5/mo | ~$45/mo | ~$420/mo | Cheapest below ~50K req/day |
| **Lambda + Provisioned** | ~$35/mo | ~$38/mo | ~$65/mo | ~$310/mo | Best for 50K-500K req/day with latency needs |
| **SageMaker Serverless** | $0/mo | ~$15/mo | ~$130/mo | ~$1200/mo | Only if bursty + model too big for Lambda |
| **SageMaker Real-Time** | ~$74/mo | ~$75/mo | ~$80/mo | ~$110/mo | Cheapest above ~500K req/day |

> Estimates assume: 128 MB Lambda memory, 50ms avg duration, ml.c5.large for SageMaker.
> Actual costs vary by region, instance type, and payload size.

### Cost Insight

```mermaid
flowchart LR
    A[Traffic Volume] --> B{< 50K req/day?}
    B -->|Yes| C{Latency sensitive?}
    B -->|No| D{< 500K req/day?}
    C -->|No| E["Lambda<br/>~$1-5/mo"]
    C -->|Yes| F["Lambda + Provisioned<br/>~$35-40/mo"]
    D -->|Yes| F
    D -->|No| G["SageMaker Real-Time<br/>~$74+/mo flat"]

    style E fill:#9f9
    style F fill:#ff9
    style G fill:#f99
```

**Key insight:** SageMaker Real-Time has high fixed cost but nearly flat scaling. At high traffic volumes, it becomes the cheapest per-request because you're spreading the instance cost across more requests. Lambda is cheap when idle but expensive at scale because you pay per invocation.

---

## Comparison Summary

```mermaid
quadrantChart
    title Cost vs Latency Consistency
    x-axis "High Idle Cost" --> "Low Idle Cost"
    y-axis "Inconsistent Latency" --> "Consistent Latency"
    quadrant-1 "Ideal (but impossible)"
    quadrant-2 "Production ML"
    quadrant-3 "Dev/Testing"
    quadrant-4 "Low traffic apps"
    "SageMaker Real-Time": [0.2, 0.9]
    "Lambda + Provisioned": [0.4, 0.8]
    "SageMaker Serverless": [0.7, 0.3]
    "Lambda": [0.9, 0.2]
```

---

## Decision Flowchart

```mermaid
flowchart TD
    A[Start] --> B{Model size > 250 MB?}
    B -->|Yes| C{Need consistent < 1s latency?}
    B -->|No| D{Need consistent < 1s latency?}
    C -->|Yes| E[SageMaker Real-Time Endpoint]
    C -->|No| F[SageMaker Serverless]
    D -->|Yes| G{Steady traffic?}
    D -->|No| H[Lambda]
    G -->|Yes| E
    G -->|No| I[Lambda + Provisioned Concurrency]

    style E fill:#9ff,stroke:#333
    style F fill:#9ff,stroke:#333
    style H fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
```
