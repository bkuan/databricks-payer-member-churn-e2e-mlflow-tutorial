# Healthcare Payer Member Churn Prediction
## End-to-End MLflow Demo & Tutorial

A comprehensive, production-ready demonstration of end-to-end machine learning workflows using **MLflow**, **Databricks Feature Engineering**, and **Unity Catalog** to predict member churn in the healthcare payer industry.

---

## Overview

This is a **complete tutorial and demo** that shows you how to build, deploy, and monitor a machine learning model to predict when healthcare plan members are likely to disenroll (churn). Using realistic healthcare claims data, you'll learn industry best practices for:

- **Feature Engineering** at scale with versioning and lineage tracking
- **Model Training** with experiment tracking and comparison
- **Model Management** using Champion/Challenger patterns
- **Model Deployment** for batch and real-time inference
- **Model Monitoring** with drift detection and observability

**Intended Audience:** Data Scientists, ML Engineers, Healthcare Analysts, and anyone learning MLflow and Databricks ML capabilities.

---

## Business Problem

**Healthcare payers** (insurance companies) face significant challenges when members disenroll:
- **Revenue loss**: Lost premiums and lifetime value
- **Acquisition costs**: Replacing members is 5-7x more expensive than retention
- **Care continuity**: Disrupted patient care and outcomes
- **Network instability**: Harder to negotiate provider contracts

**Solution:** Predict which members are at risk of churning so retention teams can intervene proactively with targeted programs.

---

## Key Features

### Complete ML Lifecycle
- End-to-end workflow from raw data to production deployment
- Real-world healthcare use case with domain-specific features
- Production-ready code with best practices throughout

### MLflow Integration
- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Version control with Unity Catalog
- **Model Serving**: Deploy models as REST APIs
- **Model Monitoring**: Track performance and drift over time

### Databricks Feature Engineering
- **Feature Store**: Centralized feature repository with lineage
- **Point-in-time Correctness**: Prevent data leakage
- **Feature Reuse**: Share features across teams and projects
- **Online/Offline Consistency**: Same features for training and serving

### Advanced ML Patterns
- **Champion/Challenger**: Safe model updates with A/B testing
- **Hyperparameter Tuning**: Automated optimization with cross-validation
- **Model Comparison**: Comprehensive evaluation frameworks
- **Explainability**: Feature importance and prediction insights

### Production Deployment
- **Batch Inference**: Score large datasets with Spark
- **Real-time Serving**: Low-latency REST API endpoints
- **SQL Integration**: Query models directly from SQL
- **Monitoring Dashboards**: Track model health and business metrics

---

## Learning Objectives

By working through this tutorial, you will:

### 1. Understand Healthcare Analytics
- Member churn patterns and risk factors
- Claims-based feature engineering
- Risk-based intervention strategies

### 2. Master MLflow Workflows
- Setting up experiments and tracking runs
- Logging models with signatures and input examples
- Managing model versions and aliases
- Deploying models to production

### 3. Leverage Databricks Capabilities
- Feature Engineering with lineage tracking
- Unity Catalog for model governance
- Lakehouse architecture for ML
- Integration with Spark for scale

### 4. Implement Production ML
- Champion/Challenger model comparison
- Hyperparameter tuning best practices
- Batch and real-time deployment patterns
- Monitoring and drift detection

### 5. Connect ML to Business Impact
- Translating predictions to interventions
- Measuring retention program ROI
- Building stakeholder-friendly dashboards

---

## Quick Start

### Prerequisites

**Required:**
- Databricks workspace (AWS, Azure, or GCP)
- Databricks Runtime **16.3.x-cpu-ml-scala2.12** or higher
- Unity Catalog enabled
- Cluster with single user or shared mode

**Data:**
- Healthcare claims data in CSV format (sample structure provided)
- Uploaded to Databricks Volumes at `/Volumes/demo/hls/payer_detailed_claims/`

**Knowledge Level:**
- Intermediate Python and SQL
- Basic understanding of machine learning concepts
- Familiarity with Databricks (helpful but not required)

### Setup Instructions

#### Step 1: Upload the Notebooks
```bash
# Option A: Import via Databricks UI
1. Log into your Databricks workspace
2. Go to Workspace → Import
3. Upload all three notebooks:
   - 1_Feature_Engineering.ipynb
   - 2_Model_Training_and_Comparison.ipynb
   - 3_Deployment_and_Monitoring.ipynb

# Option B: Use Databricks CLI
databricks workspace import \
  "1_Feature_Engineering.ipynb" \
  /Workspace/Users/<your-email>/healthcare-churn-demo/

databricks workspace import \
  "2_Model_Training_and_Comparison.ipynb" \
  /Workspace/Users/<your-email>/healthcare-churn-demo/

databricks workspace import \
  "3_Deployment_and_Monitoring.ipynb" \
  /Workspace/Users/<your-email>/healthcare-churn-demo/
```

#### Step 2: Prepare Your Data
```sql
-- Create catalog and schema structure
CREATE CATALOG IF NOT EXISTS demo;
CREATE SCHEMA IF NOT EXISTS demo.hls;

-- Upload your claims data to Databricks Volumes
-- Expected location: /Volumes/demo/hls/payer_detailed_claims/*.csv

-- Verify data upload
SELECT * FROM READ_FILES('/Volumes/demo/hls/payer_detailed_claims/*.csv', format => 'csv')
LIMIT 10;
```

#### Step 3: Configure Cluster
```yaml
Cluster Configuration:
  - Runtime: 16.3.x-cpu-ml-scala2.12 or higher
  - Access Mode: Single User or Shared
  - Unity Catalog: Enabled
  - Libraries: databricks-feature-engineering (auto-installed in notebook)
```

#### Step 4: Run the Notebooks Sequentially
1. **Start with Notebook 1**: `1_Feature_Engineering.ipynb`
   - Attach to your configured cluster
   - Run all cells (Runtime: ~10-15 minutes)
   - Verify feature tables are created

2. **Continue with Notebook 2**: `2_Model_Training_and_Comparison.ipynb`
   - Run all cells (Runtime: ~20-30 minutes)
   - Monitor experiments in the Experiments sidebar
   - View models in Unity Catalog

3. **Finish with Notebook 3**: `3_Deployment_and_Monitoring.ipynb`
   - Run all cells (Runtime: ~5-10 minutes)
   - Explore deployment options
   - Set up monitoring dashboards

**Total Runtime: approximately 35-55 minutes**

---

## Sample Data Structure

Your claims data should have the following structure:

```csv
member_id,service_date,dob,claim_type,provider_id,status,charge_amount,paid_amount,allowed_amount
M001,2024-01-15,1985-03-20,Outpatient,P123,Approved,500.00,450.00,475.00
M001,2024-02-10,1985-03-20,Pharmacy,P456,Approved,150.00,120.00,135.00
M002,2024-01-20,1990-07-15,Inpatient,P789,Denied,2500.00,0.00,2000.00
```

**Required Fields:**
- `member_id`: Unique identifier for each member
- `service_date`: Date of service (YYYY-MM-DD)
- `dob`: Member date of birth (YYYY-MM-DD)
- `claim_type`: Type of claim (Inpatient, Outpatient, Pharmacy, Professional)
- `provider_id`: Provider identifier
- `status`: Claim status (Approved, Denied, Pending)
- `charge_amount`: Billed amount
- `paid_amount`: Amount paid by payer
- `allowed_amount`: Allowed amount per contract

---

## Repository Structure

```
Healthcare End-to-End-ML-Flow/
├── README.md                                          # This file
├── 1_Feature_Engineering.ipynb                        # Part 1: Data & Features
├── 2_Model_Training_and_Comparison.ipynb              # Part 2: Model Training
├── 3_Deployment_and_Monitoring.ipynb                  # Part 3: Deployment
└── sample_data/                                       # (Optional) Sample data
```

**Note:** The tutorial is split into 3 notebooks for easier navigation and execution. Run them sequentially.

---

## Notebook Architecture

The tutorial is split into **3 notebooks** covering **11 comprehensive sections**:

---

### Notebook 1: Feature Engineering
**File:** `1_Feature_Engineering.ipynb` (20 cells)

Covers data preparation and feature creation:

### Section 1: Introduction & Setup (Cells 0-4)
- Business context and problem definition
- Technology stack overview
- Data loading and validation

### Section 2: Feature Engineering (Cells 5-6)
- Temporal features (claim frequency, recency)
- Utilization metrics (provider diversity, claim counts)
- Financial indicators (payment ratios, denial rates)
- Demographics (age calculations)

### Section 3: Feature Store Management (Cells 7-19)
- Register features in Unity Catalog
- Version control and lineage tracking
- Point-in-time feature lookups
- On-demand feature functions

---

### Notebook 2: Model Training and Comparison
**File:** `2_Model_Training_and_Comparison.ipynb` (41 cells)

Covers model development, evaluation, and selection:

### Section 4: Model Training - Champion (Cells 20-34)
- Random Forest baseline model
- MLflow experiment tracking
- Feature importance analysis
- Model registration with Feature Store lineage

### Section 5: Model Management & Versioning (Cells 35-42)
- Unity Catalog model governance
- Champion/Challenger aliasing
- Model metadata and tags
- Loading models for inference

### Section 6: MLflow Tracing & Observability (Cells 43-45)
- Prediction tracing with spans
- Input/output logging
- Performance monitoring

### Section 7: Challenger Model Training (Cells 46-49)
- Gradient Boosting alternative
- Model comparison framework
- Setting Challenger alias

### Section 8: Model Comparison & Selection (Cells 50-59)
- Comprehensive evaluation metrics (ROC AUC, F1, Precision, Recall, Brier Score)
- Hyperparameter tuning with Randomized Search CV
- Champion promotion workflow

---

### Notebook 3: Deployment and Monitoring
**File:** `3_Deployment_and_Monitoring.ipynb` (14 cells)

Covers production deployment and operations:

### Section 9: Deployment Options (Cells 60-66)
- Batch inference with Feature Engineering Client
- Real-time model serving setup
- SQL-based inference with AI_QUERY
- REST API integration

### Section 10: Monitoring & Operations (Cells 67-70)
- Model performance monitoring
- Drift detection with Lakehouse Monitoring
- Databricks SQL dashboards
- Genie (conversational AI) integration

### Section 11: Business Actions & Next Steps (Cells 71-72)
- Risk-based intervention strategies
- ROI measurement framework
- Roadmap for production deployment

---

## Learning Paths

### For Data Scientists
**Focus Areas:** Feature engineering, model training, evaluation
- Start with sections 2-4 (Feature Engineering & Model Training)
- Deep dive into feature importance and model comparison
- Experiment with different algorithms and hyperparameters

### For ML Engineers
**Focus Areas:** Deployment, monitoring, production workflows
- Focus on sections 5, 9-10 (Model Management & Deployment)
- Study the Champion/Challenger pattern
- Learn batch and real-time serving patterns

### For Healthcare Analysts
**Focus Areas:** Business context, interventions, ROI
- Read sections 1, 11 (Introduction & Business Actions)
- Review feature definitions and their healthcare meaning
- Study the risk-based intervention strategies

### For Business Stakeholders
**Focus Areas:** Business value, decision-making, impact
- Review README and section 11 (Business Actions)
- Understand how predictions drive retention programs
- Learn about ROI measurement and success metrics

---

## Expected Outcomes

After running this tutorial, you will have:

- A trained model predicting member churn with 80%+ ROC AUC
- Feature store with 15+ engineered features and full lineage
- Model registry in Unity Catalog with versioned models
- Champion/Challenger setup for safe model updates
- Deployment patterns for batch and real-time scoring
- Monitoring framework for tracking model performance
- Business insights on churn risk factors and interventions

---

## Customization Guide

### Adapt to Your Data

**1. Update Data Source**
```python
# In Cell 2, modify the data path:
df = spark.table("YOUR_CATALOG.YOUR_SCHEMA.YOUR_CLAIMS_TABLE")
```

**2. Customize Features**
```python
# In Cell 6, add domain-specific features:
features_df = features_df.withColumn(
    "your_custom_feature",
    # Your custom logic here
)
```

**3. Adjust Churn Definition**
```python
# In Cell 6, modify the churn threshold (default: 180 days):
label_features_df = features_df.withColumn(
    "disenrolled", 
    when(col("days_since_last_claim") > YOUR_THRESHOLD, 1).otherwise(0)
)
```

**4. Add New Algorithms**
```python
# In Cell 47+, add your preferred algorithm:
from sklearn.ensemble import XGBoostClassifier
xgb = XGBoostClassifier(...)
xgb.fit(X_train, y_train)
```

---

## Troubleshooting

### Common Issues

**Issue:** `databricks-feature-engineering` not found
```bash
# Solution: Install in notebook cell 3
%pip install databricks-feature-engineering
dbutils.library.restartPython()
```

**Issue:** Unity Catalog not enabled
```sql
-- Solution: Enable Unity Catalog in workspace settings
-- Admin Console → Workspace Settings → Unity Catalog
```

**Issue:** Cluster permissions error
```yaml
# Solution: Use Single User or Shared access mode
# Cluster Configuration → Access Mode → Single User
```

**Issue:** Feature table already exists
```sql
-- Solution: Drop and recreate or use different name
DROP TABLE IF EXISTS demo.hls.member_features_versioned;
```

---

## Additional Resources

### Documentation
- [Databricks MLflow Guide](https://docs.databricks.com/mlflow/index.html)
- [Feature Engineering Documentation](https://docs.databricks.com/machine-learning/feature-store/index.html)
- [Unity Catalog for ML](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html)
- [Model Serving Guide](https://docs.databricks.com/machine-learning/model-serving/index.html)

### Tutorials
- [MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html)
- [Databricks Feature Store Tutorial](https://docs.databricks.com/machine-learning/feature-store/tutorial.html)
- [Unity Catalog Best Practices](https://docs.databricks.com/data-governance/unity-catalog/best-practices.html)

### Original Source
This tutorial is adapted from the [Databricks MLflow End-to-End Classic ML Example](https://docs.databricks.com/aws/en/notebooks/source/mlflow/mlflow-classic-ml-e2e-mlflow-3.html) and customized for healthcare payer use cases.

---

## Contributing

We welcome contributions. Here's how you can help:

- **Report Issues:** Found a bug? Open an issue with details
- **Suggest Features:** Have ideas for improvements? Let us know
- **Share Use Cases:** Adapted this for another industry? Share your story
- **Improve Documentation:** See something unclear? Submit a PR

---

## License

This project is provided as-is for educational and demonstration purposes. Feel free to use and adapt it for your organization's needs.

---

## Healthcare Data Privacy Notice

**Important:** This tutorial uses synthetic/sample healthcare data for demonstration purposes only. When working with real Protected Health Information (PHI):

- Ensure HIPAA compliance in your Databricks workspace
- Use appropriate data encryption and access controls
- Implement audit logging for all data access
- Follow your organization's data governance policies
- De-identify data when possible for ML workflows
- Obtain proper approvals before using real patient data

---

## Key Takeaways

1. **MLflow + Unity Catalog** = Complete model lifecycle management
2. **Feature Store** ensures consistency between training and serving
3. **Champion/Challenger** pattern enables safe model updates in production
4. **Monitoring** is essential for long-term ML success
5. **Business impact** (reduced churn) is the ultimate success metric

---

## Support

**Questions?** Here's where to get help:

- **Databricks Community:** [community.databricks.com](https://community.databricks.com)
- **MLflow Discussions:** [GitHub Discussions](https://github.com/mlflow/mlflow/discussions)
- **Documentation:** [docs.databricks.com](https://docs.databricks.com)

---

## Getting Started

1. Clone or download all three notebooks
2. Follow the Quick Start guide above
3. Run the notebooks sequentially (1 → 2 → 3)
4. Explore results in MLflow UI and Unity Catalog

---

**Built using Databricks, MLflow, and Apache Spark**
