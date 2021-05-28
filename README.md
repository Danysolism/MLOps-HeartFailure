---
page_type: sample
languages:
  - python
products:
  - azure
  - azure-machine-learning-service
  - azure-devops
description: ""
---

# MLOps- Heart Failure

This project presents a Machine Learining model to predict moratity by heart failure.  
The solution provides an example of how to operationalise Machine Learning development cycle. We will be using the **Azure DevOps Project** for build and release/deployment pipelines along with **Azure Machine Learning service** for model training pipeline, model management and operationalization.

By running this project, you will have the opportunity to work with the following:

| Technology                     | Objective/Reason                                                                                         |
| ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| Azure DevOps                   | Platform that provides project management, automated builds, testing and release management capabilities |
| Azure Machine Learning Service | Service that manages Machine Learning development cycle in Azure                                         |
| Random Forest Classifier       | Machine Learning model for Heart failure detection                                                       |

# MLOps with Azure Machine Learning

- [![Build Status](https://dev.azure.com/aidemos/MLOps/_apis/build/status/microsoft.MLOps?branchName=master)](https://dev.azure.com/aidemos/MLOps/_build/latest?definitionId=96?branchName=master)
- [Microsoft MLOps Python](https://github.com/Microsoft/MLOpsPython)

## What is MLOps?

Models differ from code because they have an organic shelf life and will deteriorate unless maintained. Once deployed, they can add real business value, and this gets easier when data scientists are given the tools to adopt standard engineering practices.

Machine Learning Operations (MLOps), also known as DevOps for Machine Learning, is based on providing standard engineering practices to increase the efficiency of workflows. For example, continuous integration, delivery, and deployment.

MLOps applies these principles to artificial intelligence processes, with this aim of achieving the following objectives:

- Create reproducible models and reusable training pipelines
- Simplify model packaging, validation, and deployment for quality control, A/B testing, and more.
- Explain & observe model behaviour and automate the retraining process.

## Pipelines

### Infrastructure as a Code Pipeline

This pipeline will automatically create the resource group and the services needed in Azure. It will create the resources specified in the arm template._cloud-environment.json_

### Azure Machine Learning Pipeline

This pipeline will train the model using historical data. This pipeline will train the ML model and register it in Azure ML. Azure Machine Learning provides a whole host of functionality to accelerate the end-to-end machine learning lifecycle to help deploy ML solutions quickly and robustly. With it, we can store and version the environment in which we execute the training and the scoring. Moreover, we can track the experiment runs and the metrics defined. Once the training is complete, the model artifact will trigger the DevOps release pipeline, explained below.

### Azure DevOps Release Pipeline

Once we have got a trained, registered model, we will have a Model Deployment Pipeline. We create an image of the model test it and deploy it in a web app.

## Prerequisite

- Active Azure subscription
- At least contributor access to Azure subscription

### References

- [Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning)
- [Azure ML Python SDK ](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python)
- [Azure DevOps](https://docs.microsoft.com/en-us/azure/devops/?view=vsts)
