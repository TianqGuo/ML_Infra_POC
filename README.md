## Some basic PoC about the ML infra pipelines and related settings

### PoC1 

#### 1. The pipelines contains AWS glue, AWS lambda, AWS S3 and they are integrated by AWS step functions. 
#### 2. Please note the suitable IAM roles/users need to be setup for various groups or purposes.
#### 3. For the scalable and expandable services, components configuration need to be set correctly like S3 folder names to separate different trigger/policies.
#### 4. State machine work flow is defined in step function file.

![image](https://github.com/TianqGuo/ML_Infra_POC/assets/52896247/05cb230b-7da0-41af-a17f-3f2f5237f23e)

### PoC2

#### 1. The only differnce between this one and PoC1 is the feature store loading step.
#### 2. Please note that the feature store loading step will require two layers as shown in the code, otherwise the workflow will fail. 

![image](https://github.com/TianqGuo/ML_Infra_POC/assets/52896247/c324e661-4753-4044-a979-97a9c1b373c2)


