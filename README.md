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

### PoC3

#### 1. This is the high level design to handle the streaming data and store the features for both stream data features and batch data aggregation features.
#### 2. This is only for PoC, some details like event notifications, IAM, error handling and related configurations are not included in the code. 


![image](https://github.com/TianqGuo/ML_Infra_POC/assets/52896247/b2d88baa-5e81-4703-8df5-214c6900fc6d)


### PoC4

#### 1. This is a simplified process for the infra change and the model/api changes workflow. 
#### 2. Please note in the production there are much more elements need to be considered like different environments (DEVX, UATX, PRDX), also various tags, dependencies, input parameters, KMS...
#### 3. Docker image build process can be integrated to the Jenkins or Circle CI instead of the AWS codecommit in this example.
#### 4. Model size and performance related metrics need to be properly considered.


![image](https://github.com/TianqGuo/ML_Infra_POC/assets/52896247/98960230-220f-44da-ab50-167b7a78b213)


