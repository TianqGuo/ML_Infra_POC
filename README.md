## Some basic PoC about the ML infra pipelines and related settings

### PoC1 

#### 1. The pipelines contains AWS glue, AWS lambda, AWS S3 and they are integrated by AWS step functions. 
#### 2. Please note the suitable IAM roles/users need to be setup for various groups or purposes.
#### 3. For the scalable and expandable services, components configuration need to be set correctly like S3 folder names to separate different trigger/policies.
#### 4. State machine work flow is defined in step function file.
