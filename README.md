# Deploy and monitor a machine learning workflow for Image Classification

1. __Data Staging__
    1. Extract the data from a hosting service
    2. Transform it into a usable shape and format
    3. Explore the data
    4. Filter the objects to find the label numbers for Bicycle and Motorcycles
    5. Convert the object into the dataframe
    6. Save the data to the local machine
    7. Load it into a production system
    
2. __Model Training__
    1. Create metadata for image classification on SageMaker
    2. Upload metadata to S3 using `boto3`
    3. Get algorithm using ECR image
    4. Create estimator
    5. Add hyperparameters to the estimator
    6. Add model inputs
    7. Fit the model

3. __Getting ready to deploy__
    1. Creating data capture
    2. Model deployment and creating the endpoint
    3. Instantiating a predictor
    4. Making Prediction
    
4. __Draft Lambdas and Step Function Workflow__
    1. Lambad 1: Serialize target data from S3
    2. Lambad 2: Classification of image
    3. Lambda 3: Check for confidence threshold
    
5. __Testing and Evaluation__
    1. Generating multiple test cases (event input for lambda 1)
    2. Pulling in the JSONLines data from your inferences
    3. Plotting the results


## AWS Services Used

- **Amazon S3**: Used for storing and retrieving the dataset and model outputs. It provides a scalable and secure storage solution.
- **Amazon SageMaker**: Employed for building, training, and deploying machine learning models. SageMaker's robust and flexible environment streamlines the machine learning workflow.
- **AWS Lambda**: Utilized for creating serverless functions that interact with other AWS services like SageMaker.
- **Step Functions**: Used them to chain the lmabda function to create a ML workflow

![image](https://github.com/sidd6p/Build-a-ML-Workflow-On-Amazon-SageMaker/assets/91800813/20c32540-9cd6-407e-bef5-3f7ebbec36f9)
![image](https://github.com/sidd6p/Build-a-ML-Workflow-On-Amazon-SageMaker/assets/91800813/efad0490-5f65-413e-b6e5-d3eae0e7f66b)

