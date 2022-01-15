import os
import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
import boto3
from sagemaker import image_uris
from sagemaker.session import Session
from sagemaker.inputs import TrainingInput
import json


sagemaker_session = sagemaker.Session()

role = get_execution_role()

region = sagemaker_session.boto_session.region_name

# initialize hyperparameters
hyperparameters = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "objective":"reg:squarederror",
        "num_round":"50",
        "objective":"binary:logistic",}


# point the address of the S3 bucket where the data is stored
bucket = 'YOUR_BUCKET_NAME' # example -> 's3://sagemaker-leader-ml'



# set an output path where the trained model will be saved
output_path = '{}/output'.format(bucket)

# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
# specify the repo_version depending on your preference.
xgboost_container = sagemaker.image_uris.retrieve(framework='xgboost', region='eu-central-1', version='latest')

# construct a SageMaker estimator that calls the xgboost-container
estimator = sagemaker.estimator.Estimator(image_uri=xgboost_container, 
                                          hyperparameters=hyperparameters,
                                          role=sagemaker.get_execution_role(),
                                          instance_count=1, 
                                          instance_type='ml.m5.2xlarge', 
                                          volume_size=5, # 5 GB 
                                          output_path=output_path)

# define the data type and paths to the training and validation datasets
content_type = "text/csv"

#train_data_url = 'sagemaker-leader-ml/ml-data-train.csv' # s3://sagemaker-leader-ml/ml-data-train.csv
#train_input = TrainingInput("s3://{}".format(train_data_url), content_type=content_type)
train = sagemaker.inputs.TrainingInput(s3_data='s3://sagemaker-leader-ml/train_v1.csv', content_type=content_type)

#test_data_url = 'sagemaker-leader-ml/ml-data-test.csv' # s3://sagemaker-leader-ml
#validation_input = TrainingInput("s3://{}".format(test_data_url), content_type=content_type)
val = sagemaker.inputs.TrainingInput(s3_data='s3://sagemaker-leader-ml/val_v1.csv', content_type=content_type)                  


# execute the XGBoost training job
estimator.fit({'train': train, 'validation': val}) # {'train': train_input, 'validation': validation_input}
#estimator.fit(train,val)


predictor = estimator.deploy(instance_type='ml.t2.medium', initial_instance_count=1, endpoint_name="test-endpoint1")

 
endpoint = 'test-endpoint1'
 
runtime = boto3.Session().client('sagemaker-runtime')
 
csv_text = '16500,6500,66,270000'
# Send CSV text via InvokeEndpoint API
response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='text/csv', Body=csv_text)

# Unpack response
result = json.loads(response['Body'].read().decode())


