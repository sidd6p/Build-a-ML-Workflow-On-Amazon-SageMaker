import boto3
import json
import base64

# The name of the endpoint in AWS SageMaker to be invoked
ENDPOINT_NAME = endpoint

# Create a SageMaker runtime client using Boto3. This is used to invoke the SageMaker endpoint.
runtime = boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    """
    Handles a Lambda function invocation for image processing.

    This function takes an event containing base64-encoded image data,
    decodes it, and sends it to a SageMaker endpoint for inference.
    It then returns the inference result in the response.

    Parameters:
    - event (dict): A dictionary containing the input data for the Lambda function. 
                    Expected to have a 'body' key with 'image_data' as base64-encoded string.
    - context: Information about the runtime environment. This is unused in this function.

    Returns:
    - dict: A dictionary with the status code and the body containing the inference result.
    """

    # Decode the base64-encoded image data from the event
    image = base64.b64decode(event['body']['image_data'])
    
    # Invoke the SageMaker endpoint with the decoded image
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='image/png',
                                       Body=image)
    
    # Decode the response and add it to the event dictionary under the key 'inferences'
    event["inferences"] = json.loads(response['Body'].read().decode('utf-8'))

    # Return the status code and the modified event as the response
    return {
        'statusCode': 200,
        'body': event
    }