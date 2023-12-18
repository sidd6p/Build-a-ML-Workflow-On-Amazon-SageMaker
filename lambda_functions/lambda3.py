import json

# Define the threshold for confidence
THRESHOLD = 0.8 

def lambda_handler(event, context):
    """
    Handler function for AWS Lambda to process inferences.

    This function takes an event containing inferences and evaluates if any of the inferences
    meet a predefined confidence threshold. If the threshold is met, it returns the event data,
    otherwise, it raises an exception.

    Parameters:
    event (dict): The event data containing 'inferences' key with a list of confidence values.
    context: The context in which the Lambda function is running, provided by AWS Lambda.

    Returns:
    dict: A dictionary with statusCode and the original event data in JSON format if the threshold is met.

    Raises:
    Exception: If none of the inferences meet the required confidence threshold.
    """

    # Extract inferences from the event body
    inferences =  event['body']['inferences']
    
    # Check if any values in our inferences are above the THRESHOLD
    # Iterates through each confidence score in inferences to check against THRESHOLD
    meets_threshold = any(confidence >= THRESHOLD for confidence in inferences)
    
    # Conditional action based on whether the threshold is met
    if meets_threshold:
        # Continue if at least one inference meets the threshold
        pass
    else:
        # Raise an exception if threshold is not met
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")
    
    # Return the original event as a JSON string with a status code of 200
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }