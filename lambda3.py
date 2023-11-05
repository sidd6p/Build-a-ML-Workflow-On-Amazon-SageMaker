import json

# Define the threshold for confidence
THRESHOLD = 0.8 # You can adjust this threshold as needed

def lambda_handler(event, context):
    
    # # Grab the inferences from the event
    inferences =  event['body']['inferences']
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(confidence >= THRESHOLD for confidence in inferences)
    

    # If our threshold is met, pass our data back out of the Step Function,
    # else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")
    
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
