import boto3
import base64  # For encoding and decoding data in base64 format

# Initialize an S3 client using boto3
s3 = boto3.client('s3')

def lambda_handler(event, context):
    """
    Handles an AWS Lambda event by retrieving an image from an S3 bucket, 
    encoding it in base64 format, and returning the encoded data along with S3 metadata.

    Args:
    event (dict): A dictionary containing 's3_key' and 's3_bucket', 
                  which are the key and bucket name of the S3 object to process.
    context (LambdaContext): Provides runtime information to the handler.
    
    Returns:
    dict: A dictionary containing the status code, base64 encoded image data, 
          and the S3 bucket and key information.
    """

    # Extract the S3 key and bucket information from the event
    key = event["s3_key"]
    bucket = event["s3_bucket"]
    
    # Download the data from S3 to a temporary location on the Lambda environment
    boto3.resource('s3').Bucket(bucket).download_file(key, "/tmp/image.png")
    
    # Open the downloaded image file for reading in binary mode
    with open("/tmp/image.png", "rb") as f:
        # Read the file and encode its content in base64
        image_data = base64.b64encode(f.read())

    # Return a dictionary containing the status code, base64 encoded image data, and S3 object information
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []  # Placeholder for any additional processing results
        }
    }